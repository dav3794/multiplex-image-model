from typing import Callable
from abc import ABC

import numpy as np
import torch
from cv2 import medianBlur, GaussianBlur
from skimage.filters import butterworth
from torchvision.transforms.functional import crop

class ImageFunc(ABC):
    """Abstract base class for image transforms."""

    available_funcs = set()

    def __init__(self, func_name: str, **kwargs):
        assert (
            func_name in self.available_funcs or func_name is None
        ), (
            f"Function '{func_name}' is not supported. "
            f"Available functions: {self.available_funcs}"
        )
        self.func_name = func_name
        self.func_kwargs = kwargs
        self._func = getattr(self, func_name)

    def __call__(self, img, **img_kwargs):
        return self._func(img, **img_kwargs, **self.func_kwargs)

class OutlierPrunning(ImageFunc):
    """Class for outlier removal"""

    available_funcs = {"percentile_clip"}

    @staticmethod
    def percentile_clip(x, p=99, **kwargs):
        upper_bound = np.percentile(x, p, axis=(1, 2), keepdims=True)
        return np.clip(x, None, upper_bound)


class Preprocessing(ImageFunc):
    """Class for applying preprocessing functions to images."""

    available_funcs = {"arcsinh", "log1p"}

    @staticmethod
    def arcsinh(x, scale=5.0, **kwargs):
        return np.arcsinh(x / scale)

    @staticmethod
    def log1p(x, **kwargs):
        return np.log1p(x)


class Denoising(ImageFunc):
    """Class for applying denoising functions to images."""

    available_funcs = {"median", "gaussian", "butterworth"}

    @staticmethod
    def median(img, kernel_size=3, **kwargs):
        return medianBlur(img.astype("float32"), kernel_size)

    @staticmethod
    def gaussian(img, kernel_size=3, sigma=1.0, **kwargs):
        return GaussianBlur(img.astype("float32"), (kernel_size, kernel_size), sigma)

    @staticmethod
    def butterworth(img, cutoff_frequency_ratio=0.2, high_pass=False, **kwargs):
        return butterworth(
            img, cutoff_frequency_ratio=cutoff_frequency_ratio, high_pass=high_pass
        )

    def __call__(self, img, **img_kwargs):
        C = img.shape[0]
        denoised_channels = [self._func(img[i], **self.func_kwargs) for i in range(C)]
        return np.stack(denoised_channels)


class Scaling(ImageFunc):
    """Class for applying scaling functions to images."""

    available_funcs = {"minmax", "percentile", "global_clip"}

    def __init__(
        self,
        func_name: str,
        ds_percentiles: dict[str, float] = None,
        global_scaling_bound: float = 5.0,
        **kwargs,
    ):
        super().__init__(func_name, **kwargs)
        self.ds_percentiles = ds_percentiles
        self.global_scaling_bound = global_scaling_bound

    @staticmethod
    def minmax(img, **kwargs):
        """Scale image channels to [0, 1] range using min-max normalization."""
        min_val = np.min(img, axis=(1, 2), keepdims=True)
        max_val = np.max(img, axis=(1, 2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val, img, (img - min_val) / (max_val - min_val + 1e-8)
        )
        return np.clip(scaled_img, 0, 1)

    def percentile(self, img, dataset, **kwargs):
        """Scale image channels to [0, 1] range using dataset-specific percentile clipping."""
        upper_bound = self.ds_percentiles.get(dataset, self.global_scaling_bound)
        return np.clip(img, 0, upper_bound) / upper_bound

    def global_clip(self, img, **kwargs):
        """Scale image channels by constant global value."""
        return np.clip(img, 0, self.global_scaling_bound) / self.global_scaling_bound


class Normalization(ImageFunc):
    """Class for applying normalization functions to images."""

    available_funcs = {"zscore_ds"}

    def __init__(
        self,
        func_name: str,
        ds_marker_stats: dict[str, dict[str, tuple[int, int]]],
        **kwargs,
    ):
        super().__init__(func_name, **kwargs)
        # ds_marker_stats should be a dict of the form {dataset: {marker: (mean, std)}}
        self.ds_marker_stats = ds_marker_stats

    def zscore_ds(self, img, dataset, marker_names, **kwargs):
        normalized_channels = []
        for i, marker in enumerate(marker_names):
            mean, std = self.ds_marker_stats[dataset][marker]
            normalized_channel = (img[i] - mean) / std
            normalized_channels.append(normalized_channel)
        return np.stack(normalized_channels)


class TorchvisionTransform:
    """Class for applying torchvision transforms to images."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, **kwargs):
        return self.transform(torch.tensor(img)).numpy()


class Identity:
    """Identity transform that returns the input image unchanged."""

    def __call__(self, img, **kwargs):
        return img


class Pipeline:
    """Class for applying a sequence of image transformations in a specified order."""

    def __init__(self, transforms: dict[str, Callable], operation_order: list[str]):
        self.transforms = transforms
        self.operation_order = operation_order

        for operation in self.operation_order:
            assert operation in self.transforms, (
                f"Operation '{operation}' is not defined in transforms. "
                f"Available transforms: {list(self.transforms.keys())}"
            )

    def __call__(self, img, dataset=None, marker_names=None):
        for operation in self.operation_order:
            transform = self.transforms[operation]
            img = transform(img, dataset=dataset, marker_names=marker_names)
        return img


class TestCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Crop the image and mask
        h, w = img.shape[-2], img.shape[-1]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        img = crop(img, top, left, self.size, self.size)

        return img
