import os
import random
from glob import glob
from typing import Literal

import numpy as np
import tifffile
import torch
from cv2 import medianBlur
from skimage import filters
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import crop


class DatasetFromTIFF(Dataset):
    def __init__(
        self,
        panels_config: dict,
        split: str,
        marker_tokenizer: dict[str, int],
        transform=None,
        use_preprocessing: bool = True,
        use_median_denoising: bool = False,
        use_butterworth_filter: bool = True,
        use_minmax_normalization: bool = False,
        use_clip_normalization: bool = True,
        global_upper_bound: float = 5.0,
        use_global_clip_limits: bool = False,
        file_extension: Literal["tiff", "npy"] = "tiff",
    ):
        """Dataset for loading multiplex images from multiple panels.

        Args:
            panels_config (dict): Configuration dictionary for panels.
            split (str): Name of the data split (e.g., 'train', 'val', 'test').
            marker_tokenizer (dict[str, int]): Tokenizer for marker names.
            transform (_type_, optional): Transform to be applied to the images. Defaults to None.
            use_preprocessing (bool, optional): Whether to use preprocessing. Defaults to True.
            use_median_denoising (bool, optional): Whether to use median denoising. Defaults to False.
            use_butterworth_filter (bool, optional): Whether to use Butterworth filter. Defaults to True.
            use_minmax_normalization (bool, optional): Whether to use min-max normalization. Defaults to True.
            use_clip_normalization (bool, optional): Whether to use clipping normalization. Defaults to False.
            global_upper_bound (float, optional): Global upper bound for clipping normalization if `clip_limits`
                is not provided in config. Defaults to 5.0.
            use_global_clip_limits (bool, optional): Whether to use global clip limits for all datasets. Defaults to False.
            file_extension (Literal['tiff', 'npy'], optional): File extension of the images. Defaults to 'tiff'.
        """
        assert "paths" in panels_config, (
            "Panels config must have 'paths' attribute with paths of splits of the data."
        )
        assert split in panels_config["paths"], (
            f"Panels config must have '{split}' attribute with data path."
        )
        assert "datasets" in panels_config, (
            "Panels config must have 'datasets' attribute with subdirectories."
        )
        assert "markers" in panels_config, (
            "Panels config must have 'markers' attribute with channel IDs."
        )

        self.channel_ids = {
            dataset: torch.tensor(
                [
                    marker_tokenizer[marker]
                    for marker in panels_config["markers"][dataset]
                ],
                dtype=torch.long,
            )
            for dataset in panels_config["datasets"]
        }

        img_path = panels_config["paths"][split]
        self.imgs = []  # tuples of (img_path, dataset)
        for dataset in panels_config["datasets"]:
            tiffs = glob(os.path.join(img_path, dataset, "imgs", f"*.{file_extension}"))
            self.imgs.extend([(tiff, dataset) for tiff in tiffs])

        if use_global_clip_limits:
            self.clip_limits = {}
        else:
            self.clip_limits = panels_config.get("clip_limits", {})
        self.global_upper_bound = global_upper_bound

        self.transform = transform
        self.use_denoising = use_median_denoising
        self.use_butterworth = use_butterworth_filter
        self.use_minmax_normalization = use_minmax_normalization
        self.use_clip_normalization = use_clip_normalization
        self.use_preprocessing = use_preprocessing
        self.file_extension = file_extension
        self.read_file_func = (
            tifffile.imread if self.file_extension == "tiff" else np.load
        )

    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)

    @staticmethod
    def denoise(img):
        denoised_channels = [
            medianBlur(img[i].astype("float32"), 3) for i in range(img.shape[0])
        ]
        return np.stack(denoised_channels)

    @staticmethod
    def butterworth(img):
        filtered_channels = [
            filters.butterworth(img[i], cutoff_frequency_ratio=0.2, high_pass=False)
            for i in range(img.shape[0])
        ]
        return np.stack(filtered_channels)

    @staticmethod
    def norm_minmax(img):
        min_val = np.min(img, axis=(1, 2), keepdims=True)
        max_val = np.max(img, axis=(1, 2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val, img, (img - min_val) / (max_val - min_val + 1e-8)
        )
        scaled_img = np.clip(scaled_img, 0, 1)
        return scaled_img

    def norm_clip(self, img, dataset):
        """Normalize image channels to [0, 1] range using clipping."""
        upper_bound = self.clip_limits.get(dataset, self.global_upper_bound)
        img = np.clip(img, 0, upper_bound) / upper_bound
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]

        img = self.read_file_func(img_path)
        if self.use_preprocessing:
            img = self.preprocess(img)

        if self.transform:
            img = self.transform(torch.tensor(img)).numpy()

        if self.use_butterworth:
            img = self.butterworth(img)

        if self.use_denoising:
            img = self.denoise(img)

        if self.use_clip_normalization:
            img = self.norm_clip(img, dataset)

        elif self.use_minmax_normalization:
            img = self.norm_minmax(img)

        img = torch.tensor(img)

        return img, channel_ids, dataset, img_path


class PanelBatchSampler(Sampler):
    """Sampler that yields batches of indices grouped by panels."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by panel
        self.panel_to_indices = {}
        for idx, (_, panel_idx) in enumerate(dataset.imgs):
            if panel_idx not in self.panel_to_indices:
                self.panel_to_indices[panel_idx] = []
            self.panel_to_indices[panel_idx].append(idx)

        # Convert to list of (panel, indices) pairs for easier random selection
        self.panels = list(self.panel_to_indices.keys())

        self.epoch_batches = []  # Store batches for an epoch
        self._generate_batches()  # Prepare the first epoch

    def _generate_batches(self):
        """Generate batches ensuring each sample is used exactly once per epoch."""
        self.epoch_batches = []  # Reset batches for the new epoch

        # Shuffle panels if needed
        if self.shuffle:
            random.shuffle(self.panels)

        for panel in self.panels:
            indices = self.panel_to_indices[panel]

            # Shuffle indices within the panel if needed
            if self.shuffle:
                random.shuffle(indices)

            # Split indices into batches of batch_size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                self.epoch_batches.append(batch)

        # Shuffle the final batch order for diversity
        if self.shuffle:
            random.shuffle(self.epoch_batches)

    def __iter__(self):
        """Yield batches, ensuring all images are used exactly once per epoch."""
        for batch in self.epoch_batches:
            yield batch
        self._generate_batches()  # Prepare for next epoch

    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.epoch_batches)


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
