import os
import random
from glob import glob
from typing import Literal

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, Sampler

from .transforms import (
    OutlierPrunning,
    Preprocessing,
    Denoising,
    Scaling,
    Normalization,
    TorchvisionTransform,
    Pipeline,
    Identity,
)

class MultiplexDataset(Dataset):
    def __init__(
        self,
        panels_config: dict,
        split: str,
        marker_tokenizer: dict[str, int],
        transform=None,
        preprocessing_func: Literal["arcsinh", "log1p"] | None = "arcsinh",
        scaling_func: Literal["minmax", "percentile", "global_clip"] | None = "percentile",
        global_scaling_bound: float = 5.0,
        denoising_func: Literal["median", "gaussian", "butterworth"] | None = "butterworth",
        normalization_func: Literal["zscore_ds"] | None = None,
        outlier_pruning_func: Literal["percentile_clip"] | None = None,
        operation_order: list[str] = [
            "transform",
            "preprocessing",
            "denoising",
            "scaling",
        ],
        file_extension: Literal["tiff", "npy"] = "tiff",
        preprocessing_kwargs: dict = {},
        denoising_kwargs: dict = {},
        scaling_kwargs: dict = {},
        normalization_kwargs: dict = {},
        outlier_pruning_kwargs: dict = {},

    ):
        """Dataset for loading multiplex images from multiple panels.
        The order of operations is given by `operation_order`.

        Args:
            panels_config (dict): Configuration dictionary for panels.
            split (str): Name of the data split (e.g., 'train', 'val', 'test').
            marker_tokenizer (dict[str, int]): Tokenizer for marker names.
            transform (_type_, optional): Transform to be applied to the images. Defaults to None.
            preprocessing_func (Literal['arcsinh', 'log1p'], optional): Function to use for preprocessing. 
                Skips preprocessing if None. Defaults to 'arcsinh'.
            denoising_func (Literal['median', 'gaussian', 'butterworth'], optional): Function to use for denoising. 
            Skips denoising if None. Defaults to 'butterworth'.
            scaling_func (Literal['minmax', 'percentile', 'global_clip'], optional): Function to use for scaling. Skips scaling if None. Defaults to 'percentile'.
            global_scaling_bound (float, optional): Global upper bound for scaling if `global_clip` is chosen as `scaling_func`. Defaults to 5.0.
            normalization_func (Literal['zscore_ds'], optional): Function to use for normalization. Skips normalization if None. Defaults to 'zscore_ds'.
            outlier_pruning_func (Literal['percentile_clip'], optional): Function to use for outlier pruning. Skips outlier pruning if None. Defaults to 'percentile_clip'.
            operation_order (list[str], optional): Order of operations to be applied.
                Defaults to ['transform', 'preprocessing', 'denoising', 'scaling', 'normalization'].
            file_extension (Literal['tiff', 'npy'], optional): File extension of the images. Defaults to 'tiff'.
            preprocessing_kwargs (dict, optional): Additional keyword arguments for preprocessing function. Defaults to {}.
            denoising_kwargs (dict, optional): Additional keyword arguments for denoising function. Defaults to {}.
            scaling_kwargs (dict, optional): Additional keyword arguments for scaling function. Defaults to {}.
            normalization_kwargs (dict, optional): Additional keyword arguments for normalization function. Defaults to {}.
            outlier_pruning_kwargs (dict, optional): Additional keyword arguments for outlier pruning function. Defaults to {}.
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

        self.ds_markers = panels_config["markers"]

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

        ds_percentiles = panels_config.get("clip_limits", None)
        ds_marker_stats = panels_config.get("marker_stats", None)

        self.preprocess = (
            Preprocessing(preprocessing_func, **preprocessing_kwargs)
            if preprocessing_func
            else Identity()
        )
        self.denoise = (
            Denoising(denoising_func, **denoising_kwargs)
            if denoising_func
            else Identity()
        )
        self.scale = (
            Scaling(
                scaling_func, ds_percentiles, global_scaling_bound, **scaling_kwargs
            )
            if scaling_func
            else Identity()
        )
        self.norm = (
            Normalization(normalization_func, ds_marker_stats, **normalization_kwargs)
            if normalization_func
            else Identity()
        )
        self.outlier_pruning = (
            OutlierPrunning(outlier_pruning_func, **outlier_pruning_kwargs)
            if outlier_pruning_func
            else Identity()
        )
        self.transform = TorchvisionTransform(transform) if transform else Identity()

        self.pipeline = Pipeline(
            transforms={
                "preprocessing": self.preprocess,
                "denoising": self.denoise,
                "scaling": self.scale,
                "normalization": self.norm,
                "transform": self.transform,
                "outlier_pruning": self.outlier_pruning,
            },
            operation_order=operation_order,
        )

        self.file_extension = file_extension
        self.read_file_func = (
            tifffile.imread if self.file_extension == "tiff" else np.load
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]
        marker_names = self.ds_markers[dataset]

        img = self.read_file_func(img_path)
        img = self.pipeline(img, dataset=dataset, marker_names=marker_names)

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
