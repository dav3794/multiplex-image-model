import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from .transforms import (
    OutlierPruning,
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
        unsupported_marker_behavior: Literal["error", "drop"] = "error",
        transform=None,
        preprocessing_func: Literal["arcsinh", "log1p"] | None = "arcsinh",
        scaling_func: Literal["minmax", "percentile", "global_clip"]
        | None = "percentile",
        global_scaling_bound: float = 5.0,
        denoising_func: Literal["median", "gaussian", "butterworth"]
        | None = "butterworth",
        normalization_func: Literal["zscore_ds"] | None = None,
        outlier_pruning_func: Literal["percentile_clip"] | None = None,
        operation_order: list[str] = [
            "transform",
            "preprocessing",
            "denoising",
            "scaling",
        ],
        file_extension: Literal["tiff", "npy"] = "tiff",
        skip_too_small: bool = False,
        min_image_size: int | tuple[int, int] | None = None,
        image_size_workers: int = 4,
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
            unsupported_marker_behavior (Literal['error', 'drop'], optional): Behavior when encountering unsupported markers. Defaults to 'error'.
                `error`: Raises an error if unsupported markers are found.
                `drop`: Drops unsupported markers and the corresponding channels and continues processing.
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
            skip_too_small (bool, optional): Whether to exclude images smaller than
                `min_image_size` during dataset construction. Defaults to False.
            min_image_size (int or tuple[int, int], optional): Minimum image height and
                width. An integer applies the same threshold to both dimensions.
            image_size_workers (int, optional): Number of parallel workers used to inspect
                images not found in the size metadata cache. Defaults to 4.
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

        # Load tokenized channel IDs and scan for markers not present in the tokenizer
        self.channel_ids: dict[str, torch.Tensor] = {}
        self.unsupported_markers_per_ds: dict[str, list[str]] = {}

        for dataset in panels_config["datasets"]:
            unsupported_markers = []
            tokenized_markers = []
            for marker in panels_config["markers"][dataset]:
                marker_token = marker_tokenizer.get(
                    marker, -1
                )  # Use -1 for unsupported markers
                tokenized_markers.append(marker_token)
                if marker_token == -1:
                    unsupported_markers.append(marker)

            self.channel_ids[dataset] = torch.tensor(
                tokenized_markers, dtype=torch.long
            )

            if len(unsupported_markers) > 0:
                self.unsupported_markers_per_ds[dataset] = unsupported_markers

        if self.unsupported_markers_per_ds:
            msg = f"Unsupported markers found in the panels config (dataset: {{markers}}): {self.unsupported_markers_per_ds}."
            if unsupported_marker_behavior == "error":
                raise ValueError(
                    f"{msg} Provided tokenizer does not recognize these markers. Set unsupported_marker_behavior='drop' to ignore them."
                )

            print(f"{msg} These markers will be dropped from the dataset.")

        # Load image paths for each dataset
        img_path = panels_config["paths"][split]
        self.imgs = []  # tuples of (img_path, dataset)
        for dataset in panels_config["datasets"]:
            tiffs = glob(os.path.join(img_path, dataset, "imgs", f"*.{file_extension}"))
            self.imgs.extend([(tiff, dataset) for tiff in tiffs])

        self.file_extension = file_extension
        self.read_file_func = (
            tifffile.imread if self.file_extension == "tiff" else np.load
        )

        if skip_too_small:
            if min_image_size is None:
                raise ValueError("min_image_size must be set when skip_too_small=True")
            if isinstance(min_image_size, int):
                min_height = min_width = min_image_size
            else:
                min_height, min_width = min_image_size
            if min_height <= 0 or min_width <= 0:
                raise ValueError("min_image_size dimensions must be positive")
            if image_size_workers <= 0:
                raise ValueError("image_size_workers must be positive")

            image_sizes = self._load_image_sizes(
                img_path,
                split,
                image_size_workers,
            )

            kept_images = []
            size_filter_stats = {
                dataset: {"total": 0, "kept": 0, "skipped": 0}
                for dataset in panels_config["datasets"]
            }
            for image_path, dataset in self.imgs:
                stats = size_filter_stats[dataset]
                stats["total"] += 1
                if self._has_minimum_size(
                    image_path,
                    min_height,
                    min_width,
                    image_sizes[image_path],
                ):
                    kept_images.append((image_path, dataset))
                    stats["kept"] += 1
                else:
                    stats["skipped"] += 1
            self.imgs = kept_images

            print(
                f"Image-size filtering for split '{split}' "
                f"(minimum {min_height}x{min_width}):"
            )
            for dataset, stats in size_filter_stats.items():
                print(
                    f"  {dataset}: kept {stats['kept']}, skipped {stats['skipped']}, "
                    f"total {stats['total']}"
                )
            total_images = sum(stats["total"] for stats in size_filter_stats.values())
            total_kept = sum(stats["kept"] for stats in size_filter_stats.values())
            print(
                f"  Overall: kept {total_kept}, skipped {total_images - total_kept}, "
                f"total {total_images}"
            )

        # Transformations declaration
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
            OutlierPruning(outlier_pruning_func, **outlier_pruning_kwargs)
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

    def _load_image_sizes(
        self,
        image_root: str,
        split: str,
        workers: int,
    ) -> dict[str, tuple[int, int]]:
        cache_path = Path(image_root) / ".multiplex_image_sizes.json"
        try:
            with cache_path.open() as cache_file:
                cache_data = json.load(cache_file)
            if cache_data.get("version") != 2:
                cache_data = {"version": 2, "images": {}}
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            cache_data = {"version": 2, "images": {}}

        cached_images = cache_data.setdefault("images", {})
        image_sizes = {}
        images_to_read = []
        for image_path, _ in self.imgs:
            cache_key = os.path.relpath(image_path, image_root)
            cached = cached_images.get(cache_key)
            if cached:
                image_sizes[image_path] = (cached["height"], cached["width"])
            else:
                images_to_read.append((image_path, cache_key))

        with tqdm(
            total=len(self.imgs),
            initial=len(image_sizes),
            desc=f"Checking {split} image sizes",
            unit="image",
        ) as progress:
            if images_to_read:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    sizes = executor.map(
                        self._read_image_size,
                        (image_path for image_path, _ in images_to_read),
                    )
                    for (image_path, cache_key), (height, width) in zip(
                        images_to_read,
                        sizes,
                    ):
                        image_sizes[image_path] = (height, width)
                        cached_images[cache_key] = {
                            "height": height,
                            "width": width,
                        }
                        progress.update()

        if images_to_read:
            temporary_path = cache_path.with_name(
                f"{cache_path.name}.{os.getpid()}.tmp"
            )
            with temporary_path.open("w") as cache_file:
                json.dump(cache_data, cache_file)
            os.replace(temporary_path, cache_path)

        return image_sizes

    def _read_image_size(self, image_path: str) -> tuple[int, int]:
        if self.file_extension == "tiff":
            with tifffile.TiffFile(image_path) as image_file:
                shape = image_file.series[0].shape
        else:
            shape = np.load(image_path, mmap_mode="r").shape
        if len(shape) < 2:
            return 0, 0
        height, width = shape[-2:]
        return int(height), int(width)

    def _has_minimum_size(
        self,
        image_path: str,
        min_height: int,
        min_width: int,
        image_size: tuple[int, int] | None = None,
    ) -> bool:
        """Check spatial dimensions using the configured image reader."""
        height, width = image_size or self._read_image_size(image_path)
        return height >= min_height and width >= min_width

    def _prune_unsupported_markers(
        self,
        img,
        channel_ids,
        marker_names,
        dataset,
    ) -> tuple[np.ndarray, torch.Tensor, list[str]]:
        """Remove channels corresponding to unsupported markers."""
        if dataset in self.unsupported_markers_per_ds:
            supported_channel_mask = channel_ids != -1  # Mask for supported channels
            img = img[supported_channel_mask]
            channel_ids = channel_ids[supported_channel_mask]
            marker_names = [
                marker
                for marker, is_supported in zip(marker_names, supported_channel_mask)
                if is_supported
            ]

        return img, channel_ids, marker_names

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]
        marker_names = self.ds_markers[dataset]

        img = self.read_file_func(img_path)
        img, channel_ids, marker_names = self._prune_unsupported_markers(
            img, channel_ids, marker_names, dataset
        )
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
