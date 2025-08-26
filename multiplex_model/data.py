import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import rotate, crop
from typing import Dict, Tuple, List
from glob import glob
import tifffile
from cv2 import medianBlur
from skimage import filters


class DatasetFromTIFF(Dataset):
    def __init__(
            self, 
            panels_config: Dict,
            split: str,
            marker_tokenizer: Dict[str, int],
            transform=None,
            use_median_denoising: bool = False,
            use_butterworth_filter: bool = True,
            use_minmax_normalization: bool = True,
            use_clip_normalization: bool = False,
        ):
        assert 'paths' in panels_config, "Panels config must have 'paths' attribute with paths of splits of the data."
        assert split in panels_config['paths'], f"Panels config must have '{split}' attribute with data path."
        assert 'datasets' in panels_config, "Panels config must have 'datasets' attribute with subdirectories."
        assert 'markers' in panels_config, "Panels config must have 'markers' attribute with channel IDs."

        
        # self.active_channels = {
        #     dataset: np.array(list(panels_config['markers'][dataset].keys()))
        #     for dataset in panels_config['datasets']
        # }
        self.channel_ids = {
            dataset: torch.tensor([
                marker_tokenizer[marker]
                for marker in panels_config['markers'][dataset]
            ], dtype=torch.long)
            for dataset in panels_config['datasets']
        }

        img_path = panels_config['paths'][split]
        self.imgs = [] # tuples of (img_path, dataset)
        for dataset in panels_config['datasets']:
            tiffs = glob(os.path.join(img_path, dataset, 'imgs', '*.tiff'))
            self.imgs.extend([(tiff, dataset) for tiff in tiffs])

        self.transform = transform
        self.use_denoising = use_median_denoising
        self.use_butterworth = use_butterworth_filter
        self.use_minmax_normalization = use_minmax_normalization
        self.use_clip_normalization = use_clip_normalization

    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)
    
    @staticmethod
    def denoise(img):
        denoised_channels = [
            medianBlur(img[i].astype('float32'), 3) 
            for i in range(img.shape[0])
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
        min_val = np.min(img, axis=(1,2), keepdims=True)
        max_val = np.max(img, axis=(1,2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val,
            img,
            (img - min_val) / (max_val - min_val + 1e-8)
        )
        scaled_img = np.clip(scaled_img, 0, 1)
        return torch.tensor(scaled_img)
    
    @staticmethod
    def norm_clip(img, upper_bound=5):
        """Normalize image channels to [0, 1] range using clipping."""
        img = np.clip(img, 0, upper_bound) / upper_bound
        return torch.tensor(img)
    
    # def strip_channels(self, img, dataset):
    #     """Strip absent channels from the image based on the dataset config."""
    #     active_channels = self.active_channels[dataset]
    #     stripped_img = img[active_channels]
    #     return stripped_img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, dataset = self.imgs[idx]
        channel_ids = self.channel_ids[dataset]

        img = tifffile.imread(img_path)
        # img = self.strip_channels(img, dataset)
        img = self.preprocess(img)

        if self.transform:
            img = self.transform(torch.tensor(img)).numpy()

        if self.use_butterworth:
            img = self.butterworth(img)

        if self.use_denoising:
            img = self.denoise(img)

        if self.use_clip_normalization:
            img = self.norm_clip(img)

        if self.use_minmax_normalization:
            img = self.norm_minmax(img)
        
        return img, channel_ids, dataset, img_path


class SegmentationDataset(Dataset):
    def __init__(
            self, 
            data_paths: List[str],
            channel_ids: List[List[int]],
            celltypes_ids: List[List[int]],
            transform=None,
            mask_labels_mapping: Dict[int, int] = None,
        ):
        assert len(data_paths) == len(channel_ids) == len(celltypes_ids), 'Each image path has to have a corresponding channel ids and celltype ids'
        self.channel_ids = channel_ids
        self.celltypes_ids = celltypes_ids
        self.transform = transform
        self.mask_labels_mapping = mask_labels_mapping
        self.imgs_with_masks = [] # tuples of (img_path, mask_path, channel_ids index)
        self.imgs = []

        for i, data_path in enumerate(data_paths):
            imgs = glob(os.path.join(data_path, 'imgs/*.tiff'))
            for img in imgs:
                mask = f'{data_path}/masks/{os.path.basename(img)}'
                if os.path.exists(mask):
                    self.imgs_with_masks.append((img, mask, i))
                    self.imgs.append((img, i))
                else:
                    print(f'Mask {mask} does not exist. Skipping image {img}')

    @staticmethod
    def preprocess(img):
        return np.arcsinh(img / 5.0)
    
    @staticmethod
    def butterworth(img):
        filtered_channels = [
            filters.butterworth(img[i], cutoff_frequency_ratio=0.2, high_pass=False)
            for i in range(img.shape[0])
        ]
        return np.stack(filtered_channels)

    @staticmethod
    def norm_minmax(img):
        min_val = np.min(img, axis=(1,2), keepdims=True)
        max_val = np.max(img, axis=(1,2), keepdims=True)
        scaled_img = np.where(
            max_val == min_val,
            img,
            (img - min_val) / (max_val - min_val + 1e-8)
        )
        return torch.tensor(scaled_img)
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, mask_path, panel_channels_idx = self.imgs_with_masks[idx]
        channel_ids = self.channel_ids[panel_channels_idx]
        celltypes_ids = self.celltypes_ids[panel_channels_idx]

        img = tifffile.imread(img_path)   
        img = self.preprocess(img)

        img = self.butterworth(img)
        img = self.norm_minmax(img)

        mask = tifffile.imread(mask_path)
        if self.mask_labels_mapping is not None:
            mask = np.vectorize(self.mask_labels_mapping.get)(mask)
            
        mask = torch.tensor(mask)
        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask, channel_ids, celltypes_ids, panel_channels_idx


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
                batch = indices[i:i + self.batch_size]
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


class SegmRotateAndCrop:
    def __init__(self, size: int, angle: int):
        self.size = size
        self.angle = angle

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly rotate the image
        angle = random.randint(-self.angle, self.angle)
        img = rotate(img, angle)
        mask = rotate(mask.unsqueeze(0), angle).squeeze(0)

        # Randomly crop the image and mask
        h, w = img.shape[-2], img.shape[-1]
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        img = crop(img, top, left, self.size, self.size)
        mask = crop(mask, top, left, self.size, self.size)

        return img, mask
    

class SegmRotateAndResize:
    def __init__(self, size: int, angle: int):
        self.size = size
        self.angle = angle

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Randomly rotate the image
        angle = random.randint(-self.angle, self.angle)
        img = rotate(img, angle)
        mask = rotate(mask.unsqueeze(0), angle).squeeze(0)

        # Resize the image and mask
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(self.size, self.size), mode='bilinear', align_corners=False
        ).squeeze(0)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(self.size, self.size), mode='nearest'
        ).squeeze(0).squeeze(0)

        return img, mask


class SegmTestCrop:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Crop the image and mask
        h, w = img.shape[-2], img.shape[-1]
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        img = crop(img, top, left, self.size, self.size)
        mask = crop(mask, top, left, self.size, self.size)

        return img, mask


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
