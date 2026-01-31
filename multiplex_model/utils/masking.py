"""Masking utilities for channel and spatial masking of multiplex images."""

import numpy as np
import torch


def apply_channel_masking(
    img: torch.Tensor,
    channel_ids: torch.Tensor,
    min_channels_frac: float = 0.75,
    fully_masked_channels_max_frac: float = 0.5,
    apply_channel_subset_sampling: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply channel masking to the input images.

    This function performs two types of masking:
    1. Random channel subset selection (keeping min_channels_frac to 100% of channels) - optional
    2. Full channel masking (dropping up to fully_masked_channels_max_frac of remaining channels)

    Args:
        img (torch.Tensor): Input images [B, C, H, W]
        channel_ids (torch.Tensor): Channel IDs [B, C]
        min_channels_frac (float): Minimum fraction of channels to keep in initial sampling
        fully_masked_channels_max_frac (float): Maximum fraction of channels to fully mask
        apply_channel_subset_sampling (bool): Whether to apply channel subset sampling before masking.
            Set to False for validation/testing to only do full channel masking.

    Returns:
        tuple: (original_img, original_channel_ids, masked_img, active_channel_ids)
            - original_img: Image with sampled channels [B, C_sampled, H, W]
            - original_channel_ids: IDs of sampled channels [B, C_sampled]
            - masked_img: Image with some channels fully dropped [B, C_active, H, W]
            - active_channel_ids: IDs of active (non-dropped) channels [B, C_active]
    """
    batch_size, num_channels, H, W = img.shape

    # Step 1: Randomly sample a subset of channels to keep (optional)
    num_sampled_channels = num_channels
    if apply_channel_subset_sampling:
        min_channels = int(np.ceil(num_channels * min_channels_frac))
        num_sampled_channels = np.random.randint(min_channels, num_channels + 1)

        if num_sampled_channels < num_channels:
            new_img = []
            new_channel_ids = []
            for b_i in range(batch_size):
                channels_subset_idx = torch.randperm(num_channels)[:num_sampled_channels]
                new_img.append(img[b_i : b_i + 1, channels_subset_idx, :, :])
                new_channel_ids.append(channel_ids[b_i : b_i + 1, channels_subset_idx])
            img = torch.cat(new_img, dim=0)
            channel_ids = torch.cat(new_channel_ids, dim=0)

    # Step 2: Sample full channels to mask (drop)
    max_channels_to_mask = int(np.ceil(num_sampled_channels * fully_masked_channels_max_frac))
    num_channels_to_mask = np.random.randint(1, max_channels_to_mask + 1)

    masked_img = []
    active_channel_ids = []
    for b_i in range(batch_size):
        channels_to_keep = torch.randperm(num_sampled_channels)[num_channels_to_mask:]
        masked_img.append(img[b_i : b_i + 1, channels_to_keep, :, :])
        active_channel_ids.append(channel_ids[b_i : b_i + 1, channels_to_keep])

    masked_img = torch.cat(masked_img, dim=0)  # [B, C_active, H, W]
    active_channel_ids = torch.cat(active_channel_ids, dim=0)  # [B, C_active]

    return img, channel_ids, masked_img, active_channel_ids


def apply_spatial_masking(
    img: torch.Tensor,
    spatial_masking_ratio: float = 0.6,
    mask_patch_size: int = 8,
    mask_fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply spatial patch masking to the input images.

    Args:
        img (torch.Tensor): Input images [B, C, H, W]
        spatial_masking_ratio (float): Fraction of patches to mask
        mask_patch_size (int): Size of each square patch to mask
        mask_fill_value (float): Value to fill in masked pixels (default: 0.0)

    Returns:
        tuple: (masked_img, pixel_mask)
            - masked_img: Image with masked patches set to zero [B, C, H, W]
            - pixel_mask: Boolean mask indicating masked pixels [B, C, H, W]
    """
    batch_size, num_channels, H, W = img.shape

    # Create patch-level mask
    h = w = H // mask_patch_size
    mask = torch.rand((batch_size, num_channels, h, w), device=img.device)
    mask = mask <= spatial_masking_ratio

    # Upsample mask to pixel level
    pixel_mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)

    # Apply mask
    masked_img = img.clone()
    masked_img[pixel_mask] = mask_fill_value

    return masked_img, pixel_mask
