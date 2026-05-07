import numpy as np
import torch
from cv2 import medianBlur
from skimage.filters import butterworth

from core.utils.virtues import _assign_patch_tokens_to_cells, _get_uniform_crops


def build_model_config_from_yaml(
    yaml_cfg: dict, tokenizer: dict, config_name: str = ""
):
    def get_hyperkernel(cfg):
        if "hyperkernel" in cfg:
            return cfg["hyperkernel"]
        if "hyperkernel_config" in cfg:
            return cfg["hyperkernel_config"]
        raise KeyError("Missing hyperkernel / hyperkernel_config")

    encoder_yaml = yaml_cfg["encoder"]
    decoder_yaml = yaml_cfg["decoder"]

    encoder_config = {
        "ma_layers_blocks": encoder_yaml["ma_layers_blocks"],
        "ma_embedding_dims": encoder_yaml["ma_embedding_dims"],
        "pm_layers_blocks": encoder_yaml["pm_layers_blocks"],
        "pm_embedding_dims": encoder_yaml["pm_embedding_dims"],
        "hyperkernel_config": get_hyperkernel(encoder_yaml),
    }

    if "use_mask_token" in encoder_yaml:
        encoder_config["use_mask_token"] = encoder_yaml["use_mask_token"]

    if "mask_token_init" in encoder_yaml:
        encoder_config["mask_token_init"] = encoder_yaml["mask_token_init"]

    if "use_hyperkernel_act" in encoder_yaml:
        encoder_config["use_hyperkernel_act"] = encoder_yaml["use_hyperkernel_act"]

    if "beta" in config_name.lower():
        encoder_config["use_latent_norm"] = True

    if "encoder_type" in encoder_yaml:
        encoder_config["encoder_type"] = encoder_yaml["encoder_type"]

    decoder_config = {
        "decoded_embed_dim": decoder_yaml["decoded_embed_dim"],
        "num_blocks": decoder_yaml["num_blocks"],
        "hyperkernel_config": get_hyperkernel(decoder_yaml),
    }

    if "block_type" in decoder_yaml:
        decoder_config["block_type"] = decoder_yaml["block_type"]

    return {
        "num_channels": len(tokenizer),
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
    }


def apply_immuvis_preprocessing(
    img: torch.Tensor,  # shape (C, H, W) float32
    dataset_name: str = "hn",
    use_arcsinh: bool = True,
    use_butterworth: bool = True,
    use_median_denoising: bool = False,
    use_clip_normalization: bool = True,
    use_minmax_normalization: bool = False,
    clip_upper_bound: float = 5.0,
    clip_limits: dict | None = None,
) -> torch.Tensor:
    """
    Replicates the exact preprocessing chain used in ImmuVis DatasetFromTIFF
    """
    img = img.float().numpy()

    if use_arcsinh:
        img = np.arcsinh(img / 5.0)

    if use_butterworth:
        filtered = []
        for ch in range(img.shape[0]):
            filt = butterworth(img[ch], cutoff_frequency_ratio=0.2, high_pass=False)
            filtered.append(filt)
        img = np.stack(filtered)

    if use_median_denoising:
        denoised = []
        for ch in range(img.shape[0]):
            d = medianBlur(img[ch].astype(np.float32), ksize=3)
            denoised.append(d)
        img = np.stack(denoised)

    if use_clip_normalization:
        if clip_limits is not None and dataset_name in clip_limits:
            ub = clip_limits[dataset_name]
        else:
            ub = clip_upper_bound
        img = np.clip(img, 0, ub) / ub

    elif use_minmax_normalization:
        minv = np.min(img, axis=(1, 2), keepdims=True)
        maxv = np.max(img, axis=(1, 2), keepdims=True)
        denom = maxv - minv + 1e-8
        img = np.where(maxv == minv, img, (img - minv) / denom)
        img = np.clip(img, 0, 1)

    return torch.from_numpy(img).float()


def compute_cell_tokens_immuvis(
    model,  # "MultiplexAutoencoder"
    img: torch.Tensor,
    channel: torch.Tensor,
    segmentation_mask: torch.Tensor,
    device="cuda",
    crop_size=128,
    patch_size=8,
    stride=42,
    chunk_size=128,
) -> torch.Tensor:
    """
    Compute cell tokens using ImmuVis for patch embeddings.
    Similar to original, but embed 8x8 patches with ImmuVis.
    """
    crops, indices = _get_uniform_crops(img, stride, crop_size=crop_size)
    crops = [crops.to(device) for crops in crops]
    channel = channel.to(device)

    cell_tokens = {}
    weights = {}
    for i in range(0, len(crops), chunk_size):
        crops_chunk = crops[i : i + chunk_size]
        for crop_idx, crop in enumerate(crops_chunk):
            patches = []
            for pr in range(0, crop_size, patch_size):
                for pc in range(0, crop_size, patch_size):
                    patch = crop[:, pr : pr + patch_size, pc : pc + patch_size]
                    patches.append(patch)
            patches = torch.stack(patches).to(device)

            num_patches = patches.shape[0]
            ch_ids_chunk = channel.repeat(num_patches, 1).to(device)
            with torch.no_grad():
                # with torch.amp.autocast("cuda", enabled=True):
                # no autocasting for immuvis
                encoder_output = model.encode(patches, ch_ids_chunk)["output"]

                patch_tokens = encoder_output.mean(dim=(2, 3))
                patch_tokens = torch.nn.functional.normalize(
                    patch_tokens, p=2, dim=1
                )  # immuvis specific
                patch_tokens = patch_tokens.cpu()

            grid_h = crop_size // patch_size
            grid_w = crop_size // patch_size
            patch_tokens = patch_tokens.view(grid_h, grid_w, -1)

            row, col = indices[i + crop_idx]
            crop_mask = segmentation_mask[row : row + crop_size, col : col + crop_size]

            crop_cell_tokens, crop_weights = _assign_patch_tokens_to_cells(
                patch_tokens, crop_mask, patch_size
            )
            for cell_id, tokens in crop_cell_tokens.items():
                if cell_id not in cell_tokens:
                    cell_tokens[cell_id] = []
                    weights[cell_id] = []
                cell_tokens[cell_id].extend(tokens)
                weights[cell_id].extend(crop_weights[cell_id])

    avg_cell_tokens = {}
    for cell_id, tokens in cell_tokens.items():
        tokens = torch.stack(tokens)
        weights_array = torch.tensor(
            weights[cell_id], dtype=tokens.dtype, device=tokens.device
        )
        weights_array = weights_array / weights_array.sum()
        avg_cell_token = torch.sum(tokens * weights_array[:, None], dim=0)
        avg_cell_tokens[cell_id] = avg_cell_token
    cell_ids = list(avg_cell_tokens.keys())
    cell_ids.sort()
    cell_tokens = torch.stack([avg_cell_tokens[cell_id] for cell_id in cell_ids])
    return cell_ids, cell_tokens, None, indices
