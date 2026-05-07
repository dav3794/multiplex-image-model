import os
from pathlib import Path

from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML

from core.utils import add_model_repo
from core.utils.immuvis import (
    apply_immuvis_preprocessing,
    _assign_patch_tokens_to_cells,
    _get_uniform_crops,
)


def pad_image_to_target(x, target_size=224):
    """
    Expects x of shape (B, C, H, W).
    Center-pads spatial dimensions with 0 to reach target_size.
    """
    B, C, H, W = x.shape
    pad_h = max(0, target_size - H)
    pad_w = max(0, target_size - W)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return F.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )


def setup_eva(
    dataset_name,
    conf,
    repo_path,
    embeddings_path,
    panel_conf_path,
    scheme: str,
    device=None,
    batch_size=128,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("marker_embeddings", exist_ok=True)
    target_symlink = "marker_embeddings/GenePT_embedding.pkl"

    if not os.path.exists(target_symlink):
        if os.path.exists(embeddings_path):
            os.symlink(embeddings_path, target_symlink)
        else:
            raise RuntimeError(
                f"[ERROR] GenePT embedding not found at {embeddings_path}."
            )

    add_model_repo(repo_path)
    from dotenv import load_dotenv
    from Eva.utils import load_from_hf

    load_dotenv()

    eva_conf = OmegaConf.load(conf)
    model = load_from_hf(repo_id="yandrewl/Eva", conf=eva_conf, device=device)
    model = model.to(device).eval()

    PANEL_CONFIG = YAML().load(open(Path(panel_conf_path)))

    CLIP_LIMITS = PANEL_CONFIG["clip_limits"]

    MARKERS = [
        m for m in PANEL_CONFIG["markers"][dataset_name] if m not in ["DNA1", "DNA2"]
    ]

    def get_channels(tid: int, dataset):
        return MARKERS

    if scheme == "context":

        def prepare_fn(x_raw: torch.Tensor, mask_raw: torch.Tensor, tid: int, dataset):
            x = apply_immuvis_preprocessing(
                x_raw,
                dataset_name=dataset_name,
                use_arcsinh=True,
                use_butterworth=True,
                use_median_denoising=False,
                use_clip_normalization=True,
                use_minmax_normalization=False,
                clip_upper_bound=5.0,
                clip_limits=CLIP_LIMITS,
            )

            pad_size = 112
            x = torch.nn.functional.pad(
                x,
                pad=(pad_size, pad_size, pad_size, pad_size),
                mode="constant",
                value=0,
            )
            mask = (
                torch.nn.functional.pad(
                    mask_raw,
                    pad=(pad_size, pad_size, pad_size, pad_size),
                    mode="constant",
                    value=0,
                )
                if mask_raw is not None
                else None
            )
            return x, mask

        def compute_fn(model, x, channels, mask, device, batch_size):
            crop_size = 224
            stride = 112

            eva_patch_size = getattr(model, "token_size")

            crops, indices = _get_uniform_crops(x, stride, crop_size=crop_size)
            crops = [c.to(device) for c in crops]

            cell_tokens = {}
            weights = {}

            for i in range(0, len(crops), batch_size):
                crops_chunk = crops[i : i + batch_size]

                # [B, C, 224, 224] -> [B, 224, 224, C]
                x_batch = torch.stack(crops_chunk).permute(0, 2, 3, 1)
                bms = [channels for _ in range(x_batch.shape[0])]

                with torch.no_grad():
                    features = model.extract_features(
                        patch=x_batch,
                        bms=bms,
                        device=device,
                        cls=False,
                        channel_mode="full",
                        pool=False,
                    )

                B, N, D = features.shape
                grid_dim = int(N**0.5)

                patch_tokens_batch = features.view(B, grid_dim, grid_dim, D).cpu()

                for b_idx in range(B):
                    row, col = indices[i + b_idx]
                    crop_mask = mask[row : row + crop_size, col : col + crop_size]

                    crop_cell_tokens, crop_weights = _assign_patch_tokens_to_cells(
                        patch_tokens_batch[b_idx], crop_mask, patch_size=eva_patch_size
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

            if len(cell_ids) == 0:
                embed_dim = features.shape[-1] if "features" in locals() else 768
                return [], torch.empty((0, embed_dim)), None, indices

            out_tokens = torch.stack([avg_cell_tokens[cell_id] for cell_id in cell_ids])
            return cell_ids, out_tokens, None, indices

    elif scheme == "patch":

        def prepare_fn(x_raw, mask_raw, tid, dataset):
            x = apply_immuvis_preprocessing(
                x_raw,
                dataset_name=dataset_name,
                use_arcsinh=True,
                use_butterworth=True,
                use_median_denoising=False,
                use_clip_normalization=True,
                use_minmax_normalization=False,
                clip_upper_bound=5.0,
                clip_limits=CLIP_LIMITS,
            )

            return x, mask_raw

        def compute_fn(model, x, channels, mask, device, batch_size):
            if isinstance(x, (list, tuple)):
                x_batch = torch.stack([item.to(device) for item in x])
            else:
                x_batch = x.to(device)

            x_batch = pad_image_to_target(x_batch, 224)  # Eva expects 224x224 inputs

            x_batch = x_batch.permute(0, 2, 3, 1)  # Eva expects channels last

            with torch.no_grad():
                features = model.extract_features(
                    patch=x_batch,
                    bms=channels,
                    device=device,
                    cls=False,
                    channel_mode="full",
                    pool=True,
                )

            return None, features, None, None

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return model, prepare_fn, get_channels, compute_fn, device
