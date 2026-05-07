from pathlib import Path

import torch
from ruamel.yaml import YAML

from core.utils import load_yaml, add_model_repo
from core.utils.immuvis import (
    build_model_config_from_yaml,
    compute_cell_tokens_immuvis,
)


def setup_immuvis_hardsigmoid(
    dataset_name: str,
    repo_path: str,
    checkpoint_path: str,
    conf: str,
    panel_conf_path,
    tokenizer_path,
    scheme: str,
    device: torch.device = None,
    batch_size: int = 128,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    add_model_repo(repo_path)
    from multiplex_model.modules import MultiplexAutoencoder

    PANEL_CONFIG = YAML().load(open(Path(panel_conf_path)))
    TOKENIZER = YAML().load(open(Path(tokenizer_path)))
    MODEL_YAML = load_yaml(conf)

    CLIP_LIMITS = PANEL_CONFIG["clip_limits"]

    MARKERS = [
        m for m in PANEL_CONFIG["markers"][dataset_name] if m not in ["DNA1", "DNA2"]
    ]

    IMMUVIS_CHANNEL_IDS = torch.tensor(
        [TOKENIZER[m] for m in MARKERS],
        dtype=torch.long,
    )

    model_config = build_model_config_from_yaml(MODEL_YAML, TOKENIZER, config_name=conf)

    model = MultiplexAutoencoder(**model_config)

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True,
    )

    model = model.to(device).eval()

    def custom_preprocess(x: torch.Tensor, tid, dataset):
        """
        Replicates Virtues preprocessing (percentile clipping, log1p, gaussian blur)
        but replaces the final z-score transformation with ImmuVis clip scaling.
        """
        # 1. Clipping at 99th tissue percentile
        quantiles = dataset.quantiles.loc[tid].values[dataset.channel_mask]
        quantiles = torch.from_numpy(quantiles).float().to(x.device)[:, None, None]
        min_ = torch.zeros_like(quantiles)
        x = torch.clamp(x, min=min_, max=quantiles)

        # 2. Log1p normalization
        x = torch.log1p(x)

        # 3. Gaussian Blur
        x = dataset.gaussian_blur(x)

        # 4. Clip + Normalization (Replaces Log-Standardization)
        if CLIP_LIMITS is not None and dataset_name in CLIP_LIMITS:
            ub = float(CLIP_LIMITS[dataset_name])
        else:
            ub = 5.0

        x = torch.clamp(x, min=0.0, max=ub) / ub

        return x

    def get_channels(tid, dataset):
        return IMMUVIS_CHANNEL_IDS

    if scheme == "context":

        def prepare_fn(x_raw: torch.Tensor, mask_raw: torch.Tensor, tid: int, dataset):
            pad_size = 120
            x = torch.nn.functional.pad(
                x_raw,
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
            x = custom_preprocess(x, tid, dataset)
            return x, mask

        def compute_fn(model, x, channels, mask, device, batch_size):
            return compute_cell_tokens_immuvis(
                model, x, channels, mask, device=device, chunk_size=batch_size
            )

    elif scheme == "patch":

        def prepare_fn(x_raw: torch.Tensor, mask_raw: torch.Tensor, tid: int, dataset):
            x = custom_preprocess(x_raw, tid, dataset)

            return x, mask_raw

        def compute_fn(model, x, channels, mask, device, batch_size):
            if isinstance(x, (list, tuple)):
                x_list = [item.to(device, dtype=torch.float32) for item in x]
                channels_list = [ch.to(device) for ch in channels]
            else:
                x_list = [x.to(device, dtype=torch.float32)]
                channels_list = [channels.to(device)]

            crop_b = torch.stack(x_list)  # (B, C, H, W)
            ch_b = torch.stack(channels_list)  # (B, C)

            with torch.no_grad():
                output = model.encode(crop_b, ch_b)["output"]
                if output.dim() != 4:
                    print(
                        f"[WARNING] ImmuVis patch: encode output expected 4D, got {output.shape}"
                    )

                cell_tokens = output.mean(dim=(2, 3))  # (B, latent_dim)

                uncertainty = None
                if hasattr(model, "decode"):
                    try:
                        decoded_output = model.decode(output, ch_b)
                        _, logsigma = decoded_output.unbind(dim=-1)
                        uncertainty = torch.exp(logsigma)
                    except Exception as e:
                        print(f"[WARNING] Failed to decode for uncertainty: {e}")

            if cell_tokens.shape[0] != len(x_list):
                print(
                    f"[WARNING] ImmuVis patch: produced {cell_tokens.shape[0]} embeddings "
                    f"but had {len(x_list)} input crops"
                )

            return None, cell_tokens, uncertainty, None

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return model, prepare_fn, get_channels, compute_fn, device
