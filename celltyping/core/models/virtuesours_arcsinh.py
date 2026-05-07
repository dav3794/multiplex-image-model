from pathlib import Path

import torch
from ruamel.yaml import YAML
from safetensors import safe_open

from core.utils import add_model_repo
from core.utils.immuvis import apply_immuvis_preprocessing


def setup_virtuesours_arcsinh(
    dataset_name: str,
    marker_embeddings,
    repo_path,
    virtues_conf,
    panel_conf_path,
    ckpt_path,
    scheme: str,
    device=None,
    batch_size=128,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    add_model_repo(repo_path)
    from modules.multiplex_virtues import MultiplexVirtues
    from utils.cell_tokens import compute_cell_tokens

    PANEL_CONFIG = YAML().load(open(Path(panel_conf_path)))
    CLIP_LIMITS = PANEL_CONFIG["clip_limits"]

    model = MultiplexVirtues(
        use_default_config=False,
        custom_config=None,
        prior_bias_embeddings=marker_embeddings,
        prior_bias_embedding_type="esm",
        prior_bias_embedding_fusion_type="add",
        patch_size=virtues_conf.model.patch_size,
        model_dim=virtues_conf.model.model_dim,
        feedforward_dim=virtues_conf.model.feedforward_dim,
        encoder_pattern=virtues_conf.model.encoder_pattern,
        num_encoder_heads=virtues_conf.model.num_encoder_heads,
        decoder_pattern=virtues_conf.model.decoder_pattern,
        num_decoder_heads=virtues_conf.model.num_decoder_heads,
        num_hidden_layers=virtues_conf.model.num_decoder_hidden_layers,
        positional_embedding_type=virtues_conf.model.positional_embedding_type,
        dropout=virtues_conf.model.dropout,
        group_layers=virtues_conf.model.group_layers,
        norm_after_encoder_decoder=virtues_conf.model.norm_after_encoder_decoder,
        verbose=False,
    )

    weights = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    model.load_state_dict(weights)
    model = model.to(device).eval()

    def get_channels(tid: int, dataset):
        return dataset.get_marker_indices()

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

            pad_size = 120
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
            return compute_cell_tokens(
                model, x, channels, mask, device=device, chunk_size=batch_size
            )

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
                x = [item.to(device) for item in x]
                channels = [ch.to(device) for ch in channels]
            else:
                x = x.to(device)
                channels = channels.to(device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
                    encoder_output = model.encoder.forward_list(
                        x,  # list of crops
                        channels,  # list of marker indices
                        multiplex_mask=None,
                    )

            patch_summaries = encoder_output.patch_summary_tokens
            if len(patch_summaries) != len(x):
                print(
                    f"[WARNING] Number of patch summaries ({len(patch_summaries)}) "
                    f"!= number of crops ({len(x)})"
                )

            embeddings = []
            for cell_embedding in patch_summaries:
                emb = cell_embedding.mean(dim=(0, 1))
                embeddings.append(emb)

            cell_tokens = torch.stack(embeddings)  # (N_crops, latent_dim)

            if cell_tokens.shape[0] != len(x):
                print(
                    f"[WARNING] Final embeddings count ({cell_tokens.shape[0]}) "
                    f"!= crops ({len(x)})"
                )

            return None, cell_tokens, None, None

    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return model, prepare_fn, get_channels, compute_fn, device
