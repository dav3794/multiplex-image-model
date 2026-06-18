"""Kronos ImmuVis v2 — generate patch embeddings.

Uses ImmuvisDINO model with DatasetFromTIFF and CLS-token extraction.

Usage:
    python -m clinical.models.immuvis_kronos --config configs/generate/exotic.yaml
"""

import argparse
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.utils import add_repo, load_yaml


def get_all_patches(img: torch.Tensor, patch_size: int = 128):
    C, H, W = img.shape[1:]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h or pad_w:
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        H, W = img.shape[2:]
    patches = []
    coords = []
    for i0 in range(0, H, patch_size):
        i1 = i0 + patch_size
        for j0 in range(0, W, patch_size):
            j1 = j0 + patch_size
            patches.append(img[:, :, i0:i1, j0:j1])
            coords.append(((i0, j0), (i1, j1)))
    return patches, coords


def _embed_images(model, dataloader, device, patch_size=128, kronos_patch_size=8,
                  outpath=None, split_name=None, model_prefix=None, save_every=200):
    model.eval()
    embeddings = []
    metadata = []
    batch_idx = 0
    for i, (img, channel_ids, panel_idx, img_path) in enumerate(
        tqdm(dataloader, desc=f"Embedding {split_name} images")
    ):
        B, C, H, W = img.shape
        if H < patch_size or W < patch_size:
            print(f"Image smaller than patch size: {img.shape} at {img_path[0]}")
            continue
        for b in range(B):
            sample_img_path = os.path.realpath(img_path[b])
            sample_img = img[b: b + 1].float().to(device)
            sample_channel_ids = channel_ids[b: b + 1].to(device=device, dtype=torch.long)
            for patch, (coords0, coords1) in zip(*get_all_patches(sample_img, patch_size)):
                metadata.append((sample_img_path, str(panel_idx[b]), coords0, coords1))
                with torch.no_grad():
                    B_p, C_p, H_p, W_p = patch.shape
                    x_hk = patch.reshape(B_p * C_p, 1, H_p, W_p)
                    x_enc = model.hyperkernel(x_hk, sample_channel_ids)
                    x_enc = x_enc.flatten(2).transpose(1, 2)
                    h_p, w_p = H_p // kronos_patch_size, W_p // kronos_patch_size
                    N = h_p * w_p
                    cls_tokens = model.cls_token.expand(B_p, -1, -1)
                    x_enc = torch.cat((cls_tokens, x_enc), dim=1)
                    x_enc = x_enc + model.pos_embed[:, :N + 1, :]
                    for blk in model.blocks:
                        x_enc = blk(x_enc)
                    x_enc = model.norm(x_enc)
                    latent = x_enc[0, 0, None, None]
                    embeddings.append(latent.cpu().numpy())
        if (i + 1) % save_every == 0:
            if outpath and embeddings:
                np.save(
                    os.path.join(outpath, f"{model_prefix}_{split_name}_image_patches_embeddings_batch_{batch_idx}.npy"),
                    np.stack(embeddings),
                )
                pd.DataFrame(metadata, columns=["img_path", "panel", "coords0", "coords1"]).to_csv(
                    os.path.join(outpath, f"{model_prefix}_{split_name}_image_patches_metadata_batch_{batch_idx}.csv"),
                    index=False,
                )
                embeddings = []
                metadata = []
                batch_idx += 1
    if embeddings:
        np.save(
            os.path.join(outpath, f"{model_prefix}_{split_name}_image_patches_embeddings_batch_{batch_idx}.npy"),
            np.stack(embeddings),
        )
        pd.DataFrame(metadata, columns=["img_path", "panel", "coords0", "coords1"]).to_csv(
            os.path.join(outpath, f"{model_prefix}_{split_name}_image_patches_metadata_batch_{batch_idx}.csv"),
            index=False,
        )


def generate(cfg: dict) -> None:
    """Generate embeddings for a kronos_immuvis_v2 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["name"]
    embeddings_path = cfg["output_dir"]
    repo_path = cfg["repo_path"]
    checkpoint_path = cfg["checkpoint_path"]
    model_config_path = cfg["model_config_path"]
    strict_loading = cfg.get("strict_loading", True)
    data_overrides = cfg.get("data_overrides", {})

    os.makedirs(embeddings_path, exist_ok=True)
    add_repo(repo_path)
    for key in list(sys.modules):
        if key.startswith("multiplex_model"):
            del sys.modules[key]

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler
    from train_kronos_immuvis_v2 import ImmuvisDINO

    panel_config_dict = load_yaml(cfg["panel_config"])
    tokenizer = load_yaml(cfg["tokenizer_config"])
    model_config = load_yaml(model_config_path)
    if cfg.get("datasets"):
        panel_config_dict = {**panel_config_dict, "datasets": cfg["datasets"]}

    num_channels = len(tokenizer)
    kronos_ps = model_config.get("patch_size", 8)
    embed_dim = model_config.get("embed_dim", 768)
    depth = model_config.get("depth", 12)
    num_heads = model_config.get("num_heads", 12)
    out_dim = model_config.get("out_dim", 65536)

    print(f"\n{'=' * 80}")
    print(f"Processing model: {model_name}")
    print(f"  embed_dim={embed_dim}, depth={depth}, heads={num_heads}, patch_size={kronos_ps}")
    print(f"  output: {embeddings_path}")
    print(f"{'=' * 80}")

    model = ImmuvisDINO(
        num_markers=num_channels, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads,
        patch_size=kronos_ps, out_dim=out_dim,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("student_state_dict", ckpt.get("model_state_dict", ckpt.get("model", ckpt.get("state_dict", ckpt))))
    try:
        model.load_state_dict(state_dict, strict=strict_loading)
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    for split_name in cfg["splits"]:
        dataset = DatasetFromTIFF(
            panels_config=panel_config_dict, split=split_name,
            marker_tokenizer=tokenizer, transform=None,
            use_preprocessing=data_overrides.get("use_preprocessing", False),
            use_median_denoising=data_overrides.get("use_median_denoising", False),
            use_butterworth_filter=data_overrides.get("use_butterworth_filter", True),
            use_minmax_normalization=data_overrides.get("use_minmax_normalization", False),
            use_clip_normalization=data_overrides.get("use_clip_normalization", True),
            file_extension=data_overrides.get("file_extension", "npy"),
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=PanelBatchSampler(dataset, cfg["batch_size"], shuffle=False),
            num_workers=cfg["num_workers"],
        )
        print(f"\nEmbedding {split_name} ({len(dataset)} images)...")
        _embed_images(
            model, dataloader, device, patch_size=cfg["patch_size"],
            kronos_patch_size=kronos_ps,
            outpath=embeddings_path, split_name=split_name,
            model_prefix=model_name, save_every=cfg["save_every"],
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kronos ImmuVis v2 embeddings.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for entry in cfg.get("generation", cfg.get("models", [])):
        generate({k: v for k, v in cfg.items() if k not in ("generation", "models")} | entry)


if __name__ == "__main__":
    main()
