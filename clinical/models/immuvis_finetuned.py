"""Fine-tuned ImmuVis models — generate patch embeddings.

Uses MultiplexAutoencoder.load_from_checkpoint() so no model_config_path
is needed. Uses MultiplexDataset from the finetuning-multiplex-image-model repo.

Usage:
    python -m clinical.generate --config configs/generate/finetuned.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML
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


def _embed_images(model, dataloader, device, patch_size=128,
                  outpath=None, split_name=None, model_prefix=None, save_every=200):
    model.eval()
    embeddings = []
    metadata = []
    batch_idx = 0
    for i, (img, channel_ids, ds_name, img_path) in enumerate(
        tqdm(dataloader, desc=f"Embedding {split_name} images")
    ):
        B, C, H, W = img.shape
        if H < patch_size or W < patch_size:
            print(f"Image smaller than patch size: {img.shape} at {img_path[0]}")
            continue
        channel_ids = channel_ids.to(device=device, dtype=torch.long)
        for b in range(B):
            sample_img = img[b: b + 1].to(torch.float32).to(device)
            sample_channel_ids = channel_ids[b: b + 1]
            sample_img_path = os.path.realpath(img_path[b])
            sample_panel = str(ds_name[b])
            for patch, (coords0, coords1) in zip(*get_all_patches(sample_img, patch_size)):
                metadata.append((sample_img_path, sample_panel, coords0, coords1))
                with torch.no_grad():
                    latent = model.encode(patch, sample_channel_ids)["output"]
                    embeddings.append(latent.cpu().numpy().squeeze(0))
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
    """Generate embeddings for a fine-tuned model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["name"]
    embeddings_path = cfg.get("output_dir") or cfg.get("embeddings_path")
    repo_path = cfg["repo_path"]
    checkpoint_path = cfg["checkpoint_path"]
    strict_loading = cfg.get("strict_loading", True)
    data_overrides = cfg.get("data_overrides", {})

    os.makedirs(embeddings_path, exist_ok=True)
    add_repo(repo_path)
    for key in list(sys.modules):
        if key.startswith("multiplex_model"):
            del sys.modules[key]

    from multiplex_model.data import MultiplexDataset, PanelBatchSampler
    from multiplex_model.modules.immuvis import MultiplexAutoencoder

    panel_config_dict = load_yaml(cfg["panel_config"])
    tokenizer = load_yaml(cfg["tokenizer_config"])

    if cfg.get("datasets"):
        panel_config_dict = {**panel_config_dict, "datasets": cfg["datasets"]}

    num_channels = len(tokenizer)
    print(f"\n{'=' * 80}")
    print(f"Processing model: {model_name}")
    print(f"  repo: {repo_path}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  output: {embeddings_path}")
    print(f"{'=' * 80}")

    model = MultiplexAutoencoder.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=strict_loading,
    )
    model = model.to(device).eval()
    print(f"Model loaded (strict={strict_loading})")

    for split_name in cfg["splits"]:
        dataset = MultiplexDataset(
            panels_config=panel_config_dict, split=split_name,
            marker_tokenizer=tokenizer, transform=None,
            unsupported_marker_behavior=data_overrides.get(
                "unsupported_marker_behavior", "drop"
            ),
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
            outpath=embeddings_path, split_name=split_name,
            model_prefix=model_name, save_every=cfg["save_every"],
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings from fine-tuned models.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    for entry in cfg.get("generation", cfg.get("models", [])):
        generate({k: v for k, v in cfg.items() if k not in ("generation", "models")} | entry)


if __name__ == "__main__":
    main()
