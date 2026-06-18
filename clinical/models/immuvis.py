"""ImmuVis family — generate patch embeddings.

Usage (standalone):
    python -m clinical.models.immuvis --config configs/generate/603.yaml
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from core.utils import load_yaml, normalize_optional


DEFAULTS = {
    "models_path": "/raid_encrypted/immucan/models",
    "panel_config": "/storage_ssd_3/immuvis_celltyping/multiplex-image-model/configs/all_panels_config.yaml",
    "tokenizer_config": "/storage_ssd_3/immuvis_celltyping/multiplex-image-model/configs/all_markers_tokenizer.yaml",
    "repo_path": None,
    "data_config": None,
    "datasets": ["cords", "danenberg"],
    "splits": ["test", "train"],
    "patch_size": 128,
    "batch_size": 1,
    "num_workers": 8,
    "save_every": 200,
    "strict_loading": False,
    "include_zeroshot": False,
    "file_extension": "npy",
    "seed": 42,
}


def get_all_patches(img: torch.Tensor, patch_size: int = 128):
    H, W = img.shape[2:]
    i0, j0 = 0, 0
    i1, j1 = patch_size, patch_size
    patches = []
    coords = []
    while True:
        while True:
            patch = img[:, :, i0:i1, j0:j1]
            patches.append(patch)
            coords.append([(i0, j0), (i1, j1)])
            j1 += patch_size
            if j1 > W:
                break
            j0 = j1 - patch_size
        i1 += patch_size
        if i1 > H:
            break
        i0 = i1 - patch_size
        j0 = 0
        j1 = patch_size
    return patches, coords


def _embed_images(
    model, dataloader, device, patch_size=128,
    outpath=None, split_name=None, model_prefix=None, save_every=200,
):
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
        channel_ids = channel_ids.to(device=device, dtype=torch.long)
        for b in range(B):
            sample_img = img[b: b + 1]
            sample_channel_ids = channel_ids[b: b + 1]
            sample_img_path = os.path.realpath(img_path[b])
            sample_panel = str(panel_idx[b])
            for patch, (coords0, coords1) in zip(
                *get_all_patches(sample_img, patch_size)
            ):
                patch = patch.to(torch.float32).to(device)
                metadata.append((sample_img_path, sample_panel, coords0, coords1))
                with torch.no_grad():
                    latent = model.encode(patch, sample_channel_ids)["output"]
                    embeddings.append(latent.cpu().numpy().squeeze(0))
        if (i + 1) % save_every == 0:
            print(f"Processed {i + 1} images, saving batch {batch_idx}...")
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
    print(f"Finished embedding {split_name} images!")


def _resolve_repo_path(cfg_repo_path: str | None) -> str | None:
    if cfg_repo_path is not None:
        return cfg_repo_path
    try:
        import multiplex_model
        return None
    except ImportError:
        pass
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "multiplex-image-model",
        Path(__file__).resolve().parent.parent.parent / "beta-multiplex-image-model",
    ]
    for c in candidates:
        if c.is_dir():
            return str(c)
    raise RuntimeError("Cannot locate multiplex-image-model repo. Set repo_path in config.")


def generate(cfg: dict) -> None:
    """Generate batch-based patch embeddings for an ImmuVis model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    repo_path = _resolve_repo_path(cfg.get("repo_path"))
    if repo_path is not None:
        sys.path.insert(0, repo_path)

    from multiplex_model.data import MultiplexDataset, PanelBatchSampler
    from multiplex_model.modules.immuvis import MultiplexAutoencoder
    from multiplex_model.utils.configuration import (
        DataConfig, DecoderConfig, EncoderConfig, load_panel_config, load_tokenizer_config,
    )

    model_name = cfg["name"]
    model_path = os.path.join(cfg["models_path"], f"{model_name}.pth")
    model_config_path = os.path.join(cfg["models_path"], f"config.{model_name}.yaml")
    embeddings_path = cfg.get("output_dir", cfg.get("embeddings_path"))

    print(f"\n{'=' * 80}")
    print(f"Processing model: {model_name}")
    print(f"  checkpoint: {model_path}")
    print(f"  output: {embeddings_path}")
    print(f"{'=' * 80}")

    if not os.path.exists(model_config_path):
        print(f"Skipping {model_name}: config not found at {model_config_path}")
        return
    os.makedirs(embeddings_path, exist_ok=True)

    model_config_dict = load_yaml(model_config_path)
    panel_config_dict = load_panel_config(cfg["panel_config"])
    tokenizer = load_tokenizer_config(cfg["tokenizer_config"])

    data_config = {}
    if cfg.get("data_config"):
        external = load_yaml(cfg["data_config"])
        data_config.update(DataConfig(**external).model_dump())
    data_overrides = {
        k: normalize_optional(cfg.get(k))
        for k in ["preprocessing_func", "denoising_func", "scaling_func",
                   "normalization_func", "file_extension", "global_scaling_bound",
                   "operation_order"]
    }
    cli_overrides = {k: v for k, v in data_overrides.items() if v is not None}
    if cli_overrides:
        data_config.update(cli_overrides)
    data_config = DataConfig(**data_config).model_dump()

    panel_config_dict = {**panel_config_dict}
    if cfg.get("datasets"):
        panel_config_dict["datasets"] = cfg["datasets"]

    num_channels = len(tokenizer)
    print(f"Number of channels: {num_channels}")

    encoder_config = EncoderConfig(**model_config_dict["encoder"])
    decoder_config = DecoderConfig(**model_config_dict["decoder"])
    fallback = {
        "num_channels": num_channels,
        "encoder_config": encoder_config.model_dump(),
        "decoder_config": decoder_config.model_dump(),
    }

    model = MultiplexAutoencoder.load_from_checkpoint(
        checkpoint=model_path, map_location="cpu",
        model_config=fallback, strict=cfg.get("strict_loading", False),
    ).to(device)
    model.eval()
    print("Model loaded successfully!")

    for split_name in cfg["splits"]:
        dataset = MultiplexDataset(
            panels_config=panel_config_dict, split=split_name,
            marker_tokenizer=tokenizer, transform=None, **data_config,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=PanelBatchSampler(dataset, cfg["batch_size"], shuffle=False),
            num_workers=cfg["num_workers"],
        )
        print(f"\nEmbedding {split_name} ({len(dataset)} images)...")
        _embed_images(
            model, dataloader, device,
            patch_size=cfg["patch_size"],
            outpath=embeddings_path, split_name=split_name,
            model_prefix=model_name, save_every=cfg["save_every"],
        )

    print(f"Completed embedding for {model_name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ImmuVis-family embeddings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models-path")
    parser.add_argument("--embeddings-path")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--versions", type=int, nargs="+")
    parser.add_argument("--file-extension")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = {**DEFAULTS, **raw}
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    cfg.update(cli_overrides)

    if cfg.get("generation"):
        for entry in cfg["generation"]:
            generate({**cfg, **entry})
    elif cfg.get("model_names"):
        for name in cfg["model_names"]:
            generate({**cfg, "name": name, "output_dir": cfg.get("embeddings_path")})
    else:
        patterns = cfg.get("checkpoint_glob") or [f"Immu*-6{v:02d}-*.pth" for v in cfg.get("versions", [])] or ["Immu*-6*.pth"]
        model_files = sorted(glob(f"{cfg['models_path']}/{patterns[0]}"))
        if not cfg.get("include_zeroshot"):
            model_files = [p for p in model_files if "zeroshot" not in os.path.basename(p)]
        for path in model_files:
            name = os.path.basename(path).split(".")[0]
            generate({**cfg, "name": name, "output_dir": cfg.get("embeddings_path")})


if __name__ == "__main__":
    main()
