"""Eva family — generate patch embeddings.

Loads an EvaMAE model from HuggingFace Hub, runs inference with 8 TTA
transforms and marker filtering, and saves per-dataset per-split embeddings.

Usage:
    python -m clinical.models.eva --config configs/generate/eva.yaml
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.utils import add_repo, load_yaml


TRANSFORMS = [
    ("original", lambda x: x),
    ("rot90", lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
    ("rot180", lambda x: torch.rot90(x, k=2, dims=[-2, -1])),
    ("rot270", lambda x: torch.rot90(x, k=3, dims=[-2, -1])),
    ("flip", lambda x: torch.flip(x, dims=[-2])),
    ("flip_rot90", lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2, -1])),
    ("flip_rot180", lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=2, dims=[-2, -1])),
    ("flip_rot270", lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=3, dims=[-2, -1])),
]


def _inference_amp_context(device: str):
    if device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda")
    return contextlib.nullcontext()


def _supported_markers(model, panel_markers: list[str]):
    genept = set(model.model.marker_embed.genept_embeddings.keys())
    unknown = set(model.model.marker_embed.unknown_marker_embeddings.keys())
    supported = genept | unknown
    kept = [(i, m) for i, m in enumerate(panel_markers) if m in supported]
    return [i for i, _ in kept], [m for _, m in kept]


def _extract_features(model, image_bchw: torch.Tensor, markers: list[str], device: str) -> np.ndarray:
    image_bchw = image_bchw.float()
    marker_in = [markers for _ in range(image_bchw.shape[0])]
    with torch.no_grad(), _inference_amp_context(device):
        image_out, _ = model.model.forward_encoder(image_bchw, marker_in)
        features = image_out[:, :, 1:, :].mean(2).squeeze(1)
    return features.float().detach().cpu().numpy()


def _embed_split(model, dataloader, kept_indices, kept_markers,
                 dataset_name, split, output_root, crop_size, device):
    metadata_records = []
    embedding_batches = []
    kept_idx_tensor = torch.tensor(kept_indices, dtype=torch.long)
    batch_idx = 0

    for img, _channel_ids, _panel_idx, img_path in tqdm(dataloader, desc=f"{dataset_name} {split}"):
        if img.shape[0] != 1:
            raise ValueError("Full-image tiling mode currently requires batch_size=1.")
        img = img.index_select(dim=1, index=kept_idx_tensor).to(device)
        _b, _c, height, width = img.shape
        for i in range(height // crop_size):
            for k in range(width // crop_size):
                top, left = i * crop_size, k * crop_size
                crop = img[:, :, top: top + crop_size, left: left + crop_size]
                for transform_name, transform_fn in TRANSFORMS:
                    aug_crop = transform_fn(crop)
                    features = _extract_features(model, aug_crop, kept_markers, device)
                    embedding_batches.append(features)
                    metadata_records.append({
                        "image_paths": img_path[0],
                        "coords0": f"({top}, {left})",
                        "coords1": f"({top + crop_size}, {left + crop_size})",
                        "transform": transform_name,
                        "dataset": dataset_name,
                    })
                    batch_idx += 1
                    del aug_crop
                del crop
                if device.startswith("cuda") and batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
        del img

    embeddings = np.concatenate(embedding_batches, axis=0) if embedding_batches else np.empty((0, 0))
    metadata = pd.DataFrame(metadata_records)
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    metadata.to_csv(os.path.join(output_dir, f"{split}_metadata.csv"), index=False)
    np.save(os.path.join(output_dir, f"{split}_embeddings.npy"), embeddings)
    print(f"Saved {split} embeddings ({embeddings.shape}) to {output_dir}")


def generate(cfg: dict) -> None:
    """Generate embeddings for an Eva model."""
    os.chdir(cfg["eva_dir"])
    add_repo(cfg["mim_dir"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    SEED = cfg.get("seed", 42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    conf_path = cfg.get("conf") or os.path.join(cfg["eva_dir"], "config.yaml")
    eva_conf = OmegaConf.load(conf_path)
    image_size = eva_conf.ds.patch_size
    strategy = OmegaConf.select(eva_conf, "inference.image_size_strategy") or "interpolate_pos_embed"
    ckpt_size = OmegaConf.select(eva_conf, "inference.checkpoint_patch_size") or image_size
    loader_crop_size = ckpt_size if strategy == "checkpoint_center_crop" else image_size

    from Eva.utils import load_from_hf
    model = load_from_hf(repo_id=cfg["repo_id"], conf=eva_conf, device=device)
    model.to(torch.float16)
    model.to(device)
    model.eval()

    from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler

    for dataset_name in cfg["datasets"]:
        print(f"\nProcessing dataset: {dataset_name}")

        _YAML = YAML(typ="safe")
        panel_config = _YAML.load(open(cfg["panel_config"]))
        tokenizer = _YAML.load(open(cfg["tokenizer_config"]))
        panel_config["datasets"] = [dataset_name]

        panel_markers = list(panel_config["markers"][dataset_name])
        kept_indices, kept_markers = _supported_markers(model, panel_markers)
        print(f"  Panel markers: {len(panel_markers)}, kept: {len(kept_markers)}")
        if not kept_markers:
            print(f"  Skipping {dataset_name}: no Eva-supported markers.")
            continue

        for split in cfg["splits"]:
            dataset = DatasetFromTIFF(
                panels_config=panel_config, split=split,
                marker_tokenizer=tokenizer, transform=None,
                use_median_denoising=False,
                use_butterworth_filter=True,
                use_clip_normalization=True,
                use_minmax_normalization=False,
                clip_per_dataset=True,
            )
            dataloader = DataLoader(
                dataset,
                batch_sampler=PanelBatchSampler(dataset, cfg["batch_size"], shuffle=False),
                num_workers=cfg["num_workers"],
            )
            _embed_split(
                model, dataloader, kept_indices, kept_markers,
                dataset_name, split, cfg["output_root"],
                loader_crop_size, device,
            )

    print(f"\nAll datasets processed. Output root: {cfg['output_root']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Eva-family embeddings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--splits", nargs="+")
    args = parser.parse_args()

    raw = load_yaml(args.config)
    cfg = {**raw}
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    generate(cfg)


if __name__ == "__main__":
    main()
