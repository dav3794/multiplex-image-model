"""VIRTUES family — generate patch embeddings.

Loads a MultiplexVirtues model from safetensors, runs inference with 8 TTA
transforms, and saves per-dataset per-split embeddings + metadata.

Usage:
    python -m clinical.models.virtues --config configs/generate/virtues.yaml
"""

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from tqdm import tqdm

from core.utils import add_repo, load_yaml


TRANSFORMS = [
    ("original",        lambda x: x),
    ("rot90",           lambda x: torch.rot90(x, k=1, dims=[-2, -1])),
    ("rot180",          lambda x: torch.rot90(x, k=2, dims=[-2, -1])),
    ("rot270",          lambda x: torch.rot90(x, k=3, dims=[-2, -1])),
    ("flip",           lambda x: torch.flip(x, dims=[-2])),
    ("flip_rot90",     lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=1, dims=[-2, -1])),
    ("flip_rot180",    lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=2, dims=[-2, -1])),
    ("flip_rot270",    lambda x: torch.rot90(torch.flip(x, dims=[-2]), k=3, dims=[-2, -1])),
]

PATCH_SIZE = 128


def _process_dataloader(model, dataloader, dataset_name, split, output_root):
    image_paths = []
    coords0 = []
    coords1 = []
    transforms_col = []
    all_embeddings = []
    batch_count = 0

    for img, channel_ids, _, _, img_name in tqdm(dataloader, desc=f"{dataset_name} {split}"):
        img = img[0]
        channel_ids = channel_ids[0]
        H, W = img.shape[-2], img.shape[-1]
        for i in range(H // PATCH_SIZE):
            for k in range(W // PATCH_SIZE):
                crop = img[:, i * PATCH_SIZE: (i + 1) * PATCH_SIZE, k * PATCH_SIZE: (k + 1) * PATCH_SIZE].cuda()
                for tf_name, tf_fn in TRANSFORMS:
                    aug_crop = tf_fn(crop)
                    batch_count += 1
                    with torch.amp.autocast(device_type="cuda"):
                        out = model.encoder.forward_list(
                            [aug_crop], [channel_ids]
                        ).patch_summary_tokens[0]
                    out = out.float().numpy(force=True)
                    image_paths.append(img_name[0])
                    coords0.append(f"({i * PATCH_SIZE}, {k * PATCH_SIZE})")
                    coords1.append(f"({(i + 1) * PATCH_SIZE}, {(k + 1) * PATCH_SIZE})")
                    transforms_col.append(tf_name)
                    all_embeddings.append(out.mean((0, 1)).reshape(1, -1))
                    del out, aug_crop
                del crop
                if batch_count % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        del img, channel_ids, img_name

    embeddings = np.concatenate(all_embeddings, axis=0)
    metadata = pd.DataFrame({
        "image_paths": image_paths,
        "coords0": coords0,
        "coords1": coords1,
        "transform": transforms_col,
    })
    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    metadata.to_csv(os.path.join(out_dir, f"{split}_metadata.csv"), index=False)
    np.save(os.path.join(out_dir, f"{split}_embeddings.npy"), embeddings)
    print(f"Saved {split} embeddings ({embeddings.shape}) to {out_dir}")


def generate(cfg: dict) -> None:
    """Generate embeddings for a VIRTUES model."""
    SEED = cfg.get("seed", 42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    repo_path = cfg.get("repo_path") or str(Path(__file__).resolve().parent.parent.parent / "virtues")
    add_repo(repo_path)

    base_config_path = cfg.get("base_config") or os.path.join(repo_path, "configs/base_config.yaml")
    datasets_config_path = cfg.get("datasets_config") or os.path.join(repo_path, "configs/datasets/example_config_multiple_datasets.yaml")

    from modules.multiplex_virtues import MultiplexVirtues
    from datasets.multiplex_dataset import MultiplexDataset
    from utils.utils import load_marker_embeddings

    conf = OmegaConf.load(base_config_path)
    ds_confs = OmegaConf.load(datasets_config_path)["datasets"]
    marker_embeddings = load_marker_embeddings(cfg["marker_embeddings_dir"])

    model = MultiplexVirtues(
        use_default_config=False, custom_config=None,
        prior_bias_embeddings=marker_embeddings,
        prior_bias_embedding_type="esm",
        prior_bias_embedding_fusion_type="add",
        patch_size=conf.model.patch_size,
        model_dim=conf.model.model_dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        decoder_pattern=conf.model.decoder_pattern,
        num_decoder_heads=conf.model.num_decoder_heads,
        num_hidden_layers=conf.model.num_decoder_hidden_layers,
        positional_embedding_type=conf.model.positional_embedding_type,
        dropout=conf.model.dropout,
        group_layers=conf.model.group_layers,
        norm_after_encoder_decoder=conf.model.norm_after_encoder_decoder,
        verbose=False,
    )
    print(f"Loading checkpoint from: {cfg['checkpoint']}")
    weights = {}
    with safe_open(cfg["checkpoint"], framework="pt", device="cpu") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)
    model.load_state_dict(weights)
    model = model.cuda().eval()
    print("Model loaded successfully!")

    output_root = cfg["output_root"]
    for ds_name in cfg["datasets"]:
        print(f"\nProcessing dataset: {ds_name}")
        ds_conf = ds_confs[ds_name]
        common_kwargs = dict(
            tissue_dir=ds_conf.tissue_dir, crop_dir=ds_conf.crop_dir,
            mask_dir=ds_conf.mask_dir, tissue_index=ds_conf.tissue_index,
            crop_index=ds_conf.crop_index, channels_file=ds_conf.channels_file,
            quantiles_file=ds_conf.quantiles_file, means_file=ds_conf.means_file,
            stds_file=ds_conf.stds_file,
            marker_embedding_dir=cfg["marker_embeddings_dir"],
            crop_size=conf.data.crop_size, patch_size=conf.model.patch_size,
            masking_ratio=[0.0, 0.0], channel_fraction=[1.0, 1.0],
            disable_augmentation=True, full_image=True, strategy="good",
        )
        for split in cfg["splits"]:
            dataset = MultiplexDataset(**common_kwargs, split=split)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
            _process_dataloader(model, dataloader, ds_name, split, output_root)

    print(f"\nAll datasets processed. Output root: {cfg['output_root']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VIRTUES-family embeddings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint")
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
