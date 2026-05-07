import argparse
import importlib
import inspect
import os
import random
import sys
import yaml

import numpy as np
import torch
from omegaconf import OmegaConf

from core.embeddings.context import generate_embeddings_context
from core.embeddings.patch import generate_embeddings_patch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

ROOT = os.path.abspath(os.path.join("..", "virtues"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.multiplex_dataset import MultiplexDataset
from utils.utils import load_marker_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dataset YAML config"
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="./core/models/registry.yaml",
        help="Path to the model registry YAML",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model key from the config (e.g., immuvis475)",
    )
    parser.add_argument(
        "--scheme", type=str, choices=["context", "patch"], default="context"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    with open(args.registry, "r") as f:
        registry = yaml.safe_load(f)

    virtues_conf = OmegaConf.load(
        conf.get("virtues_config_path", "./core/models/virtues_config.yaml")
    )

    processed_dir = conf["processed_dir"]
    path_marker_embeddings = os.path.join(
        conf["virtues_embeddings_dir"], "esm2_t30_150M_UR50D"
    )
    annotations_path = os.path.join(processed_dir, "sce_annotations.csv")

    out_dir = os.path.join(
        conf["base_path"], "embeddings", f"{args.model}_{args.scheme}"
    )
    os.makedirs(out_dir, exist_ok=True)

    dataset = MultiplexDataset(
        tissue_dir=os.path.join(processed_dir, "images"),
        crop_dir=os.path.join(processed_dir, "crops"),
        mask_dir=os.path.join(processed_dir, "masks"),
        tissue_index=os.path.join(processed_dir, "tissue_index.csv"),
        crop_index=os.path.join(processed_dir, "crop_index.csv"),
        channels_file=os.path.join(processed_dir, "channels.csv"),
        quantiles_file=os.path.join(processed_dir, "quantiles.csv"),
        means_file=os.path.join(processed_dir, "means.csv"),
        stds_file=os.path.join(processed_dir, "stds.csv"),
        marker_embedding_dir=path_marker_embeddings,
        split="test",
        crop_size=virtues_conf.data.crop_size,
        patch_size=virtues_conf.model.patch_size,
        masking_ratio=virtues_conf.data.masking_ratio,
        channel_fraction=virtues_conf.data.channel_fraction,
    )

    models_dict = registry.get("models", {})
    if args.model not in models_dict:
        raise ValueError(
            f"Model '{args.model}' not defined in {args.registry} under 'models:'"
        )

    model_kwargs = models_dict[args.model].copy()
    setup_func_path = model_kwargs.pop("setup_func")

    try:
        module_name, func_name = setup_func_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        setup_func = getattr(module, func_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to dynamically load setup function '{setup_func_path}': {e}"
        )

    model_kwargs["scheme"] = args.scheme
    model_kwargs["batch_size"] = args.batch_size

    sig = inspect.signature(setup_func)

    if "dataset_name" in sig.parameters:
        model_kwargs["dataset_name"] = conf["dataset_name"]

    if "marker_embeddings" in sig.parameters:
        model_kwargs["marker_embeddings"] = load_marker_embeddings(
            path_marker_embeddings
        )

    if "virtues_conf" in sig.parameters:
        model_kwargs["virtues_conf"] = virtues_conf

    model, prepare_fn, get_channels, compute_fn, device = setup_func(**model_kwargs)

    runner_kwargs = dict(
        dataset=dataset,
        model=model,
        prepare_fn=prepare_fn,
        get_channels=get_channels,
        compute_fn=compute_fn,
        annotations_path=annotations_path,
        out_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
    )

    if args.scheme == "context":
        generate_embeddings_context(**runner_kwargs)
    elif args.scheme == "patch":
        generate_embeddings_patch(**runner_kwargs)
