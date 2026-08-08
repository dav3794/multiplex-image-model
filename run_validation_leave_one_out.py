import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from multiplex_model.data import DatasetFromTIFF, TestCrop
from multiplex_model.modules.immuvis import MultiplexAutoencoder
from multiplex_model.utils.configuration import DecoderConfig, EncoderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leave-one-out validation (mask one channel at a time)."
    )
    parser.add_argument(
        "--versions",
        type=int,
        nargs="+",
        default=list(range(0, 19)),
        help="Model versions to evaluate (e.g. --versions 14 15).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Direct path to a single model checkpoint (bypasses --versions glob).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Direct path to model config YAML (required with --checkpoint).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of test images to evaluate per model (default: all).",
    )
    parser.add_argument(
        "--models-path",
        default="/raid_encrypted/immucan/models",
        help="Path to model checkpoints and configs.",
    )
    parser.add_argument(
        "--results-dir",
        default="/raid_encrypted/immucan/results/with_reconstructs",
        help="Where to save CSV outputs.",
    )
    parser.add_argument(
        "--recon-dir",
        default="/raid_encrypted/immucan/recons/immuvis-beta",
        help="Where to save leave-one-out reconstructions (npz).",
    )
    parser.add_argument(
        "--save-reconstructions",
        action="store_true",
        help="Save leave-one-out reconstructions to NPZ files.",
    )
    parser.add_argument(
        "--panel-config",
        default="/home/mzmyslowski/marcin_multiplex/configs/all_panels_config.yaml",
    )
    parser.add_argument(
        "--tokenizer-config",
        default="/home/mzmyslowski/marcin_multiplex/configs/all_markers_tokenizer.yaml",
    )
    return parser.parse_args()


def create_leave_one_out_batch(
    img: torch.Tensor, channel_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create leave-one-out batch for a single image.

    Args:
        img: [C, H, W]
        channel_ids: [C]

    Returns:
        masked_img: [C, C-1, H, W]
        active_channel_ids: [C, C-1]
        output_channel_ids: [C, 1]
        masked_indices: [C]
    """
    num_channels, height, width = img.shape
    keep_mask = ~torch.eye(num_channels, dtype=torch.bool, device=img.device)

    img_expand = img.unsqueeze(0).expand(num_channels, -1, -1, -1)
    masked_img = img_expand[keep_mask].view(num_channels, num_channels - 1, height, width)

    channel_ids_expand = channel_ids.unsqueeze(0).expand(num_channels, -1)
    active_channel_ids = channel_ids_expand[keep_mask].view(num_channels, num_channels - 1)

    output_channel_ids = channel_ids.view(num_channels, 1)
    masked_indices = torch.arange(num_channels, device=img.device)

    return masked_img, active_channel_ids, output_channel_ids, masked_indices


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yaml = YAML(typ="safe")
    with open(args.panel_config, "r") as f:
        panel_config_dict = yaml.load(f)

    panel_config_dict["datasets"] = ["hn"]

    with open(args.tokenizer_config, "r") as f:
        tokenizer = yaml.load(f)

    inv_tokenizer = {v: k for k, v in tokenizer.items()}
    num_channels = len(tokenizer)

    print(f"Number of channels: {num_channels}")
    print(f"Sample markers: {list(tokenizer.keys())[:5]}")

    test_transform = TestCrop(128)
    test_dataset = DatasetFromTIFF(
        panels_config=panel_config_dict,
        split="test",
        marker_tokenizer=tokenizer,
        use_preprocessing=False,
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
        transform=test_transform,
    )

    print(f"Test dataset size: {len(test_dataset)} images")
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.checkpoint:
        if not args.model_config:
            raise ValueError("--model-config is required when using --checkpoint")
        model_entries = [(args.checkpoint, args.model_config)]
    else:
        patterns = [f"Immu*-6{v:02d}-beta-*.pth" for v in args.versions]
        model_files: list[str] = []
        for pattern in patterns:
            model_files.extend(glob(f"{args.models_path}/{pattern}"))

        if not model_files:
            raise FileNotFoundError(
                f"No model checkpoints found in {args.models_path} for versions {args.versions}."
            )
        model_entries = []
        for mf in sorted(model_files):
            cfg = f"{args.models_path}/config.{Path(mf).stem}.yaml"
            model_entries.append((mf, cfg))

    os.makedirs(args.results_dir, exist_ok=True)

    for model_weights_path, model_config_path in model_entries:
        model_checkpoint = os.path.basename(model_weights_path)
        model_idx = Path(model_weights_path).stem

        with open(model_config_path, "r") as f:
            model_config_dict = yaml.load(f)

        encoder_config = EncoderConfig(**model_config_dict["encoder"])
        decoder_config = DecoderConfig(**model_config_dict["decoder"])

        model = MultiplexAutoencoder(
            num_channels=num_channels,
            encoder_config=encoder_config.model_dump(),
            decoder_config=decoder_config.model_dump(),
        ).to(device)

        print(f"Loading model weights from: {model_weights_path}")
        checkpoint = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        all_mse = []
        all_uncertainties = []
        all_pearson_r = []
        all_channel_ids = []
        all_dataset_names = []
        all_image_paths = []

        recon_dir = Path(args.recon_dir) / f"immuvis_{model_idx}_loo"
        if args.save_reconstructions:
            recon_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for img_idx, (img, channel_ids, ds_name, img_path) in enumerate(
                tqdm(dataloader, desc="Leave-one-out validation")
            ):
                if args.max_images is not None and img_idx >= args.max_images:
                    break

                img = img.squeeze(0).to(device, dtype=torch.float32)
                channel_ids = channel_ids.squeeze(0).to(device, dtype=torch.long)

                (
                    masked_img,
                    active_channel_ids,
                    output_channel_ids,
                    masked_indices,
                ) = create_leave_one_out_batch(img=img, channel_ids=channel_ids)

                output = model(masked_img, active_channel_ids, output_channel_ids)[
                    "output"
                ]

                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi).squeeze(1)
                logvar = logvar.squeeze(1)

                target_channels = img[masked_indices]
                mse = (mi - target_channels).pow(2).mean(dim=(1, 2))

                mi_mean = mi.mean(dim=(1, 2), keepdim=True)
                target_mean = target_channels.mean(dim=(1, 2), keepdim=True)
                pearson_r = ((mi - mi_mean) * (target_channels - target_mean)).mean(
                    dim=(1, 2)
                ) / (mi.std(dim=(1, 2)) * target_channels.std(dim=(1, 2)) + 1e-8)

                all_mse.append(mse.flatten().cpu().numpy())
                all_uncertainties.append(logvar.mean(dim=(1, 2)).flatten().cpu().numpy())
                all_pearson_r.append(pearson_r.flatten().cpu().numpy())
                all_channel_ids.append(channel_ids[masked_indices].flatten().cpu().numpy())

                num_observations = masked_indices.numel()
                all_dataset_names.extend([ds_name[0]] * num_observations)
                all_image_paths.extend([img_path[0]] * num_observations)

                if args.save_reconstructions:
                    masked_channel_ids = channel_ids.detach().cpu().numpy()
                    masked_marker_names = [
                        inv_tokenizer.get(int(cid), "Unknown")
                        for cid in masked_channel_ids.tolist()
                    ]
                    metadata = {
                        "image_index": int(img_idx),
                        "image_path": str(img_path[0]),
                        "dataset_name": str(ds_name[0]),
                        "masked_strategy": "leave_one_out",
                        "num_channels": int(masked_channel_ids.shape[0]),
                    }
                    out_path = recon_dir / f"recn-{img_idx:05d}.npz"
                    np.savez_compressed(
                        out_path,
                        recon=mi.detach().cpu().numpy(),
                        variance=torch.exp(logvar).detach().cpu().numpy(),
                        target=img.detach().cpu().numpy(),
                        channel_ids=masked_channel_ids,
                        marker_names=np.array(masked_marker_names),
                        masked_channel_ids=masked_channel_ids,
                        masked_marker_names=np.array(masked_marker_names),
                        metadata=np.array(json.dumps(metadata)),
                    )

        mses = np.concatenate(all_mse, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        pearson_rs = np.concatenate(all_pearson_r, axis=0)
        masked_channel_ids = np.concatenate(all_channel_ids, axis=0)

        all_vals = np.stack(
            [mses, uncertainties, pearson_rs, masked_channel_ids],
            axis=1,
        )

        df = pd.DataFrame(all_vals, columns=["mse", "logsigma", "pearson", "Channel_ID"])
        df["marker"] = df["Channel_ID"].map(lambda x: inv_tokenizer.get(x, "Unknown"))
        df["masked"] = "leave_one_out"
        df["masked_count"] = 1
        df["model"] = f"ImmuViT-{model_idx}"
        df["dataset_name"] = all_dataset_names
        df["image_path"] = all_image_paths

        output_file = os.path.join(args.results_dir, f"immuvis_{model_idx}_loo.csv")
        print(f"Saving results to: {output_file}")
        df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
