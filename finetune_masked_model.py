import copy
import os
import sys

import comet_ml  # noqa: F401
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode

from train_masked_model import train_masked
from multiplex_model.data import MultiplexDataset, PanelBatchSampler, TestCrop
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    FinetuneConfig,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    finish_experiment,
)


def _extract_model_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    raise ValueError(
        "Could not find model weights in checkpoint. Expected key model_state_dict."
    )


def _normalize_model_config(model_config: dict, num_channels: int) -> dict:
    resolved = copy.deepcopy(model_config)
    if "encoder" in resolved and "encoder_config" not in resolved:
        resolved["encoder_config"] = resolved.pop("encoder")
        if "hyperkernel" in resolved["encoder_config"]:
            resolved["encoder_config"]["hyperkernel_config"] = resolved["encoder_config"].pop(
                "hyperkernel"
            )
    if "decoder" in resolved and "decoder_config" not in resolved:
        resolved["decoder_config"] = resolved.pop("decoder")
        if "hyperkernel" in resolved["decoder_config"]:
            resolved["decoder_config"]["hyperkernel_config"] = resolved["decoder_config"].pop(
                "hyperkernel"
            )
    resolved["num_channels"] = num_channels
    return resolved


def _extend_tokenizer(old_tokenizer: dict[str, int], panel_config: dict) -> tuple[dict, list[str]]:
    markers_map = panel_config.get("markers", {})
    datasets = panel_config.get("datasets", list(markers_map.keys()))

    new_tokenizer = dict(old_tokenizer)
    new_markers: list[str] = []
    next_idx = max(old_tokenizer.values()) + 1

    for dataset in datasets:
        for marker in markers_map.get(dataset, []):
            if marker not in new_tokenizer:
                new_tokenizer[marker] = next_idx
                next_idx += 1
                new_markers.append(marker)

    return new_tokenizer, new_markers


def _load_with_extended_embeddings(
    model: MultiplexAutoencoder,
    checkpoint_state: dict,
    num_old_channels: int,
) -> None:
    new_state = model.state_dict()
    embed_keys = {
        "encoder.hyperkernel.hyperkernel_weights.weight",
        "decoder.channel_embed.hyperkernel_weights.weight",
        "decoder.channel_embed.hyperkernel_bias.weight",
    }

    with torch.no_grad():
        for key, new_tensor in new_state.items():
            if key not in checkpoint_state:
                print(f"Warning: key {key} not found in checkpoint, skipping.")
                continue
            old_tensor = checkpoint_state[key]

            if old_tensor.shape == new_tensor.shape:
                new_state[key] = old_tensor.to(new_tensor.device)
                continue

            if key in embed_keys:
                if old_tensor.shape[1:] != new_tensor.shape[1:]:
                    raise ValueError(
                        f"Shape mismatch for {key}: {old_tensor.shape} vs {new_tensor.shape}"
                    )
                new_state[key][:num_old_channels].copy_(
                    old_tensor.to(new_tensor.device)
                )
                print(f"Extended embedding for {key}: copied {num_old_channels} channels from checkpoint, "
                      f"initialized {new_tensor.shape[0] - num_old_channels} new channels.")
                continue

            print(
                f"Warning: skipping incompatible key {key} ({old_tensor.shape} -> {new_tensor.shape})"
            )

    model.load_state_dict(new_state, strict=True)


def _freeze_all_but_new_markers(
    model: MultiplexAutoencoder,
    num_old_channels: int,
) -> None:
    """Freeze all parameters except those for new markers (in hyperkernel)."""

    for param in model.parameters():
        param.requires_grad = False

    trainable_params = [
        model.encoder.hyperkernel.hyperkernel_weights.weight,
        model.decoder.channel_embed.hyperkernel_weights.weight,
    ]
    if hasattr(model.decoder.channel_embed, "hyperkernel_bias"):
        trainable_params.append(model.decoder.channel_embed.hyperkernel_bias.weight)

    for param in trainable_params:
        param.requires_grad = True

        def _mask_old_rows(grad, num_old=num_old_channels):
            if grad is None:
                return None
            grad[:num_old].zero_()
            return grad

        param.register_hook(_mask_old_rows)


def _freeze_marker_agnostic_backbone(model: MultiplexAutoencoder) -> None:
    for param in model.encoder.marker_agnostic_encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.decoder.parameters():
        param.requires_grad = False
    for param in model.decoder.pred.parameters():
        param.requires_grad = False


def _save_tokenizer(tokenizer: dict[str, int], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    yaml = YAML()
    with open(output_path, "w") as handle:
        yaml.dump(tokenizer, handle)


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    config = FinetuneConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    old_tokenizer = config.tokenizer_config
    panel_config = config.panel_config
    updated_tokenizer, new_markers = _extend_tokenizer(old_tokenizer, panel_config)

    if new_markers:
        print(f"Adding {len(new_markers)} new markers to tokenizer: {new_markers}")
    else:
        print("No new markers detected in panel config.")

    inv_tokenizer = {v: k for k, v in updated_tokenizer.items()}

    size = config.input_image_size
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(size),
            RandomHorizontalFlip(),
        ]
    )
    test_transform = TestCrop(size[0])

    dataset_kwargs = config.data_config.model_dump()

    train_dataset = MultiplexDataset(
        panels_config=panel_config,
        split="train",
        marker_tokenizer=updated_tokenizer,
        transform=train_transform,
        **dataset_kwargs,
    )
    test_dataset = MultiplexDataset(
        panels_config=panel_config,
        split="test",
        marker_tokenizer=updated_tokenizer,
        transform=test_transform,
        **dataset_kwargs,
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, batch_size)
    test_batch_sampler = PanelBatchSampler(test_dataset, batch_size, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    checkpoint = torch.load(config.pretrained_checkpoint, map_location=device)
    checkpoint_state = _extract_model_state_dict(checkpoint)

    num_old_channels = len(old_tokenizer)
    checkpoint_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    if checkpoint_config is None:
        checkpoint_config = {
            "num_channels": num_old_channels,
            "encoder_config": config.encoder_config.model_dump(),
            "decoder_config": config.decoder_config.model_dump(),
        }

    model_config = _normalize_model_config(checkpoint_config, len(updated_tokenizer))
    model = MultiplexAutoencoder(**model_config).to(device)

    _load_with_extended_embeddings(model, checkpoint_state, num_old_channels)

    if config.freeze_mode == "freeze_all_but_new_markers":
        if not new_markers:
            raise ValueError(
                "Error: freeze_mode=freeze_all_but_new_markers but no new markers were found. "
                "Training will not update any parameters."
            )
        _freeze_all_but_new_markers(model, num_old_channels)
    elif config.freeze_mode == "freeze_marker_agnostic":
        _freeze_marker_agnostic_backbone(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("Warning: no trainable parameters found after freezing.")

    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    optimizer = optim.AdamW(
        trainable_params,
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        num_annealing_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )

    comet_config = config.model_dump()
    init_experiment(comet_config)

    run_name = get_run_name()
    if config.updated_tokenizer_path:
        tokenizer_output_path = config.updated_tokenizer_path
    else:
        tokenizer_output_path = os.path.join(
            config.checkpoints_dir,
            f"tokenizer-{run_name}.yaml",
        )

    _save_tokenizer(updated_tokenizer, tokenizer_output_path)
    print(f"Saved updated tokenizer to: {tokenizer_output_path}")

    train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=inv_tokenizer,
        epochs=config.epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        beta=config.beta,
        min_channels_frac=config.min_channels_frac,
        spatial_masking_ratio=config.spatial_masking_ratio,
        fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
        mask_patch_size=config.mask_patch_size,
        save_checkpoint_every=config.save_checkpoint_freq,
        checkpoints_path=config.checkpoints_dir,
    )

    finish_experiment()
