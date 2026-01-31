import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import DatasetFromTIFF, PanelBatchSampler, TestCrop
from multiplex_model.losses import RankMe, beta_nll_loss, intrinsic_dimension, nll_loss
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    TrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_wandb_run,
    get_run_name,
    get_scheduler_with_warmup,
    init_wandb_run,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
)


def train_masked(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    epochs=10,
    gradient_accumulation_steps=1,
    beta=1.0,
    min_channels_frac=0.75,
    fully_masked_channels_max_frac=0.5,
    spatial_masking_ratio=0.6,
    mask_patch_size=8,
    start_epoch=0,
    save_checkpoint_every=5,
    checkpoints_path="checkpoints",
):
    """Train a masked autoencoder (decode the remaining channels) with the given parameters."""
    model.train()
    scaler = GradScaler()
    run_name = get_run_name()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # Apply channel masking with channel subset sampling
            img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                min_channels_frac,
                fully_masked_channels_max_frac,
                apply_channel_subset_sampling=True,
            )

            # Apply spatial masking
            masked_img, _ = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size
            )

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(masked_img, active_channel_ids, channel_ids)["output"]
                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)

                loss = beta_nll_loss(img, mi, logvar, beta=beta)

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                log_training_metrics(
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0],
                    mu=mi.mean().item(),
                    logvar=logvar.mean().item(),
                    mae=torch.abs(img - mi).mean().item(),
                    mse=torch.square(img - mi).mean().item(),
                )

        val_loss = test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            spatial_masking_ratio=spatial_masking_ratio,
            fully_masked_channels_max_frac=fully_masked_channels_max_frac,
            mask_patch_size=mask_patch_size,
            marker_names_map=marker_names_map,
        )
        print(f"Validation loss: {val_loss:.4f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if (epoch + 1) % save_checkpoint_every == 0:
            torch.save(
                checkpoint,
                f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth",
            )
        torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

    final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
    print(f"Training completed. Saving final model at {final_model_path}...")
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, final_model_path)


def test_masked(
    model,
    test_dataloader,
    device,
    epoch,
    marker_names_map,
    num_plots=5,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    plot_indices = np.random.choice(
        np.arange(len(test_dataloader)), size=num_plots, replace=False
    )
    plot_indices = set(plot_indices)

    all_latents = []
    all_channel_variances = []
    all_channel_maes = []

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(test_dataloader, desc=f"Testing epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            # Apply channel masking (only full channel masking for validation, no channel dropping)
            _, _, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                apply_channel_subset_sampling=False,
            )

            # Apply spatial masking
            masked_img, pixel_mask = apply_spatial_masking(
                masked_img, spatial_masking_ratio, mask_patch_size
            )

            latent = model.encode(masked_img, active_channel_ids)["output"]
            output = model.decode(latent, channel_ids)
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)

            latent = normalize(latent.mean(dim=(2, 3)), p=2, dim=1)
            all_latents.append(latent.cpu())

            # Accumulate per-channel statistics for correlation analysis
            variance_per_channel = torch.exp(logvar).mean(
                dim=(0, 2, 3)
            )  # Mean variance per channel
            mae_per_channel = torch.abs(img - mi).mean(dim=(0, 2, 3))  # MAE per channel
            all_channel_variances.append(variance_per_channel.cpu())
            all_channel_maes.append(mae_per_channel.cpu())

            loss = nll_loss(img, mi, logvar)
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()

            if idx in plot_indices:
                unactive_channels = [
                    i for i in channel_ids[0] if i not in active_channel_ids[0]
                ]
                masked_channels_names = "\n".join(
                    [marker_names_map[i.item()] for i in unactive_channels]
                )

                reconstr_img = plot_reconstructs_with_masks(
                    img,
                    mi,
                    pixel_mask,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    ncols=9,
                )
                log_validation_images(
                    fig=reconstr_img,
                    panel_idx=panel_idx[0],
                    img_path=img_path[0],
                    epoch=epoch,
                    masked_channels_names=masked_channels_names,
                )
                plt.close("all")

    val_loss = running_loss / len(test_dataloader)
    val_mae = running_mae / len(test_dataloader)
    val_mse = running_mse / len(test_dataloader)

    all_latents = torch.cat(all_latents)
    rankme = RankMe(all_latents)
    intinsic_dim = intrinsic_dimension(all_latents)

    # Calculate Pearson correlation between predicted variances and MAEs per channel
    all_channel_variances = torch.cat(all_channel_variances)
    all_channel_maes = torch.cat(all_channel_maes)
    # Calculate Pearson correlation using flattened data across all batches
    variance_mae_corr = torch.corrcoef(
        torch.stack([all_channel_variances.flatten(), all_channel_maes.flatten()])
    )[0, 1].item()

    log_validation_metrics(
        val_loss=val_loss,
        val_mae=val_mae,
        val_mse=val_mse,
        latent_rankme=rankme,
        latent_intrinsic_dim=intinsic_dim,
        epoch=epoch,
        variance_mae_correlation=variance_mae_corr,
    )

    return val_loss


if __name__ == "__main__":
    # Load the configuration file
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    # Validate configuration using Pydantic model
    config = TrainingConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(SIZE),
            RandomHorizontalFlip(),
        ]
    )

    test_transform = TestCrop(SIZE[0])

    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
    )

    test_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        use_preprocessing=False,  # saved data is already preprocessed
        use_median_denoising=False,
        use_butterworth_filter=True,
        use_minmax_normalization=False,
        use_clip_normalization=True,
        file_extension="npy",
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Build model configuration
    num_channels = len(TOKENIZER)
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=config.decoder_config.model_dump(),
    ).to(device)

    # Setup optimizer and scheduler
    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    optimizer = optim.AdamW(
        model.parameters(), lr=config.peak_lr, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        num_annealing_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )

    # # Initialize wandb
    wandb_config = config.model_dump()
    init_wandb_run(wandb_config)

    # Load checkpoint if specified
    start_epoch = 0
    if config.resolve_checkpoint():
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Train the model
    train_masked(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        epochs=config.epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        min_channels_frac=config.min_channels_frac,
        spatial_masking_ratio=config.spatial_masking_ratio,
        fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
        mask_patch_size=config.mask_patch_size,
        save_checkpoint_every=config.save_checkpoint_freq,
        checkpoints_path=config.checkpoints_dir,
    )

    finish_wandb_run()
