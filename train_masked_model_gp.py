"""
Training script for masked multiplex image model with GP-based loss.

This script extends the standard training with a Gaussian Process-based negative
log-likelihood loss that models spatial correlations using the GP covariance module.
"""

import os
import sys

import comet_ml  # noqa: F401
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
from multiplex_model.losses import HybridGPNLLLoss, RankMe, beta_nll_loss, nll_loss
from multiplex_model.modules.gp_covariance import LowRankTimesSpatialCovariance
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    ClampWithGrad,
    TrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
)


def train_masked_gp(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    gp_covariance_module=None,
    use_gp_loss=True,
    lambda_gp=0.1,
    gp_max_cg_iterations=50,
    gp_downscale_factor=1,
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
    """
    Train a masked autoencoder with optional GP-based loss.
    
    Args:
        model: The autoencoder model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Computation device
        marker_names_map: Mapping from token IDs to marker names
        gp_covariance_module: GP covariance module (LowRankTimesSpatialCovariance)
        use_gp_loss: Whether to use GP loss
        lambda_gp: Weight for GP loss component
        gp_max_cg_iterations: Max CG iterations for GP loss
        gp_downscale_factor: Downsampling factor for GP loss
        epochs: Number of training epochs
        gradient_accumulation_steps: Number of gradient accumulation steps
        beta: Beta parameter for beta-NLL loss
        min_channels_frac: Minimum fraction of channels to keep
        fully_masked_channels_max_frac: Max fraction of fully masked channels
        spatial_masking_ratio: Ratio of spatial masking
        mask_patch_size: Size of spatial mask patches
        start_epoch: Starting epoch (for resuming)
        save_checkpoint_every: Checkpoint save frequency
        checkpoints_path: Directory to save checkpoints
    """
    model.train()
    scaler = GradScaler()
    run_name = get_run_name()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    # Initialize GP loss if enabled
    gp_loss_fn = None
    if use_gp_loss and gp_covariance_module is not None:
        gp_loss_fn = HybridGPNLLLoss(
            covariance_module=gp_covariance_module,
            lambda_gp=lambda_gp,
            max_cg_iterations=gp_max_cg_iterations,
            downscale_factor=gp_downscale_factor,
            device=device,
        )
        print(f"Using GP loss with lambda_gp={lambda_gp}")

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss_components = {
            "standard_nll": [],
            "gp_nll": [],
            "total_loss": [],
        }
        
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

            output = model(masked_img, active_channel_ids, channel_ids)["output"]
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
            logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)

            if use_gp_loss and gp_loss_fn is not None:
                # Use hybrid GP loss
                loss, loss_dict = gp_loss_fn(img, mi, logvar)
                
                # Track loss components
                for key in loss_dict:
                    epoch_loss_components[key].append(loss_dict[key])
            else:
                # Fall back to standard beta-NLL loss
                loss = beta_nll_loss(img, mi, logvar, beta=beta)
                epoch_loss_components["total_loss"].append(loss.item())

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # Prepare metrics for logging
                metrics = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "mu": mi.mean().item(),
                    "logvar": logvar.mean().item(),
                    "mae": torch.abs(img - mi).mean().item(),
                    "mse": torch.square(img - mi).mean().item(),
                    "step": step,
                }
                
                # Add GP loss components if applicable
                if use_gp_loss and batch_idx > 0 and epoch_loss_components["gp_nll"]:
                    metrics["standard_nll"] = epoch_loss_components["standard_nll"][-1]
                    metrics["gp_nll"] = epoch_loss_components["gp_nll"][-1]
                
                log_training_metrics(**metrics)
                step += 1

        # Print epoch statistics
        if use_gp_loss and epoch_loss_components["gp_nll"]:
            avg_standard_nll = np.mean(epoch_loss_components["standard_nll"])
            avg_gp_nll = np.mean(epoch_loss_components["gp_nll"])
            avg_total = np.mean(epoch_loss_components["total_loss"])
            print(f"\nEpoch {epoch} Loss Components:")
            print(f"  Standard NLL: {avg_standard_nll:.4f}")
            print(f"  GP NLL: {avg_gp_nll:.4f}")
            print(f"  Total Loss: {avg_total:.4f}")

        # Validation
        test_masked_gp(
            model,
            val_dataloader,
            device,
            epoch,
            gp_covariance_module=gp_covariance_module,
            gp_loss_fn=gp_loss_fn,
            spatial_masking_ratio=spatial_masking_ratio,
            fully_masked_channels_max_frac=fully_masked_channels_max_frac,
            mask_patch_size=mask_patch_size,
            marker_names_map=marker_names_map,
            use_gp_loss=use_gp_loss,
        )

        # Save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if gp_covariance_module is not None:
             checkpoint["gp_covariance_state_dict"] = gp_covariance_module.state_dict()
        
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


def test_masked_gp(
    model,
    test_dataloader,
    device,
    epoch,
    gp_covariance_module,
    gp_loss_fn,
    marker_names_map,
    num_plots=4,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
    use_gp_loss=True,
):
    """
    Validation loop with optional GP loss evaluation.
    """
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    running_mse = 0.0
    
    running_standard_nll = 0.0
    running_gp_nll = 0.0
    
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

            # Apply channel masking
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

            # Per-channel statistics
            variance_per_channel = torch.exp(logvar).mean(dim=(0, 2, 3))
            mae_per_channel = torch.abs(img - mi).mean(dim=(0, 2, 3))
            all_channel_variances.append(variance_per_channel.cpu())
            all_channel_maes.append(mae_per_channel.cpu())

            # Compute loss
            if use_gp_loss and gp_loss_fn is not None:
                loss, loss_dict = gp_loss_fn(img, mi, logvar)
                running_standard_nll += loss_dict["standard_nll"]
                running_gp_nll += loss_dict["gp_nll"]
            else:
                loss = nll_loss(img, mi, logvar)
            
            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()

            # Visualization
            if idx in plot_indices:
                unactive_channels = [
                    i for i in channel_ids[0] if i not in active_channel_ids[0]
                ]
                masked_channels_names = " | ".join(
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
                    img_idx=idx,
                )
                plt.close("all")

    val_loss = running_loss / len(test_dataloader)
    val_mae = running_mae / len(test_dataloader)
    val_mse = running_mse / len(test_dataloader)

    all_latents = torch.cat(all_latents)
    rankme = RankMe(all_latents)

    # Variance-MAE correlation
    all_channel_variances = torch.cat(all_channel_variances)
    all_channel_maes = torch.cat(all_channel_maes)
    variance_mae_corr = torch.corrcoef(
        torch.stack([all_channel_variances.flatten(), all_channel_maes.flatten()])
    )[0, 1].item()

    val_metrics = {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "latent_rankme": rankme,
        "variance_mae_correlation": variance_mae_corr,
        "epoch": epoch,
    }
    
    if use_gp_loss and gp_loss_fn is not None:
        val_metrics["val_standard_nll"] = running_standard_nll / len(test_dataloader)
        val_metrics["val_gp_nll"] = running_gp_nll / len(test_dataloader)

    log_validation_metrics(**val_metrics)

    print(f"{'=' * 40} EPOCH {epoch + 1} {'=' * 40}")
    print(f"Total Loss: {val_loss:.4f}")
    if use_gp_loss and gp_loss_fn is not None:
        print(f"Standard NLL: {val_metrics['val_standard_nll']:.4f}")
        print(f"GP NLL: {val_metrics['val_gp_nll']:.4f}")
    print(f"MAE: {val_mae:.6f}")
    print(f"MSE: {val_mse:.6f}")
    print(f"Pearson MAE vs Var: {variance_mae_corr:.4f}")
    print("=" * 90)
    print()

    return val_metrics


if __name__ == "__main__":
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "train_masked_config.yaml"
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    # Validate configuration
    config = TrainingConfig(**raw_config)

    device = config.device
    print(f"Using device: {device}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = YAML().load(open(config.panel_config))
    TOKENIZER = YAML().load(open(config.tokenizer_config))
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    # Data transforms
    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(SIZE),
            RandomHorizontalFlip(),
        ]
    )
    test_transform = TestCrop(SIZE[0])

    # Datasets
    train_dataset = DatasetFromTIFF(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        use_preprocessing=False,
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
        use_preprocessing=False,
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

    # Build model
    num_channels = len(TOKENIZER)
    model = MultiplexAutoencoder(
        num_channels=num_channels,
        encoder_config=config.encoder_config.model_dump(),
        decoder_config=config.decoder_config.model_dump(),
    ).to(device)

    # GP Loss Configuration
    use_gp_loss = getattr(config, "use_gp_loss", False)
    lambda_gp = getattr(config, "lambda_gp", 0.1)
    gp_kernel_jitter = getattr(config, "gp_kernel_jitter", 1e-2)
    gp_lengthscale = getattr(config, "gp_lengthscale", 5.0)
    gp_max_cg_iterations = getattr(config, "gp_max_cg_iterations", 50)
    gp_downscale_factor = getattr(config, "gp_downscale_factor", 1)
    gp_learn_lengthscale = getattr(config, "gp_learn_lengthscale", False)

    print("\nGP Loss Configuration:")
    print(f"  Use GP Loss: {use_gp_loss}")
    print(f"  Lambda GP: {lambda_gp}")
    print(f"  Kernel Jitter: {gp_kernel_jitter}")
    print(f"  Lengthscale: {gp_lengthscale}")
    print(f"  Max CG Iterations: {gp_max_cg_iterations}")
    print(f"  Downscale Factor: {gp_downscale_factor}")
    print(f"  Learn Lengthscale: {gp_learn_lengthscale}\n")

    # Initialize GP covariance module
    gp_covariance_module = None
    if use_gp_loss:
        # Create grid coordinates for GP
        H, W = SIZE
        H_gp = H // gp_downscale_factor
        W_gp = W // gp_downscale_factor
        
        # Create meshgrid [0, 1] normalized
        y = torch.linspace(0, 1, H_gp, device=device)
        x = torch.linspace(0, 1, W_gp, device=device)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        grid_coords = torch.stack([Y, X], dim=-1).reshape(-1, 2)

        gp_covariance_module = LowRankTimesSpatialCovariance(
            grid_coords=grid_coords,
            kernel_jitter=gp_kernel_jitter,
            spatial_matern_kernel_length_scale=gp_lengthscale,
            learn_lengthscale=gp_learn_lengthscale,
            device=device,
        )

    # Optimizer and scheduler
    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    # Include GP covariance parameters in optimization if learnable
    params_to_optimize = list(model.parameters())
    if use_gp_loss and gp_learn_lengthscale and gp_covariance_module is not None:
        params_to_optimize += list(gp_covariance_module.parameters())

    optimizer = optim.AdamW(
        params_to_optimize,
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

    # Initialize experiment tracking
    comet_config = config.model_dump()
    comet_config.update({
        "use_gp_loss": use_gp_loss,
        "lambda_gp": lambda_gp,
        "gp_kernel_jitter": gp_kernel_jitter,
        "gp_lengthscale": gp_lengthscale,
        "gp_max_cg_iterations": gp_max_cg_iterations,
        "gp_downscale_factor": gp_downscale_factor,
        "gp_learn_lengthscale": gp_learn_lengthscale,
    })
    init_experiment(comet_config)

    # Load checkpoint if specified
    start_epoch = 0
    if config.resolve_checkpoint():
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if gp_covariance_module is not None and "gp_covariance_state_dict" in checkpoint:
             gp_covariance_module.load_state_dict(checkpoint["gp_covariance_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Train the model
    train_masked_gp(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        gp_covariance_module=gp_covariance_module,
        use_gp_loss=use_gp_loss,
        lambda_gp=lambda_gp,
        gp_max_cg_iterations=gp_max_cg_iterations,
        gp_downscale_factor=gp_downscale_factor,
        epochs=config.epochs,
        start_epoch=start_epoch,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        min_channels_frac=config.min_channels_frac,
        spatial_masking_ratio=config.spatial_masking_ratio,
        fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
        mask_patch_size=config.mask_patch_size,
        save_checkpoint_every=config.save_checkpoint_freq,
        checkpoints_path=config.checkpoints_dir,
        beta=config.beta,
    )

    finish_experiment()
