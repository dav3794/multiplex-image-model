"""Training script combining learnable spatial mask token with Kronecker marker GP loss.

Merges the mask-token flow from `train_masked_model_learnmask.py` with the
Kronecker + marker covariance GP loss from `train_masked_model_gp.py`.
"""

import logging
import math
import os
import sys
from typing import Any

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
from multiplex_model.losses import (
    HybridKroneckerMarkerGPNLLLoss,
    RankMe,
    beta_nll_loss,
    nll_loss,
)
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.modules.gp_covariance import KroneckerMarkerCovariance
from multiplex_model.utils import (
    ClampWithGrad,
    TrainingConfig,
    apply_channel_masking,
    finish_experiment,
    get_pixel_mask,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
    log_validation_batch_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
    plot_reconstructs_with_uncertainty,
)

logger = logging.getLogger(__name__)


def train_masked_learnmask_gp(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    gp_covariance_module,
    gp_loss_fn,
    total_steps,
    use_gp_loss=True,
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
    model.train()
    scaler = GradScaler()
    run_name = get_run_name()

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    step = start_epoch * (len(train_dataloader) // gradient_accumulation_steps)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss_components: dict[str, list[float]] = {
            "standard_nll": [],
            "gp_nll": [],
            "total_loss": [],
        }

        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                min_channels_frac,
                fully_masked_channels_max_frac,
                apply_channel_subset_sampling=True,
            )

            pixel_mask = get_pixel_mask(masked_img, spatial_masking_ratio, mask_patch_size)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    masked_img, active_channel_ids, channel_ids, spatial_mask=pixel_mask
                )["output"]
                mi, logvar = output.unbind(dim=-1)
                mi = torch.sigmoid(mi)
                logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)

            # GP loss runs in float32: linalg.solve in Woodbury (gp_covariance.py)
            # rejects mixed bfloat16/float32 dtypes used by the precomputed Kronecker eigs.
            if use_gp_loss and gp_loss_fn is not None:
                marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)
                loss, loss_dict = gp_loss_fn(
                    img.float(), mi.float(), logvar.float(), marker_emb.float()
                )
                for key in loss_dict:
                    epoch_loss_components[key].append(loss_dict[key])
            else:
                loss = beta_nll_loss(img, mi, logvar, beta=beta)
                epoch_loss_components["total_loss"].append(loss.item())

            if not loss.isfinite():
                logger.warning("Non-finite loss at step %d epoch %d, skipping batch", batch_idx, epoch)
                optimizer.zero_grad()
                continue

            scaler.scale(loss / gradient_accumulation_steps).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                mask_token_value = model.encoder.mask_token.item() if model.encoder.mask_token is not None else None

                metrics: dict[str, Any] = {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "mu": mi.mean().item(),
                    "logvar": logvar.mean().item(),
                    "mae": torch.abs(img - mi).mean().item(),
                    "mse": torch.square(img - mi).mean().item(),
                    "step": step,
                    "mask_token": mask_token_value,
                }
                if use_gp_loss and epoch_loss_components["gp_nll"]:
                    metrics["standard_nll"] = epoch_loss_components["standard_nll"][-1]
                    metrics["gp_nll"] = epoch_loss_components["gp_nll"][-1]

                log_training_metrics(**metrics)
                step += 1

        if use_gp_loss and epoch_loss_components["gp_nll"]:
            avg_standard_nll = float(np.mean(epoch_loss_components["standard_nll"]))
            avg_gp_nll = float(np.mean(epoch_loss_components["gp_nll"]))
            avg_total = float(np.mean(epoch_loss_components["total_loss"]))
            print(f"\nEpoch {epoch} Loss Components:")
            print(f"  Standard NLL: {avg_standard_nll:.4f}")
            print(f"  GP NLL:       {avg_gp_nll:.4f}")
            print(f"  Total Loss:   {avg_total:.4f}")

        test_masked_learnmask_gp(
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

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "total_steps": total_steps,
        }
        if gp_covariance_module is not None:
            checkpoint["gp_covariance_state_dict"] = gp_covariance_module.state_dict()
        if hasattr(model, "get_architecture_config"):
            checkpoint["model_config"] = model.get_architecture_config()

        if (epoch + 1) % save_checkpoint_every == 0:
            torch.save(checkpoint, f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth")
        torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

    final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
    print(f"Training completed. Saving final model at {final_model_path}...")
    final_checkpoint: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if hasattr(model, "get_architecture_config"):
        final_checkpoint["model_config"] = model.get_architecture_config()
    torch.save(final_checkpoint, final_model_path)


def test_masked_learnmask_gp(
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

    all_latents: list[torch.Tensor] = []
    all_channel_variances: list[torch.Tensor] = []
    all_channel_maes: list[torch.Tensor] = []
    all_channel_mses: list[torch.Tensor] = []

    with torch.no_grad():
        for idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(test_dataloader, desc=f"Testing epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            _, _, masked_img, active_channel_ids = apply_channel_masking(
                img,
                channel_ids,
                fully_masked_channels_max_frac=fully_masked_channels_max_frac,
                apply_channel_subset_sampling=False,
            )

            pixel_mask = get_pixel_mask(masked_img, spatial_masking_ratio, mask_patch_size)

            latent = model.encode(masked_img, active_channel_ids, spatial_mask=pixel_mask)["output"]
            output = model.decode(latent, channel_ids)
            mi, logvar = output.unbind(dim=-1)
            mi = torch.sigmoid(mi)
            logvar = torch.clamp(logvar, -15.0, 15.0)

            latent = normalize(latent.mean(dim=(2, 3)), p=2, dim=1)
            all_latents.append(latent.cpu())

            variance_per_channel = torch.exp(logvar).mean(dim=(0, 2, 3))
            mae_per_channel = torch.abs(img - mi).mean(dim=(0, 2, 3))
            mse_per_channel = torch.square(img - mi).mean(dim=(0, 2, 3))
            all_channel_variances.append(variance_per_channel.cpu())
            all_channel_maes.append(mae_per_channel.cpu())
            all_channel_mses.append(mse_per_channel.cpu())

            batch_var_mse_corr = torch.corrcoef(
                torch.stack([variance_per_channel.cpu(), mse_per_channel.cpu()])
            )[0, 1].item()
            if math.isfinite(batch_var_mse_corr):
                log_validation_batch_metrics(
                    variance_mse_correlation_per_batch=batch_var_mse_corr,
                    step=epoch * len(test_dataloader) + idx,
                )

            if use_gp_loss and gp_loss_fn is not None:
                marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)
                loss, loss_dict = gp_loss_fn(img, mi, logvar, marker_emb)
                running_standard_nll += loss_dict["standard_nll"]
                running_gp_nll += loss_dict["gp_nll"]
                if idx == 0:
                    _, _, K_C = gp_covariance_module._compute_marker_eigen(marker_emb[0])
                    eigvals = torch.linalg.eigvalsh(K_C)
                    print(
                        f"  Marker cov diagnostics — "
                        f"min_eigval: {eigvals.min().item():.4f}, "
                        f"condition_number: {(eigvals.max() / eigvals.min()).item():.2f}"
                    )
            else:
                loss = nll_loss(img, mi, logvar)

            running_loss += loss.item()
            running_mae += torch.abs(img - mi).mean().item()
            running_mse += torch.square(img - mi).mean().item()

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

                sigma = torch.exp(0.5 * logvar)
                uncertainty_img = plot_reconstructs_with_uncertainty(
                    img,
                    mi,
                    sigma,
                    channel_ids,
                    unactive_channels,
                    markers_names_map=marker_names_map,
                    ncols=9,
                )
                log_validation_images(
                    fig=uncertainty_img,
                    panel_idx=panel_idx[0],
                    img_path=img_path[0],
                    epoch=epoch,
                    masked_channels_names=masked_channels_names,
                    img_idx=idx,
                    name_suffix="_sigma",
                )
                plt.close("all")

    val_loss = running_loss / len(test_dataloader)
    val_mae = running_mae / len(test_dataloader)
    val_mse = running_mse / len(test_dataloader)

    latents_cat = torch.cat(all_latents)
    rankme = RankMe(latents_cat)

    all_variances = torch.cat(all_channel_variances)
    all_maes = torch.cat(all_channel_maes)
    all_mses = torch.cat(all_channel_mses)
    variance_mae_corr = torch.corrcoef(
        torch.stack([all_variances.flatten(), all_maes.flatten()])
    )[0, 1].item()
    variance_mse_corr = torch.corrcoef(
        torch.stack([all_variances.flatten(), all_mses.flatten()])
    )[0, 1].item()

    val_metrics: dict[str, Any] = {
        "val_loss": val_loss,
        "val_mae": val_mae,
        "val_mse": val_mse,
        "latent_rankme": rankme,
        "variance_mae_correlation": variance_mae_corr,
        "variance_mse_correlation": variance_mse_corr,
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
        print(f"GP NLL:       {val_metrics['val_gp_nll']:.4f}")
    print(f"MAE: {val_mae:.6f}")
    print(f"MSE: {val_mse:.6f}")
    print(f"Pearson MAE vs Var: {variance_mae_corr:.4f}")
    print(f"Pearson MSE vs Var: {variance_mse_corr:.4f}")
    print("=" * 90)
    print()

    return val_metrics


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

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

    num_channels = len(TOKENIZER)
    model_config: dict[str, Any] = {
        "num_channels": num_channels,
        "encoder_config": config.encoder_config.model_dump(),
        "decoder_config": config.decoder_config.model_dump(),
    }

    use_gp_loss = getattr(config, "use_gp_loss", False)
    use_kronecker_gp = getattr(config, "use_kronecker_gp", False)
    use_marker_covariance = getattr(config, "use_marker_covariance", False)
    lambda_gp = getattr(config, "lambda_gp", 0.1)
    gp_kernel_jitter = getattr(config, "gp_kernel_jitter", 1e-2)
    gp_lengthscale = getattr(config, "gp_lengthscale", 5.0)
    gp_downscale_factor = getattr(config, "gp_downscale_factor", 1)
    marker_embed_dim = getattr(config, "marker_embed_dim", 32)
    marker_jitter = getattr(config, "marker_jitter", 1e-2)

    assert use_gp_loss and use_kronecker_gp and use_marker_covariance, (
        "This script combines learnmask with Kronecker marker GP loss. "
        "Set use_gp_loss=true, use_kronecker_gp=true, use_marker_covariance=true."
    )

    print("\nGP Loss Configuration:")
    print(f"  Lambda GP:        {lambda_gp}")
    print(f"  Kernel Jitter:    {gp_kernel_jitter}")
    print(f"  Lengthscale:      {gp_lengthscale}")
    print(f"  Downscale Factor: {gp_downscale_factor}")
    print(f"  Marker Embed Dim: {marker_embed_dim}")
    print(f"  Marker Jitter:    {marker_jitter}\n")

    H, W = SIZE
    H_gp = H // gp_downscale_factor
    W_gp = W // gp_downscale_factor
    assert H_gp == W_gp, (
        f"Kronecker GP requires square spatial grid, got {H_gp}x{W_gp}."
    )

    hk_cfg = config.encoder_config
    if len(hk_cfg.ma_layers_blocks) == 0:
        hk_input_dim = 1
    else:
        hk_input_dim = hk_cfg.ma_embedding_dims[-1]
    hk_embed_dim = hk_cfg.pm_embedding_dims[0]
    hk_kernel_size = hk_cfg.hyperkernel_config.kernel_size
    hyperkernel_model_dim = hk_embed_dim * (hk_kernel_size ** 2) * hk_input_dim

    gp_covariance_module = KroneckerMarkerCovariance(
        grid_size=H_gp,
        marker_embed_dim=marker_embed_dim,
        hyperkernel_model_dim=hyperkernel_model_dim,
        kernel_jitter=gp_kernel_jitter,
        marker_jitter=marker_jitter,
        spatial_matern_kernel_length_scale=gp_lengthscale,
        device=device,
    ).to(device)

    gp_loss_fn = HybridKroneckerMarkerGPNLLLoss(
        covariance_module=gp_covariance_module,
        lambda_gp=lambda_gp,
        downscale_factor=gp_downscale_factor,
        device=device,
    )
    print(f"Using Kronecker Marker GP loss with lambda_gp={lambda_gp}")

    start_epoch = 0
    checkpoint = None
    if config.resolve_checkpoint():
        assert config.from_checkpoint is not None
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        model = MultiplexAutoencoder.load_from_checkpoint(
            checkpoint, model_config=model_config
        ).to(device)
        if "gp_covariance_state_dict" in checkpoint:
            gp_covariance_module.load_state_dict(checkpoint["gp_covariance_state_dict"])
        else:
            logger.warning(
                "Checkpoint missing 'gp_covariance_state_dict' — "
                "KroneckerMarkerCovariance starts from random init"
            )
        start_epoch = checkpoint.get("epoch", -1) + 1
    else:
        model = MultiplexAutoencoder(**model_config).to(device)

    # When extending training (bumping config.epochs), use reset_lr_schedule: true to get
    # a fresh cosine cycle. Without it, total_steps is reused from the checkpoint, and if
    # config.epochs > original epochs the scheduler may be past its annealing boundary.
    if checkpoint is not None and "total_steps" in checkpoint and not config.reset_lr_schedule:
        total_steps = checkpoint["total_steps"]
    else:
        remaining_epochs = config.epochs - start_epoch
        total_steps = len(train_dataloader) * remaining_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    params_to_optimize = list(model.parameters()) + list(gp_covariance_module.parameters())
    optimizer = optim.AdamW(
        params_to_optimize, lr=config.peak_lr, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        num_annealing_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )

    if checkpoint is not None and not config.reset_lr_schedule:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    comet_config = config.model_dump()
    comet_config.update(
        {
            "use_gp_loss": use_gp_loss,
            "use_kronecker_gp": use_kronecker_gp,
            "use_marker_covariance": use_marker_covariance,
            "lambda_gp": lambda_gp,
            "gp_kernel_jitter": gp_kernel_jitter,
            "gp_lengthscale": gp_lengthscale,
            "gp_downscale_factor": gp_downscale_factor,
            "marker_embed_dim": marker_embed_dim,
            "marker_jitter": marker_jitter,
        }
    )
    init_experiment(comet_config)

    train_masked_learnmask_gp(
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        gp_covariance_module=gp_covariance_module,
        gp_loss_fn=gp_loss_fn,
        total_steps=total_steps,
        use_gp_loss=use_gp_loss,
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
