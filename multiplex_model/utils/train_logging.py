"""Logging and visualization utilities for training and validation."""

from math import ceil
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def plot_reconstructs_with_uncertainty(
    orig_img: torch.Tensor,
    reconstructed_img: torch.Tensor,
    sigma_plot: torch.Tensor,
    channel_ids: torch.Tensor,
    masked_ids: torch.Tensor,
    markers_names_map: Dict[int, str],
    ncols: int = 9,
    scale_by_max: bool = True,
    partially_masked_ids: List[int] = [],
):
    """Plot the original image and the reconstructed image with uncertainty.

    Args:
        orig_img (torch.Tensor): Original image
        reconstructed_img (torch.Tensor): Reconstructed image
        sigma_plot (torch.Tensor): Uncertainty image
        channel_ids (torch.Tensor): Indices of the original channels
        masked_ids (torch.Tensor): Indices of the masked/reconstructed channels
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 9.
        scale_by_max (bool, optional): Whether to scale the images by their maximum value. Defaults to True.
        partially_masked_ids (List[int], optional): List of channel IDs that were only partially masked. Defaults to [].

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # plot original image
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig_orig, axs_orig = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs_orig.flatten()
    for i in range(0, len(ax_flat), 3):
        j = i // 3

        # first original image
        ax_img = ax_flat[i]
        ax_img.axis("off")

        ax_reconstructed = ax_flat[i + 1]
        ax_reconstructed.axis("off")

        ax_uncertainty = ax_flat[i + 2]
        ax_uncertainty.axis("off")

        if j < num_channels:
            marker_name = markers_names_map[channel_ids[0, j].item()]
            ax_img.imshow(orig_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1)
            ax_img.set_title(f"Original\n{marker_name}")

            ax_reconstructed.imshow(
                reconstructed_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1
            )
            is_masked = channel_ids[0, j].item() in masked_ids
            is_partially_masked = channel_ids[0, j].item() in partially_masked_ids
            if is_partially_masked:
                masked_str = " (partially masked)"
            elif is_masked:
                masked_str = " (masked)"
            else:
                masked_str = ""
            ax_reconstructed.set_title(f"Reconstructed{masked_str}\n{marker_name}")

            if scale_by_max:
                var_min = sigma_plot[0, j].min().item()
                var_max = sigma_plot[0, j].max().item()
            else:
                var_min = None
                var_max = None

            ax_uncertainty.imshow(
                sigma_plot[0, j].cpu().numpy(),
                cmap="CMRmap",
                vmin=var_min,
                vmax=var_max,
            )
            ax_uncertainty.set_title(f"Variance\n{marker_name}")

    fig_orig.tight_layout()

    return fig_orig


def plot_reconstructs_with_masks(
    orig_img: torch.Tensor,
    reconstructed_img: torch.Tensor,
    pixel_masks: torch.Tensor,
    channel_ids: torch.Tensor,
    fully_masked_ids: List[int],
    markers_names_map: Dict[int, str],
    ncols: int = 9,
):
    """Plot the original image, masked image (with white pixels where masked), and reconstruction.

    Args:
        orig_img (torch.Tensor): Original image [B, C, H, W]
        reconstructed_img (torch.Tensor): Reconstructed image [B, C_all, H, W] (all channels)
        pixel_masks (torch.Tensor): Boolean pixel-level masks [B, C_active, H, W] where True = masked
        channel_ids (torch.Tensor): Indices of all channels [B, C_all]
        fully_masked_ids (List[int]): List of channel IDs that were fully masked (dropped)
        markers_names_map (Dict[int, str]): Channel index to marker name mapping
        ncols (int, optional): Number of columns on the plot. Defaults to 9.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    num_channels = orig_img.shape[1]

    nrows = ceil(num_channels / (ncols // 3))
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    ax_flat = axs.flatten()

    # Create mapping from channel_id to index in masked_img
    active_channel_ids = [
        cid for cid in channel_ids[0].tolist() if cid not in fully_masked_ids
    ]
    channel_to_masked_idx = {cid: idx for idx, cid in enumerate(active_channel_ids)}

    for i in range(0, len(ax_flat), 3):
        j = i // 3

        ax_orig = ax_flat[i]

        ax_masked = ax_flat[i + 1]

        ax_reconstructed = ax_flat[i + 2]

        if j < num_channels:
            channel_id = channel_ids[0, j].item()
            marker_name = markers_names_map[channel_id]

            # Show original
            ax_orig.imshow(orig_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1)
            ax_orig.set_title(f"Original\n{marker_name}")
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            # Add black frame
            for spine in ax_orig.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
                spine.set_visible(True)

            # Show masked version
            if channel_id in fully_masked_ids:
                # Fully masked channel - show all white (RGBA)
                white_img = np.ones((*orig_img[0, j].shape, 4))
                white_img[..., :3] = 1.0  # RGB = white
                white_img[..., 3] = 1.0  # Alpha = 100% opaque
                ax_masked.imshow(white_img)
                ax_masked.set_title(f"Masked (fully)\n{marker_name}")
                ax_masked.set_xticks([])
                ax_masked.set_yticks([])
                # Add black frame
                for spine in ax_masked.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1)
                    spine.set_visible(True)
            else:
                # Partially masked channel - show with white pixels where masked
                masked_idx = channel_to_masked_idx[channel_id]

                # Convert grayscale to RGBA using colormap (image already normalized to 0-1)
                cmap = plt.cm.CMRmap
                img_data = orig_img[0, j].cpu().numpy()
                rgba_img = cmap(img_data)  # Apply colormap directly

                # Set masked pixels to pure white with 100% opacity
                mask_np = pixel_masks[0, masked_idx].cpu().numpy()
                rgba_img[mask_np] = [1.0, 1.0, 1.0, 1.0]  # Pure white, fully opaque

                ax_masked.imshow(rgba_img)
                ax_masked.set_title(f"Masked\n{marker_name}")
                ax_masked.set_xticks([])
                ax_masked.set_yticks([])
                # Add black frame
                for spine in ax_masked.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1)
                    spine.set_visible(True)

            # Show reconstruction
            ax_reconstructed.imshow(
                reconstructed_img[0, j].cpu().numpy(), cmap="CMRmap", vmin=0, vmax=1
            )
            ax_reconstructed.set_title(f"Reconstructed\n{marker_name}")
            ax_reconstructed.set_xticks([])
            ax_reconstructed.set_yticks([])
            # Add black frame
            for spine in ax_reconstructed.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1)
                spine.set_visible(True)
        else:
            # Turn off empty subplots
            ax_orig.axis("off")
            ax_masked.axis("off")
            ax_reconstructed.axis("off")

    fig.tight_layout()
    return fig


def init_wandb_run(config: Dict[str, Any]) -> None:
    """Initialize wandb run with the given configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing wandb settings
    """
    wandb.init(
        project=config["wandb_project"],
        config=config,
        tags=config.get("tags", []),
        name=config.get("run_name", None),
    )


def log_training_metrics(
    loss: float,
    lr: float,
    mu: float,
    logvar: float,
    mae: float,
    mse: float,
    step: Optional[int] = None,
) -> None:
    """Log training metrics to wandb.

    Args:
        loss (float): Training loss
        lr (float): Learning rate
        mu (float): Mean of predicted values
        logvar (float): Log variance
        mae (float): Mean absolute error
        mse (float): Mean squared error
        step (Optional[int]): Step number for logging
    """
    metrics = {
        "train/loss": loss,
        "train/lr": lr,
        "train/µ": mu,
        "train/logvar": logvar,
        "train/mae": mae,
        "train/mse": mse,
    }
    wandb.log(metrics, step=step)


def log_validation_metrics(
    val_loss: float,
    val_mae: float,
    val_mse: float,
    latent_rankme: float,
    latent_intrinsic_dim: float,
    epoch: int,
    variance_mae_correlation: Optional[float] = None,
) -> None:
    """Log validation metrics to wandb.

    Args:
        val_loss (float): Validation loss
        val_mae (float): Validation MAE
        val_mse (float): Validation MSE
        latent_rankme (float): RankMe metric for latent representations
        latent_intrinsic_dim (float): Intrinsic dimension of latent representations
        epoch (int): Current epoch number
        variance_mae_correlation (Optional[float]): Pearson correlation between predicted variances and MAEs per channel
    """
    metrics = {
        "val/loss": val_loss,
        "val/mae": val_mae,
        "val/mse": val_mse,
        "val/latent_RankMe": latent_rankme,
        "val/latent_intinsic_dim": latent_intrinsic_dim,
        "epoch": epoch,
    }
    if variance_mae_correlation is not None:
        metrics["val/variance_mae_correlation"] = variance_mae_correlation
    wandb.log(metrics)


def log_validation_images(
    fig: plt.Figure,
    panel_idx: int,
    img_path: str,
    epoch: int,
    masked_channels_names: str,
) -> None:
    """Log validation reconstruction images to wandb.

    Args:
        fig (plt.Figure): Matplotlib figure to log
        panel_idx (int): Panel index
        img_path (str): Path to the image
        epoch (int): Current epoch number
        masked_channels_names (str): Names of masked channels
    """
    caption = (
        f"Resulting outputs (dataset {panel_idx}, image {img_path}, epoch {epoch+1})"
        f"\n\nMasked channels: {masked_channels_names}"
    )
    wandb.log({"val/reconstructions": wandb.Image(fig, caption=caption)})


def get_run_name() -> str | None:
    """Get the current wandb run name.

    Returns:
        str | None : Current run name or None if no run is active
    """
    return wandb.run.name if wandb.run else None


def finish_wandb_run() -> None:
    """Finish the current wandb run."""
    wandb.finish()
