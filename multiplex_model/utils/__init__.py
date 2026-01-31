"""Utilities package for multiplex image model training."""

# Configuration
from .configuration import (
    DecoderConfig,
    EncoderConfig,
    HyperkernelConfig,
    TrainingConfig,
    build_wandb_config,
)

# Masking
from .masking import (
    apply_channel_masking,
    apply_spatial_masking,
)
from .optim import (
    ClampWithGrad,
    get_scheduler_with_warmup,
)

# Logging and visualization
from .train_logging import (
    finish_wandb_run,
    get_run_name,
    init_wandb_run,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
    plot_reconstructs_with_uncertainty,
)

__all__ = [
    # Configuration
    "HyperkernelConfig",
    "EncoderConfig",
    "DecoderConfig",
    "TrainingConfig",
    "build_wandb_config",
    # Masking
    "apply_channel_masking",
    "apply_spatial_masking",
    # Logging
    "plot_reconstructs_with_uncertainty",
    "plot_reconstructs_with_masks",
    "init_wandb_run",
    "log_training_metrics",
    "log_validation_metrics",
    "log_validation_images",
    "get_run_name",
    "finish_wandb_run",
    # Optim
    "ClampWithGrad",
    "get_scheduler_with_warmup",
]
