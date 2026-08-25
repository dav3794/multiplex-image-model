"""Utilities package for multiplex image model training."""

# Configuration
from .configuration import (
    DecoderConfig,
    DINOHeadConfig,
    DINOTrainingConfig,
    EncoderConfig,
    HyperkernelConfig,
    ModuleConfig,
    FinetuneConfig,
    TrainingConfig,
)

# Masking
from .masking import (
    apply_channel_masking,
    apply_spatial_masking,
)
from .optim import (
    ClampWithGrad,
    cosine_schedule_at,
    get_scheduler_with_warmup,
)

# Logging and visualization
from .train_logging import (
    finish_experiment,
    get_run_name,
    init_experiment,
    log_training_metrics,
    log_validation_images,
    log_validation_metrics,
    plot_reconstructs_with_masks,
    plot_reconstructs_with_uncertainty,
)

__all__ = [
    # Configuration
    "HyperkernelConfig",
    "ModuleConfig",
    "EncoderConfig",
    "DecoderConfig",
    "DINOHeadConfig",
    "DINOTrainingConfig",
    "FinetuneConfig",
    "TrainingConfig",
    # Masking
    "apply_channel_masking",
    "apply_spatial_masking",
    # Logging
    "plot_reconstructs_with_uncertainty",
    "plot_reconstructs_with_masks",
    "init_experiment",
    "log_training_metrics",
    "log_validation_metrics",
    "log_validation_images",
    "get_run_name",
    "finish_experiment",
    # Optim
    "ClampWithGrad",
    "cosine_schedule_at",
    "get_scheduler_with_warmup",
]
