"""Configuration models and utilities using Pydantic for validation."""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .train_logging import get_run_name


class ModuleConfig(BaseModel):
    """Configuration for a module type (block, encoder, etc.).

    Can be specified as:
    1. A string (module type name): "convnext"
    2. A dict with type and parameters: {"type": "convnext", "module_parameters": {...}}
    """

    type: str = Field(..., description="Module type name (e.g., 'convnext')")
    module_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the module constructor",
    )

    @classmethod
    def from_string_or_dict(cls, value: str | dict[str, Any]) -> "ModuleConfig":
        """Create ModuleConfig from string or dict.

        Args:
            value: Either a string (module type) or dict with 'type' and optional 'module_parameters'

        Returns:
            ModuleConfig instance
        """
        if isinstance(value, str):
            return cls(type=value)
        elif isinstance(value, dict):
            return cls(**value)
        else:
            raise ValueError(f"ModuleConfig must be string or dict, got {type(value)}")

    class Config:
        extra = "forbid"


class HyperkernelConfig(BaseModel):
    """Configuration for Hyperkernel module."""

    kernel_size: int = Field(1, gt=0, description="Kernel size for convolution")
    padding: int = Field(0, ge=0, description="Padding for convolution")
    stride: int = Field(1, gt=0, description="Stride for convolution")
    use_bias: bool = Field(True, description="Whether to use bias in the hyperkernel")

    class Config:
        extra = "forbid"


class EncoderConfig(BaseModel):
    """Configuration for MultiplexImageEncoder."""

    ma_layers_blocks: list[int] = Field(
        ..., description="Number of blocks in each marker-agnostic layer"
    )
    ma_embedding_dims: list[int] = Field(
        ..., description="Embedding dimensions for marker-agnostic layers"
    )
    pm_layers_blocks: list[int] = Field(
        ..., description="Number of blocks in each pan-marker layer"
    )
    pm_embedding_dims: list[int] = Field(
        ..., description="Embedding dimensions for pan-marker layers"
    )
    hyperkernel_config: HyperkernelConfig = Field(
        ..., description="Hyperkernel configuration", alias="hyperkernel"
    )
    use_latent_norm: bool = Field(
        default=False,
        description="Whether to apply LayerNorm to the latent representation",
    )
    encoder_type: str | ModuleConfig | None = Field(
        default="convnext",
        description=(
            "Encoder type to use for marker-agnostic and pan-marker encoders. "
            "Can be a string (e.g., 'convnext') or a dict with 'type' and 'module_parameters'."
        ),
    )

    @field_validator("ma_layers_blocks", "pm_layers_blocks")
    @classmethod
    def validate_blocks(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            raise ValueError("All block counts must be positive")
        return v

    @field_validator("ma_embedding_dims", "pm_embedding_dims")
    @classmethod
    def validate_embedding_dims(cls, v: list[int]) -> list[int]:
        if any(x <= 0 for x in v):
            raise ValueError("All embedding dimensions must be positive")
        return v

    @field_validator("ma_embedding_dims")
    @classmethod
    def validate_ma_lengths(cls, v: list[int], info) -> list[int]:
        if "ma_layers_blocks" in info.data:
            blocks = info.data["ma_layers_blocks"]
            if len(v) != len(blocks):
                raise ValueError(
                    f"ma_embedding_dims length ({len(v)}) must match ma_layers_blocks length ({len(blocks)})"
                )
        return v

    @field_validator("pm_layers_blocks")
    @classmethod
    def validate_pm_not_empty(cls, v: list[int]) -> list[int]:
        if len(v) == 0:
            raise ValueError(
                "pm_layers_blocks cannot be empty - at least one pan-marker layer is required"
            )
        return v

    @field_validator("pm_embedding_dims")
    @classmethod
    def validate_pm_lengths(cls, v: list[int], info) -> list[int]:
        if len(v) == 0:
            raise ValueError(
                "pm_embedding_dims cannot be empty - at least one pan-marker layer is required"
            )
        if "pm_layers_blocks" in info.data:
            blocks = info.data["pm_layers_blocks"]
            if len(v) != len(blocks):
                raise ValueError(
                    f"pm_embedding_dims length ({len(v)}) must match pm_layers_blocks length ({len(blocks)})"
                )
        return v

    class Config:
        extra = "forbid"


class DecoderConfig(BaseModel):
    """Configuration for MultiplexImageDecoder."""

    decoded_embed_dim: int = Field(
        ..., gt=0, description="Embedding dimension of decoded tensor"
    )
    num_blocks: int = Field(
        ..., gt=0, description="Number of ConvNeXt blocks in decoder"
    )
    hyperkernel_config: HyperkernelConfig = Field(
        ..., description="Hyperkernel configuration", alias="hyperkernel"
    )
    num_outputs: int = Field(
        default=2, gt=0, description="Number of outputs per marker channel"
    )
    block_type: str | ModuleConfig | None = Field(
        default="convnext",
        description="Block type to use in decoder. Can be string or dict with 'type' and 'module_parameters'.",
    )

    @field_validator("block_type", mode="before")
    @classmethod
    def validate_block_type(cls, v) -> ModuleConfig:
        if v is None:
            return ModuleConfig(type="convnext")
        if isinstance(v, ModuleConfig):
            return v
        return ModuleConfig.from_string_or_dict(v)

    class Config:
        extra = "forbid"


class TrainingConfig(BaseModel):
    """Pydantic model for training configuration with validation."""

    # Data parameters
    device: str = Field(
        ..., description="Device to use for training (e.g., 'cuda', 'cpu')"
    )
    input_image_size: tuple[int, int] = Field(
        ..., description="Input image size (height, width)"
    )
    batch_size: int = Field(..., gt=0, description="Batch size for training")
    num_workers: int = Field(..., ge=0, description="Number of data loading workers")

    # Config file paths
    panel_config: str = Field(..., description="Path to panel configuration file")
    tokenizer_config: str = Field(
        ..., description="Path to tokenizer configuration file"
    )

    # Training parameters
    peak_lr: float = Field(..., gt=0, description="Peak learning rate", alias="lr")
    final_lr: float = Field(
        ..., gt=0, description="Final learning rate after annealing"
    )
    frac_warmup_steps: float = Field(
        ..., ge=0, le=1, description="Fraction of steps for warmup"
    )
    weight_decay: float = Field(..., ge=0, description="Weight decay for optimizer")
    gradient_accumulation_steps: int = Field(
        ..., gt=0, description="Number of gradient accumulation steps"
    )
    epochs: int = Field(..., gt=0, description="Number of training epochs")
    beta: float = Field(..., ge=0, description="Beta parameter for beta-NLL loss")

    # Masking parameters
    min_channels_frac: float = Field(
        ..., gt=0, le=1, description="Minimum fraction of channels to keep"
    )
    spatial_masking_ratio: float = Field(
        ..., ge=0, le=1, description="Fraction of spatial patches to mask"
    )
    fully_masked_channels_max_frac: float = Field(
        ..., ge=0, le=1, description="Maximum fraction of channels to fully mask"
    )
    mask_patch_size: int = Field(..., gt=0, description="Size of spatial mask patches")

    # Model architecture
    encoder_config: EncoderConfig = Field(
        ..., description="Encoder configuration", alias="encoder"
    )
    decoder_config: DecoderConfig = Field(
        ..., description="Decoder configuration", alias="decoder"
    )

    # Checkpoint parameters
    from_checkpoint: str | None = Field(
        None,
        description="Path to checkpoint to resume from. Use 'last' to load last checkpoint if available",
    )
    checkpoints_dir: str = Field(
        "checkpoints", description="Directory to save checkpoints"
    )
    save_checkpoint_freq: int = Field(
        ..., gt=0, description="Frequency of checkpoint saving (in epochs)"
    )

    # Comet.ml parameters
    comet_project: str = Field(..., description="Comet.ml project name")
    comet_workspace: str | None = Field(None, description="Comet.ml workspace name")
    comet_api_key: str | None = Field(
        None, description="Comet.ml API key (can also be set via COMET_API_KEY env var)"
    )
    tags: list[str] = Field(
        default_factory=list, description="Tags for Comet.ml experiment"
    )
    run_name: str | None = Field(None, description="Name for Comet.ml experiment")

    def resolve_checkpoint(self) -> bool:
        """Resolve checkpoint path and determine if checkpoint should be loaded.

        If from_checkpoint is 'last', attempts to find the last checkpoint file.
        Updates from_checkpoint to the actual path or None if not found.

        Returns:
            bool: True if checkpoint should be loaded, False otherwise
        """
        if not self.from_checkpoint:
            return False

        if self.from_checkpoint == "last":
            if not self.run_name:
                self.run_name = get_run_name()

            last_possible_checkpoint = (
                f"{self.checkpoints_dir}/last_checkpoint-{self.run_name}.pth"
            )
            if os.path.exists(last_possible_checkpoint):
                self.from_checkpoint = last_possible_checkpoint
                return True
            else:
                print(
                    f"No last checkpoint found at {last_possible_checkpoint}, starting from scratch."
                )
                self.from_checkpoint = None
                return False

        return True

    class Config:
        extra = "forbid"  # Raise error on unknown fields
