"""Multiplex Image Model Modules

This package contains the modular components for building multiplex image models.

Base Abstractions:
- Block: Base class for all blocks
- Encoder: Base class for all encoders
- Identity: Identity layer with dict output
- LayerNorm: Layer normalization supporting different data formats

ConvNeXt Architecture:
- GlobalResponseNormalization: GRN layer
- ConvNextBlock: ConvNeXt block implementation
- ConvNeXtEncoder: ConvNeXt encoder backbone

Vision Transformer (ViT) Architecture:
- ViTTransformerBlock: Vision Transformer block
- ViTEncoder: Vision Transformer encoder backbone

Swin Transformer Architecture:
- SwinTransformerBlock: Swin Transformer block with shifted window attention
- SwinTransformerEncoder: Swin Transformer encoder backbone

ResNet Architecture:
- ResNetBasicBlock: ResNet basic block (2 conv layers)
- ResNetBottleneck: ResNet bottleneck block (1x1 -> 3x3 -> 1x1)
- ResNetEncoder: ResNet encoder backbone

Multiplex-specific Components:
- Hyperkernel: Dynamic kernel generation for marker-specific processing
- MultiplexImageEncoder: Encoder for multiplex images
- MultiplexImageDecoder: Decoder for multiplex images
- MultiplexAutoencoder: Complete autoencoder model

Registry System:
- BLOCK_REGISTRY: Registry for block types
- ENCODER_REGISTRY: Registry for encoder types
- build_from_config: Build instances from configuration dictionaries

To add new architectures, extend the base classes (Block, Encoder)
and register them using the appropriate registry decorator.
"""

# Registry system
from .registry import (
    BLOCK_REGISTRY,
    ENCODER_REGISTRY,
    build_from_config,
    resolve_block_class,
    resolve_encoder_class,
)

# Base abstractions
from .base_modules import (
    Block,
    Encoder,
    Identity,
    LayerNorm,
)

# ConvNeXt architecture
from .convext import (
    ConvNextBlock,
    ConvNeXtEncoder,
    GlobalResponseNormalization,
)

# Vision Transformer architecture
from .vit import (
    ViTTransformerBlock,
    ViTEncoder,
)

# Swin Transformer architecture
from .swin import (
    SwinTransformerBlock,
    SwinTransformerEncoder,
)

# ResNet architecture
from .resnet import (
    ResNetBasicBlock,
    ResNetBottleneck,
    ResNetEncoder,
)

# Multiplex-specific components
from .immuvis import (
    Hyperkernel,
    MultiplexAutoencoder,
    MultiplexImageDecoder,
    MultiplexImageEncoder,
)


__all__ = [
    # Registry system
    "BLOCK_REGISTRY",
    "ENCODER_REGISTRY",
    "build_from_config",
    "resolve_block_class",
    "resolve_encoder_class",
    # Base abstractions
    "Block",
    "Encoder",
    "Identity",
    "LayerNorm",
    # ConvNeXt architecture
    "GlobalResponseNormalization",
    "ConvNextBlock",
    "ConvNeXtEncoder",
    # Vision Transformer architecture
    "ViTTransformerBlock",
    "ViTEncoder",
    # Swin Transformer architecture
    "SwinTransformerBlock",
    "SwinTransformerEncoder",
    # ResNet architecture
    "ResNetBasicBlock",
    "ResNetBottleneck",
    "ResNetEncoder",
    # Multiplex-specific components
    "Hyperkernel",
    "MultiplexImageEncoder",
    "MultiplexImageDecoder",
    "MultiplexAutoencoder",
]
