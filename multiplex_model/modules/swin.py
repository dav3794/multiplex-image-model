import torch
import torch.nn as nn
from torchvision.models.swin_transformer import (
    SwinTransformerBlock as TorchvisionSwinBlock,
)

from .base_modules import Block, Encoder, LayerNorm
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY


@BLOCK_REGISTRY.register("swin")
class SwinTransformerBlock(Block):
    """Swin Transformer Block wrapper using torchvision's SwinTransformerBlock."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int | list[int] = [7, 7],
        shift_size: int | list[int] = [0, 0],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize Swin Transformer block.

        Args:
            dim (int): Embedding dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            window_size (int | list[int], optional): Window size [H, W] or single int for both. Defaults to [7, 7].
            shift_size (int | list[int], optional): Shift size [H, W] or single int for both. Defaults to [0, 0].
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            attn_dropout (float, optional): Attention dropout rate. Defaults to 0.0.
        """
        super().__init__()

        # Normalize window_size and shift_size to lists
        if isinstance(window_size, int):
            window_size = [window_size, window_size]
        if isinstance(shift_size, int):
            shift_size = [shift_size, shift_size]

        self.block = TorchvisionSwinBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor [B, C, H, W]
        """
        # Convert to [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        # Apply Swin transformer block
        x = self.block(x)

        # Convert back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x


@ENCODER_REGISTRY.register("swin")
class SwinTransformerEncoder(Encoder):
    """Swin Transformer Encoder backbone for encoding images."""

    def __init__(
        self,
        input_channels: int,
        layers_blocks: list[int],
        embedding_dims: list[int],
        stem: bool = True,
        patch_size: int = 2,
        block_parameters: dict | None = None,
    ):
        """Initialize the Swin Transformer Encoder.

        Args:
            input_channels (int): Number of input channels.
            layers_blocks (list[int]): Number of blocks in each layer.
            embedding_dims (list[int]): Embedding dimensions for each layer.
            stem (bool, optional): Whether to use a stem layer. Defaults to True.
            patch_size (int, optional): Size of patches for patch embedding. Defaults to 2.
            block_parameters (dict | None, optional): Additional parameters to pass to Swin block constructor.
                Can include: num_heads, window_size, mlp_ratio, dropout, attn_dropout.
                Defaults to None (uses SwinTransformerBlock defaults).
        """
        super().__init__()

        block_parameters = block_parameters or {}

        self.patch_embeds = nn.ModuleList()

        # Initial patch embedding (stem)
        if stem:
            self.patch_embeds.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        embedding_dims[0],
                        kernel_size=patch_size,
                        stride=patch_size,
                    ),
                    LayerNorm(embedding_dims[0], data_format="channels_first"),
                )
            )
        else:
            self.patch_embeds.append(nn.Identity())

        # Patch merging between stages
        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.patch_embeds.append(
                nn.Sequential(
                    LayerNorm(input_dim, data_format="channels_first"),
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, stride=2),
                )
            )

        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            stage_blocks = []
            for i in range(blocks):
                # Alternate between W-MSA and SW-MSA
                win_size = block_parameters.get("window_size", 7)
                shift = win_size // 2 if (i % 2 == 1) else 0
                num_heads = block_parameters.get("num_heads", max(1, dim // 32))

                # Create block kwargs without duplicates
                block_call_kwargs = block_parameters.copy()
                block_call_kwargs["window_size"] = [win_size, win_size]
                block_call_kwargs["shift_size"] = [shift, shift]
                block_call_kwargs["num_heads"] = num_heads

                stage_blocks.append(
                    SwinTransformerBlock(
                        dim,
                        **block_call_kwargs,
                    )
                )
            self.blocks.append(nn.Sequential(*stage_blocks))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """Forward pass of the Swin Transformer.

        Args:
            x (torch.Tensor): Images batch tensor with shape [B, C, H, W]
            return_features (bool, optional): If True, returns the features after each stage. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        for patch_embed, blocks in zip(self.patch_embeds, self.blocks):
            x = patch_embed(x)
            x = blocks(x)

            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs
