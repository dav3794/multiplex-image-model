import torch
import torch.nn as nn
from torchvision.models.vision_transformer import EncoderBlock as TransformerBlock

from .base_modules import Block, Encoder, LayerNorm
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY


@BLOCK_REGISTRY.register("vit")
class ViTTransformerBlock(Block):
    """Vision Transformer Block wrapper using torchvision's EncoderBlock."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize ViT Transformer block.

        Args:
            dim (int): Embedding dimension.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            attn_dropout (float, optional): Attention dropout rate. Defaults to 0.0.
        """
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.block = TransformerBlock(
            num_heads=num_heads,
            hidden_dim=dim,
            mlp_dim=mlp_hidden_dim,
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
        # Convert from [B, C, H, W] to sequence format
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Apply ViT transformer block
        x = self.block(x)

        # Convert back to [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


@ENCODER_REGISTRY.register("vit")
class ViTEncoder(Encoder):
    """Vision Transformer Encoder backbone for encoding images."""

    def __init__(
        self,
        input_channels: int,
        layers_blocks: list[int],
        embedding_dims: list[int],
        stem: bool = True,
        patch_size: int = 8,
        image_size: int = 128,
        block_parameters: dict | None = None,
    ):
        """Initialize the ViT Encoder.

        Args:
            input_channels (int): Number of input channels.
            layers_blocks (list[int]): Number of blocks in each layer.
            embedding_dims (list[int]): Embedding dimensions for each layer.
            stem (bool, optional): Whether to use a stem layer. Defaults to True.
            patch_size (int, optional): Size of patches for patch embedding. Defaults to 8.
            image_size (int, optional): Input image size for positional embedding. Defaults to 128.
            block_parameters (dict | None, optional): Additional parameters to pass to ViT Transformer block constructor.
                Can include: num_heads, mlp_ratio, dropout, attn_dropout.
                Defaults to None (uses ViTTransformerBlock defaults).
        """
        super().__init__()

        # Prepare block kwargs
        block_parameters = block_parameters or {}

        self.patch_embeds = nn.ModuleList()

        # Initial patch embedding (stem)
        if stem or patch_size > 1:
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

        # Downsampling between stages
        if len(embedding_dims) > 1:
            for i, out_dim in enumerate(embedding_dims[1:]):
                input_dim = embedding_dims[i]
                self.patch_embeds.append(
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, stride=2)
                )

        # Positional embedding
        current_size = image_size // patch_size if stem else image_size
        num_patches = current_size * current_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, embedding_dims[0]) * 0.02
        )

        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            stage_blocks = []
            for _ in range(blocks):
                stage_blocks.append(ViTTransformerBlock(dim, **block_parameters))
            self.blocks.append(nn.Sequential(*stage_blocks))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """Forward pass of the ViT.

        Args:
            x (torch.Tensor): Images batch tensor with shape [B, C, H, W]
            return_features (bool, optional): If True, returns the features after each stage. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        for i, (patch_embed, blocks) in enumerate(zip(self.patch_embeds, self.blocks)):
            x = patch_embed(x)

            # Add positional embedding only to first stage
            if i == 0:
                B, C, H, W = x.shape
                x_seq = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
                x_seq = x_seq + self.pos_embed[:, : x_seq.size(1), :]
                x = x_seq.transpose(1, 2).reshape(B, C, H, W)

            x = blocks(x)

            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs
