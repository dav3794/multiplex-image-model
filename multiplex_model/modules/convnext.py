import torch
import torch.nn as nn
from typing import Dict, List

from .base_modules import Block, Encoder, LayerNorm
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY


class GlobalResponseNormalization(nn.Module):
    """Global Response Normalization (GRN) layer
    from https://arxiv.org/pdf/2301.00808"""

    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, H, W, E = x.shape

        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


@BLOCK_REGISTRY.register("convnext")
class ConvNextBlock(Block):
    """ConvNext2 block"""

    def __init__(
        self,
        dim: int,
        inter_dim: int = None,
        kernel_size: int = 7,
        padding: int = 3,
    ):
        super().__init__()
        inter_dim = inter_dim or dim * 4
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.ln = nn.LayerNorm(dim)
        self.conv2 = nn.Linear(
            dim, inter_dim
        )  # equivalent to nn.Conv2d(dim, inter_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNormalization(inter_dim)
        self.conv3 = nn.Linear(
            inter_dim, dim
        )  # equivalent to nn.Conv2d(inter_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, C, H, W = x.shape
        residual = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.ln(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.conv3(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = x + residual

        return x


@ENCODER_REGISTRY.register("convnext")
class ConvNeXtEncoder(Encoder):
    """ConvNeXT Encoder backbone for encoding images."""

    def __init__(
        self,
        input_channels: int,
        layers_blocks: List[int],
        embedding_dims: List[int],
        stem: bool = True,
        block_parameters: Dict = None,
    ):
        """Initialize the ConvNeXT Encoder.

        Args:
            input_channels (int): Number of input channels.
            layers_blocks (List[int]): Number of blocks in each layer.
            embedding_dims (List[int]): Embedding dimensions for each layer.
            stem (bool, optional): Whether to use a stem layer. Defaults to True.
            block_parameters (Dict, optional): Additional parameters to pass to ConvNextBlock constructor.
                Can include: kernel_size, padding, inter_dim. Defaults to None (uses ConvNextBlock defaults).
        """
        super().__init__()

        # Use ConvNextBlock with optional custom parameters
        block_parameters = block_parameters or {}

        self.norm_poolings = nn.ModuleList()
        if stem:
            stem_layer = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    embedding_dims[0],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                LayerNorm(embedding_dims[0], data_format="channels_first"),
            )
        else:
            stem_layer = nn.Identity()

        self.norm_poolings.append(stem_layer)

        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.norm_poolings.append(
                nn.Sequential(
                    LayerNorm(input_dim, data_format="channels_first"),
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, padding=0, stride=2),
                )
            )

        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(
                nn.Sequential(
                    *[ConvNextBlock(dim, **block_parameters) for _ in range(blocks)]
                )
            )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        """Forward pass of the ConvNeXT.

        Args:
            x (torch.Tensor): Images batch tensor with shape [B, C, H, W]
            return_features (bool, optional): If True, returns the features after each block. Defaults to False.

        Returns:
            Dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []
        for norm_pooling, blocks in zip(self.norm_poolings, self.blocks):
            x = norm_pooling(x)
            x = blocks(x)

            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs
