import torch
import torch.nn as nn

from .base_modules import Block, Encoder
from .registry import BLOCK_REGISTRY, ENCODER_REGISTRY


@BLOCK_REGISTRY.register("resnetbasic")
class ResNetBasicBlock(Block):
    """ResNet Basic Block (2 conv layers)."""

    expansion = 1

    def __init__(
        self,
        dim: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        """Initialize Basic Block.

        Args:
            dim (int): Number of channels.
            stride (int, optional): Stride for first conv layer. Defaults to 1.
            downsample (nn.Module | None, optional): Downsample layer for residual. Defaults to None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor [B, C, H, W]
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BLOCK_REGISTRY.register("resnetbottleneck")
class ResNetBottleneck(Block):
    """ResNet Bottleneck Block (1x1 -> 3x3 -> 1x1 conv layers)."""

    expansion = 4

    def __init__(
        self,
        dim: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        bottleneck_ratio: int = 4,
    ):
        """Initialize Bottleneck Block.

        Args:
            dim (int): Number of output channels.
            stride (int, optional): Stride for 3x3 conv layer. Defaults to 1.
            downsample (nn.Module | None, optional): Downsample layer for residual. Defaults to None.
            bottleneck_ratio (int, optional): Expansion ratio for bottleneck. Defaults to 4.
        """
        super().__init__()
        width = dim // bottleneck_ratio

        self.conv1 = nn.Conv2d(dim, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]

        Returns:
            torch.Tensor: Output tensor [B, C, H, W]
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@ENCODER_REGISTRY.register("resnet")
class ResNetEncoder(Encoder):
    """ResNet Encoder backbone for encoding images."""

    def __init__(
        self,
        input_channels: int,
        layers_blocks: list[int],
        embedding_dims: list[int],
        stem: bool = True,
        block_type: str = "resnetbottleneck",
        block_parameters: dict | None = None,
    ):
        """Initialize the ResNet Encoder.

        Args:
            input_channels (int): Number of input channels.
            layers_blocks (list[int]): Number of blocks in each layer.
            embedding_dims (list[int]): Embedding dimensions for each layer.
            stem (bool, optional): Whether to use a stem layer. Defaults to True.
            block_type (str, optional): Type of residual block ('resnetbasic' or 'resnetbottleneck'). Defaults to 'resnetbottleneck'.
            block_parameters (dict | None, optional): Additional parameters to pass to block constructor.
                Defaults to None (uses block defaults).
        """
        super().__init__()

        # Select block class
        if block_type == "resnetbasic":
            block_class = ResNetBasicBlock
        elif block_type == "resnetbottleneck":
            block_class = ResNetBottleneck
        else:
            raise ValueError(
                f"Unknown block_type: {block_type}. Use 'resnetbasic' or 'resnetbottleneck'."
            )

        block_parameters = block_parameters or {}

        # Stem
        if stem:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    embedding_dims[0],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.BatchNorm2d(embedding_dims[0]),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Identity()

        # Build layers
        self.layers = nn.ModuleList()
        for idx, (num_blocks, dim) in enumerate(zip(layers_blocks, embedding_dims)):
            # Downsample if not first layer and dimension changes
            stride = 1 if idx == 0 else 2
            prev_dim = embedding_dims[idx - 1] if idx > 0 else embedding_dims[0]

            layer_blocks = []
            for block_idx in range(num_blocks):
                # First block of each layer (except first layer) performs downsampling
                block_stride = stride if block_idx == 0 and idx > 0 else 1

                # Add downsample projection if needed
                downsample = None
                if block_stride != 1 or prev_dim != dim:
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            prev_dim,
                            dim,
                            kernel_size=1,
                            stride=block_stride,
                            bias=False,
                        ),
                        nn.BatchNorm2d(dim),
                    )

                layer_blocks.append(
                    block_class(
                        dim,
                        stride=block_stride,
                        downsample=downsample,
                        **block_parameters,
                    )
                )
                prev_dim = dim  # Update for next block in same layer

            self.layers.append(nn.Sequential(*layer_blocks))

    def forward(self, x: torch.Tensor, return_features: bool = False) -> dict:
        """Forward pass of the ResNet.

        Args:
            x (torch.Tensor): Images batch tensor with shape [B, C, H, W]
            return_features (bool, optional): If True, returns the features after each layer. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        x = self.stem(x)

        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features
        return outputs
