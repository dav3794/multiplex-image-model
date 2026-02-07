from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module, ABC):
    """Base block class for all blocks

    All blocks should inherit from this class and implement the forward method.
    The forward method should return a torch.Tensor (not a Dict for flexibility).
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the block

        Args:
            x (torch.Tensor): Input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Output tensor
        """
        raise NotImplementedError


class Encoder(nn.Module, ABC):
    """Base encoder class for all encoders

    Encoders should implement the forward method that returns a Dict
    containing at least 'output' key, and optionally 'features' key.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self, x: torch.Tensor, return_features: bool = False, *args, **kwargs
    ) -> dict:
        """Forward pass of the encoder

        Args:
            x (torch.Tensor): Input tensor
            return_features (bool): Whether to return intermediate features
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict: Dictionary with 'output' and optionally 'features' keys
        """
        raise NotImplementedError


class Identity(nn.Identity):
    """Identity layer that returns a dictionary for compatibility"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> dict:
        return {"output": super().forward(x)}


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first
    from https://github.com/facebookresearch/ConvNeXt-V2/blob/2553895753323c6fe0b2bf390683f5ea358a42b9/models/utils.py#L79
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
