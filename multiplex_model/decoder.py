import torch
from typing import Optional, Type
import torch.nn as nn
from skimage import filters
import numpy as np

class FromNumpyToTensor(nn.Module):
    """
    Converts numpy.ndarray, with shape [C] x H x W, to torch.Tensor, while preserving dtype.
    If channels dimension is not present it is added to tensor as the first dimension.
    """
    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        return x


class ArcsinhNormalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arcsinh(x / 5)


class MinMaxNormalize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - x.amin(dim=[1, 2], keepdim=True)) / x.amax(dim=[1, 2], keepdim=True)
    
class ButterworthFilter(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3, f'Incorrect shape {x.shape}'
        x = x.numpy()
        for i in range(x.shape[0]):
            x[i] = filters.butterworth(
                        x[i],
                        cutoff_frequency_ratio=0.2,
                        order=3.0,
                        high_pass=False,
                        squared_butterworth=True,
                        npad=0,
                    )
        x = torch.from_numpy(x)
        return x


class GlobalNormalize(nn.Module):
    # Clip everything to [0, 3] and divide by 3
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, 0, 3) / 3


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

        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x

class ConvNextBlock(nn.Module):
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
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.ln = nn.LayerNorm(dim)
            self.conv2 = nn.Linear(dim, inter_dim) # equivalent to nn.Conv2d(dim, inter_dim, kernel_size=1)
            self.act = nn.GELU()
            self.grn = GlobalResponseNormalization(inter_dim)
            self.conv3 = nn.Linear(inter_dim, dim) # equivalent to nn.Conv2d(inter_dim, dim, kernel_size=1)

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
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        x = x + residual

        return x

class MultiplexImageDecoder(nn.Module):
    """Decoder for restoring the multiplex image from the embedding tensor."""
    
    def __init__(
            self,
            input_embedding_dim: int,
            decoded_embed_dim: int,
            num_blocks: int,
            scaling_factor: int,
            num_channels: int,
            decoder_layer_type: Optional[Type] = ConvNextBlock,
            **kwargs
        ) -> None:
            """
            Args:
                input_embedding_dim (int): Embedding dimension of the input tensor.
                decoded_embed_dim (int): Embedding dimension of the decoded tensor (before last projections).
                num_blocks (int): Number of multiplex blocks in each intermediate layer.
                scaling_factor (int): Scaling factor for the upsampling.
                num_channels (int): Number of possible output channels/markers.
                decoder_layer_type (Type, optional): Type of the decoder layer. Defaults to ConvNextBlock.
            """
            super().__init__()
            self.scaling_factor = scaling_factor
            self.num_channels = num_channels
            self.decoded_embed_dim = decoded_embed_dim
            self.num_outputs = 2

            self.channel_embed = nn.Embedding(num_channels, input_embedding_dim * decoded_embed_dim) # input projection
            self.channel_biases = nn.Embedding(num_channels, decoded_embed_dim)

            self.decoder = nn.Sequential(*[
                decoder_layer_type(
                    decoded_embed_dim, 
                    **kwargs
                ) for _ in range(num_blocks)
            ])
            self.pred = nn.Conv2d(
                decoded_embed_dim, 
                scaling_factor**2 * self.num_outputs, 
                kernel_size=1
            )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multiplex Image Decoder.

        Args:
            x (torch.Tensor): Input tensor (embedding).
            indices (torch.Tensor): Indices of the markers.

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        B, I, H, W = x.shape
        E, A, O = self.decoded_embed_dim, self.scaling_factor, self.num_outputs

        channel_embeds = self.channel_embed(indices) # [B, C, I*E]
        channel_biases = self.channel_biases(indices) # [B, C, E]
        C = channel_embeds.shape[1]
        N = B * C
        channel_embeds = channel_embeds.reshape(B, C, I, E)
        channel_biases = channel_biases.reshape(B, C, E, 1, 1)

        x = torch.einsum('bihw, bcie -> bcehw', x, channel_embeds)
        x += channel_biases
        x = x.reshape(N, E, H, W)

        x = self.decoder(x)
        x = self.pred(x)

        x = x.reshape(N, A, A, O, H, W).reshape(B, C, A, A, O, H, W)
        x = torch.einsum('bcxyohw -> bchxwyo', x)

        x = x.reshape(B, C, H*A, W*A, O)

        return x
    