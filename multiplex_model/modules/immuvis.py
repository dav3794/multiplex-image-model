from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_modules import Block, Encoder, Identity, LayerNorm
from .registry import resolve_block_class, resolve_encoder_class


class Hyperkernel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        input_dim: int,
        embedding_dim: int,
        module_type: Literal["encoder", "decoder"],
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        use_bias: bool = True,
    ):
        """Initialize the Hyperkernel model

        Args:
            num_channels (int): Number of channels in the input tensor
            input_dim (int): Input dimension of each channel
            embedding_dim (int): Embedding dimension for the input tensor
            module_type (Literal['encoder', 'decoder']): Whether the Hyperkernel is used in encoder or decoder
            kernel_size (int, optional): Kernel size for the conv layer (already squared). Model embedding will be embedding_dim*kernel_size**2.
            padding (int, optional): Padding for the conv layer. Defaults to 1.
            stride (int, optional): Stride for the conv layer. Defaults to 1.
            use_bias (bool, optional): Whether to use bias in the conv layer. Defaults to True.
        """
        super(Hyperkernel, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.num_channels = num_channels
        if kernel_size == stride == 1 and padding == 0:
            self.layer_type = "linear"
            self.kernel_size = 1
        else:
            self.layer_type = "conv"
            self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.module_type = module_type

        self.out_dim = self.embedding_dim * self.kernel_size**2
        self.model_dim = self.out_dim * self.input_dim
        self.hyperkernel_weights = nn.Embedding(num_channels, self.model_dim)

        self.use_bias = use_bias
        if use_bias:
            if module_type == "encoder":
                self.hyperkernel_bias = nn.Parameter(
                    torch.zeros(1, self.embedding_dim, 1, 1)
                )
            else:
                self.hyperkernel_bias = nn.Embedding(num_channels, self.embedding_dim)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Returns the superkernel weights for the given indices.

        Args:
            x (torch.Tensor): Input tensor of shape (B, X, H, W).
                X is C*I for encoder and I for decoder.
            indices (torch.Tensor): Indices of the markers in the input tensor.
                Shape: (B, C), where B is batch size and C is number of channels.

        Returns:
            torch.Tensor: Superkernel-transformed tensor.
                Shape: (B, E, H, W) for encoder and (B, C, E, H, W) for decoder.
        """
        B, C = indices.shape
        I = self.input_dim
        E = self.embedding_dim
        O = self.out_dim  # E or E*K*K
        CI = C * I
        spatial_shape = x.shape[-2:]

        weights = self.hyperkernel_weights(indices).to(x.dtype)  # (B, C, I*O)
        weights = weights.reshape(B, C, I, O)

        if self.layer_type == "conv":
            K = self.kernel_size
            weights = weights.reshape(B, C, I, E, K, K)

        tailing_weights_shape = weights.shape[3:]

        if self.module_type == "encoder":
            weights = weights.reshape(B, CI, *tailing_weights_shape)

            if self.layer_type == "conv":
                # treat batch as group for conv
                weights = weights.transpose(1, 2).reshape(
                    B * E, CI, K, K
                )  # (B*E, C*I, K, K)
                x = x.reshape(1, B * CI, *spatial_shape)  # (1, B*C*I, H, W)
                x = F.conv2d(
                    x, weights, padding=self.padding, stride=self.stride, groups=B
                )
                spatial_shape = x.shape[-2:]
                x = x.reshape(B, E, *spatial_shape)  # (B, E, H, W)
            else:
                x = torch.einsum("bchw, bce -> behw", x, weights)

            if self.use_bias:
                x = x + self.hyperkernel_bias

        else:  # decoder
            if self.layer_type == "conv":
                # treat batch and channels as groups for conv
                x = x.unsqueeze(1).expand(-1, C, -1, -1, -1)  # (B, C, I, H, W)
                x = x.reshape(1, B * C * I, *spatial_shape)  # (1, B*C*I, H, W)

                weights = (
                    weights.reshape(B * C, I, E, K, K)
                    .transpose(1, 2)
                    .reshape(B * C * E, I, K, K)
                )  # (B*C*E, I, K, K)

                x = F.conv2d(
                    x, weights, padding=self.padding, stride=self.stride, groups=B * C
                )
                spatial_shape = x.shape[-2:]
                x = x.reshape(B, C, E, *spatial_shape)  # (B, C, E, H, W)

            else:
                x = torch.einsum("bihw, bcie -> bcehw", x, weights)

            if self.use_bias:
                channel_biases = self.hyperkernel_bias(indices)  # [B, C, E]
                channel_biases = channel_biases.unsqueeze(-1).unsqueeze(
                    -1
                )  # [B, C, E, 1, 1]
                x = x + channel_biases

        return x


class MultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

    def __init__(
        self,
        num_channels: int,
        ma_layers_blocks: list[int],
        ma_embedding_dims: list[int],
        hyperkernel_config: dict,
        pm_layers_blocks: list[int],
        pm_embedding_dims: list[int],
        use_latent_norm: bool = False,
        encoder_type: str | type[Encoder] | dict = "convnext",
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            num_channels (int): Number of all possible channels/markers.
            ma_layers_blocks (List[int]): Number of blocks in each marker-agnostic layer.
            ma_embedding_dims (List[int]): Embedding dimensions for each marker-agnostic layer.
            hyperkernel_config (Dict): Configuration for the hyperkernel.
            pm_layers_blocks (List[int]): Number of blocks in each pan-marker layer.
            pm_embedding_dims (List[int]): Embedding dimensions for each pan-marker layer.
            use_latent_norm (bool, optional): Whether to apply LayerNorm to the latent representation. Defaults to False.
            encoder_type (Union[str, Type[Encoder], Dict], optional): Type of encoder to use.
                Can be a string (registry name), Encoder class, or config dict with 'type' and 'module_parameters'.
                For ConvNeXtEncoder, module_parameters can include 'block_parameters' dict with ConvNextBlock parameters
                (e.g., kernel_size, padding, inter_dim).
                Defaults to "convnext".
        """
        super().__init__()

        # Resolve encoder class
        encoder_cls = resolve_encoder_class(encoder_type)

        # Prepare encoder kwargs - extract only module_parameters if it's a dict
        encoder_kwargs = {}
        if isinstance(encoder_type, dict) and "module_parameters" in encoder_type:
            encoder_kwargs = encoder_type["module_parameters"].copy()

        # channel-agnostic part
        if len(ma_layers_blocks) == 0:
            self.marker_agnostic_encoder = Identity()
            hyperkernel_input_dim = 1
        else:
            # Build marker-agnostic encoder with required parameters
            self.marker_agnostic_encoder = encoder_cls(
                input_channels=1,
                layers_blocks=ma_layers_blocks,
                embedding_dims=ma_embedding_dims,
                stem=True,
                **encoder_kwargs,
            )
            hyperkernel_input_dim = ma_embedding_dims[-1]
        hyperkernel_embedding_dim = pm_embedding_dims[0]

        self.hyperkernel = Hyperkernel(
            num_channels=num_channels,
            input_dim=hyperkernel_input_dim,
            embedding_dim=hyperkernel_embedding_dim,
            module_type="encoder",
            **hyperkernel_config,
        )
        self.norm = LayerNorm(hyperkernel_embedding_dim, data_format="channels_first")

        # pan-marker part
        self.pan_marker_encoder = encoder_cls(
            input_channels=hyperkernel_embedding_dim,
            layers_blocks=pm_layers_blocks,
            embedding_dims=pm_embedding_dims,
            stem=False,
            **encoder_kwargs,
        )

        self.latent_norm = (
            LayerNorm(pm_embedding_dims[-1], data_format="channels_first")
            if use_latent_norm
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Multiplex images batch tensor with shape [B, C, H, W]
            encoded_indices (torch.Tensor): Indices of the markers in channels tensor with shape [B, C].
            return_features (bool, optional): If True, returns the features after each block. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        x = self.marker_agnostic_encoder(x, return_features=return_features)
        if return_features:
            features += x["features"]
        x = x["output"]
        _, E_ma, H_ma, W_ma = x.shape
        x = x.reshape(B, C, E_ma, H_ma, W_ma).reshape(B, C * E_ma, H_ma, W_ma)

        x = self.hyperkernel(x, encoded_indices)

        x = self.norm(x)
        x = self.pan_marker_encoder(x, return_features=return_features)
        if return_features:
            features += x["features"]
        x = x["output"]
        x = self.latent_norm(x)

        outputs["output"] = x
        if return_features:
            outputs["features"] = features

        return outputs


class MultiplexImageDecoder(nn.Module):
    """Decoder for restoring the multiplex image from the embedding tensor."""

    def __init__(
        self,
        input_embedding_dim: int,
        decoded_embed_dim: int,
        num_blocks: int,
        scaling_factor: int,
        num_channels: int,
        hyperkernel_config: dict,
        num_outputs: int = 2,
        block_type: str | type[Block] | dict = "convnext",
    ) -> None:
        """
        Args:
            input_embedding_dim (int): Embedding dimension of the input tensor.
            decoded_embed_dim (int): Embedding dimension of the decoded tensor (before last projections).
            num_blocks (int): Number of multiplex blocks in each intermediate layer.
            scaling_factor (int): Scaling factor for the upsampling.
            num_channels (int): Number of possible output channels/markers.
            hyperkernel_config (dict): Configuration for the hyperkernel.
            num_outputs (int, optional): Number of output channels per marker. Defaults to 2.
            block_type (str | Type[Block] | dict, optional): Type of block to use.
                Can be a string (registry name), Block class, or config dict. Defaults to "convnext".
        """
        super().__init__()
        self.scaling_factor = scaling_factor
        self.num_channels = num_channels
        self.decoded_embed_dim = decoded_embed_dim
        self.num_outputs = num_outputs

        # Resolve block class and parameters
        block_cls = resolve_block_class(block_type)
        block_kwargs = {}
        if isinstance(block_type, dict) and "module_parameters" in block_type:
            block_kwargs = block_type["module_parameters"]

        # self.channel_embed = nn.Embedding(num_channels, input_embedding_dim * decoded_embed_dim) # input projection
        self.channel_embed = Hyperkernel(
            num_channels=num_channels,
            input_dim=input_embedding_dim,
            embedding_dim=decoded_embed_dim,
            module_type="decoder",
            **hyperkernel_config,
        )

        self.decoder = nn.Sequential(
            *[
                block_cls(
                    decoded_embed_dim,
                    **block_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.pred = nn.Conv2d(
            decoded_embed_dim, scaling_factor**2 * self.num_outputs, kernel_size=1
        )

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multiplex Image Decoder.

        Args:
            x (torch.Tensor): Input tensor (embedding).
            indices (torch.Tensor): Indices of the markers.

        Returns:
            torch.Tensor: Reconstructed image tensor
        """
        B, _, H, W = x.shape
        C = indices.shape[1]
        N = B * C
        E, A, O = self.decoded_embed_dim, self.scaling_factor, self.num_outputs

        x = self.channel_embed(x, indices)  # [B, C, E, H, W]
        x = x.reshape(N, E, H, W)

        x = self.decoder(x)
        x = self.pred(x)

        x = x.reshape(N, A, A, O, H, W).reshape(B, C, A, A, O, H, W)
        x = torch.einsum("bcxyohw -> bchxwyo", x)

        x = x.reshape(B, C, H * A, W * A, O)

        return x


class MultiplexAutoencoder(nn.Module):
    """Multiplex image Autoencoder with Hyperkernel and Multiplex Image Encoder-Decoder."""

    def __init__(
        self,
        num_channels: int,
        encoder_config: dict,
        decoder_config: dict,
    ):
        """Initialize the Multiplex Autoencoder model.

        Args:
            num_channels (int): Number of all possible channels/markers.
            encoder_config (dict): Configuration for the encoder.
            decoder_config (dict): Configuration for the decoder.
        """
        super().__init__()
        self.latent_dim = encoder_config["pm_embedding_dims"][-1]
        self.num_channels = num_channels

        self.encoder = MultiplexImageEncoder(
            num_channels=self.num_channels, **encoder_config
        )

        hyperkernels_scaling_factor = (
            encoder_config["hyperkernel_config"]["stride"]
            * decoder_config["hyperkernel_config"]["stride"]
        )
        scaling_factor = hyperkernels_scaling_factor * 2 ** len(
            encoder_config["ma_layers_blocks"] + encoder_config["pm_layers_blocks"][:-1]
        )
        self.decoder = MultiplexImageDecoder(
            input_embedding_dim=self.latent_dim,
            scaling_factor=scaling_factor,
            num_channels=self.num_channels,
            **decoder_config,
        )

    def encode(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """Encode the input images using the encoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels.
            return_features (bool, optional): If True, returns the features after encoding. Defaults to False.

        Returns:
            dict: A dictionary containing the encoded images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encoder(
            x, encoded_indices, return_features=return_features
        )
        outputs = {"output": encoding_output["output"]}

        if return_features:
            outputs["features"] = encoding_output["features"]
        return outputs

    def decode(
        self,
        x: torch.Tensor,
        decoded_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the encoded images using the decoder.

        Args:
            x (torch.Tensor): Encoded images tensor with shape (B, E', H', W').
            decoded_indices (torch.Tensor): Indices of the markers in channels for decoding.

        Returns:
            torch.Tensor: Decoded images tensor with shape (B, C, H, W).
        """
        x = self.decoder(x, decoded_indices)
        return x

    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        decoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        """Forward pass of the Multiplex Autoencoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels
                for encoding.
            decoded_indices (torch.Tensor): Indices of the markers in channels
                for decoding.

        Returns:
            dict: A dictionary containing the reconstructed images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encode(
            x, encoded_indices, return_features=return_features
        )
        x = encoding_output["output"]
        x = self.decode(x, decoded_indices)
        outputs = {"output": x}
        if return_features:
            outputs["features"] = encoding_output["features"]
        return outputs
