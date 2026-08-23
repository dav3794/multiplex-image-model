import copy
import os
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_modules import Block, Encoder, Identity, LayerNorm
from .registry import resolve_block_class, resolve_encoder_class


def load_marker_embeddings(
    source: str | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """Load a precomputed marker embedding table.

    Args:
        source: Either an in-memory table or a path to a `.npy`/`.pt` file holding a
            dense (vocabulary_size, embedding_dim) matrix. Row order must match the
            marker indices produced by the tokenizer.

    Returns:
        torch.Tensor: Float32 embedding table of shape (vocabulary_size, embedding_dim).
    """
    if isinstance(source, str):
        path = os.path.expanduser(source)
        if not os.path.exists(path):
            raise ValueError(f"Marker embeddings not found: {path}")
        if path.endswith(".npy"):
            table = torch.from_numpy(np.load(path))
        elif path.endswith((".pt", ".pth")):
            table = torch.load(path, map_location="cpu")
        else:
            raise ValueError(
                f"Unsupported marker embeddings format: {path}. Use '.npy' or '.pt'."
            )
    elif isinstance(source, np.ndarray):
        table = torch.from_numpy(source)
    else:
        table = source

    if not isinstance(table, torch.Tensor) or table.ndim != 2:
        raise ValueError(
            "Marker embeddings must be a 2D (vocabulary_size, embedding_dim) tensor."
        )

    return table.detach().to(torch.float32).contiguous()


def build_marker_projector(
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    num_layers: int = 2,
    zero_init_output: bool = False,
) -> nn.Sequential:
    """Build an MLP projecting external marker embeddings.

    Args:
        input_dim: Dimension of the external marker embeddings.
        output_dim: Dimension of the produced coefficients (the hyperkernel rank).
        hidden_dim: Hidden dimension of the MLP; defaults to `input_dim`.
        num_layers: Total number of linear layers (>= 1).
        zero_init_output: Whether to initialize the final projection to zero.

    Returns:
        nn.Sequential: Projector applied to embeddings of shape (..., input_dim).
    """
    if num_layers < 1:
        raise ValueError("`num_layers` must be at least 1 for the marker projector.")

    hidden_dim = hidden_dim or input_dim
    layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
    dim = input_dim
    for _ in range(num_layers - 1):
        layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
        dim = hidden_dim

    final_layer = nn.Linear(dim, output_dim)
    if zero_init_output:
        nn.init.zeros_(final_layer.weight)
    else:
        nn.init.normal_(final_layer.weight, std=dim**-0.5)
    nn.init.zeros_(final_layer.bias)
    layers.append(final_layer)

    return nn.Sequential(*layers)


class Hyperkernel(nn.Module):
    def __init__(
        self,
        num_channels: int | None,
        input_dim: int,
        embedding_dim: int,
        module_type: Literal["encoder", "decoder"],
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        use_bias: bool = True,
        low_rank: bool = False,
        rank: int | None = None,
        marker_embeddings: str | np.ndarray | torch.Tensor | None = None,
        projector_hidden_dim: int | None = None,
        projector_num_layers: int = 2,
    ):
        """Initialize the Hyperkernel model

        Args:
            num_channels (int, optional): Number of channels in the input tensor.
                Only used for the learnable per-marker tables; may be None when
                `marker_embeddings` is provided (unbounded marker vocabulary).
            input_dim (int): Input dimension of each channel
            embedding_dim (int): Embedding dimension for the input tensor
            module_type (Literal['encoder', 'decoder']): Whether the Hyperkernel is used in encoder or decoder
            kernel_size (int, optional): Kernel size for the conv layer (already squared). Model embedding will be embedding_dim*kernel_size**2.
            padding (int, optional): Padding for the conv layer. Defaults to 1.
            stride (int, optional): Stride for the conv layer. Defaults to 1.
            use_bias (bool, optional): Whether to use bias in the conv layer. Defaults to True.
            low_rank (bool, optional): If True, factorize the per-marker weight table as a
                shared basis combined with per-marker coefficients
                (W_m = sum_r coeff[m, r] * basis[r]), reducing parameters from
                num_channels * model_dim to rank * model_dim + num_channels * rank.
                Defaults to False.
            rank (int, optional): Number of shared basis components. Required when
                low_rank is True. Defaults to None.
            marker_embeddings (str | np.ndarray | torch.Tensor, optional): Precomputed
                marker embeddings (e.g. from an LLM) as a dense
                (vocabulary_size, embedding_dim) table, or a path to a `.npy`/`.pt` file
                holding it. Rows are addressed by the marker indices coming from the
                tokenizer. When given, the learnable per-marker tables are replaced by an
                MLP projecting these embeddings to the low-rank coefficients (and to the
                decoder bias), so the marker vocabulary is unbounded. Requires
                low_rank=True. Defaults to None.
            projector_hidden_dim (int, optional): Hidden dimension of the projector MLP.
                Defaults to the external embedding dimension.
            projector_num_layers (int, optional): Number of linear layers in the projector
                MLP. Defaults to 2.
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

        self.low_rank = low_rank
        self.use_marker_embeddings = marker_embeddings is not None
        if self.use_marker_embeddings and not low_rank:
            raise ValueError(
                "`marker_embeddings` requires `low_rank=True`; the projected embedding "
                "is used as the low-rank coefficient vector."
            )
        if not self.use_marker_embeddings and num_channels is None:
            raise ValueError(
                "`num_channels` is required when no `marker_embeddings` are provided."
            )

        if self.use_marker_embeddings:
            table = load_marker_embeddings(marker_embeddings)
            # Non-persistent: the table is precomputed and reloaded from disk, so it
            # stays out of checkpoints and the vocabulary can grow after training.
            self.register_buffer("marker_embedding_table", table, persistent=False)
            self.marker_embedding_dim = table.shape[1]

        if low_rank:
            if rank is None or rank <= 0:
                raise ValueError(
                    "`rank` must be a positive integer when `low_rank` is True."
                )
            self.rank = rank
            self.hyperkernel_basis = nn.Parameter(torch.empty(rank, self.model_dim))
            # Init so that initial per-marker weights match the std of the
            # full-rank nn.Embedding (~N(0, 1)) elementwise.
            nn.init.normal_(self.hyperkernel_basis, std=rank**-0.5)
            if self.use_marker_embeddings:
                self.coeff_projector = build_marker_projector(
                    self.marker_embedding_dim,
                    rank,
                    hidden_dim=projector_hidden_dim,
                    num_layers=projector_num_layers,
                )
            else:
                # Per-marker coefficients over a shared basis of weight matrices.
                self.hyperkernel_coeff = nn.Embedding(num_channels, rank)
                nn.init.normal_(self.hyperkernel_coeff.weight)
        else:
            self.rank = None
            self.hyperkernel_weights = nn.Embedding(num_channels, self.model_dim)

        self.use_bias = use_bias
        if use_bias:
            if module_type == "encoder":
                self.hyperkernel_bias = nn.Parameter(
                    torch.zeros(1, self.embedding_dim, 1, 1)
                )
            elif self.use_marker_embeddings:
                self.hyperkernel_bias = build_marker_projector(
                    self.marker_embedding_dim,
                    self.embedding_dim,
                    hidden_dim=projector_hidden_dim,
                    num_layers=projector_num_layers,
                    zero_init_output=True,
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

        marker_embeds = None
        if self.use_marker_embeddings:
            marker_embeds = self.marker_embedding_table[indices]  # (B, C, D)

        if self.low_rank:
            if self.use_marker_embeddings:
                coeff = self.coeff_projector(marker_embeds).to(x.dtype)  # (B, C, R)
            else:
                coeff = self.hyperkernel_coeff(indices).to(x.dtype)  # (B, C, R)
            weights = coeff @ self.hyperkernel_basis.to(x.dtype)  # (B, C, I*O)
        else:
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
                if self.use_marker_embeddings:
                    channel_biases = self.hyperkernel_bias(marker_embeds).to(x.dtype)
                else:
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
        use_mask_token: bool = False,
        mask_token_init: float = 0.0,
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
            use_mask_token (bool, optional): Whether to replace masked pixels with a learnable token. Defaults to False.
            mask_token_init (float, optional): Initial value for the mask token. Defaults to 0.0.
            encoder_type (Union[str, Type[Encoder], Dict], optional): Type of encoder to use.
                Can be a string (registry name), Encoder class, or config dict with 'type' and 'module_parameters'.
                For ConvNeXtEncoder, module_parameters can include 'block_parameters' dict with ConvNextBlock parameters
                (e.g., kernel_size, padding, inter_dim).
                Defaults to "convnext".
        """
        super().__init__()

        self.use_mask_token = use_mask_token
        self.mask_token = (
            nn.Parameter(torch.tensor(mask_token_init)) if use_mask_token else None
        )

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
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Multiplex images batch tensor with shape [B, C, H, W]
            encoded_indices (torch.Tensor): Indices of the markers in channels tensor with shape [B, C].
            spatial_mask (torch.Tensor, optional): Boolean mask for masked pixels [B, C, H, W].
            return_features (bool, optional): If True, returns the features after each block. Defaults to False.

        Returns:
            dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []

        B, C, H, W = x.shape
        if self.use_mask_token and spatial_mask is not None:
            mask_token = self.mask_token.to(dtype=x.dtype)
            x = torch.where(spatial_mask, mask_token, x)
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
        share_hyperkernel_coeff: bool = False,
    ):
        """Initialize the Multiplex Autoencoder model.

        Args:
            num_channels (int): Number of all possible channels/markers.
            encoder_config (dict): Configuration for the encoder.
            decoder_config (dict): Configuration for the decoder.
            share_hyperkernel_coeff (bool, optional): If True, tie the per-marker
                low-rank coefficient table (`hyperkernel_coeff`) of the decoder
                hyperkernel to the encoder hyperkernel, so a single per-marker code
                drives both encoding and decoding. Requires both hyperkernels to use
                low_rank=True with matching rank. Defaults to False.
        """
        super().__init__()
        self._architecture_config = {
            "num_channels": num_channels,
            "encoder_config": copy.deepcopy(encoder_config),
            "decoder_config": copy.deepcopy(decoder_config),
            "share_hyperkernel_coeff": share_hyperkernel_coeff,
        }

        self.latent_dim = encoder_config["pm_embedding_dims"][-1]
        self.num_channels = num_channels
        self.share_hyperkernel_coeff = share_hyperkernel_coeff

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

        if share_hyperkernel_coeff:
            self._tie_hyperkernel_coeff()

    def _tie_hyperkernel_coeff(self) -> None:
        """Tie the decoder hyperkernel's per-marker coefficients to the encoder's.

        Both hyperkernels must be built with low_rank=True and the same rank, so that
        the shared `hyperkernel_coeff` table (num_channels x rank) is compatible.
        """
        encoder_hk = self.encoder.hyperkernel
        decoder_hk = self.decoder.channel_embed
        if not (encoder_hk.low_rank and decoder_hk.low_rank):
            raise ValueError(
                "share_hyperkernel_coeff requires both encoder and decoder hyperkernels "
                "to use low_rank=True."
            )
        if encoder_hk.rank != decoder_hk.rank:
            raise ValueError(
                "share_hyperkernel_coeff requires matching ranks for the encoder "
                f"({encoder_hk.rank}) and decoder ({decoder_hk.rank}) hyperkernels."
            )
        if encoder_hk.use_marker_embeddings != decoder_hk.use_marker_embeddings:
            raise ValueError(
                "share_hyperkernel_coeff requires the encoder and decoder hyperkernels "
                "to both use (or both not use) external marker embeddings."
            )
        if encoder_hk.use_marker_embeddings:
            if encoder_hk.marker_embedding_dim != decoder_hk.marker_embedding_dim:
                raise ValueError(
                    "share_hyperkernel_coeff requires matching marker embedding "
                    f"dimensions for the encoder ({encoder_hk.marker_embedding_dim}) "
                    f"and decoder ({decoder_hk.marker_embedding_dim}) hyperkernels."
                )
            decoder_hk.coeff_projector = encoder_hk.coeff_projector
        else:
            decoder_hk.hyperkernel_coeff = encoder_hk.hyperkernel_coeff

    def get_architecture_config(self, by_alias: bool = False) -> dict:
        """Return the model architecture configuration.

        Args:
            by_alias: If True, uses config aliases (e.g., 'hyperkernel').

        Returns:
            dict: Architecture configuration for rebuilding the model.
        """
        config = copy.deepcopy(self._architecture_config)
        if by_alias:
            config = config.copy()
            config["encoder"] = config.pop("encoder_config")
            config["decoder"] = config.pop("decoder_config")
            config["encoder"]["hyperkernel"] = config["encoder"].pop(
                "hyperkernel_config"
            )
            config["decoder"]["hyperkernel"] = config["decoder"].pop(
                "hyperkernel_config"
            )
        return config

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str | dict,
        map_location: str | torch.device | None = None,
        model_config: dict | None = None,
        marker_embeddings: str | None = None,
        strict: bool = True,
    ) -> "MultiplexAutoencoder":
        """Create a model and load weights from a checkpoint.

        Args:
            checkpoint: Path to checkpoint file or loaded checkpoint dict.
            map_location: Optional map_location passed to torch.load when checkpoint is a path.
            model_config: Model config to use if checkpoint lacks 'model_config'.
            marker_embeddings: Optional path overriding the external marker embedding
                table in both encoder and decoder hyperkernel configurations.
            strict: Whether to strictly enforce that the keys in state_dict match the model.

        Returns:
            MultiplexAutoencoder: Model with weights loaded from checkpoint.
        """
        if isinstance(checkpoint, dict):
            checkpoint_data = checkpoint
        else:
            checkpoint_data = torch.load(checkpoint, map_location=map_location, weights_only=True)

        resolved_config = checkpoint_data.get("model_config", model_config)
        if resolved_config is None:
            raise ValueError(
                "Checkpoint is missing 'model_config'; provide model_config to load the model."
            )

        resolved_config = copy.deepcopy(resolved_config)
        if marker_embeddings is not None:
            resolved_config["encoder_config"]["hyperkernel_config"][
                "marker_embeddings"
            ] = marker_embeddings
            resolved_config["decoder_config"]["hyperkernel_config"][
                "marker_embeddings"
            ] = marker_embeddings

        model = cls(**resolved_config)
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)
        return model

    def encode(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
        """Encode the input images using the encoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels.
            spatial_mask (torch.Tensor, optional): Boolean mask for masked pixels [B, C, H, W].
            return_features (bool, optional): If True, returns the features after encoding. Defaults to False.

        Returns:
            dict: A dictionary containing the encoded images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encoder(
            x,
            encoded_indices,
            spatial_mask=spatial_mask,
            return_features=return_features,
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
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
        """Forward pass of the Multiplex Autoencoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels
                for encoding.
            decoded_indices (torch.Tensor): Indices of the markers in channels
                for decoding.
            spatial_mask (torch.Tensor, optional): Boolean mask for masked pixels [B, C, H, W].

        Returns:
            dict: A dictionary containing the reconstructed images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encode(
            x,
            encoded_indices,
            spatial_mask=spatial_mask,
            return_features=return_features,
        )
        x = encoding_output["output"]
        x = self.decode(x, decoded_indices)
        outputs = {"output": x}
        if return_features:
            outputs["features"] = encoding_output["features"]
        return outputs
