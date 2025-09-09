import e2cnn.group.groups

import torch

# --- Monkey-patch Tensor.__setitem__ to accept uint8 masks -------------
_orig_setitem = torch.Tensor.__setitem__

def _safe_setitem(self, key, value):
    # intercept "tensor[mask] = …" where mask is uint8
    if isinstance(key, torch.Tensor) and key.dtype is torch.uint8:
        key = key.to(torch.bool)
    return _orig_setitem(self, key, value)

torch.Tensor.__setitem__ = _safe_setitem

import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type, Callable, Literal
from typing_extensions import Literal
from torch import einsum
from torch.nn import functional as F
# import logging


# import e2cnn
# import e2cnn.nn as e2nn
# from e2cnn.group import directsum

import escnn.nn as e2nn
import escnn

from multiplex_model.modules import Superkernel, MultiplexImageDecoder


class GRNByIrrep(nn.Module):
    """
    One γ, β per *copy* of an irrep (no matter its dimension).
    This keeps equivariance for any ESCNN FieldType.
    """

    def __init__(self, field_type: e2nn.FieldType, eps=1e-6):
        super().__init__()
        self.eps = eps

        # Build a list:  [(slice, dim)]     one entry per field
        # `slice`   → the channel range in the tensor
        # `dim`     → irrep dimension (1 for trivial, 2 for a real harmonic, ...)
        self.fields = []
        start = 0
        for r in field_type:
            size = r.size
            self.fields.append((slice(start, start+size), size))
            start += size

        # One γ, β per field copy
        self.gamma = nn.Parameter(torch.zeros(1, len(self.fields), 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, len(self.fields), 1, 1))
        self.ftype = field_type

    # def forward(self, x: e2nn.GeometricTensor):
    #     t = x.tensor                                    # [B,C,H,W]
    #     B, C, H, W = t.shape

    #     # compute per-field L2 norm  → shape  [B, n_fields, 1, 1]
    #     norms = []
    #     for sl, d in self.fields:
    #         # sum squares over irrep dimension `d` and spatial dims H,W
    #         n = t[:, sl].reshape(B, d, -1)              # [B,d,H*W]
    #         n = torch.linalg.norm(n, ord=2, dim=(1,2), keepdim=True)
    #         norms.append(n)
    #     gx = torch.cat(norms, dim=1)                    # [B,nf,1,1]

    #     # same rescaling as in the paper
    #     nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)   # [B,nf,1,1]

    #     # broadcast γ, β to the corresponding channels
    #     out = t.clone()
    #     # for i, (sl, _) in enumerate(self.fields):
    #     #     out[:, sl] = (self.gamma[:, i] * (t[:, sl] * nx[:, i])
    #     #                   + self.beta[:, i] + t[:, sl])
    #     for i, (sl, _) in enumerate(self.fields):
    #         gamma_i = self.gamma[:, i:i+1, ...]      # (1,1,1,1)
    #         beta_i  = self.beta[:,  i:i+1, ...]      # (1,1,1,1)
    #         nx_i    = nx[:,   i:i+1, ...]            # (B,1,1,1)

    #         out[:, sl] = gamma_i * (t[:, sl] * nx_i) + beta_i + t[:, sl]
    #     return e2nn.GeometricTensor(out, self.ftype)
    def forward(self, x: e2nn.GeometricTensor):
        t = x.tensor                                # [B, C, H, W]
        B, C, H, W = t.shape

        # ----------------------------------------------------------
        # 1) per-field L2 norm  →  [B, n_fields, 1, 1]
        # ----------------------------------------------------------
        norms = []
        # for sl, _ in self.fields:
        #     # no need to flatten – just reduce across channel & space
        #     n = torch.linalg.norm(t[:, sl], ord=2, dim=(1, 2, 3), keepdim=True)
        #     norms.append(n)                         # each n: [B, 1, 1, 1]
        norms = []
        for sl, _ in self.fields:
            n = torch.sqrt((t[:, sl] ** 2).sum(dim=(1, 2, 3), keepdim=True))
            norms.append(n)                       # [B,1,1,1]

        gx = torch.cat(norms, dim=1)                # [B, n_fields, 1, 1]
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)

        # ----------------------------------------------------------
        # 2) apply γ, β per field (keep axis 1!)
        # ----------------------------------------------------------
        out = t.clone()
        for i, (sl, _) in enumerate(self.fields):
            gamma_i = self.gamma[:, i:i+1]          # [1,1,1,1]
            beta_i  = self.beta[:,  i:i+1]          # [1,1,1,1]
            nx_i    = nx[:,   i:i+1]                # [B,1,1,1]

            out[:, sl] = gamma_i * (t[:, sl] * nx_i) + beta_i + t[:, sl]

        return e2nn.GeometricTensor(out, self.ftype)


# logging.basicConfig(level=logging.DEBUG)

class EquivariantMultiplexAutoencoderOld(nn.Module):
    """Multiplex image Transformer with Superkernel and Multiplex Image Decoder."""

    def __init__(
            self, 
            num_channels: int,
            input_image_size: int,
            superkernel_embedding_dim: int,
            superkernel_depth: int,
            superkernel_heads: int,
            superkernel_layer_type: Literal['conv', 'linear'],
            encoder_config: Dict,
            decoder_config: Dict,
            superkernel_kernel_size: int = None,
            superkernel_conv_padding: int = None,
            superkernel_conv_stride: int = 1,
            mlp_ratio: float = 4.,
            **kwargs
            ):
        """Initialize the Multiplex Transformer model.

        Args:
            num_channels (int): Number of channels/markers in the dataset.
            input_image_size (int): Size of the input image.
            superkernel_embedding_dim (int): Embedding dimension for the Superkernel.
            superkernel_depth (int): Number of layers in the Superkernel model.
            superkernel_heads (int): Number of heads per channel embedding in the Superkernel model.
            superkernel_layer_type (Literal['conv', 'linear']): Type of the Superkernel layer.
            encoder_config (Dict): Configuration for the encoder.
            decoder_config (Dict): Configuration for the decoder.
            superkernel_kernel_size (int, optional): Size of Superkernel kernel if conv type. Defaults to None.
            superkernel_conv_padding (int, optional): Convolution padding if conv type. Defaults to None.
            superkernel_conv_stride (int, optional): Convolution stride if conv type. Defaults to 1.
            mlp_ratio (float, optional): MLP ratio. Defaults to 4..
        """
        super().__init__()
        self.num_channels = num_channels
        self.input_image_size = input_image_size
        self.superkernel_embedding_dim = superkernel_embedding_dim
        self.superkernel_depth = superkernel_depth
        self.superkernel_heads = superkernel_heads
        self.superkernel_layer_type = superkernel_layer_type
        self.superkernel_kernel_size = superkernel_kernel_size
        self.superkernel_conv_padding = superkernel_conv_padding
        self.superkernel_conv_stride = superkernel_conv_stride
        self.mlp_ratio = mlp_ratio


        self.superkernel = Superkernel(
            num_channels=num_channels, 
            embedding_dim=superkernel_embedding_dim, 
            num_layers=superkernel_depth, 
            num_heads=superkernel_heads, 
            mlp_ratio=mlp_ratio, 
            layer_type=superkernel_layer_type,
            kernel_size=superkernel_kernel_size,
            **kwargs
        )
        self.act = nn.GELU()

        self.encoder = EscnnMultiplexImageEncoder(
            **encoder_config
        )

        self.decoder = MultiplexImageDecoder(
            **decoder_config
        )

    def forward(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor, 
            decoded_indices: torch.Tensor
        ) -> torch.Tensor:
        # print("shape", x.shape)
        B = x.shape[0]
        # print(f'Input shape: {x.shape}, Encoded indices shape: {encoded_indices.shape}, Decoded indices shape: {decoded_indices.shape}')
        # print((f"Input isnan: {torch.isnan(x).any()}, "))
        # print((f"Encoded indices isnan: {torch.isnan(encoded_indices).any()}, "))
        # print((f"Decoded indices isnan: {torch.isnan(decoded_indices).any()}, "))
        superkernel_weights = self.superkernel(encoded_indices)


        # print((f"Superkernel weights isnan: {torch.isnan(superkernel_weights).any()}, "))
        if self.superkernel_layer_type == 'conv':
            x = torch.cat([
                F.conv2d(
                    x[i].unsqueeze(0), 
                    superkernel_weights[i].to(x.dtype), 
                    padding=self.superkernel_conv_padding,
                    stride=self.superkernel_conv_stride
                )
                for i in range(B)
            ])
            
        else:
            x = torch.einsum('bchw, bce -> behw', x, superkernel_weights.to(x.dtype))

        # print(f'After superkernel shape: {x.shape}')
        # print((f"After superkernel isnan: {torch.isnan(x).any()}, "))
        x = self.act(x)

        # print(f'After activation shape: {x.shape}')
        # print((f"After activation isnan: {torch.isnan(x).any()}, "))
        x = self.encoder(x)
        # print(f"latent isnan: {torch.isnan(x).any()}")

        # print(f'After encoder shape: {x.shape}')
        # print((f"After encoder isnan: {torch.isnan(x).any()}, "))
        latent = x
        # print(f'Latent shape: {latent.shape}')
        # latent, features = x[:, 0], x[:, 1:]
        # latent = x.mean(dim=(2, 3))
        # x = features.permute(0, 2, 1).reshape(B, 768, 14, 14)

        x = self.decoder(x, decoded_indices)
        # x = x[:, :, ]
        
        # print(f"decoded isnan: {torch.isnan(x).any()}")
        # print(f'After decoder shape: {x.shape}')
        # print((f"After decoder isnan: {torch.isnan(x).any()}, "))        
        return x, latent


class EquivariantMultiplexAutoencoder(nn.Module):
    """Multiplex image Autoencoder with Superkernel and Multiplex Image Encoder-Decoder."""

    def __init__(
            self,
            num_channels: int,
            superkernel_config: Dict,
            encoder_config: Dict,
            decoder_config: Dict,
            ):
        """Initialize the Multiplex Autoencoder model.

        Args:
            num_channels (int): Number of all possible channels/markers.
            superkernel_config (Dict): Configuration for the superkernel.
            encoder_config (Dict): Configuration for the encoder.
            decoder_config (Dict): Configuration for the decoder.
        """
        super().__init__()
        self.superkernel_dim = superkernel_config['embedding_dim']
        self.latent_dim = encoder_config['embedding_dims'][-1]
        self.decoder_dim = decoder_config['decoded_embed_dim']
        self.num_channels = num_channels
        self.superkernel_conv_padding = (superkernel_config.get('kernel_size') or 0) // 2
        self.superkernel_conv_stride = superkernel_config.get('stride', 1)

        self.act = nn.GELU()

        self.superkernel = Superkernel(
            num_channels=self.num_channels,
            **superkernel_config
        )

        # self.pixel_shift_superkernel = Superkernel(
        #     num_channels=self.num_channels,
        #     embedding_dim=1,
        #     layer_type='linear',
        #     num_layers=0,
        #     num_heads=None,
        #     mlp_ratio=None,
        #     kernel_size=None,
        # )

        self.encoder = EquivariantMultiplexImageEncoder( # finish thi
            channel_embedding_dim=self.superkernel_dim,
            # num_all_channels=self.num_channels,
            **encoder_config
        )

        scaling_factor = 2 ** len(encoder_config['layers_blocks'])
        self.decoder = MultiplexImageDecoder(
            input_embedding_dim=self.latent_dim,
            scaling_factor=scaling_factor,
            num_channels=self.num_channels,
            **decoder_config
        )

    def embed_images(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor,
            return_features: bool = False,
        ) -> Dict:
        """Embed the input images using the superkernel.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels.
            return_features (bool, optional): If True, returns the superkernel weights. Defaults to False.

        Returns:
            Dict: A dictionary containing the embedded images tensor (under 'output') and optionally the superkernel weights.
        """
        B, C = x.shape[0], x.shape[1]
        superkernel_weights = self.superkernel(encoded_indices)
        # pixel_shift_weights = self.pixel_shift_superkernel(encoded_indices)

        # pixel_shift_weights = pixel_shift_weights.reshape(B, C, 1, 1)
        # x = x + pixel_shift_weights
        # print(f'Input shape: {x.shape}, Encoded indices shape: {encoded_indices.shape}')

        # print(f'Superkernel weights shape: {superkernel_weights.shape}')
        if self.superkernel.layer_type == 'conv':
            x = torch.cat([
                F.conv2d(
                    x[i].unsqueeze(0), 
                    superkernel_weights[i].to(x.dtype), 
                    padding=self.superkernel_conv_padding,
                    stride=self.superkernel_conv_stride
                )
                for i in range(B)
            ])
            
        else:
            x = torch.einsum('bchw, bce -> behw', x, superkernel_weights.to(x.dtype))
        
        x = self.act(x)
        outputs = {'output': x}
        if return_features:
            outputs['features'] = [superkernel_weights]
        return outputs

    def encode_images(
            self, 
            x: torch.Tensor, 
            encoded_indices: torch.Tensor,
            return_features: bool = False,
        ) -> Dict:
        """Encode the input images using the encoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels.
            return_features (bool, optional): If True, returns the features after encoding. Defaults to False.

        Returns:
            Dict: A dictionary containing the encoded images tensor (under 'output') and optionally the features.
        """
        embedding_output = self.embed_images(x, encoded_indices, return_features=return_features)
        x = embedding_output['output']
        encoding_output = self.encoder(x, return_features=return_features)
        outputs = {'output': encoding_output['output']}
        # print(f'Encoding output shape: {outputs["output"].shape}')

        if return_features:
            outputs['features'] = embedding_output['features'] + encoding_output['features']
        return outputs

    def decode_images(
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
        ) -> Dict:
        """Forward pass of the Multiplex Autoencoder.

        Args:
            x (torch.Tensor): Input images tensor with shape (B, C, H, W).
            encoded_indices (torch.Tensor): Indices of the markers in channels
                for encoding.
            decoded_indices (torch.Tensor): Indices of the markers in channels
                for decoding.

        Returns:
            Dict: A dictionary containing the reconstructed images tensor (under 'output') and optionally the features.
        """
        encoding_output = self.encode_images(x, encoded_indices, return_features=return_features)
        x = encoding_output['output']
        x = self.decode_images(x, decoded_indices)
        outputs = {'output': x}
        if return_features:
            outputs['features'] = encoding_output['features']
        return outputs




class EscnnMultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

    def __init__(
            self,
            encoder_class: Type,
            reshape_fn: Callable = nn.Identity(),
            **kwargs
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            encoder_class (Type): Encoder class to use.
            reshape_fn (Type, optional): Reshape function to apply to the output of the encoder. Defaults to nn.Identity.
        """
        super().__init__()
        self.encoder = encoder_class(**kwargs)
        self.last_conv = Regular2Trivial(self.encoder.out_type, n_scalars=kwargs["embedding_dims"][-1])
        self.reshape_fn = reshape_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.encoder.forward_features(x)
        x = self.last_conv(x)  # [B, C, H, W] → [B, E, H, W]
        x = x.tensor  # Extract the raw tensor from GeometricTensor
        x = self.reshape_fn(x)
        return x



class Regular2Trivial(nn.Module):
    """
    Point-wise R2Conv that projects any FieldType to *n_scalars*
    copies of the trivial representation.
    """

    def __init__(self, in_type: e2nn.FieldType, n_scalars: int = 1):
        super().__init__()

        gspace   = in_type.gspace
        out_type = e2nn.FieldType(gspace,
                                  [gspace.trivial_repr] * n_scalars)

        # kernel_size = 1 → no spatial mixing, only channel mixing
        self.proj = e2nn.R2Conv(in_type, out_type,
                                kernel_size=1, bias=True, 
                                initialize=True # TODO initialize=True
                                )

        self.in_type, self.out_type = in_type, out_type

    def forward(self, x: e2nn.GeometricTensor):
        return self.proj(x)


class BLConvNeXtBlock(nn.Module):
    def __init__(self,
                 in_type: e2nn.FieldType,
                 expansion: int = 4,
                 ksize: int = 7):
        super().__init__()
        # initialize=False
        initialize=True

        gspace = in_type.gspace
        # For the point-wise MLP we keep the same irrep set but multiply the copies
        mid_type = e2nn.FieldType(
            gspace,
            expansion * list(in_type.representations)
        )

        # depth-wise ⇒ no mixing across *copies*
        # easiest robust choice: a *full* equivariant conv
        self.depthwise = e2nn.R2Conv(
            in_type, in_type,
            kernel_size=ksize, padding=ksize//2, bias=True,
            initialize=initialize
        )

        self.norm = e2nn.IIDBatchNorm2d(in_type, affine=True)
        # self.norm = e2nn.GNormBatchNorm(in_type, affine=True)

        self.pw_up   = e2nn.R2Conv(in_type, mid_type, kernel_size=1, bias=True, initialize=initialize)
        self.gelu    = e2nn.NormNonLinearity(mid_type)
        # self.grn     = GRNByIrrep(mid_type)
        self.pw_down = e2nn.R2Conv(mid_type, in_type, kernel_size=1, bias=True, initialize=initialize)

        self.in_type  = in_type
        self.out_type = in_type

    def forward(self, x: e2nn.GeometricTensor):
        y = self.depthwise(x)
        y = self.norm(y)

        y = self.pw_up(y)
        y = self.gelu(y)
        # y = self.grn(y)
        y = self.pw_down(y)

        return x + y

# ---------------------------------------------------------------------
# 1.  A stack of BL-ConvNeXt blocks that preserves the FieldType
# ---------------------------------------------------------------------
class EscnnConvNeXtBlocks(nn.Module):
    """
    Sequence of `num_blocks` BLConvNeXtBlock, all sharing the *same* FieldType.
    """

    def __init__(
        self,
        in_type: e2nn.FieldType,
        num_blocks: int = 1,
        ksize: int = 7,
        expansion: int = 4,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BLConvNeXtBlock(
                    in_type=in_type,
                    ksize=ksize,
                    expansion=expansion,
                )
                for _ in range(num_blocks)
            ]
        )
        self.in_type = self.out_type = in_type

    def forward(self, x: e2nn.GeometricTensor):
        for blk in self.blocks:
            x = blk(x)
        return x
    

class EquivariantMultiplexImageEncoder(nn.Module):
    r"""
    *Input*  : B × C_in × H × W  (plain tensor, **C_in = `channel_embedding_dim`**)  
    *Output* : GeometricTensor whose FieldType is the last stage’s one.
    """

    def __init__(
        self,
        layers_blocks,            # e.g. [2, 2, 2]
        embedding_dims,           # e.g. [192, 384, 768]  ← **must** be multiples of (1+2*max_freq)
        channel_embedding_dim,    # e.g. 96 trivial copies
        include_stem: bool = True,
        maximum_frequency: int = 3,
    ):
        super().__init__()

        # ───────────────────────────────────────────────────────────────
        # 0.  Group bookkeeping
        # ───────────────────────────────────────────────────────────────
        self.max_freq  = maximum_frequency
        # self.r2_act    = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2)
        self.r2_act    = escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2) - axis zero was throwing
        # self.r2_act    = e2cnn.gspaces.Rot2dOnR2(N=-1, maximum_frequency=maximum_frequency)        # SO(2)
        self.G: e2cnn.group.groups.O2         = self.r2_act.fibergroup
        bl_repr        = self.G.bl_regular_representation(self.max_freq)  # dim = 1+2*max_freq
        # irreps = [self.G.irrep(0)] + [self.G.irrep(k) for k in range(1, self.max_freq + 1)]
        # bl_repr = directsum(irreps, name=f"bl_reg_{self.max_freq}")
        # print(bl_repr)
        repr_dim       = 1 + 2 * self.max_freq                           # 7 for max_freq = 3

        # ------------------------------------------------------------------
        # 1.  Build FieldTypes for every stage
        # ------------------------------------------------------------------
        def _make_stage_type(channels: int) -> e2nn.FieldType:
            # if channels % repr_dim != 0:
            #     raise ValueError(
            #         f"embedding_dim (= {channels}) must be a multiple of "
            #         f"the BL-regular representation dimension (= {repr_dim})"
            #     )
            # reps = [bl_repr] * (channels // repr_dim)
            reps = [bl_repr] * channels
            # print(reps)
            return e2nn.FieldType(self.r2_act, reps)

        self.input_type = e2nn.FieldType(
            self.r2_act, [self.r2_act.trivial_repr] * channel_embedding_dim
        )

        stage_types = [_make_stage_type(d) for d in embedding_dims]

        # ------------------------------------------------------------------
        # 2.  Stem / inter-stage down-sampling convolutions
        #     (stride-2, 2×2 kernel, no padding = “pixel unshuffle” like ConvNeXt)
        self.poolings = nn.ModuleList()
        prev_type = self.input_type

        # ------------------------------------------------------------------
        for idx, out_type in enumerate(stage_types):
            if idx == 0 and not include_stem:
                # identity shortcut if the user opts out of a stem
                self.poolings.append(e2nn.IdentityModule(prev_type))
            else:
                # print(f"Pooling from {prev_type} to {out_type}")
                self.poolings.append(
                    # nn.Sequential(
                    #     e2nn.PointwiseAvgPoolAntialiased(
                    #         prev_type,
                    #         sigma=0.5, # 0.66,
                    #         stride=2,
                    #         padding = 0,
                    #         # padding=0,  # no padding, like ConvNeXt
                    #         # initialize=True,  # TODO: set to True if you want to initialize
                    #     ),
                    #     # e2nn.PointwiseAvgPool2D(
                    #     #     prev_type,
                    #     #     kernel_size=2,
                    #     # ),
                    #     e2nn.R2Conv(                   # channel *expander*
                    #         prev_type, out_type,
                    #         kernel_size=1, stride=1, padding=0,
                    #         bias=True, initialize=True
                    #     )
                    # )
                    # nn.Sequential(
                    #     e2nn.PointwiseAvgPoolAntialiased(prev_type, sigma=0.66, stride=2),
                    #     e2nn.R2Conv(
                    #         prev_type,
                    #         out_type,
                    #         kernel_size=3,
                    #         stride=1,
                    #         padding=0,
                    #         bias=True,
                    #         initialize=True,  # TODO: set to True if you want to initialize
                    #     )
                    # )

                    
                    # nn.Sequential(
                    #     # AsymmetricPad2d(prev_type, (1, 0, 1, 0)),  # pad right and bottom by 1
                    #     AsymmetricPad2d(prev_type, (0, 1, 0, 1)),  # pad right and bottom by 1
                    #     e2nn.R2Conv(
                    #         prev_type,
                    #         out_type,
                    #         kernel_size=3,
                    #         stride=2,
                    #         padding=1,  # no padding, like ConvNeXt
                    #         bias=True,
                    #         initialize=True,  # TODO: set to True if you want to initialize
                    #     ),
                    #     # CropRightBottom(out_type, cropright=1, cropbottom=1)  if idx > 0 else nn.Identity()
                    # )
                    e2nn.R2Conv(
                        prev_type, out_type,
                        kernel_size=3, # was 3x3
                        stride=2,
                        padding=1, # change wrt 74, 75
                        bias=True,
                        initialize=True, # TODO: set to True if you want to initialize
                    )
                    
                    # this is what the 74, and 75 versions had
                    # e2nn.R2Conv(
                    #     prev_type, out_type,
                    #     kernel_size=3,
                    #     stride=2,
                    #     padding=1,
                    #     bias=True,
                    #     initialize=True, # TODO: set to True if you want to initialize
                    # )
                )
            prev_type = out_type

        # ------------------------------------------------------------------
        # 3.  Stage-wise NormNonLinearities (GELU analogue) and blocks
        # ------------------------------------------------------------------
        self.acts = nn.ModuleList(
            [e2nn.NormNonLinearity(t) for t in stage_types]
        )

        self.blocks = nn.ModuleList(
            [
                EscnnConvNeXtBlocks(
                    in_type=t,
                    num_blocks=n,
                )
                for t, n in zip(stage_types, layers_blocks)
            ]
        )

        self.out_type = self.blocks[-1].out_type  # last stage’s FieldType

        self.regular2trivial = Regular2Trivial(
            in_type=self.out_type,
            n_scalars=embedding_dims[-1]
        )

    # a convenience wrapper identical to vanilla ConvNeXt API
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        """Forward pass of the ConvNeXT.

        Args:
            x (torch.Tensor): Multiplex images batch tensor with shape [B, C, H, W]
            return_features (bool, optional): If True, returns the features after each block. Defaults to False.

        Returns:
            Dict: A dictionary containing the output tensor and optionally the features.
        """
        outputs = {}
        features = []
        g = e2nn.GeometricTensor(x, self.input_type)
        for i, (pool, act, blk) in enumerate(zip(self.poolings, self.acts, self.blocks)):
            pooled = pool(g)
            # print(f"After pooling: {pooled.tensor.shape}, type: {pooled.type}")
            # print(f"Pooling type: {type(pool)}")
            # print(f"Activation in type: {act.in_type}")
            g = act(pooled)
            # print(f"After pooling: {g.tensor.shape}, type: {g.type}")
            g = blk(g)
            if i == len(self.blocks) - 1:
                g = self.regular2trivial(g)
                g = g.tensor  # Extract the raw tensor from GeometricTensor
            if return_features:
                features.append(g)
        # print(f'Latent shape: {g.shape}')
        outputs['output'] = g
        if return_features:
            outputs['features'] = features
        return outputs