import torch
from typing import List, Dict, Optional, Type, Literal

import torch.nn as nn
import torch.nn.functional as F


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


class MLP(nn.Module):
    """Standard MLP module"""
    def __init__(
            self, 
            embedding_dim: int,
            mlp_dim: int,
            mlp_bias: bool = True,
            act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim, bias=mlp_bias),
            act(),
            nn.Linear(mlp_dim, embedding_dim, bias=mlp_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttnBlock(nn.Module):
    def __init__(
            self, 
            embedding_dim : int, 
            num_heads: int, 
            mlp_ratio: float=4.,
            mlp_bias: bool=True,
        ):
        super(AttnBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = embedding_dim

        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        self.proj = nn.Linear(embedding_dim, embedding_dim)

        self.mlp = MLP(
            embedding_dim=embedding_dim,
            mlp_dim=int(embedding_dim * mlp_ratio),
            mlp_bias=mlp_bias
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.proj(x)
        x += res

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += res

        return x


# class TransformerEncoder(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float = 4.0):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=embed_dim, 
#                 nhead=num_heads, 
#                 dim_feedforward=int(embed_dim * mlp_ratio),
#                 activation="gelu",
#                 batch_first=True,
#             ) for _ in range(num_layers)
#         ])

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x


class Superkernel(nn.Module):
    def __init__(
            self, 
            num_channels: int,
            embedding_dim: int, 
            num_layers: int,
            num_heads: int,
            mlp_ratio: float,
            layer_type: Literal['conv', 'linear'],
            kernel_size: int = None,
            **kwargs
        ):
        """Initialize the Superkernel model

        Args:
            num_channels (int): Number of channels in the input tensor
            embedding_dim (int): Embedding dimension for the input tensor
            num_layers (int): Number of layers in the model
            num_heads (int): Number of heads per channel embedding
            mlp_ratio (float): MLP ratio for the model
            layer_type (Literal['conv', 'linear']): Whether the output of superkernel should be a convolutional or linear layer weights
            kernel_size (int, optional): Kernel size for the conv layer (already squared). Model embedding will be embedding_dim*kernel_size**2. 
                Number of heads should be a multiplication of squared kernel_size. Defaults to None.
        """
        super(Superkernel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layer_type = layer_type
        self.kernel_size = kernel_size

        model_dim = embedding_dim * kernel_size**2 if layer_type == 'conv' else embedding_dim 
        self.embedder = nn.Embedding(num_channels, model_dim)

        self.encoder = nn.Sequential(*[
            AttnBlock(
                embedding_dim=model_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                mlp_bias=True
            ) 
            for _ in range(num_layers)
        ])

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns the superkernel weights for the given indices.

        Args:
            indices (torch.Tensor): Indices of the markers in the input tensor. 
                Shape: (B, C), where B is batch size and C is number of channels.

        Returns:
            torch.Tensor: Superkernel weights for the given indices.
        """
        B, C = indices.shape
        x = self.embedder(indices) # (B, C, E)
        
        if self.num_layers > 0:
            x = self.encoder(x) # (B, C, E)

        if self.layer_type == 'conv':
            # (B, W, C, K, K)
            x = x.reshape(B, C, self.kernel_size, self.kernel_size, self.embedding_dim).permute(0, 4, 1, 2, 3) 
            
        return x


# class InterSuperkernel(nn.Module):
#     def __init__(
#             self, 
#             in_channels: int, 
#             all_channels: int,
#             inner_channels: int = 128,
#             kernel_size: int = 3, 
#             num_heads: int = 4, 
#             num_layers: int = 3
#         ): 
#         """
#         Super Kernel Channel Block
#         :param in_channels: k (input channels)
#         :param inner_channels: l (inner channels)
#         :param kernel_size: s (kernel area)
#         :params num_heads: Number of attention heads in Transformer
#         :param num_layers: Number of Transformer layers in decoder
#         """
#         super().__init__()
#         self.in_channels = in_channels
#         self.inner_channels = inner_channels
#         self.kernel_size = kernel_size
#         self.kernel_dim = inner_channels * kernel_size ** 2
        
#         self.channel_embedding = nn.Embedding(all_channels, self.kernel_dim)

#         self.kernel_embedding = torch.nn.Parameter(torch.zeros(1, inner_channels, self.kernel_dim))
#         self.kernel_base = torch.nn.Parameter(torch.zeros(inner_channels, inner_channels, kernel_size, kernel_size))
#         nn.init.xavier_uniform_(self.kernel_embedding)      
#         nn.init.xavier_uniform_(self.kernel_base)

#         self.decoder = TransformerEncoder(embed_dim=self.kernel_dim, num_heads=num_heads, num_layers=num_layers)

#         # print(in_channels)
#         self.inner_projection = nn.Linear(in_channels, inner_channels)
#         self.outer_projection = nn.Linear(inner_channels, in_channels)
#         self.bn_inner = nn.BatchNorm2d(inner_channels)
#         self.bn_outer = nn.BatchNorm2d(in_channels)
#         self.activation = nn.LeakyReLU()
  
#     def forward(self, x, indices):
#         """
#         :param x: input tensor of shape (B, K, H, W)"
#         :param indices: input tensor of shape (B, C)
#         """

#         # B, K, H, W = x.shape
#         B, C = indices.shape
#         channel_embedding = self.channel_embedding(indices) # (B, C, L * S^2)
#         batch_kernel_embeddings = self.kernel_embedding.expand(B, -1, -1) # (B, L, L * S^2)
#         decoder_input = torch.concat([channel_embedding, batch_kernel_embeddings], dim=1) # (B, C + L, L * S^2))
#         decoder_output = self.decoder(decoder_input) # (B, C + L, L * S^2)
#         output_kernel = decoder_output[:, C:, :] # (B, L, L * S^2)
#         output_kernel = output_kernel.view(B, self.inner_channels, self.inner_channels, self.kernel_size, self.kernel_size)
#         final_kernel = output_kernel + self.kernel_base
        
#         # print(x.shape)
#         inner_x = self.inner_projection(x.permute(0, 2, 3, 1)) # (B, H, W, L) # TODO: CONV 1x1
#         inner_x = inner_x.permute(0, 3, 1, 2) # (B, L, H, W)

#         inner_x = self.bn_inner(inner_x)
#         inner_x = self.activation(inner_x)

#         # super_kernel_x = F.conv2d(inner_x, final_kernel, padding=self.kernel_size // 2, stride=1) # (B, C, H, W)
#         super_kernel_x = torch.cat([
#                 F.conv2d(
#                     inner_x[i].unsqueeze(0), 
#                     final_kernel[i].to(x.dtype), 
#                     padding=self.kernel_size // 2,
#                     stride=1
#                 )
#                 for i in range(B)
#             ])

#         projected_x = self.outer_projection(super_kernel_x.permute(0, 2, 3, 1)) # (B, H, W, C)
#         projected_x = projected_x.permute(0, 3, 1, 2) # (B, C, H, W)
#         projected_x = self.bn_outer(projected_x)
#         projected_x = self.activation(projected_x)

#         return x + projected_x


class MultiplexImageEncoder(nn.Module):
    """Encoder backbone for encoding multiplex images."""

    def __init__(
            self,
            layers_blocks: List[int],
            embedding_dims: List[int],
            channel_embedding_dim: int,
            # num_all_channels: int,
            **kwargs
    ):
        """Initialize the Multiplex Image Encoder.

        Args:
            layers_blocks (List[int]): Number of blocks in each layer.
            embedding_dims (List[int]): Embedding dimensions for each layer.
            channel_embedding_dim (int): Embedding dimension for the channels.
        """
        super().__init__()

        self.poolings = nn.ModuleList()
        self.poolings.append(
            nn.Conv2d(
                channel_embedding_dim, 
                embedding_dims[0], 
                kernel_size=2, 
                padding=0, 
                stride=2
            )
        )

        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.poolings.append(
                nn.Conv2d(input_dim, out_dim, kernel_size=2, padding=0, stride=2)
            )

        self.act = nn.GELU()
             
        self.blocks = nn.ModuleList()
        # self.inter_superkernels = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(
                nn.Sequential(*[
                    ConvNextBlock(dim)
                    for _ in range(blocks)
                ])
            )
            
            # self.inter_superkernels.append(
            #     nn.ModuleList([
            #         InterSuperkernel(
            #             in_channels=dim,
            #             all_channels=num_all_channels,
            #         )
            #         for _ in range(blocks)
            #     ])
            # )

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
        for pooling, blocks in zip(self.poolings, self.blocks):
            x = self.act(pooling(x))
            # for block in blocks:
            x = blocks(x)
                # x = inter_superkernel(x, indices)

            if return_features:
                features.append(x)

        outputs['output'] = x
        if return_features:
            outputs['features'] = features
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
    

class MultiplexAutoencoder(nn.Module):
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

        self.encoder = MultiplexImageEncoder(
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


class SegmentationDecoder(nn.Module):
    """Decoder for U-net-like segmentation tasks."""
    
    def __init__(
            self,
            input_embedding_dim: int,
            decoded_embed_dims: List[int],
            num_blocks: List[int],
            num_all_outputs: int,
            decoder_layer_type: Type = ConvNextBlock,
            **kwargs
        ) -> None:
            """
            Args:
                input_embedding_dim (int): Embedding dimension of the input tensor.
                decoded_embed_dims (List[int]): Embedding dimensions of the decoded tensors (before last projections).
                num_blocks (List[int]): Number of multiplex blocks in each intermediate layer.
                num_all_outputs (int): Number of all possible output predictions.
                decoder_layer_type (Type, optional): Type of the decoder layer. Defaults to ConvNextBlock.
            """
            super().__init__()
            self.num_all_outputs = num_all_outputs
            self.decoded_embed_dims = decoded_embed_dims

            self.proj = nn.Linear(
                decoded_embed_dims[-1], 
                num_all_outputs
            )

            self.act = nn.GELU()

            self.up_convs = nn.ModuleList()
            self.feat_agg_convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            for i, (decoded_embed_dim, num_block) in enumerate(zip(decoded_embed_dims, num_blocks)):
                up_conv = nn.ConvTranspose2d(
                    input_embedding_dim if i == 0 else decoded_embed_dims[i-1],
                    decoded_embed_dim,
                    kernel_size=2,
                    stride=2,
                )
                feat_agg_conv = nn.Conv2d(
                    decoded_embed_dim * 2, 
                    decoded_embed_dim, 
                    kernel_size=1
                )
                norm = nn.LayerNorm(decoded_embed_dim)
                decoder_block = nn.Sequential(*[
                    decoder_layer_type(
                        decoded_embed_dim, 
                        **kwargs
                    ) for _ in range(num_block)
                ])
                self.up_convs.append(up_conv)
                self.feat_agg_convs.append(feat_agg_conv)
                self.norms.append(norm)
                self.decoder_blocks.append(decoder_block)



    def forward(self, x: torch.Tensor, encoded_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multiplex Image Decoder.

        Args:
            x (torch.Tensor): Input tensor (embedding).
            indices (torch.Tensor): Indices of the cell types to predict.
            encoded_features (torch.Tensor): Encoded features from the encoder.

        Returns:
            torch.Tensor: Segmentation mask tensor
        """
        encoded_features = list(reversed(encoded_features)) 

        assert len(encoded_features) == len(self.decoded_embed_dims), \
            f'Expected {len(self.decoded_embed_dims)} encoded features, got {len(encoded_features)}'
        
        for i, (up_conv, feat_agg_conv, norm, decoder_block) in enumerate(zip(
            self.up_convs, self.feat_agg_convs, self.norms, self.decoder_blocks
        )):
            x = up_conv(x) 
            x = torch.cat([x, encoded_features[i]], dim=1)
            x = feat_agg_conv(x)
            x = self.act(x)
            x = norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = decoder_block(x)

        x = self.proj(x.permute(0, 2, 3, 1))

        return x


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        encoder_depth: int,
        decoder_depth: int,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
    ):
        """Vector Quantizer module for encoding and decoding image latents/embeddings.

        Args:
            encoder_depth (int): Depth of the quantizer encoder.
            decoder_depth (int): Depth of the quantizer decoder.
            num_embeddings (int): Number of embeddings in the quantizer.
            embedding_dim (int): Dimension of the input embeddings.
            beta (float, optional): Weighting factor for the loss. Defaults to 0.25.
        """
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.encoder = nn.Sequential(*[
            ConvNextBlock(dim=embedding_dim)
            for _ in range(encoder_depth)
        ])

        self.decoder = nn.Sequential(*[
            ConvNextBlock(dim=embedding_dim)
            for _ in range(decoder_depth)
        ])

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings) 

    def forward(self, x, return_losses=False) -> Dict:
        BATCH_SIZE, embedding_dim, H, W = x.shape

        inputs = self.encoder(x) # (B, C, H, W)

        inputs = inputs.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        flat_input = inputs.reshape(-1, embedding_dim) # (B*H*W, C)

        embeddings = self.embedding.weight # (num_embeddings, C)
        dists = torch.cdist(flat_input, embeddings, p=2) # (B*H*W, num_embeddings)
        encoding_indices = torch.argmin(dists, dim=1) # (B*H*W)
        
        quantized_latent = torch.index_select(embeddings, 0, encoding_indices)
        quantized_latent = quantized_latent.view_as(inputs)

        matched_indices = encoding_indices.view(BATCH_SIZE, H, W)

        if return_losses:
            q_loss = F.mse_loss(inputs.detach(), quantized_latent) # sg[z_e(x)], e
            e_loss = F.mse_loss(inputs, quantized_latent.detach()) # z_e(x), sg[e]
            vq_loss = q_loss + self.beta * e_loss

        quantized_latent = quantized_latent.detach() + inputs - inputs.detach()
        quantized_latent = quantized_latent.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W) 

        decoded_latent = self.decoder(quantized_latent)

        outputs = {
            'quantized_latent': quantized_latent,
            'decoded_latent': decoded_latent,
            'matched_indices': matched_indices
        }
        if return_losses:
            mae_loss = F.l1_loss(x, decoded_latent)
            outputs['vq_loss'] = vq_loss
            outputs['e_loss'] = e_loss
            outputs['q_loss'] = q_loss
            outputs['mae_loss'] = mae_loss

        return outputs
