#!/usr/bin/env python3
# modules_vit.py
# -*- coding: utf-8 -*-


import timm 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchvision.models import VisionTransformer
from torchvision.models.vision_transformer import Encoder
from typing import List, Dict, Optional, Type, Callable, Tuple


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GlobalResponseNormalization(nn.Module):
    """Global Response Normalization (GRN) layer.
    """
    def __init__(self, dim, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNextBlock(nn.Module):
    """ConvNext2 block.
    """
    def __init__(self, dim: int, inter_dim: int = None, kernel_size: int = 7, padding: int = 3):
            super().__init__()
            inter_dim = inter_dim or dim * 4
            self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
            self.ln = nn.LayerNorm(dim)
            self.conv2 = nn.Linear(dim, inter_dim)
            self.act = nn.GELU()
            self.grn = GlobalResponseNormalization(inter_dim)
            self.conv3 = nn.Linear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.conv3(x)
        x = x.permute(0, 3, 1, 2)
        x = x + residual
        return x

class ConvNeXtEncoder(nn.Module):
    """ ConvNeXT Encoder backbone for encoding images.
    """
    def __init__(self, input_channels: int, layers_blocks: List[int], embedding_dims: List[int], stem: bool = True):
        super().__init__()
        self.norm_poolings = nn.ModuleList()
        if stem:
            stem_layer = nn.Sequential(
                    nn.Conv2d(input_channels, embedding_dims[0], kernel_size=2, stride=2, padding=0),
                    LayerNorm(embedding_dims[0], data_format="channels_first")
                )
        else:
            stem_layer = nn.Sequential(
                    LayerNorm(input_channels, data_format="channels_first"),
                    nn.Conv2d(input_channels, embedding_dims[0], kernel_size=2, stride=2, padding=0),
                )

        self.norm_poolings.append(stem_layer)
        for i, out_dim in enumerate(embedding_dims[1:]):
            input_dim = embedding_dims[i]
            self.norm_poolings.append(
                nn.Sequential(
                    LayerNorm(input_dim, data_format="channels_first"),
                    nn.Conv2d(input_dim, out_dim, kernel_size=2, padding=0, stride=2)
                )
            )

        self.blocks = nn.ModuleList()
        for blocks, dim in zip(layers_blocks, embedding_dims):
            self.blocks.append(nn.Sequential(*[ConvNextBlock(dim) for _ in range(blocks)]))
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        outputs = {}
        features = []
        for norm_pooling, blocks in zip(self.norm_poolings, self.blocks):
            x = norm_pooling(x)
            x = blocks(x)
            if return_features:
                features.append(x)

        outputs['output'] = x
        if return_features:
            outputs['features'] = features

        return outputs


class Hyperkernel(nn.Module):
    def __init__(self, num_channels, input_dim, embedding_dim, module_type, kernel_size=1, padding=0, stride=1, use_bias=True):
        super(Hyperkernel, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.layer_type = 'linear' if kernel_size == stride == 1 and padding == 0 else 'conv'
        self.kernel_size = 1 if self.layer_type == 'linear' else kernel_size
        self.padding = padding
        self.stride = stride
        self.module_type = module_type
        
        # Calculate full model dimension for embedding
        self.out_dim = self.embedding_dim * self.kernel_size**2
        self.model_dim = self.out_dim * self.input_dim
        
        self.hyperkernel_weights = nn.Embedding(num_channels, self.model_dim)
        self.use_bias = use_bias
        
        if use_bias:
            if module_type == 'encoder':
                self.hyperkernel_bias = nn.Parameter(torch.zeros(1, self.embedding_dim, 1, 1))
            else:
                self.hyperkernel_bias = nn.Embedding(num_channels, self.embedding_dim)

    def forward(self, x, indices):
        B, C = indices.shape
        weights = self.hyperkernel_weights(indices).to(x.dtype)
        weights = weights.reshape(B, C, self.input_dim, self.embedding_dim, self.kernel_size, self.kernel_size)

        if self.module_type == 'encoder':
            if self.layer_type == 'conv':
                w = weights.permute(0, 3, 1, 2, 4, 5).contiguous()
                w = w.view(B * self.embedding_dim, C * self.input_dim, self.kernel_size, self.kernel_size)
                x_reshaped = x.reshape(1, B * C * self.input_dim, *x.shape[-2:])
                x = F.conv2d(x_reshaped, w, padding=self.padding, stride=self.stride, groups=B)
                x = x.reshape(B, self.embedding_dim, x.shape[-2], x.shape[-1])
                
            else:
                w = weights.reshape(B, C, self.input_dim, self.embedding_dim)
                x = x.reshape(B, C, self.input_dim, *x.shape[-2:])
                x = torch.einsum('bcihw, bcie -> behw', x, w)
            
            if self.use_bias:
                x = x + self.hyperkernel_bias

        else:        
            if self.layer_type == 'conv': 
                x_expanded = x.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(1, B * C * self.input_dim, *x.shape[-2:])
                w = weights.permute(0, 1, 3, 2, 4, 5).contiguous()
                w = w.view(B * C * self.embedding_dim, self.input_dim, self.kernel_size, self.kernel_size)
                
                # Convolution
                x = F.conv2d(x_expanded, w, padding=self.padding, stride=self.stride, groups=B*C)
                
                # Reshape output: (1, B*C*E, H, W) -> (B, C, E, H, W)
                x = x.reshape(B, C, self.embedding_dim, x.shape[-2], x.shape[-1])
                
            else:
                w = weights.reshape(B, C, self.input_dim, self.embedding_dim)
                # Einsum: b=batch, c=channel, i=input, e=embed
                x = torch.einsum('bihw, bcie -> bcehw', x, w)
                
            if self.use_bias:
                # Decoder bias is per-channel
                x = x + self.hyperkernel_bias(indices).unsqueeze(-1).unsqueeze(-1)
                
        return x


def to_2tuple(x) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict:
        return {'output': super().forward(x)}


# SWIN TRANSFORMER MODULES
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0., **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (head_dim or dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        ) 

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nD = mask.shape[0]
            attn = attn.view(B_ // nD, nD, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int] = None, norm_layer: Type[nn.Module] = nn.LayerNorm, device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim, **dd)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        pad_values = (0, 0, 0, W % 2, 0, H % 2)
        x = nn.functional.pad(x, pad_values)
        _, H, W, _ = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        return self.reduction(self.norm(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=4, head_dim=None, window_size=7, shift_size=0,
                 always_partition=False, dynamic_mask=False, mlp_ratio=4., qkv_bias=True, 
                 proj_drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 device=None, dtype=None):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.target_shift_size = to_2tuple(shift_size)
        self.always_partition = always_partition
        self.dynamic_mask = dynamic_mask
        self.window_size, self.shift_size = self._calc_window_shift(window_size, shift_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim, **dd)
        self.attn = WindowAttention(dim, num_heads=num_heads, head_dim=head_dim, window_size=self.window_size,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop, **dd)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, **dd)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop, **dd)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.register_buffer("attn_mask", None, persistent=False)
        self.apply(self._init_weights)
        
        if not self.norm1.weight.is_meta:
            self.reset_parameters()

    def reset_parameters(self):
        self._init_buffers()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_buffers(self):
        if not self.dynamic_mask:
            attn_mask = self.get_attn_mask(device=self.norm1.weight.device, dtype=self.norm1.weight.dtype)
            self.register_buffer("attn_mask", attn_mask, persistent=False)

    def get_attn_mask(self, x=None, device=None, dtype=None):
        if any(self.shift_size):
            if x is not None:
                H, W = x.shape[1], x.shape[2]
                device, dtype = x.device, x.dtype
            else:
                H, W = self.input_resolution

            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1), dtype=dtype, device=device)
            cnt = 0
            for h in ((0, -self.window_size[0]), (-self.window_size[0], -self.shift_size[0]), (-self.shift_size[0], None)):
                for w in ((0, -self.window_size[1]), (-self.window_size[1], -self.shift_size[1]), (-self.shift_size[1], None)):
                    img_mask[:, h[0]:h[1], w[0]:w[1], :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        return attn_mask

    def _calc_window_shift(self, target_window_size, target_shift_size=None):
        target_window_size = to_2tuple(target_window_size)
        if target_shift_size is None:
            target_shift_size = self.target_shift_size
            if any(target_shift_size):
                target_shift_size = (target_window_size[0] // 2, target_window_size[1] // 2)
        else:
            target_shift_size = to_2tuple(target_shift_size)

        if self.always_partition:
            return target_window_size, target_shift_size

        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def set_input_size(self, feat_size, window_size, always_partition=None):
        self.input_resolution = feat_size
        if always_partition is not None:
            self.always_partition = always_partition

        self.window_size, self.shift_size = self._calc_window_shift(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self._init_buffers()

    def _attn(self, x):
        _, H, W, C = x.shape
        shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2)) if any(self.shift_size) else x
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
        _, Hp, Wp, _ = shifted_x.shape
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_area, C)
        attn_mask = self.get_attn_mask(shifted_x) if getattr(self, 'dynamic_mask', False) else self.attn_mask
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size[0], self.window_size[1], C), self.window_size, Hp, Wp)[:, :H, :W, :].contiguous()
        return torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2)) if any(self.shift_size) else shifted_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = x + self.drop_path1(self._attn(self.norm1(x))).reshape(B, H, W, C)
        x = x + self.drop_path2(self.mlp(self.norm2(x))).reshape(B, H, W, C)
        return x
    
    def init_non_persistent_buffers(self) -> None:
        self._init_buffers()


class SwinTransformerStage(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, downsample=True, num_heads=4, window_size=7, mlp_ratio=4., drop_path=0., **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        self.downsample = PatchMerging(dim=dim, out_dim=out_dim) if downsample else nn.Identity()
        shift_size = tuple([to_2tuple(window_size)[0] // 2] * 2)
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim, input_resolution=self.output_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size, mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, **kwargs
            ) for i in range(depth)])

    def set_input_size(self, feat_size, window_size, always_partition=None):
        self.input_resolution = feat_size
        self.output_resolution = tuple(i // 2 for i in feat_size) if not isinstance(self.downsample, nn.Identity) else feat_size
        for block in self.blocks: block.set_input_size(self.output_resolution, window_size, always_partition)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.downsample(x))


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding (Stem).
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, output_fmt='NHWC'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.output_fmt = output_fmt

    def forward(self, x):
        x = self.proj(x)
        if self.output_fmt == 'NHWC':
            x = x.permute(0, 2, 3, 1)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SwinHierarchicalEncoder(nn.Module):
    """ Swin Encoder mimicking ConvNeXtEncoder structure with Stages.
    """
    def __init__(self, input_channels: int, layers_blocks: List[int], embedding_dims: List[int], input_resolution: Tuple[int, int], stem: bool = True):
        super().__init__()
        self.stages = nn.ModuleList()
        curr_res, in_dim = input_resolution, input_channels
        
        if stem:
            patch_size, embed_dim = 2, embedding_dims[0]
            self.stem = PatchEmbed(img_size=input_resolution, patch_size=patch_size, in_chans=input_channels, embed_dim=embed_dim, output_fmt='NHWC')
            curr_res, in_dim = (input_resolution[0] // patch_size, input_resolution[1] // patch_size), embed_dim
        else:
            self.stem = nn.Identity()

        for i, (depth, out_dim) in enumerate(zip(layers_blocks, embedding_dims)):
            do_downsample = not (stem and i == 0)
            stage = SwinTransformerStage(
                dim=in_dim, out_dim=out_dim, input_resolution=curr_res, depth=depth,
                downsample=do_downsample, num_heads=max(out_dim // 32, 1), window_size=8, drop_path=0.0
            )
            self.stages.append(stage)
            curr_res, in_dim = stage.output_resolution, out_dim

    def forward(self, x, return_features=False):
        outputs, features = {}, []
        x = self.stem(x) if not isinstance(self.stem, nn.Identity) else x.permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
            if return_features:
                features.append(x.permute(0, 3, 1, 2))

        outputs['output'] = x.permute(0, 3, 1, 2)
        if return_features: outputs['features'] = features
        return outputs


# ViT & DINOv2 / DINOv3 MODULES
class TransformerEncoder(Encoder):
    def __init__(self, seq_length: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: float, attention_dropout: float, norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
    def forward(self, input: torch.Tensor):
        return self.ln(self.layers(self.dropout(input)))

class VitEncoder(VisionTransformer):
    """ Standard ViT (feature extractor).
    """
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, superkernel_dim, dropout = 0, attention_dropout = 0, num_classes = 1000, representation_size = None, norm_layer = partial(nn.LayerNorm, eps=1e-6), conv_stem_configs = None):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size, norm_layer, conv_stem_configs)
        self.conv_proj = nn.Conv2d(in_channels=superkernel_dim, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.encoder = TransformerEncoder(seq_length=self.seq_length, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=mlp_dim, dropout=dropout, attention_dropout=attention_dropout, norm_layer=norm_layer)
        self.hidden_dim, self.patch_size, self.image_size = hidden_dim, patch_size, image_size

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        x = self.conv_proj(x)
        return x.reshape(n, self.hidden_dim, (h // self.patch_size) * (w // self.patch_size)).permute(0, 2, 1)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0, return_features: bool = False) -> Dict:
        n, _, h, w = x.shape
        x = self._process_input(x)
        x = x + self.encoder.pos_embedding[:, :x.shape[1], :]
        x = self.encoder(x)
        x = x.permute(0, 2, 1).reshape(n, self.hidden_dim, h // self.patch_size, w // self.patch_size)
        outputs = {"output": x}
        if return_features: outputs["features"] = [x]
        return outputs


class DinoV2Encoder(nn.Module):
    def __init__(self, input_dim: int, model_name: str = 'dinov2_vits14'):
        super().__init__()
        # Using timm to load DINOv2 models
        timm_names = {
            'dinov2_vits14': 'vit_small_patch14_dinov2.lvd142m',
            'dinov2_vitb14': 'vit_base_patch14_dinov2.lvd142m',
            'dinov2_vitl14': 'vit_large_patch14_dinov2.lvd142m',
            'dinov2_vitg14': 'vit_giant_patch14_dinov2.lvd142m',
        }
        timm_name = timm_names.get(model_name, model_name)
        
        self.model = timm.create_model(
            timm_name, 
            pretrained=True, 
            num_classes=0, 
            dynamic_img_size=True 
        )
        
        # Adapter
        original_conv = self.model.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels=input_dim, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        self.model.patch_embed.proj = new_conv
        self.embed_dim = self.model.num_features

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        x_tokens = self.model.forward_features(x)
        prefix_tokens = self.model.num_prefix_tokens
        patch_tokens = x_tokens[:, prefix_tokens:]
        
        # Reshape [B, N, D] -> [B, D, H, W]
        B, N, D = patch_tokens.shape
        H_feat = W_feat = int(N ** 0.5)
        x_out = patch_tokens.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        
        ret = {'output': x_out}
        if return_features: ret['features'] = [x_out]
        return ret


class DinoV3Encoder(nn.Module):
    def __init__(self, input_dim: int, model_name: str = 'vit_7b_patch16_dinov3.sat493m'):
        super().__init__()
        self.model = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            dynamic_img_size=True 
        )
        
        # Adapter to match input dimensions (e.g. from Hyperkernel output) to model input
        original_conv = self.model.patch_embed.proj
        new_conv = nn.Conv2d(
            in_channels=input_dim, 
            out_channels=original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding
        )
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        self.model.patch_embed.proj = new_conv
        self.embed_dim = self.model.num_features

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict:
        # DINOv3 with RoPE handles dynamic sizes well.
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x_tokens = self.model.forward_features(x)
        
        # Remove special tokens (CLS, registers, etc.)
        prefix_tokens = self.model.num_prefix_tokens
        patch_tokens = x_tokens[:, prefix_tokens:]
        
        # Reshape [B, N, D] -> [B, D, H, W]
        B, N, D = patch_tokens.shape
        H_feat = W_feat = int(N ** 0.5)
        x_out = patch_tokens.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
        
        ret = {'output': x_out}
        if return_features: ret['features'] = [x_out]
        return ret


class MultiplexImageEncoder(nn.Module):
    """Updated Encoder backbone supporting ConvNeXt, ViT, Swin, and DinoV3.
    """
    def __init__(self, num_channels, ma_layers_blocks, ma_embedding_dims, hyperkernel_config, pm_layers_blocks=None, pm_embedding_dims=None, encoder_type='convnext', vit_config=None, swin_config=None, dinov2_config=None, dinov3_config=None, input_image_size=(128, 128), **kwargs):
        super().__init__()
        self.encoder_type = encoder_type
        
        # Agnostic
        if not ma_layers_blocks:
            self.marker_agnostic_encoder = Identity()
            self.agnostic_is_identity = True
            hk_input_dim = 1
            curr_res = input_image_size
        else:
            self.agnostic_is_identity = False
            if encoder_type == 'swin':
                self.marker_agnostic_encoder = SwinHierarchicalEncoder(1, ma_layers_blocks, ma_embedding_dims, input_resolution=input_image_size, stem=True)
                ds_factor = 2 
                curr_res = (input_image_size[0] // ds_factor, input_image_size[1] // ds_factor)
            else:
                self.marker_agnostic_encoder = ConvNeXtEncoder(1, ma_layers_blocks, ma_embedding_dims)
                curr_res = input_image_size
            hk_input_dim = ma_embedding_dims[-1]

        # Hyperkernel
        hk_config = hyperkernel_config.copy()
        if 'input_dim' in hk_config: del hk_config['input_dim']
        self.hyperkernel = Hyperkernel(num_channels, hk_input_dim, module_type='encoder', **hk_config)
        self.act = nn.GELU()

        # Pan-Marker
        if encoder_type == 'vit':
            if vit_config is None: raise ValueError("vit_config required for vit")
            vit_config['superkernel_dim'] = self.hyperkernel.embedding_dim
            self.pan_marker_encoder = VitEncoder(**vit_config)
        elif encoder_type == 'swin':
            self.pan_marker_encoder = SwinHierarchicalEncoder(self.hyperkernel.embedding_dim, pm_layers_blocks, pm_embedding_dims, input_resolution=curr_res, stem=False)
        elif encoder_type == 'dinov2': #  DinoV2 logic
            model_name = dinov2_config.get('model_name', 'dinov2_vits14') if dinov2_config else 'dinov2_vits14'
            self.pan_marker_encoder = DinoV2Encoder(self.hyperkernel.embedding_dim, model_name)
        elif encoder_type == 'dinov3': # DinoV3 config
            config = dinov3_config if dinov3_config else dinov2_config
            model_name = config.get('model_name', 'vit_7b_patch16_dinov3.sat493m') if config else 'vit_7b_patch16_dinov3.sat493m'
            self.pan_marker_encoder = DinoV3Encoder(self.hyperkernel.embedding_dim, model_name)
        else:
            self.pan_marker_encoder = ConvNeXtEncoder(self.hyperkernel.embedding_dim, pm_layers_blocks, pm_embedding_dims, stem=False)

    def forward(self, x, encoded_indices, mask_ratio=0.0, return_features=False):
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        
        if not self.agnostic_is_identity:
            out = self.marker_agnostic_encoder(x, return_features=return_features)
            x = out['output']
            if return_features: ag_feats = out.get('features', [])
        else:
            ag_feats = []
            
        _, E_ma, H_ma, W_ma = x.shape
        x = x.reshape(B, C, E_ma, H_ma, W_ma).reshape(B, C * E_ma, H_ma, W_ma)
        x = self.hyperkernel(x, encoded_indices)
        x = self.act(x)
        
        if self.encoder_type == 'vit':
            x = self.pan_marker_encoder(x, mask_ratio=mask_ratio, return_features=return_features)
        else:
            x = self.pan_marker_encoder(x, return_features=return_features)
        
        if return_features:
            if isinstance(x, dict) and 'features' in x:
                features = ag_feats + x['features']
            elif not isinstance(x, dict):
                features = ag_feats + [x]
            else:
                features = ag_feats
            outputs = {'output': x['output'], 'features': features}
        else:
            outputs = {'output': x['output']}

        return outputs


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))     
        return x


class MultiplexImageDecoder(nn.Module):
    def __init__(self, input_embedding_dim, decoded_embed_dim, num_blocks, scaling_factor, num_channels, hyperkernel_config, decoder_type='convnext', vit_config=None):
        super().__init__()
        self.scaling_factor, self.num_channels, self.decoded_embed_dim, self.decoder_type = scaling_factor, num_channels, decoded_embed_dim, decoder_type
        hk_config = hyperkernel_config.copy(); hk_config.pop('input_dim', None)
        self.channel_embed = Hyperkernel(num_channels, input_embedding_dim, decoded_embed_dim, 'decoder', **hk_config)
        self.pred = nn.Conv2d(decoded_embed_dim, scaling_factor**2 * 2, kernel_size=1)
        if decoder_type == 'vit':
            self.decoder = nn.Sequential(*[TransformerBlock(decoded_embed_dim, vit_config.get('num_heads', 4)) for _ in range(num_blocks)])
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 2048, decoded_embed_dim))
        else:
            self.decoder = nn.Sequential(*[ConvNextBlock(decoded_embed_dim) for _ in range(num_blocks)])

    def forward(self, x, indices):
        B, _, H, W = x.shape; C = indices.shape[1]; N = B*C
        x = self.channel_embed(x, indices).reshape(N, self.decoded_embed_dim, H, W)
        if self.decoder_type == 'vit':
            x = self.decoder(x.flatten(2).transpose(1, 2) + self.decoder_pos_embed[:, :x.shape[2]*x.shape[3], :]).transpose(1, 2).reshape(N, self.decoded_embed_dim, H, W)
        else:
            x = self.decoder(x)

        return torch.einsum('bcxyohw -> bchxwyo', self.pred(x).reshape(B, C, self.scaling_factor, self.scaling_factor, 2, H, W)).reshape(B, C, H*self.scaling_factor, W*self.scaling_factor, 2)


class MultiplexAutoencoder(nn.Module):
    def __init__(self, num_channels, encoder_config, decoder_config, encoder_type='swin', vit_config=None, swin_config=None, dinov2_config=None, dinov3_config=None, input_image_size=(128, 128)):
        super().__init__()
        self.num_channels = num_channels
        self.encoder_type = encoder_config.get('encoder_type', encoder_type)
        vit_config = encoder_config.get('vit_config', vit_config)
        swin_config = encoder_config.get('swin_config', swin_config)
        dinov2_config = encoder_config.get('dinov2_config', dinov2_config)
        dinov3_config = encoder_config.get('dinov3_config', dinov3_config)

        # Encoder
        self.encoder = MultiplexImageEncoder(
            num_channels=num_channels, 
            input_image_size=input_image_size, 
            **encoder_config 
        )
        
        # Latent Dim & Scaling Factor
        if self.encoder_type == 'vit' and vit_config:
            self.latent_dim = vit_config['hidden_dim']
        elif self.encoder_type == 'swin':
            self.latent_dim = encoder_config['pm_embedding_dims'][-1]
        elif self.encoder_type == 'dinov2' or self.encoder_type == 'dinov3':
            # Handle both v2 and v3 keys
            config = dinov3_config if self.encoder_type == 'dinov3' else dinov2_config
            config = config if config else {}
            model_name = config.get('model_name', 'dinov2_vits14')
            self.latent_dim = self.encoder.pan_marker_encoder.embed_dim
        else: # convnext
            self.latent_dim = encoder_config['pm_embedding_dims'][-1]

        hk_stride = encoder_config.get('hyperkernel_config', {}).get('stride', 1)
        
        # Scaling factor calculation
        if self.encoder_type == 'vit': 
            sf = (2 ** len(encoder_config.get('ma_layers_blocks', []))) * vit_config['patch_size']
        elif self.encoder_type == 'swin': 
            sf = 2 ** (len(encoder_config.get('ma_layers_blocks', [])) + len(encoder_config.get('pm_layers_blocks', [])))
        elif self.encoder_type == 'dinov2': 
            sf = (2 ** len(encoder_config.get('ma_layers_blocks', []))) * 8 
        elif self.encoder_type == 'dinov3':
            # DinoV3 requested is patch16, so scaling factor is 16
            sf = (2 ** len(encoder_config.get('ma_layers_blocks', []))) * 16
        else: 
            sf = 2 ** (len(encoder_config.get('ma_layers_blocks', [])) + len(encoder_config.get('pm_layers_blocks', [])))
        
        sf = int(sf * hk_stride)
        
        # Decoder
        self.decoder = MultiplexImageDecoder(
            input_embedding_dim=self.latent_dim, 
            decoded_embed_dim=decoder_config['decoded_embed_dim'], 
            num_blocks=decoder_config['num_blocks'], 
            scaling_factor=sf, 
            num_channels=num_channels, 
            hyperkernel_config=decoder_config.get('hyperkernel_config'), 
            decoder_type=decoder_config.get('decoder_type'), 
            vit_config=decoder_config.get('vit_config')
        )

    def forward(self, x, encoded_indices, decoded_indices, mask_ratio=0.0):
        return {'output': self.decode(self.encode(x, encoded_indices, mask_ratio)['output'], decoded_indices)['output']}
    
    def encode(self, x, encoded_indices, mask_ratio=0.0): 
        return self.encoder(x, encoded_indices, mask_ratio)
    
    def decode(self, x, decoded_indices): 
        return {'output': self.decoder(x, decoded_indices)}
