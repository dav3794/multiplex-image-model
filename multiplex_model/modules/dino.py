"""DINO projection head for self-distillation training.

Implements the DINO/DINOv2 projection head that maps a pooled feature vector to a
distribution over ``out_dim`` prototypes. iBOT (patch-level) heads are intentionally
omitted; only the global (CLS-equivalent) head is provided here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear(m: nn.Module) -> None:
    """Truncated-normal init for Linear layers (bias zeroed)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def build_mlp(
    in_dim: int, out_dim: int, hidden_dim: int, n_layers: int
) -> nn.Module:
    """Build a GELU MLP; a single Linear when ``n_layers == 1``."""
    n_layers = max(n_layers, 1)
    if n_layers == 1:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """MLP projection head producing prototype logits for DINO self-distillation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        n_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        """Initialize the DINO head.

        Args:
            in_dim (int): Dimension of the input (pooled) feature vector.
            out_dim (int, optional): Number of prototypes (output logits). Defaults to 65536.
            hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to 2048.
            bottleneck_dim (int, optional): Dimension of the L2-normalized bottleneck. Defaults to 256.
            n_layers (int, optional): Number of MLP layers (>=1). Defaults to 3.
            use_bn (bool, optional): Whether to use BatchNorm between MLP layers. Defaults to False.
            norm_last_layer (bool, optional): If True, freeze the weight-norm magnitude of the
                last layer (recommended for stability early in training). Defaults to True.
        """
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(_init_linear)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a pooled feature vector to prototype logits.

        Args:
            x (torch.Tensor): Pooled features of shape [B, in_dim].

        Returns:
            torch.Tensor: Prototype logits of shape [B, out_dim].
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOvMFHead(nn.Module):
    """Continuous MLP projection head for the vMF iBOT objective.

    Produces an unnormalized ``out_dim``-dimensional continuous projection (no discrete
    prototype layer). Shared architecture between student and EMA teacher.
    L2-normalization and centering are applied later in the loss. Operates on the last
    dimension, so it supports both pooled inputs ``[B, in_dim]`` and dense per-cell
    inputs ``[B, N, in_dim]``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden_dim: int = 2048,
        n_layers: int = 3,
    ):
        """Initialize the vMF projection head.

        Args:
            in_dim (int): Dimension of the input feature vector.
            out_dim (int, optional): Dimension of the continuous projection (the vMF
                hypersphere dimension D). Defaults to 256.
            hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to 2048.
            n_layers (int, optional): Number of MLP layers (>=1). Defaults to 3.
        """
        super().__init__()
        self.mlp = build_mlp(in_dim, out_dim, hidden_dim, n_layers)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to a continuous vMF-space vector (applied on the last dim)."""
        return self.mlp(x)


class VMFPredictor(nn.Module):
    """Student-only predictor producing vMF parameters (mean direction + concentration).

    Maps a projection of dimension ``dim`` to ``dim + 1`` outputs: the first ``dim`` are
    the unnormalized mean direction and the last is the raw concentration scalar (mapped
    to a positive concentration via softplus inside the loss). This head is BYOL/SimSiam
    style and is applied only on the student; the EMA teacher does not use it. Operates on
    the last dimension, supporting pooled ``[B, dim]`` and dense ``[B, N, dim]`` inputs.
    """

    def __init__(self, dim: int, hidden_dim: int = 2048, n_layers: int = 2):
        """Initialize the vMF predictor.

        Args:
            dim (int): Dimension of the vMF projection space (D). The output is dim + 1.
            hidden_dim (int, optional): Hidden dimension of the MLP. Defaults to 2048.
            n_layers (int, optional): Number of MLP layers (>=1). Defaults to 2.
        """
        super().__init__()
        self.dim = dim
        self.mlp = build_mlp(dim, dim + 1, hidden_dim, n_layers)
        self.apply(_init_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict vMF parameters; returns ``[..., dim + 1]`` (mean ++ raw concentration)."""
        return self.mlp(x)
