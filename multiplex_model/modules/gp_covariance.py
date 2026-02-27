"""
Gaussian Process covariance module for multiplex image modeling.

This module provides the GP covariance structure that can be used as part of
the model architecture to predict spatially-correlated uncertainties.
"""

import torch
import torch.nn as nn
import gpytorch
from linear_operator.operators import LowRankRootLinearOperator, DiagLinearOperator


class LowRankPlusSpatialCovariance(nn.Module):
    """
    Covariance structure combining spatial GP kernel with low-rank uncertainties.
    
    This module computes:
    K = K_spatial + sigma @ sigma^T + jitter * I
    
    where:
    - K_spatial: Matérn kernel over spatial coordinates (captures spatial correlation)
    - sigma @ sigma^T: Low-rank per-pixel uncertainty structure
    - jitter * I: Numerical stability term (diagonal noise)
    
    This is designed to be used as part of the model's prediction head to model
    spatially-correlated uncertainties in multiplex image reconstruction.
    """
    
    def __init__(
        self,
        kernel_jitter: float = 1e-2,
        spatial_matern_kernel_nu: float = 1.5,
        spatial_matern_kernel_length_scale: float = 5.0,
        learn_lengthscale: bool = False,
        device=None,
    ):
        """
        Initialize the GP covariance module.
        
        Args:
            kernel_jitter: Diagonal noise for numerical stability (σ²_noise)
            spatial_matern_kernel_nu: Smoothness parameter for Matérn kernel
                - 0.5: exponential kernel (very rough)
                - 1.5: once differentiable (moderately smooth)
                - 2.5: twice differentiable (smooth)
            spatial_matern_kernel_length_scale: Initial spatial correlation scale
                in normalized units (image coordinates are normalized to [0, 1])
            learn_lengthscale: Whether to make lengthscale a learnable parameter
            device: Computation device (cuda/cpu)
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.kernel_jitter = kernel_jitter
        
        # Initialize spatial kernel
        self.spatial_kernel = gpytorch.kernels.MaternKernel(nu=spatial_matern_kernel_nu)
        self.spatial_kernel.lengthscale = spatial_matern_kernel_length_scale
        
        if not learn_lengthscale:
            # Fix the lengthscale (not learned during training)
            self.spatial_kernel.raw_lengthscale.requires_grad = False
        
        self.spatial_kernel = self.spatial_kernel.to(device)
    
    def forward(
        self,
        sigma: torch.Tensor,
        grid_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the full covariance matrix.
        
        Args:
            sigma: Per-pixel uncertainty vector [..., N]
                This can be [N] or [Batch, N]
            grid_coords: Spatial coordinates [N, 2] in normalized space [0, 1]
        
        Returns:
            Covariance matrix as a GPyTorch LazyTensor [..., N, N]
        """
        # Ensure sigma has correct feature dimension [..., N, 1]
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        elif sigma.dim() == 2:
            sigma = sigma.unsqueeze(-1) # [B, N, 1]
        
        # Compute spatial kernel covariance
        # This will be [N, N]
        k_spatial = self.spatial_kernel(grid_coords)
        
        # Low-rank structure from predicted uncertainties
        # This represents: sigma @ sigma^T
        # Result will match batch shape of sigma: [..., N, N]
        K_low_rank = LowRankRootLinearOperator(sigma)
        
        # Jitter term for numerical stability
        N = grid_coords.size(0)
        jitter = DiagLinearOperator(
            torch.ones(N, device=self.device) * self.kernel_jitter
        )
        
        # Combine all components
        # k_spatial and jitter will broadcast to match K_low_rank batch dimensions
        return k_spatial + K_low_rank + jitter
    
    def get_lengthscale(self) -> float:
        """Get current lengthscale value."""
        return self.spatial_kernel.lengthscale.item()
    
    def set_lengthscale(self, value: float):
        """Set lengthscale value."""
        self.spatial_kernel.lengthscale = value


class LowRankTimesSpatialCovariance(nn.Module):
    """
    Covariance structure combining spatial GP kernel with low-rank uncertainties via multiplication.
    
    This module computes:
    K = K_spatial * (sigma @ sigma^T) + jitter * I
    
    where:
    - K_spatial: Matérn kernel over spatial coordinates
    - sigma @ sigma^T: Low-rank per-pixel uncertainty structure
    - jitter * I: Numerical stability term
    """
    
    def __init__(
        self,
        grid_coords: torch.Tensor,
        kernel_jitter: float = 1e-2,
        spatial_matern_kernel_nu: float = 1.5,
        spatial_matern_kernel_length_scale: float = 0.05,
        learn_lengthscale: bool = True,
        device=None,
    ):
        """
        Initialize the GP covariance module.
        
        Args:
            grid_coords: Fixed spatial coordinates [N, 2]
            kernel_jitter: Diagonal noise for numerical stability
            spatial_matern_kernel_nu: Smoothness parameter for Matérn kernel
            spatial_matern_kernel_length_scale: Initial spatial correlation scale
            learn_lengthscale: Whether to learn the lengthscale
            device: Computation device
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.device = device
        self.kernel_jitter = kernel_jitter
        
        # Initialize spatial kernel
        self.spatial_kernel = gpytorch.kernels.MaternKernel(nu=spatial_matern_kernel_nu)
        self.spatial_kernel.lengthscale = spatial_matern_kernel_length_scale
        
        if not learn_lengthscale:
            self.spatial_kernel.raw_lengthscale.requires_grad = False
            
        self.spatial_kernel = self.spatial_kernel.to(device)
        
        # Store grid_coords and jitter as buffers (move with module, no grad)
        self.register_buffer("grid_coords", grid_coords)
        N = grid_coords.size(0)
        self.register_buffer("jitter_val", torch.ones(N, device=device) * kernel_jitter)

    def forward(
        self, 
        sigma: torch.Tensor,
        grid_coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the full covariance matrix.
        
        Args:
            sigma: Per-pixel uncertainty vector [..., N]
            grid_coords: Spatial coordinates [N, 2]. Ignored if precomputed,
                but included for compatibility with other covariance modules.
            
        Returns:
            Covariance matrix as a GPyTorch LinearOperator
        """
        # Ensure sigma has correct feature dimension
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        elif sigma.dim() == 2:
            sigma = sigma.unsqueeze(-1)

        # Compute spatial kernel fresh each forward pass so the computation
        # graph is recreated (required when lengthscale is learnable).
        coords = self.grid_coords.to(dtype=sigma.dtype)
        k_spatial = self.spatial_kernel(coords)

        # Low-rank structure: sigma @ sigma^T
        K_low_rank = LowRankRootLinearOperator(sigma)

        # Jitter for numerical stability
        jitter = DiagLinearOperator(self.jitter_val.to(dtype=sigma.dtype))

        # Element-wise multiplication of spatial kernel and low-rank structure
        return k_spatial * K_low_rank + jitter
