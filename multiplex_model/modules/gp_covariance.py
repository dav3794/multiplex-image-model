"""
Gaussian Process covariance module for multiplex image modeling.

This module provides the GP covariance structure that can be used as part of
the model architecture to predict spatially-correlated uncertainties.
"""

import math

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


class KroneckerPlusSpatialCovariance(nn.Module):
    """
    GP covariance using Kronecker structure + Woodbury identity.

    Models K = (K_x ⊗ K_y) + U·Uᵀ + jitter·I

    K_x and K_y are the same 1D Matérn kernel evaluated on the row and column
    pixel axes respectively — a separable approximation to the isotropic Matérn.
    The full N×N covariance is never materialised.

    Advantages over LowRankPlusSpatialCovariance / LowRankTimesSpatialCovariance:
    - No CG: log-prob is computed analytically via eigendecomposition + Woodbury
    - Eigendecomposition is done once at init (two n×n matrices, not n²×n²)
    - log_prob_all_markers batches all C channels in a single A⁻¹ call
    - Complexity: O(n³) init, O(n²·C) per image vs O(n²·C·CG_iters) for CG

    Constraint: does not support learnable lengthscale. The eigendecomposition
    is computed at init and cached for the lifetime of the module. If the
    lengthscale needs tuning, set it before instantiation.
    """

    def __init__(
        self,
        grid_size: int,
        kernel_jitter: float = 1e-2,
        spatial_matern_kernel_nu: float = 1.5,
        spatial_matern_kernel_length_scale: float = 5.0,
        device=None,
    ):
        """
        Args:
            grid_size: Number of pixels along each spatial axis (assumes square
                images: total pixels N = grid_size²). Must equal H = W of the
                images passed to log_prob / log_prob_all_markers.
            kernel_jitter: Diagonal noise for numerical stability (σ²_noise).
                Absorbed into Kronecker eigenvalues at init — no overhead at
                call time.
            spatial_matern_kernel_nu: Matérn smoothness (0.5 / 1.5 / 2.5).
            spatial_matern_kernel_length_scale: Spatial correlation range in
                normalised [0, 1] coordinates.
            device: Computation device.
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kernel_jitter = kernel_jitter
        self.grid_size = grid_size
        self.N = grid_size * grid_size

        # 1D grid in [0, 1] — same for both axes (separable kernel)
        x1d = torch.linspace(0, 1, grid_size, device=device).unsqueeze(-1)

        k1d = gpytorch.kernels.MaternKernel(nu=spatial_matern_kernel_nu).to(device)
        k1d.lengthscale = spatial_matern_kernel_length_scale
        # Fix lengthscale — Kronecker approach does not support online updates
        k1d.raw_lengthscale.requires_grad = False

        with torch.no_grad():
            K1d = k1d(x1d).evaluate()            # [n, n]
            lam, V = torch.linalg.eigh(K1d)      # K1d = V diag(lam) Vᵀ

        self.register_buffer("lam", lam)          # [n]
        self.register_buffer("V", V)              # [n, n]

        # Kronecker eigenvalues of (K_x ⊗ K_y + jitter·I): λᵢ·λⱼ + jitter
        # Jitter is absorbed here so _A_solve is a pure arithmetic operation.
        kron_eigs = torch.outer(lam, lam) + kernel_jitter   # [n, n]
        self.register_buffer("kron_eigs", kron_eigs)

    # ------------------------------------------------------------------
    # Internal solver
    # ------------------------------------------------------------------

    def _A_solve(self, v: torch.Tensor) -> torch.Tensor:
        """
        Solve A⁻¹v where A = K_x ⊗ K_y + jitter·I, analytically.

        Exploits A = (V⊗V) diag(kron_eigs) (V⊗V)ᵀ so that:
            A⁻¹v = (V⊗V) diag(1/kron_eigs) (V⊗V)ᵀ v

        Applied via four einsum contractions — no matrix materialisation.

        Args:
            v: [N] or [N, m]

        Returns:
            A⁻¹v, same shape as v.
        """
        n = self.grid_size
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(-1)
        m = v.shape[-1]

        V3  = v.reshape(n, n, m)
        # (V⊗V)ᵀ v  ≡  V.T @ V3 @ V  (eigenvectors on both axes)
        tmp = torch.einsum("abm,bj->ajm", V3, self.V)
        tmp = torch.einsum("ai,ajm->ijm", self.V, tmp)
        # Scale by 1/eigenvalue
        tmp = tmp / self.kron_eigs.unsqueeze(-1)
        # (V⊗V) tmp  ≡  V @ tmp @ V.T
        tmp = torch.einsum("ijm,bj->ibm", tmp, self.V)
        tmp = torch.einsum("ai,ibm->abm", self.V, tmp)

        result = tmp.reshape(self.N, m)
        return result.squeeze(-1) if squeeze else result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_prob(
        self,
        mu: torch.Tensor,
        U: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        log p(target | mu, K) for a single marker. No CG.

        K = (K_x ⊗ K_y) + U·Uᵀ + jitter·I

        Uses matrix determinant lemma + Woodbury identity:
            log det(A + UUᵀ) = log det(A) + log det(I + UᵀA⁻¹U)
            (A + UUᵀ)⁻¹e   = A⁻¹e − A⁻¹U (I + UᵀA⁻¹U)⁻¹ Uᵀ A⁻¹e

        Args:
            mu:     [N]    predicted mean
            U:      [N, k] low-rank uncertainty factor (typically k=1)
            target: [N]    ground truth
        """
        e = target - mu
        k = U.shape[-1]

        log_det_A = self.kron_eigs.log().sum()
        AU        = self._A_solve(U)                                     # [N, k]
        M_mat     = torch.eye(k, device=U.device) + U.T @ AU            # [k, k]
        log_det_K = log_det_A + torch.linalg.slogdet(M_mat)[1]

        Ae      = self._A_solve(e)
        K_inv_e = Ae - AU @ torch.linalg.solve(M_mat, U.T @ Ae)
        mahal   = e @ K_inv_e

        return -0.5 * (mahal + log_det_K + self.N * math.log(2 * math.pi))

    def log_prob_all_markers(
        self,
        mu_all: torch.Tensor,
        U_all: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched log prob across all C channels for one image. Assumes rank-1 U.

        A single batched A⁻¹ call (over all C right-hand sides at once)
        replaces a sequential loop over channels.

        Args:
            mu_all:  [N, C] predicted means, one column per channel
            U_all:   [N, C] per-pixel std dev per channel (low-rank factor, k=1)
            targets: [N, C] ground truth
        """
        N, C = targets.shape
        E = targets - mu_all                                     # [N, C]

        log_det_A = self.kron_eigs.log().sum()

        # Single batched solve for all channels
        AE = self._A_solve(E)                                    # [N, C]
        AU = self._A_solve(U_all)                                # [N, C]

        # Per-channel scalar: M_c = 1 + u_cᵀ A⁻¹ u_c
        M_vals          = 1.0 + (U_all * AU).sum(dim=0)         # [C]
        log_det_K_total = C * log_det_A + M_vals.log().sum()

        # Woodbury per channel: K_c⁻¹ e_c = A⁻¹e_c − (A⁻¹u_c)(uᵀA⁻¹e_c)/M_c
        correction = AU * ((U_all * AE).sum(dim=0) / M_vals)    # [N, C]
        K_inv_E    = AE - correction                             # [N, C]
        mahal      = (E * K_inv_E).sum()                         # scalar

        return -0.5 * (mahal + log_det_K_total + N * C * math.log(2 * math.pi))


class KroneckerMarkerCovariance(nn.Module):
    """
    GP covariance with triple Kronecker structure + marker covariance + Woodbury.

    Models K = (K_x ⊗ K_y) ⊗ K_C + U_block·U_blockᵀ + jitter·I

    K_C is computed from Hyperkernel marker embeddings projected to a lower
    dimension: K_C = E·Eᵀ + marker_jitter·I. Eigendecomposed every forward
    pass (O(C³), cheap for C ≤ 40).

    Spatial K_x, K_y are 1D Matérn kernels eigendecomposed once at init
    (same as KroneckerPlusSpatialCovariance).

    The full NC×NC covariance is never materialised. A⁻¹v is computed via
    three einsum contractions (spatial x, spatial y, marker).
    """

    def __init__(
        self,
        grid_size: int,
        marker_embed_dim: int,
        hyperkernel_model_dim: int,
        kernel_jitter: float = 1e-2,
        marker_jitter: float = 1e-2,
        spatial_matern_kernel_nu: float = 1.5,
        spatial_matern_kernel_length_scale: float = 5.0,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kernel_jitter = kernel_jitter
        self.marker_jitter = marker_jitter
        self.grid_size = grid_size
        self.N = grid_size * grid_size

        # --- Spatial eigendecomposition (identical to KroneckerPlusSpatialCovariance) ---
        x1d = torch.linspace(0, 1, grid_size, device=device).unsqueeze(-1)

        k1d = gpytorch.kernels.MaternKernel(nu=spatial_matern_kernel_nu).to(device)
        k1d.lengthscale = spatial_matern_kernel_length_scale
        k1d.raw_lengthscale.requires_grad = False

        with torch.no_grad():
            K1d = k1d(x1d).evaluate()
            lam, V = torch.linalg.eigh(K1d)

        self.register_buffer("lam", lam)
        self.register_buffer("V", V)

        # Spatial-only Kronecker eigenvalues (without jitter — jitter added in triple_eigs)
        kron_eigs = torch.outer(lam, lam)  # [n, n]
        self.register_buffer("kron_eigs", kron_eigs)

        # --- Marker embedding projection ---
        self.embedding_projection = nn.Linear(hyperkernel_model_dim, marker_embed_dim)

    def _A_solve_triple(
        self,
        v: torch.Tensor,
        V_C: torch.Tensor,
        triple_eigs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve A⁻¹v where A = (K_x ⊗ K_y) ⊗ K_C + jitter·I, analytically.

        A = (V_x ⊗ V_y ⊗ V_C) diag(triple_eigs) (V_x ⊗ V_y ⊗ V_C)ᵀ

        Applied via six einsum contractions (3 forward + divide + 3 reverse).

        Args:
            v: [NC] or [NC, m]
            V_C: [C, C] eigenvectors of K_C
            triple_eigs: [n, n, C] = kron_eigs[i,j] * lam_C[k] + jitter

        Returns:
            A⁻¹v, same shape as v.
        """
        n = self.grid_size
        C = V_C.shape[0]
        squeeze = v.dim() == 1
        if squeeze:
            v = v.unsqueeze(-1)
        m = v.shape[-1]

        # Reshape [NC, m] -> [n, n, C, m] (spatial_x, spatial_y, marker, rhs)
        V3 = v.reshape(n, n, C, m)

        # Forward transform: (V_x ⊗ V_y ⊗ V_C)ᵀ v
        # Contract marker axis with V_C
        tmp = torch.einsum("ijcm, ck -> ijkm", V3, V_C)
        # Contract spatial_y axis with V
        tmp = torch.einsum("ijkm, jb -> ibkm", tmp, self.V)
        # Contract spatial_x axis with V
        tmp = torch.einsum("ibkm, ia -> abkm", tmp, self.V)

        # Divide by eigenvalues
        tmp = tmp / triple_eigs.unsqueeze(-1)

        # Reverse transform: (V_x ⊗ V_y ⊗ V_C) tmp
        tmp = torch.einsum("abkm, jb -> ajkm", tmp, self.V)
        tmp = torch.einsum("ajkm, ia -> ijkm", tmp, self.V)
        tmp = torch.einsum("ijkm, ck -> ijcm", tmp, V_C)

        result = tmp.reshape(n * n * C, m)
        return result.squeeze(-1) if squeeze else result
