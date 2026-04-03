"""Tests for KroneckerMarkerCovariance numerical correctness."""

import math
import torch
import pytest


def _build_module(grid_size=4, marker_embed_dim=3, hyperkernel_model_dim=8, device="cpu"):
    """Helper to build a KroneckerMarkerCovariance with small dims for testing."""
    from multiplex_model.modules.gp_covariance import KroneckerMarkerCovariance

    return KroneckerMarkerCovariance(
        grid_size=grid_size,
        marker_embed_dim=marker_embed_dim,
        hyperkernel_model_dim=hyperkernel_model_dim,
        kernel_jitter=1e-2,
        marker_jitter=1e-2,
        spatial_matern_kernel_nu=1.5,
        spatial_matern_kernel_length_scale=5.0,
        device=device,
    )


def test_A_solve_triple_recovers_identity():
    """A^{-1} A v == v  for random v, using dense materialization as ground truth."""
    torch.manual_seed(42)
    n = 4
    C = 3
    N = n * n
    NC = N * C

    mod = _build_module(grid_size=n, marker_embed_dim=3, hyperkernel_model_dim=8)

    # Build K_C from random marker embeddings
    marker_emb = torch.randn(C, 8)
    E = mod.embedding_projection(marker_emb)  # [C, 3]
    K_C = E @ E.T + mod.marker_jitter * torch.eye(C)
    lam_C, V_C = torch.linalg.eigh(K_C)

    # Triple eigenvalues
    triple_eigs = (
        mod.kron_eigs.unsqueeze(-1) * lam_C.unsqueeze(0).unsqueeze(0)
        + mod.kernel_jitter
    )

    # Build dense A for ground truth
    # A = (K_x kron K_y) kron K_C + jitter * I
    V = mod.V
    lam = mod.lam
    K1d = V @ torch.diag(lam) @ V.T
    K_spatial = torch.kron(K1d, K1d)  # [N, N]
    A_dense = torch.kron(K_spatial, K_C) + mod.kernel_jitter * torch.eye(NC)

    # Random vector
    v = torch.randn(NC)
    Av = A_dense @ v

    # Solve A^{-1} (A v) should recover v
    recovered = mod._A_solve_triple(Av, V_C, triple_eigs)

    torch.testing.assert_close(recovered, v, atol=1e-4, rtol=1e-4)


def test_A_solve_triple_batched():
    """_A_solve_triple with multiple right-hand sides [NC, m]."""
    torch.manual_seed(42)
    n = 4
    C = 3
    N = n * n
    NC = N * C
    m = 5

    mod = _build_module(grid_size=n, marker_embed_dim=3, hyperkernel_model_dim=8)

    marker_emb = torch.randn(C, 8)
    E = mod.embedding_projection(marker_emb)
    K_C = E @ E.T + mod.marker_jitter * torch.eye(C)
    lam_C, V_C = torch.linalg.eigh(K_C)
    triple_eigs = (
        mod.kron_eigs.unsqueeze(-1) * lam_C.unsqueeze(0).unsqueeze(0)
        + mod.kernel_jitter
    )

    V = mod.V
    lam = mod.lam
    K1d = V @ torch.diag(lam) @ V.T
    K_spatial = torch.kron(K1d, K1d)
    A_dense = torch.kron(K_spatial, K_C) + mod.kernel_jitter * torch.eye(NC)

    v = torch.randn(NC, m)
    Av = A_dense @ v
    recovered = mod._A_solve_triple(Av, V_C, triple_eigs)

    torch.testing.assert_close(recovered, v, atol=1e-4, rtol=1e-4)
