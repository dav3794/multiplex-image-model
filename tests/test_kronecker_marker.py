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


def test_log_prob_joint_matches_dense():
    """log_prob_joint should match direct dense multivariate normal log-prob."""
    torch.manual_seed(42)
    n = 4
    C = 3
    N = n * n
    NC = N * C

    mod = _build_module(grid_size=n, marker_embed_dim=3, hyperkernel_model_dim=8)

    marker_emb = torch.randn(C, 8)

    mu_all = torch.randn(N, C)
    U_all = torch.abs(torch.randn(N, C)) * 0.1 + 0.01  # positive sigma
    targets = torch.randn(N, C)

    # Our method
    log_prob = mod.log_prob_joint(mu_all, U_all, targets, marker_emb)

    # Dense ground truth
    E = mod.embedding_projection(marker_emb)
    K_C = E @ E.T + mod.marker_jitter * torch.eye(C)

    V = mod.V
    lam = mod.lam
    K1d = V @ torch.diag(lam) @ V.T
    K_spatial = torch.kron(K1d, K1d)
    A_dense = torch.kron(K_spatial, K_C) + mod.kernel_jitter * torch.eye(NC)

    # Build U_block [NC, C] in spatial-major order: row (i*C + c) = pixel i, marker c
    U_block = torch.diag_embed(U_all).reshape(NC, C)  # [N,C] -> [N,C,C] -> [NC,C]

    K_dense = A_dense + U_block @ U_block.T

    # Dense log prob: -0.5 * (e^T K^{-1} e + log|K| + NC*log(2pi))
    e = (targets - mu_all).reshape(-1)  # [NC] spatial-major: [pix0_ch0, pix0_ch1, ..., pixN_chC]
    K_inv_e = torch.linalg.solve(K_dense, e)
    mahal = e @ K_inv_e
    log_det = torch.linalg.slogdet(K_dense)[1]
    expected = -0.5 * (mahal + log_det + NC * math.log(2 * math.pi))

    torch.testing.assert_close(log_prob, expected, atol=1e-3, rtol=1e-3)


def test_compute_marker_correlation_shape_and_diagonal():
    """compute_marker_correlation returns CxC with ones on diagonal."""
    mod = _build_module(grid_size=4, marker_embed_dim=3, hyperkernel_model_dim=8)
    marker_emb = torch.randn(5, 8)

    corr = mod.compute_marker_correlation(marker_emb)
    assert corr.shape == (5, 5)
    torch.testing.assert_close(torch.diag(corr), torch.ones(5), atol=1e-5, rtol=1e-5)
