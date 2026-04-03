# Kronecker Marker Covariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Kronecker GP framework with a triple-Kronecker `(K_x ⊗ K_y) ⊗ K_C` covariance that captures inter-marker uncertainty from Hyperkernel embeddings, plus Woodbury identity for the per-pixel low-rank update.

**Architecture:** New `KroneckerMarkerCovariance` module computes joint log-probability over N×C dimensions (pixels × markers) using triple-Kronecker eigendecomposition for the base matrix and a rank-C Woodbury correction for per-pixel sigma. A projection layer (`nn.Linear`) maps raw Hyperkernel embeddings to a lower-dimensional space before computing `K_C = E·Eᵀ + ε·I`. New loss classes wrap this module and are wired into the existing training script via config flags.

**Tech Stack:** PyTorch, GPyTorch (Matérn kernel at init only), Pydantic v2 (config validation)

**Spec:** `docs/superpowers/specs/2026-04-03-kronecker-marker-covariance-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `multiplex_model/modules/gp_covariance.py` | Add class | `KroneckerMarkerCovariance` — triple Kronecker eigensolver + Woodbury log-prob |
| `multiplex_model/losses.py` | Add classes | `KroneckerMarkerGPNLLLoss`, `HybridKroneckerMarkerGPNLLLoss` — loss wrappers |
| `multiplex_model/utils/configuration.py` | Modify | Add `use_marker_covariance`, `marker_embed_dim`, `marker_jitter` fields to `TrainingConfig` |
| `train_masked_model_gp.py` | Modify | Wire up marker covariance: extract embeddings, instantiate module, pass to loss, checkpoint, logging |
| `tests/test_kronecker_marker.py` | Create | Numerical correctness tests for triple Kronecker solver and log-prob |

---

### Task 1: Add config fields to TrainingConfig

**Files:**
- Modify: `multiplex_model/utils/configuration.py:220-244` (GP Loss parameters section)

- [ ] **Step 1: Add three new fields to TrainingConfig**

In `multiplex_model/utils/configuration.py`, add after the `use_kronecker_gp` field (line 244):

```python
    use_marker_covariance: bool = Field(
        False, description="Whether to use marker covariance in Kronecker GP loss (requires use_kronecker_gp=True)"
    )
    marker_embed_dim: int = Field(
        32, gt=0, description="Projection dimension for marker embeddings in K_C computation"
    )
    marker_jitter: float = Field(
        1e-2, ge=0, description="Jitter added to marker covariance K_C for numerical stability"
    )
```

- [ ] **Step 2: Verify config loads with new fields**

Run from the project root:

```bash
python -c "
from multiplex_model.utils import TrainingConfig
# Minimal config to validate new fields parse
c = TrainingConfig(
    device='cpu', input_image_size=(64,64), batch_size=2, num_workers=0,
    panel_config='x', tokenizer_config='x', lr=1e-3, final_lr=1e-5,
    frac_warmup_steps=0.01, weight_decay=0.0, gradient_accumulation_steps=1,
    epochs=1, beta=0.5, min_channels_frac=0.75, spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5, mask_patch_size=8, save_checkpoint_freq=1,
    comet_project='test',
    encoder={'ma_layers_blocks':[4], 'ma_embedding_dims':[16],
             'pm_layers_blocks':[4], 'pm_embedding_dims':[128],
             'hyperkernel':{'kernel_size':1,'padding':0,'stride':1,'use_bias':True}},
    decoder={'decoded_embed_dim':64, 'num_blocks':1,
             'hyperkernel':{'kernel_size':1,'padding':0,'stride':1,'use_bias':True}},
    use_marker_covariance=True, marker_embed_dim=32, marker_jitter=1e-2,
)
assert c.use_marker_covariance == True
assert c.marker_embed_dim == 32
assert c.marker_jitter == 1e-2
print('Config validation OK')
"
```

Expected: `Config validation OK`

- [ ] **Step 3: Commit**

```bash
git add multiplex_model/utils/configuration.py
git commit -m "feat: add marker covariance config fields to TrainingConfig"
```

---

### Task 2: Implement KroneckerMarkerCovariance — `_A_solve_triple`

**Files:**
- Modify: `multiplex_model/modules/gp_covariance.py` (append new class)

This task implements the core triple-Kronecker solver. The next task adds `log_prob_joint` on top.

- [ ] **Step 1: Write test for `_A_solve_triple` correctness**

Create `tests/test_kronecker_marker.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails (module doesn't exist yet)**

```bash
python -m pytest tests/test_kronecker_marker.py::test_A_solve_triple_recovers_identity -v
```

Expected: `FAILED` — `ImportError: cannot import name 'KroneckerMarkerCovariance'`

- [ ] **Step 3: Implement KroneckerMarkerCovariance with `_A_solve_triple`**

Append to `multiplex_model/modules/gp_covariance.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_kronecker_marker.py -v
```

Expected: Both `test_A_solve_triple_recovers_identity` and `test_A_solve_triple_batched` PASS.

- [ ] **Step 5: Commit**

```bash
git add multiplex_model/modules/gp_covariance.py tests/test_kronecker_marker.py
git commit -m "feat: add KroneckerMarkerCovariance with triple Kronecker solver"
```

---

### Task 3: Implement `log_prob_joint` and `compute_marker_correlation`

**Files:**
- Modify: `multiplex_model/modules/gp_covariance.py` (add methods to `KroneckerMarkerCovariance`)
- Modify: `tests/test_kronecker_marker.py` (add log_prob test)

- [ ] **Step 1: Write test for `log_prob_joint` against dense computation**

Append to `tests/test_kronecker_marker.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kronecker_marker.py::test_log_prob_joint_matches_dense -v
```

Expected: `FAILED` — `AttributeError: 'KroneckerMarkerCovariance' object has no attribute 'log_prob_joint'`

- [ ] **Step 3: Implement `log_prob_joint` and `compute_marker_correlation`**

Add these methods to `KroneckerMarkerCovariance` class in `gp_covariance.py`:

```python
    def _compute_marker_eigen(
        self, marker_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project embeddings, build K_C, eigendecompose, compute triple eigenvalues.

        Args:
            marker_embeddings: [C, hyperkernel_model_dim]

        Returns:
            (V_C, triple_eigs, K_C):
                V_C: [C, C] eigenvectors
                triple_eigs: [n, n, C] eigenvalues of A
                K_C: [C, C] marker covariance
        """
        E = self.embedding_projection(marker_embeddings)  # [C, D]
        C = E.shape[0]
        K_C = E @ E.T + self.marker_jitter * torch.eye(C, device=E.device, dtype=E.dtype)
        lam_C, V_C = torch.linalg.eigh(K_C)

        # triple_eigs[i, j, k] = kron_eigs[i,j] * lam_C[k] + kernel_jitter
        triple_eigs = self.kron_eigs.unsqueeze(-1) * lam_C.unsqueeze(0).unsqueeze(0) + self.kernel_jitter

        return V_C, triple_eigs, K_C

    def log_prob_joint(
        self,
        mu_all: torch.Tensor,
        U_all: torch.Tensor,
        targets: torch.Tensor,
        marker_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Joint log p(targets | mu, K) over all N pixels and C markers.

        K = (K_x ⊗ K_y) ⊗ K_C + U_block·U_blockᵀ + jitter·I

        Uses Woodbury identity with rank-C U_block.

        Args:
            mu_all:  [N, C] predicted means
            U_all:   [N, C] per-pixel std dev per channel
            targets: [N, C] ground truth
            marker_embeddings: [C, hyperkernel_model_dim] raw Hyperkernel embeddings

        Returns:
            Scalar log probability.
        """
        N, C = targets.shape
        NC = N * C

        V_C, triple_eigs, _ = self._compute_marker_eigen(marker_embeddings)

        # Error vector in spatial-major order: [pix0_ch0, pix0_ch1, ..., pixN_chC]
        e = (targets - mu_all).reshape(-1)  # [NC]

        # Build U_block [NC, C] in spatial-major order: row (i*C + c) = pixel i, marker c
        U_block = torch.diag_embed(U_all).reshape(NC, C)

        # log det(A)
        log_det_A = triple_eigs.log().sum()

        # A⁻¹ applied to error and U_block columns (C+1 RHS, batched)
        rhs = torch.cat([e.unsqueeze(-1), U_block], dim=-1)  # [NC, C+1]
        A_inv_rhs = self._A_solve_triple(rhs, V_C, triple_eigs)  # [NC, C+1]
        A_inv_e = A_inv_rhs[:, 0]       # [NC]
        A_inv_U = A_inv_rhs[:, 1:]      # [NC, C]

        # Woodbury inner matrix: M = I_C + U_blockᵀ A⁻¹ U_block  [C, C]
        M = torch.eye(C, device=e.device, dtype=e.dtype) + U_block.T @ A_inv_U

        # log det(K) = log det(A) + log det(M)
        log_det_K = log_det_A + torch.linalg.slogdet(M)[1]

        # K⁻¹ e = A⁻¹e - A⁻¹U M⁻¹ Uᵀ A⁻¹e
        Ut_Ainv_e = U_block.T @ A_inv_e  # [C]
        correction = A_inv_U @ torch.linalg.solve(M, Ut_Ainv_e)  # [NC]
        K_inv_e = A_inv_e - correction

        mahal = e @ K_inv_e

        return -0.5 * (mahal + log_det_K + NC * math.log(2 * math.pi))

    def compute_marker_correlation(self, marker_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute C×C correlation matrix from projected marker embeddings.

        Args:
            marker_embeddings: [C, hyperkernel_model_dim]

        Returns:
            [C, C] correlation matrix (ones on diagonal).
        """
        E = self.embedding_projection(marker_embeddings)
        K_C = E @ E.T + self.marker_jitter * torch.eye(E.shape[0], device=E.device, dtype=E.dtype)
        # Normalize to correlation: corr[i,j] = K_C[i,j] / sqrt(K_C[i,i] * K_C[j,j])
        diag_sqrt = torch.sqrt(torch.diag(K_C))
        return K_C / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_kronecker_marker.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add multiplex_model/modules/gp_covariance.py tests/test_kronecker_marker.py
git commit -m "feat: add log_prob_joint and compute_marker_correlation to KroneckerMarkerCovariance"
```

---

### Task 4: Implement loss classes

**Files:**
- Modify: `multiplex_model/losses.py` (append two new classes)
- Modify: `tests/test_kronecker_marker.py` (add loss wrapper test)

- [ ] **Step 1: Write test for HybridKroneckerMarkerGPNLLLoss**

Append to `tests/test_kronecker_marker.py`:

```python
def test_hybrid_marker_loss_forward_shape_and_components():
    """HybridKroneckerMarkerGPNLLLoss returns scalar loss and dict with expected keys."""
    from multiplex_model.losses import HybridKroneckerMarkerGPNLLLoss

    torch.manual_seed(42)
    n = 4
    B, C = 2, 3
    H = W = n

    mod = _build_module(grid_size=n, marker_embed_dim=3, hyperkernel_model_dim=8)
    loss_fn = HybridKroneckerMarkerGPNLLLoss(
        covariance_module=mod,
        lambda_gp=0.1,
        downscale_factor=1,
        device="cpu",
    )

    target = torch.rand(B, C, H, W)
    mu = torch.rand(B, C, H, W)
    logvar = torch.randn(B, C, H, W) * 0.1
    marker_embeddings = torch.randn(B, C, 8)  # [B, C, model_dim]

    total_loss, loss_dict = loss_fn(target, mu, logvar, marker_embeddings)

    assert total_loss.dim() == 0, "Loss should be scalar"
    assert total_loss.requires_grad, "Loss must be differentiable"
    assert "standard_nll" in loss_dict
    assert "gp_nll" in loss_dict
    assert "total_loss" in loss_dict

    # Verify gradient flows through marker_embeddings
    marker_embeddings_grad = torch.randn(B, C, 8, requires_grad=True)
    total_loss2, _ = loss_fn(target, mu, logvar, marker_embeddings_grad)
    total_loss2.backward()
    assert marker_embeddings_grad.grad is not None, "Gradients must flow to marker embeddings"


def test_hybrid_marker_loss_lambda_zero_equals_standard():
    """With lambda_gp=0, HybridKroneckerMarkerGPNLLLoss should equal standard NLL."""
    from multiplex_model.losses import HybridKroneckerMarkerGPNLLLoss

    torch.manual_seed(42)
    n = 4
    B, C = 1, 3
    H = W = n

    mod = _build_module(grid_size=n, marker_embed_dim=3, hyperkernel_model_dim=8)
    loss_fn = HybridKroneckerMarkerGPNLLLoss(
        covariance_module=mod,
        lambda_gp=0.0,
        downscale_factor=1,
        device="cpu",
    )

    target = torch.rand(B, C, H, W)
    mu = torch.rand(B, C, H, W)
    logvar = torch.randn(B, C, H, W) * 0.1
    marker_embeddings = torch.randn(B, C, 8)

    total_loss, loss_dict = loss_fn(target, mu, logvar, marker_embeddings)

    # Standard NLL computed directly
    var = torch.exp(logvar)
    expected_nll = torch.mean((target - mu) ** 2 / (var + 1e-8) + logvar)

    torch.testing.assert_close(total_loss, expected_nll, atol=1e-5, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_kronecker_marker.py::test_hybrid_marker_loss_forward_shape_and_components -v
```

Expected: `FAILED` — `ImportError: cannot import name 'HybridKroneckerMarkerGPNLLLoss'`

- [ ] **Step 3: Implement both loss classes**

Append to `multiplex_model/losses.py`:

```python
class KroneckerMarkerGPNLLLoss(nn.Module):
    """
    GP-based NLL loss with joint spatial + marker covariance.

    Uses KroneckerMarkerCovariance for triple Kronecker (K_x ⊗ K_y) ⊗ K_C
    plus Woodbury for per-pixel sigma. Processes one image at a time,
    computing joint log-prob over all N*C dimensions.

    Requires square images (H == W == grid_size after downscaling).
    """

    def __init__(
        self,
        covariance_module,
        downscale_factor: int = 1,
        device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.covariance_module = covariance_module
        self.downscale_factor = downscale_factor

    def _downscale(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.downscale_factor == 1:
            return tensor
        return torch.nn.functional.avg_pool2d(
            tensor,
            kernel_size=self.downscale_factor,
            stride=self.downscale_factor,
        )

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        marker_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target: [B, C, H, W] ground truth
            mu:     [B, C, H, W] predicted means
            sigma:  [B, C, H, W] per-pixel std dev (not log)
            marker_embeddings: [B, C, model_dim] Hyperkernel embeddings

        Returns:
            Scalar mean NLL per pixel per channel.
        """
        target = target.float()
        mu = mu.float()
        sigma = sigma.float()
        marker_embeddings = marker_embeddings.float()

        if self.downscale_factor > 1:
            target = self._downscale(target)
            mu = self._downscale(mu)
            sigma = self._downscale(sigma)

        B, C, H, W = target.shape
        N = H * W

        assert H == W == self.covariance_module.grid_size, (
            f"Image must be square with H == W == grid_size, "
            f"got {H}x{W} vs grid_size={self.covariance_module.grid_size}."
        )

        target_bnc = target.reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]
        mu_bnc = mu.reshape(B, C, N).permute(0, 2, 1)
        sigma_bnc = sigma.reshape(B, C, N).permute(0, 2, 1)

        total_log_prob = torch.zeros(1, device=self.device, dtype=torch.float32)
        for b in range(B):
            total_log_prob = total_log_prob + self.covariance_module.log_prob_joint(
                mu_bnc[b],
                sigma_bnc[b],
                target_bnc[b],
                marker_embeddings[b],
            )

        return -total_log_prob / (B * N * C)


class HybridKroneckerMarkerGPNLLLoss(nn.Module):
    """
    Hybrid loss: standard pixel-wise NLL + Kronecker marker GP NLL.

    L = (1 - lambda_gp) * L_standard + lambda_gp * L_kronecker_marker_gp

    Drop-in replacement for HybridKroneckerGPNLLLoss with additional
    marker_embeddings argument in forward().
    """

    def __init__(
        self,
        covariance_module,
        lambda_gp: float = 0.1,
        downscale_factor: int = 1,
        device=None,
    ):
        super().__init__()
        self.lambda_gp = lambda_gp
        self.gp_loss = KroneckerMarkerGPNLLLoss(
            covariance_module=covariance_module,
            downscale_factor=downscale_factor,
            device=device,
        )

    def forward(
        self,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        marker_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            target: [B, C, H, W] ground truth
            mu:     [B, C, H, W] predicted means
            logvar: [B, C, H, W] predicted log-variances
            marker_embeddings: [B, C, model_dim] Hyperkernel embeddings

        Returns:
            total_loss: Combined scalar loss.
            loss_dict:  {"standard_nll", "gp_nll", "total_loss"}.
        """
        var = torch.exp(logvar)
        standard_nll = torch.mean((target - mu) ** 2 / (var + 1e-8) + logvar)

        sigma = torch.sqrt(var)
        gp_nll = self.gp_loss(target, mu, sigma, marker_embeddings)

        total_loss = (1 - self.lambda_gp) * standard_nll + self.lambda_gp * gp_nll

        loss_dict = {
            "standard_nll": standard_nll.item(),
            "gp_nll": gp_nll.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, loss_dict
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_kronecker_marker.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add multiplex_model/losses.py tests/test_kronecker_marker.py
git commit -m "feat: add KroneckerMarkerGPNLLLoss and HybridKroneckerMarkerGPNLLLoss"
```

---

### Task 5: Wire up training script — module instantiation and optimizer

**Files:**
- Modify: `train_masked_model_gp.py:37-41` (imports)
- Modify: `train_masked_model_gp.py:541-594` (GP module init)
- Modify: `train_masked_model_gp.py:618-627` (optimizer params)

- [ ] **Step 1: Add import for new classes**

In `train_masked_model_gp.py`, update the import blocks:

Add `HybridKroneckerMarkerGPNLLLoss` to the losses import (line 30-36):

```python
from multiplex_model.losses import (
    HybridGPNLLLoss,
    HybridKroneckerGPNLLLoss,
    HybridKroneckerMarkerGPNLLLoss,
    RankMe,
    beta_nll_loss,
    nll_loss,
)
```

Add `KroneckerMarkerCovariance` to the gp_covariance import (line 37-40):

```python
from multiplex_model.modules.gp_covariance import (
    KroneckerMarkerCovariance,
    KroneckerPlusSpatialCovariance,
    LowRankTimesSpatialCovariance,
)
```

- [ ] **Step 2: Read new config values in `__main__` block**

After the existing GP config reads (around line 549), add:

```python
    use_marker_covariance = getattr(config, "use_marker_covariance", False)
    marker_embed_dim      = getattr(config, "marker_embed_dim", 32)
    marker_jitter         = getattr(config, "marker_jitter", 1e-2)
```

Update the print block (around line 551-559) to include:

```python
    print(f"  Use Marker Cov:      {use_marker_covariance}")
    print(f"  Marker Embed Dim:    {marker_embed_dim}")
    print(f"  Marker Jitter:       {marker_jitter}")
```

- [ ] **Step 3: Add KroneckerMarkerCovariance instantiation branch**

In the GP module init section (around line 568), add a new branch before the existing `use_kronecker_gp` check. The modified block becomes:

```python
        if use_kronecker_gp and use_marker_covariance:
            assert H_gp == W_gp, (
                f"Kronecker GP requires square spatial grid, "
                f"got {H_gp}x{W_gp}. Adjust input_image_size or gp_downscale_factor."
            )
            # Compute hyperkernel_model_dim from encoder config
            hk_cfg = config.encoder_config
            if len(hk_cfg.ma_layers_blocks) == 0:
                hk_input_dim = 1
            else:
                hk_input_dim = hk_cfg.ma_embedding_dims[-1]
            hk_embed_dim = hk_cfg.pm_embedding_dims[0]
            hk_kernel_size = hk_cfg.hyperkernel_config.kernel_size
            hyperkernel_model_dim = hk_embed_dim * (hk_kernel_size ** 2) * hk_input_dim

            gp_covariance_module = KroneckerMarkerCovariance(
                grid_size=H_gp,
                marker_embed_dim=marker_embed_dim,
                hyperkernel_model_dim=hyperkernel_model_dim,
                kernel_jitter=gp_kernel_jitter,
                marker_jitter=marker_jitter,
                spatial_matern_kernel_length_scale=gp_lengthscale,
                device=device,
            )
        elif use_kronecker_gp:
            # ... existing KroneckerPlusSpatialCovariance code unchanged ...
```

- [ ] **Step 4: Include marker covariance params in optimizer**

Modify the optimizer params section (around line 618-627). Replace:

```python
    # Include GP covariance parameters in optimization if learnable
    params_to_optimize = list(model.parameters())
    if use_gp_loss and gp_learn_lengthscale and gp_covariance_module is not None:
        params_to_optimize += list(gp_covariance_module.parameters())
```

With:

```python
    # Include GP covariance parameters in optimization if learnable
    params_to_optimize = list(model.parameters())
    if use_gp_loss and gp_covariance_module is not None:
        if use_marker_covariance:
            params_to_optimize += list(gp_covariance_module.parameters())
        elif gp_learn_lengthscale:
            params_to_optimize += list(gp_covariance_module.parameters())
```

- [ ] **Step 5: Commit**

```bash
git add train_masked_model_gp.py
git commit -m "feat: wire up KroneckerMarkerCovariance instantiation and optimizer in training script"
```

---

### Task 6: Wire up training script — forward pass and loss call

**Files:**
- Modify: `train_masked_model_gp.py:60-83` (train function signature)
- Modify: `train_masked_model_gp.py:119-138` (loss init in train function)
- Modify: `train_masked_model_gp.py:170-181` (training loop loss call)
- Modify: `train_masked_model_gp.py:656-679` (call site at bottom)

- [ ] **Step 1: Add `use_marker_covariance` param to `train_masked_gp` function**

Add `use_marker_covariance=False,` parameter after `use_gp_loss=True,` in the function signature (line 69).

- [ ] **Step 2: Update loss initialization inside `train_masked_gp`**

In the loss init block (line 119-138), add a new branch for marker covariance. The block becomes:

```python
    gp_loss_fn = None
    if use_gp_loss and gp_covariance_module is not None:
        if isinstance(gp_covariance_module, KroneckerMarkerCovariance):
            gp_loss_fn = HybridKroneckerMarkerGPNLLLoss(
                covariance_module=gp_covariance_module,
                lambda_gp=lambda_gp,
                downscale_factor=gp_downscale_factor,
                device=device,
            )
            print(f"Using Kronecker Marker GP loss with lambda_gp={lambda_gp}")
        elif isinstance(gp_covariance_module, KroneckerPlusSpatialCovariance):
            gp_loss_fn = HybridKroneckerGPNLLLoss(
                covariance_module=gp_covariance_module,
                lambda_gp=lambda_gp,
                downscale_factor=gp_downscale_factor,
                device=device,
            )
            print(f"Using Kronecker GP loss with lambda_gp={lambda_gp}")
        else:
            gp_loss_fn = HybridGPNLLLoss(
                covariance_module=gp_covariance_module,
                lambda_gp=lambda_gp,
                max_cg_iterations=gp_max_cg_iterations,
                downscale_factor=gp_downscale_factor,
                device=device,
            )
            print(f"Using CG GP loss with lambda_gp={lambda_gp}")
```

Note: `KroneckerMarkerCovariance` must be checked **before** `KroneckerPlusSpatialCovariance` since it is not a subclass. Add import of `KroneckerMarkerCovariance` at the top of the file if not already present from Task 5.

- [ ] **Step 3: Extract embeddings and pass to loss in training loop**

In the training loop (around line 175-181), modify the GP loss call:

```python
            if use_gp_loss and gp_loss_fn is not None:
                if use_marker_covariance:
                    # Extract Hyperkernel embeddings for decoded channels
                    marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)
                    # marker_emb: [B, C, model_dim]
                    loss, loss_dict = gp_loss_fn(img, mi, logvar, marker_emb)
                else:
                    loss, loss_dict = gp_loss_fn(img, mi, logvar)
```

- [ ] **Step 4: Pass `use_marker_covariance` to `train_masked_gp` call**

At the call site (around line 656-679), add the parameter:

```python
        use_marker_covariance=use_marker_covariance,
```

after the `use_gp_loss=use_gp_loss,` line.

- [ ] **Step 5: Commit**

```bash
git add train_masked_model_gp.py
git commit -m "feat: extract marker embeddings and pass to marker GP loss in training loop"
```

---

### Task 7: Wire up validation loop and checkpointing

**Files:**
- Modify: `train_masked_model_gp.py:267-280` (validation function signature)
- Modify: `train_masked_model_gp.py:347-348` (validation loss call)
- Modify: `train_masked_model_gp.py:227-239` (validation call from train)
- Modify: `train_masked_model_gp.py:242-250` (checkpoint saving)

- [ ] **Step 1: Add `use_marker_covariance` and `model` params to `test_masked_gp`**

Update `test_masked_gp` signature (line 267) to include:

```python
def test_masked_gp(
    model,
    test_dataloader,
    device,
    epoch,
    gp_covariance_module,
    gp_loss_fn,
    marker_names_map,
    num_plots=4,
    spatial_masking_ratio=0.6,
    fully_masked_channels_max_frac=0.5,
    mask_patch_size=8,
    use_gp_loss=True,
    use_marker_covariance=False,
):
```

Note: `model` is already the first parameter. We just add `use_marker_covariance=False`.

- [ ] **Step 2: Update validation loss call**

In the validation loop (around line 347-348), modify:

```python
            if use_gp_loss and gp_loss_fn is not None:
                if use_marker_covariance:
                    marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)
                    loss, loss_dict = gp_loss_fn(img, mi, logvar, marker_emb)
                else:
                    loss, loss_dict = gp_loss_fn(img, mi, logvar)
```

- [ ] **Step 3: Pass new param to validation call from train function**

In `train_masked_gp` (around line 227-239), add `use_marker_covariance=use_marker_covariance,` to the `test_masked_gp(...)` call.

- [ ] **Step 4: Add marker covariance diagnostics to validation metrics**

After the loss computation in validation loop, add diagnostic logging when `use_marker_covariance` is True. After the existing `if use_gp_loss and gp_loss_fn is not None:` block that sets `val_metrics`, add:

```python
    if use_marker_covariance and gp_covariance_module is not None:
        # Log marker covariance diagnostics using a sample embedding
        with torch.no_grad():
            sample_emb = model.encoder.hyperkernel.hyperkernel_weights.weight[:C_sample]
            _, triple_eigs, K_C = gp_covariance_module._compute_marker_eigen(sample_emb)
            eigvals = torch.linalg.eigvalsh(K_C)
            val_metrics["marker_cov_min_eigenvalue"] = eigvals.min().item()
            val_metrics["marker_cov_condition_number"] = (eigvals.max() / eigvals.min()).item()
```

Actually, this is tricky because we don't know C_sample at this point. Simpler approach — log per-batch diagnostics inside the validation loop:

In the validation loop, after the loss call when `use_marker_covariance` is True, add:

```python
                if use_marker_covariance:
                    with torch.no_grad():
                        _, _, K_C = gp_covariance_module._compute_marker_eigen(marker_emb[0])
                        eigvals = torch.linalg.eigvalsh(K_C)
                        if idx == 0:
                            log_validation_batch_metrics(
                                marker_cov_min_eigenvalue=eigvals.min().item(),
                                marker_cov_condition_number=(eigvals.max() / eigvals.min()).item(),
                                step=epoch,
                            )
```

- [ ] **Step 5: Update Comet config dict**

In the `__main__` block (around line 642-652), add marker covariance config to `comet_config`:

```python
    comet_config.update({
        # ... existing entries ...
        "use_marker_covariance": use_marker_covariance,
        "marker_embed_dim":      marker_embed_dim,
        "marker_jitter":         marker_jitter,
    })
```

- [ ] **Step 6: Commit**

```bash
git add train_masked_model_gp.py
git commit -m "feat: wire up marker covariance in validation loop and add diagnostics logging"
```

---

### Task 8: Update module exports and add example config

**Files:**
- Modify: `multiplex_model/modules/__init__.py` (add export)
- Create: `train_masked_gp_marker_config.yaml` (example config)

- [ ] **Step 1: Add KroneckerMarkerCovariance to module exports**

In `multiplex_model/modules/__init__.py`, add to the gp_covariance import (line 99-101):

```python
from .gp_covariance import (
    KroneckerMarkerCovariance,
    LowRankPlusSpatialCovariance,
)
```

And add `"KroneckerMarkerCovariance"` to `__all__` list.

- [ ] **Step 2: Create example config file**

Create `train_masked_gp_marker_config.yaml` based on existing `train_masked_gp_config.yaml`:

```yaml
# Configuration for training with Kronecker Marker GP loss
# Extends standard GP config with marker covariance from Hyperkernel embeddings

# ============================================================================
# GP LOSS CONFIGURATION
# ============================================================================
use_gp_loss: true
use_kronecker_gp: true
use_marker_covariance: true          # Enable marker covariance (K_C from embeddings)
marker_embed_dim: 32                  # Projection dim for embedding -> K_C
marker_jitter: 1e-2                   # Jitter for K_C numerical stability
lambda_gp: 0.1
gp_kernel_jitter: 1e-2
gp_lengthscale: 5.0
gp_max_cg_iterations: 50
gp_downscale_factor: 1
gp_learn_lengthscale: false           # Not applicable for Kronecker

# ============================================================================
# STANDARD TRAINING CONFIGURATION
# ============================================================================
encoder:
    ma_layers_blocks: [4,]
    ma_embedding_dims: [16,]
    pm_layers_blocks: [4, 4, 4]
    pm_embedding_dims: [128, 256, 512]
    use_latent_norm: true

    hyperkernel:
      kernel_size: 1
      padding: 0
      stride: 1
      use_bias: true

decoder:
  decoded_embed_dim: 384
  num_blocks: 1

  hyperkernel:
      kernel_size: 1
      padding: 0
      stride: 1
      use_bias: true

# Data configuration
panel_config: configs/all_panels_config.yaml
tokenizer_config: configs/all_markers_tokenizer.yaml
input_image_size: [112, 112]
num_workers: 8
batch_size: 8

# Training configuration
device: cuda
lr: 5e-4
final_lr: 1e-5
weight_decay: 0.0001
gradient_accumulation_steps: 1
epochs: 200
frac_warmup_steps: 0.01
min_channels_frac: 0.75
spatial_masking_ratio: 0.6
fully_masked_channels_max_frac: 0.5
mask_patch_size: 8
from_checkpoint: null
reset_lr_schedule: false
checkpoints_dir: checkpoints
save_checkpoint_freq: 5
beta: 0.5

# Comet.ml logging configuration
tags: ['SZARY', 'GP', 'kronecker', 'marker-covariance']
comet_project: multiplex-image-model
comet_workspace: micha-zmys-owski
comet_api_key: null
```

- [ ] **Step 3: Commit**

```bash
git add multiplex_model/modules/__init__.py train_masked_gp_marker_config.yaml
git commit -m "feat: export KroneckerMarkerCovariance and add example config"
```

---

### Task 9: End-to-end smoke test

**Files:**
- Modify: `tests/test_kronecker_marker.py` (add integration test)

- [ ] **Step 1: Write end-to-end test**

Append to `tests/test_kronecker_marker.py`:

```python
def test_end_to_end_training_step():
    """Simulate one training step: model forward -> extract embeddings -> loss -> backward."""
    torch.manual_seed(42)
    B, C_total, H, W = 2, 5, 16, 16
    C_active = 4

    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.losses import HybridKroneckerMarkerGPNLLLoss
    from multiplex_model.modules.gp_covariance import KroneckerMarkerCovariance

    model = MultiplexAutoencoder(
        num_channels=C_total,
        encoder_config={
            "ma_layers_blocks": [1],
            "ma_embedding_dims": [8],
            "pm_layers_blocks": [1],
            "pm_embedding_dims": [16],
            "hyperkernel_config": {"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
        },
        decoder_config={
            "decoded_embed_dim": 16,
            "num_blocks": 1,
            "hyperkernel_config": {"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
        },
    )

    # hyperkernel_model_dim = pm_embedding_dims[0] * kernel_size^2 * ma_embedding_dims[-1]
    # = 16 * 1 * 8 = 128
    hyperkernel_model_dim = 16 * 1 * 8

    # Image is 16x16, encoder downscales by 2^(len(ma_layers_blocks) + len(pm_layers_blocks[:-1]))
    # = 2^(1+0) = 2. Decoder upscales back. So GP grid_size depends on downscale_factor.
    # For this test use downscale_factor to match: 16 // 1 = 16
    gp_module = KroneckerMarkerCovariance(
        grid_size=H,
        marker_embed_dim=8,
        hyperkernel_model_dim=hyperkernel_model_dim,
        kernel_jitter=1e-2,
        marker_jitter=1e-2,
        device="cpu",
    )

    loss_fn = HybridKroneckerMarkerGPNLLLoss(
        covariance_module=gp_module,
        lambda_gp=0.1,
        downscale_factor=1,
        device="cpu",
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(gp_module.parameters()),
        lr=1e-3,
    )

    # Simulate forward pass
    img = torch.rand(B, C_active, H, W)
    channel_ids = torch.arange(C_active).unsqueeze(0).expand(B, -1)
    active_ids = channel_ids.clone()

    output = model(img, active_ids, channel_ids)["output"]
    mi, logvar = output.unbind(dim=-1)
    mi = torch.sigmoid(mi)
    logvar = torch.clamp(logvar, -15.0, 15.0)

    # Extract marker embeddings
    marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)

    # Compute loss
    loss, loss_dict = loss_fn(img, mi, logvar, marker_emb)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify gradients exist
    assert model.encoder.hyperkernel.hyperkernel_weights.weight.grad is not None
    assert gp_module.embedding_projection.weight.grad is not None
    assert loss.isfinite(), f"Loss is not finite: {loss.item()}"

    print(f"End-to-end smoke test passed. Loss: {loss.item():.4f}")
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/test_kronecker_marker.py -v
```

Expected: All 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kronecker_marker.py
git commit -m "test: add end-to-end smoke test for marker covariance training step"
```
