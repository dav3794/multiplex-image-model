# Kronecker Marker Covariance — Design Spec

## Goal

Extend the Kronecker GP framework to model uncertainty along the marker dimension (C) in addition to spatial dimensions (H x W). The marker covariance matrix K_C is derived end-to-end from Hyperkernel embeddings, enabling the GP loss to capture inter-marker correlations during training and expose them during inference for downstream decision-making.

## Mathematical Model

### Joint Covariance (NC dimensions: N pixels x C markers)

```
K = (K_x (x) K_y) (x) K_C  +  U_block * U_block^T  +  jitter * I_{NC}
```

Where:
- `K_x`, `K_y` in R^{n x n} — 1D Matern kernels on pixel axes (n = grid_size, N = n^2). Eigendecomposed once at init (existing behavior).
- `K_C` in R^{C x C} — Gram matrix of projected marker embeddings: `K_C = E_active * E_active^T + eps * I_C`
  - `E_active` in R^{C x D}: Hyperkernel embeddings for active markers, projected to dimension D via `nn.Linear(model_dim, marker_embed_dim)`.
  - `eps * I_C`: jitter for positive-definiteness of K_C (default eps=1e-2).
  - Eigendecomposed every forward pass (O(C^3), negligible for C <= 40).
- `U_block` in R^{NC x C} — block-diagonal matrix: column c contains per-pixel sigma vector u_c in R^N from decoder (existing sigma output), zeros elsewhere. Rank-C.
- `jitter * I_{NC}` — numerical stability, absorbed into triple eigenvalues.

### Eigendecomposition of Base Matrix A

```
A = (K_x (x) K_y) (x) K_C + jitter * I
```

Eigenvalues: `a[i,j,k] = lambda_x[i] * lambda_y[j] * lambda_C[k] + jitter`

Eigenvectors: `V_x (x) V_y (x) V_C` (never materialized — applied via einsum contractions).

### Woodbury Identity (rank-C update)

```
K = A + U_block * U_block^T

log det(K) = sum_{i,j,k} log(a[i,j,k]) + log det(I_C + U_block^T A^{-1} U_block)

K^{-1} e = A^{-1} e - A^{-1} U_block (I_C + U_block^T A^{-1} U_block)^{-1} U_block^T A^{-1} e
```

Inner matrix `I_C + U_block^T A^{-1} U_block` is C x C.

### A^{-1} v Solver (triple Kronecker)

Extends existing `_A_solve` from 2 to 3 einsum contractions:

```
v in R^{NC}  ->  reshape to [n, n, C]
1. Contract with V_C^T on marker axis:    tmp = einsum("ijc, ck -> ijk", V3, V_C)
2. Contract with V^T on spatial axis y:    tmp = einsum("ijk, jb -> ibk", tmp, V)  
3. Contract with V^T on spatial axis x:    tmp = einsum("ibk, ia -> abk", tmp, V)
4. Divide by triple_eigs[a, b, k]
5. Reverse contractions (V, V, V_C)
```

Supports batched right-hand sides: `v in R^{NC x m}` with m columns processed simultaneously.

## Architecture

### New Module: `KroneckerMarkerCovariance` (gp_covariance.py)

```python
class KroneckerMarkerCovariance(nn.Module):
    def __init__(
        self,
        grid_size: int,
        marker_embed_dim: int,       # projection dim for embeddings -> K_C
        hyperkernel_model_dim: int,   # input dim of Hyperkernel embeddings  
        kernel_jitter: float = 1e-2,
        marker_jitter: float = 1e-2,
        spatial_matern_kernel_nu: float = 1.5,
        spatial_matern_kernel_length_scale: float = 5.0,
        device=None,
    ):
        # Spatial eigendecomp (K_x, K_y) — same as KroneckerPlusSpatialCovariance
        # nn.Linear(hyperkernel_model_dim, marker_embed_dim) — embedding projection
        # marker_jitter (eps for K_C)

    def log_prob_joint(
        self,
        mu_all: Tensor,              # [N, C]
        U_all: Tensor,               # [N, C] per-pixel sigma
        targets: Tensor,             # [N, C]
        marker_embeddings: Tensor,   # [C, hyperkernel_model_dim]
    ) -> Tensor:
        # 1. Project embeddings: E = linear(marker_embeddings)  -> [C, D]
        # 2. K_C = E @ E^T + eps * I_C
        # 3. Eigendecomp K_C -> (lambda_C, V_C)
        # 4. triple_eigs[i,j,k] = lam_x[i] * lam_y[j] * lambda_C[k] + jitter
        # 5. Woodbury: log_det + mahalanobis via _A_solve_triple
        # 6. Return scalar log prob
```

### New Loss: `KroneckerMarkerGPNLLLoss` / `HybridKroneckerMarkerGPNLLLoss` (losses.py)

```python
class KroneckerMarkerGPNLLLoss(nn.Module):
    def forward(self, target, mu, sigma, marker_embeddings):
        # Loop over batch, call covariance_module.log_prob_joint per image
        # Return mean NLL per element

class HybridKroneckerMarkerGPNLLLoss(nn.Module):
    def forward(self, target, mu, logvar, marker_embeddings):
        # standard_nll: pixel-wise (unchanged)
        # gp_nll: KroneckerMarkerGPNLLLoss with marker_embeddings
        # return (1 - lambda_gp) * standard + lambda_gp * gp, loss_dict
```

### Training Script Changes (train_masked_model_gp.py)

- After model forward pass, extract embeddings:
  ```python
  marker_embeddings = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)
  # [B, C, model_dim] — pass per-batch-element to loss
  ```
- New config fields:
  - `use_marker_covariance: bool` (default false)
  - `marker_embed_dim: int` (default 32)
- Log diagnostics to Comet: `marker_cov_min_eigenvalue`, `marker_cov_condition_number`

### Inference API

New method on `KroneckerMarkerCovariance`:
```python
def compute_marker_correlation(self, marker_embeddings: Tensor) -> Tensor:
    """Returns C x C correlation matrix from K_C."""
```

This can be called post-training to extract learned marker relationships.

## Optimizer Integration

`KroneckerMarkerCovariance` contains a learnable `nn.Linear` projection layer. Its parameters must be included in the optimizer:

```python
optimizer = optim.AdamW(
    list(model.parameters()) + list(marker_cov_module.embedding_projection.parameters()),
    lr=...,
)
```

Alternatively, the marker covariance module can be saved/loaded alongside the model checkpoint (separate key in state dict).

## Backward Compatibility

- `use_marker_covariance: false` in YAML -> identical behavior to current code
- Existing checkpoints load without issues (marker covariance module saved as separate checkpoint key, absent in old checkpoints)
- `KroneckerPlusSpatialCovariance` and all existing loss classes remain unchanged
- `HybridKroneckerGPNLLLoss` continues to work as before

## Complexity

| Operation | Current (per image) | New (per image) |
|-----------|-------------------|-----------------|
| Eigendecomp init | O(n^3) spatial | O(n^3) spatial (same) |
| Eigendecomp forward | none | O(C^3) for K_C |
| A^{-1} solve | O(n^2 * C) | O(n^2 * C^2) |
| Woodbury inner | C scalar divides | C x C matrix solve |
| Memory | O(n^2) eigenvalues | O(n^2 * C) triple eigenvalues |

For n=64, C=20: current ~80K mults, new ~1.6M mults + 8K for Woodbury. Still dominated by O(n^2 * C^2).

## Risks and Mitigation

1. **Gradient instability on Hyperkernel embeddings**: Two gradient sources (reconstruction + GP). Mitigate with low `lambda_gp` (0.05-0.1), monitor gradient norms, gradient clipping per param group. Fallback: `detach()` embeddings.

2. **K_C ill-conditioned**: Similar marker embeddings -> near-singular K_C. Mitigate with `marker_jitter` (eps=1e-2). Monitor `min(lambda_C)`.

3. **Variable C per batch**: Channel masking changes active marker count. Not a problem: K_C eigendecomp is O(C^3) per forward, cheap for C <= 40.

4. **Memory**: Triple eigenvalues [n, n, C] for n=64, C=40 ~ 640KB. Negligible.

## Out of Scope

- Learnable spatial lengthscale (existing Kronecker limitation)
- Per-pixel K_C (too expensive, not needed)
- Rectangular images (existing H==W constraint remains)
- Changes to encoder/decoder architecture

## Files to Modify

1. `multiplex_model/modules/gp_covariance.py` — new `KroneckerMarkerCovariance` class
2. `multiplex_model/losses.py` — new `KroneckerMarkerGPNLLLoss` + `HybridKroneckerMarkerGPNLLLoss`
3. `train_masked_model_gp.py` — extract embeddings, new config fields, pass to loss
4. `multiplex_model/utils/configuration.py` — new config fields (`use_marker_covariance`, `marker_embed_dim`)
