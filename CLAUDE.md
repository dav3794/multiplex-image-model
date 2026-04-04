# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multiplex Image Model — a PyTorch research library for masked autoencoder training on multiplex immunofluorescence images, with optional Gaussian Process-based uncertainty estimation.

## Commands

```bash
# Install in editable mode
pip install -e ".[dev]"

# Format code
black --line-length 120 .
isort .

# Lint / type check
flake8 .
python -m mypy multiplex_model/ train_masked_model_gp.py train_masked_model.py

# Run tests
pytest tests/ -v

# Train (standard beta-NLL loss)
python train_masked_model.py train_masked_config.yaml   # uses sys.argv[1], NOT --config

# Train (GP/Kronecker loss)
python train_masked_model_gp.py train_masked_gp_config.yaml
```

## Before Submitting to Cluster

Run mypy and tests locally before rsyncing to szary:

```bash
python -m mypy multiplex_model/ train_masked_model_gp.py train_masked_model.py
pytest tests/ -v
```

mypy catches call-arg errors (wrong keyword arguments, missing args) in both the typed library code and the training scripts. `check_untyped_defs = true` ensures script bodies are checked even without type annotations.

## Architecture

### Core Components (`multiplex_model/`)

**`modules/immuvis.py`** — Main autoencoder architecture:
- `Hyperkernel`: Per-channel dynamic embedding layer. Encoder path: `(B, C*I, H, W) → (B, E, H, W)` via learned marker embeddings; decoder path: `(B, I, H, W) → (B, C, E, H, W)` per marker.
- `MultiplexImageEncoder`: Two-pathway encoder — Marker-Agnostic (MA) processes raw intensities per marker independently, then Hyperkernel maps to shared embeddings, then Pan-Marker (PM) pathway processes all markers jointly → latent `(B, E, H, W)`.
- `MultiplexImageDecoder`: Reconstructs from latent, outputs `(B, C, 2, H, W)` (mean + log-variance per marker).
- `MultiplexAutoencoder`: Full encode-decode pipeline.

**`modules/gp_covariance.py`** — GP covariance structures:
- `LowRankPlusSpatialCovariance`: `K = K_spatial + σσᵀ + jitter·I`, uses GPyTorch Matérn kernel, CG-based solver.
- `KroneckerPlusSpatialCovariance`: `K = (K_x ⊗ K_y) + U·Uᵀ + jitter·I`, separable Matérn on pixel axes, analytic Woodbury solver (~40× faster than CG at 64×64). Eigendecomposition cached at init.

**`losses.py`** — Loss functions:
- `beta_nll_loss`: Standard pixel-wise NLL with beta weighting.
- `GPNLLLoss` / `HybridGPNLLLoss`: GP NLL via conjugate gradient iterations.
- `KroneckerGPNLLLoss` / `HybridKroneckerGPNLLLoss`: Analytic GP NLL via Kronecker + Woodbury (preferred, square images only). `HybridKroneckerGPNLLLoss` = `(1-λ)·standard_NLL + λ·kronecker_NLL`.

**`modules/registry.py`** — `BLOCK_REGISTRY` / `ENCODER_REGISTRY` + `build_from_config()` factory. All architecture blocks register themselves; configs reference them by string name.

**`utils/configuration.py`** — Pydantic v2 models: `TrainingConfig`, `EncoderConfig`, `DecoderConfig`, `ModuleConfig`. All YAML configs are validated through these.

**`utils/masking.py`** — Channel masking (random subset + full dropout) and spatial patch masking for the masked autoencoder objective.

**`data.py`** — `DatasetFromTIFF`: loads multi-panel TIFF images, applies arcsinh normalization, Butterworth filtering, median denoising, and min-max/clip normalization. `PanelBatchSampler` balances batches across panels.

### Backbone Architectures

`modules/convext.py`, `vit.py`, `swin.py`, `resnet.py` — ConvNeXt, ViT, Swin, ResNet encoders. All register into `ENCODER_REGISTRY` and follow the `Encoder` base class returning `{'output': tensor, ...}`.

### Training Scripts

- `train_masked_model.py`: Standard masked autoencoder with beta-NLL.
- `train_masked_model_gp.py`: Extends standard training with GP loss. Dispatches to Kronecker vs CG solver via `use_kronecker_gp` config flag. Both scripts use gradient accumulation, mixed precision (`torch.autocast`), cosine LR with warmup, and Comet.ml experiment tracking.

### Configuration

YAML configs pass through Pydantic validation. Key top-level fields: `panel_configs` (dataset paths + markers), `encoder` / `decoder` (architecture specs), `training` (LR, batch size, masking ratios, loss weights). The `ModuleConfig` type accepts either a plain string (block name) or a `{name: ..., kwargs: ...}` dict.

## Cluster / SLURM (szary)

- SSH: `ssh mzmyslowski@bury.mimuw.edu.pl` (SSH config has `User login_on_the_cluster` — wrong for this project)
- Direct SSH bury→szary fails (no key); use `sbatch --wrap='cmd'` for one-off commands on szary
- Trained models stored at `/raid_encrypted/immucan/models/gp/` on szary
- Project dir on server: `~/marcin_multiplex/`, logs: `~/marcin_multiplex/logs/`
- SLURM: partition `common`, QOS `mzmyslowski`, node `szary`, max wall 24h — chain jobs for longer runs
- Venv: `source ~/venv/bin/activate` (set up with uv)
- Checkpoint naming: `last_checkpoint-ImVs-{N}.pth` (per epoch), `final_model-ImVs-{N}.pth` (end of run)
- `final_model-ImVs-{N}.pth` contains only model weights — use `last_checkpoint-ImVs-{N}.pth` for resumption (has epoch/optimizer/scheduler state)
- `sbatch train.sh <config_file> gp` — config is first arg, `gp` is second (not the other way around)
- When adding more epochs to a finished run: set `epochs` to total (e.g. 200 for another 100), not just the new count; use `reset_lr_schedule: true` for fresh cosine cycle

## GP Training Notes

- Stable Kronecker GP config: `gp_lengthscale: 5.0`, `frac_warmup_steps: 0.01`, `batch_size: 8`
- Kronecker kernel defaults: `kernel_jitter=1e-2`, `matern_nu=1.5` (once-differentiable); lengthscale in normalised [0,1] coords — value `5.0` ≫ image range means broad spatial correlation
- `gp_lengthscale: 0.1` → ill-conditioned kernel → divergence
- `frac_warmup_steps: 0.1` with long runs → many epochs of rising LR → instability; keep ≤ 0.01
- Occasional StdNLL spikes (~0.0 instead of ~-7) on single val epochs are normal (hard batch), not a failure
- Pearson ρ (MAE vs Var) varies 0.4–0.9 across val batches; occasional drops are normal
