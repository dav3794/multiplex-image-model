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
# tests/test_kronecker_marker.py — numerical unit tests (solver, log-prob, loss)
# tests/test_training_integration.py — validation loop, masking, logging smoke tests

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
- `KroneckerMarkerCovariance`: `(K_x ⊗ K_y) ⊗ K_C + U_block·U_blockᵀ + jitter·I` where K_C comes from Hyperkernel embeddings projected via `embedding_projection` (nn.Linear) then row-normalised. Config fields: `use_marker_covariance: bool`, `marker_embed_dim: int` (default 32), `marker_jitter: float` (default 1e-2). After instantiation call `.to(device)` explicitly — nn.Linear parameters don't move with spatial buffers.

**`losses.py`** — Loss functions:
- `beta_nll_loss`: Standard pixel-wise NLL with beta weighting.
- `GPNLLLoss` / `HybridGPNLLLoss`: GP NLL via conjugate gradient iterations.
- `KroneckerGPNLLLoss` / `HybridKroneckerGPNLLLoss`: Analytic GP NLL via Kronecker + Woodbury (preferred, square images only). `HybridKroneckerGPNLLLoss` = `(1-λ)·standard_NLL + λ·kronecker_NLL`.
- `HybridKroneckerMarkerGPNLLLoss`: Same hybrid structure but takes `marker_embeddings [B, C, model_dim]` as 4th argument; computes K_C per sample and averages GP NLL across the batch.

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
- Direct SSH bury→szary fails (no key); use `sbatch --wrap='cmd'` for one-off commands on szary (use `. ~/venv/bin/activate`, NOT `source` — SLURM uses sh not bash)
- Project dir on server: `~/marcin_multiplex/`, logs: `~/marcin_multiplex/logs/`, checkpoints: `~/marcin_multiplex/checkpoints/`
- SLURM: partition `common`, QOS `mzmyslowski`, node `szary`, max wall 7 days (`--time=7-00:00:00`)
- Venv on szary: `. ~/venv/bin/activate` (SLURM scripts must use `.` not `source`)
- Always submit training via `sbatch train.sh <config> gp` — never use `--wrap` for training; train.sh sets COMET_API_KEY and `--gres=gpu:1` (without it CUDA is not allocated even on szary)
- Checkpoint naming: `last_checkpoint-ImVs-{N}.pth` (per epoch), `final_model-ImVs-{N}.pth` (end of run); each job gets a NEW run name (N increments), so its checkpoint is saved under the new name
- `final_model-ImVs-{N}.pth` contains only model weights — use `last_checkpoint-ImVs-{N}.pth` for resumption (has epoch/optimizer/scheduler state)
- CRITICAL: after each job finishes, update `from_checkpoint` in the config to point to the latest `last_checkpoint-ImVs-{N}.pth` before submitting the next job — otherwise the next job resumes from the original checkpoint, not the latest
- `sbatch train.sh <config_file> gp` — config is first arg, `gp` is second (not the other way around)
- When adding more epochs to a finished run: set `epochs` to total (e.g. 200 for another 100), not just the new count; use `reset_lr_schedule: true` for fresh cosine cycle

## Evaluation Metrics

- **Primary metric: MSE** (Mean Squared Error) — use this when comparing runs or reporting results
- MAE is logged but secondary; Pearson ρ is reported for both MAE/Var and MSE/Var — prefer the MSE variant

## GP Training Notes

- Stable Kronecker GP config: `gp_lengthscale: 5.0`, `frac_warmup_steps: 0.01`, `batch_size: 8`
- Kronecker kernel defaults: `kernel_jitter=1e-2`, `matern_nu=1.5` (once-differentiable); lengthscale in normalised [0,1] coords — value `5.0` ≫ image range means broad spatial correlation
- `gp_lengthscale: 0.1` → ill-conditioned kernel → divergence
- `frac_warmup_steps: 0.1` with long runs → many epochs of rising LR → instability; keep ≤ 0.01
- Occasional StdNLL spikes (~0.0 instead of ~-7) on single val epochs are normal (hard batch), not a failure
- Pearson ρ (MAE vs Var) varies 0.4–0.9 across val batches; occasional drops are normal

### KroneckerMarkerCovariance numerical stability

- K_C = E @ E.T + jitter·I: embeddings E MUST be row-normalised (`nn.functional.normalize(E, p=2, dim=1)`) before this — without normalisation, K_C condition number grows with embedding scale and GP NLL goes nan immediately
- When C > marker_embed_dim (32), K_C has C-32 repeated eigenvalues at exactly `marker_jitter`; use float64 for `linalg.eigh` to avoid LAPACK convergence failure, then cast back to float32
- Symptom of broken K_C: GP NLL = nan from training epoch 0; condition number >> 1000 in diagnostics

## Evaluation Scripts

- `run_embed.py`: Extracts latent embeddings from trained model. Patches images into 128×128 tiles, encodes each, saves embeddings + metadata in batches. Config: set `datasets`, paths, and `MODEL_WEIGHTS_PATH` before running.
- `run_validation_leave_one_out.py`: Leave-one-out marker imputation benchmark (from Marcin). For each test image, masks one channel at a time (C copies with C-1 channels each), reconstructs the missing marker, reports MSE/Pearson/log-sigma per marker. Usage: `python run_validation_leave_one_out.py --versions 0 14 --panel-config <path> --tokenizer-config <path>`. Expects model naming `Immu*-6{version:02d}-beta-*.pth` with matching `config.{stem}.yaml`.
