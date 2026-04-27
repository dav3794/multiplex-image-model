# Design: Kronecker Marker Covariance + Learnable Mask Token

**Date:** 2026-04-27  
**Branch:** `feat/kronecker-learnmask` (off `feat/kronecker-marker-covariance`)

## Goal

Port two new root-level files (`immuvis.py`, `train_masked_model_learnmask.py`) into the project on the additive Kronecker marker covariance branch. The result is a training variant that combines:
- Additive K_C marker covariance (Woodbury update, from `feat/kronecker-marker-covariance`)
- Learnable spatial mask token in the encoder
- New training script using `ClampWithGrad`, `RankMe`, and `load_from_checkpoint`

## Changes

### 1. `multiplex_model/modules/immuvis.py`

**`MultiplexImageEncoder`:**
- Add `use_mask_token: bool = False` and `mask_token_init: float = 0.0` to `__init__`
- Store `self.mask_token = nn.Parameter(torch.tensor(mask_token_init)) if use_mask_token else None`
- In `forward()`: accept `spatial_mask: torch.Tensor | None = None`; when `use_mask_token` and `spatial_mask` is not None, apply `torch.where(spatial_mask, mask_token, x)` before encoding

**`MultiplexAutoencoder`:**
- In `__init__`: store `self._architecture_config = {"num_channels": ..., "encoder_config": copy.deepcopy(encoder_config), "decoder_config": copy.deepcopy(decoder_config)}`
- Add `get_architecture_config(by_alias: bool = False) -> dict` method
- Add `load_from_checkpoint(checkpoint, map_location, model_config, strict) -> MultiplexAutoencoder` classmethod
- Propagate `spatial_mask` through `encode()` and `forward()`

### 2. `multiplex_model/utils/configuration.py`

Add to `EncoderConfig`:
- `use_mask_token: bool = False`
- `mask_token_init: float = 0.0`

These flow via `**encoder_config` into `MultiplexImageEncoder.__init__`.

### 3. `multiplex_model/utils/train_logging.py`

Add `mask_token: float | None = None` to `log_training_metrics` and log it when present.

### 4. `train_masked_model_learnmask.py`

Add as new tracked file at repo root. No changes to the file itself.

## What is NOT changed

- `train_masked_model.py` and `train_masked_model_gp.py` — `spatial_mask` is optional (default `None`), so they remain unaffected
- GP covariance modules — untouched
- `TrainingConfig` — `use_mask_token` belongs in encoder config, not training config

## Testing

Existing tests in `tests/test_training_integration.py` cover the masking flow and should pass unchanged (no breaking API changes — all new params are optional with defaults).
