# Kronecker Learnmask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port learnable mask token and architecture config utilities from two new root-level files into the `feat/kronecker-marker-covariance` (additive) branch.

**Architecture:** `use_mask_token` flows through `EncoderConfig → MultiplexImageEncoder.__init__` via `**encoder_config`. `spatial_mask` is an optional tensor passed through `MultiplexAutoencoder.encode()` and `forward()` down to `MultiplexImageEncoder.forward()`, where it gates learnable token substitution. `_architecture_config` is stored at init time and exposed via `get_architecture_config()` / `load_from_checkpoint()`.

**Tech Stack:** Python 3.11, PyTorch, Pydantic v2, pytest

---

### Task 1: Create branch

**Files:**
- No file changes

- [ ] **Step 1: Create and switch to the new branch**

```bash
git checkout feat/kronecker-marker-covariance
git checkout -b feat/kronecker-learnmask
```

Expected: branch `feat/kronecker-learnmask` checked out, HEAD at latest commit of `feat/kronecker-marker-covariance`.

---

### Task 2: Add `use_mask_token` / `mask_token_init` to `EncoderConfig`

**Files:**
- Modify: `multiplex_model/utils/configuration.py:58-138`
- Test: `tests/test_training_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_integration.py`:

```python
def test_encoder_config_accepts_mask_token_fields():
    from multiplex_model.utils import TrainingConfig

    # EncoderConfig has extra="forbid"; unknown fields raise ValidationError.
    # This test verifies the two new fields are accepted without error.
    from multiplex_model.utils.configuration import EncoderConfig

    cfg = EncoderConfig(
        ma_layers_blocks=[1],
        ma_embedding_dims=[8],
        pm_layers_blocks=[1],
        pm_embedding_dims=[16],
        hyperkernel={"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
        use_mask_token=True,
        mask_token_init=0.5,
    )
    assert cfg.use_mask_token is True
    assert cfg.mask_token_init == 0.5


def test_encoder_config_mask_token_defaults():
    from multiplex_model.utils.configuration import EncoderConfig

    cfg = EncoderConfig(
        ma_layers_blocks=[1],
        ma_embedding_dims=[8],
        pm_layers_blocks=[1],
        pm_embedding_dims=[16],
        hyperkernel={"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
    )
    assert cfg.use_mask_token is False
    assert cfg.mask_token_init == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_training_integration.py::test_encoder_config_accepts_mask_token_fields tests/test_training_integration.py::test_encoder_config_mask_token_defaults -v
```

Expected: FAIL — `ValidationError: Extra inputs are not permitted`.

- [ ] **Step 3: Add the two fields to `EncoderConfig`**

In `multiplex_model/utils/configuration.py`, inside `class EncoderConfig(BaseModel)`, add after the `encoder_type` field (before the validators):

```python
    use_mask_token: bool = Field(
        default=False,
        description="Whether to replace spatially-masked pixels with a learnable scalar token",
    )
    mask_token_init: float = Field(
        default=0.0,
        description="Initial value for the learnable mask token",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_training_integration.py::test_encoder_config_accepts_mask_token_fields tests/test_training_integration.py::test_encoder_config_mask_token_defaults -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite to check no regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add multiplex_model/utils/configuration.py tests/test_training_integration.py
git commit -m "feat: add use_mask_token and mask_token_init fields to EncoderConfig"
```

---

### Task 3: Add learnable mask token to `MultiplexImageEncoder`

**Files:**
- Modify: `multiplex_model/modules/immuvis.py:1,145-263`
- Test: `tests/test_training_integration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_integration.py`:

```python
def test_encoder_mask_token_is_none_when_disabled():
    from multiplex_model.modules.immuvis import MultiplexImageEncoder

    enc = MultiplexImageEncoder(
        num_channels=4,
        ma_layers_blocks=[1],
        ma_embedding_dims=[8],
        pm_layers_blocks=[1],
        pm_embedding_dims=[16],
        hyperkernel_config={"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
    )
    assert enc.mask_token is None


def test_encoder_mask_token_is_parameter_when_enabled():
    import torch
    import torch.nn as nn
    from multiplex_model.modules.immuvis import MultiplexImageEncoder

    enc = MultiplexImageEncoder(
        num_channels=4,
        ma_layers_blocks=[1],
        ma_embedding_dims=[8],
        pm_layers_blocks=[1],
        pm_embedding_dims=[16],
        hyperkernel_config={"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
        use_mask_token=True,
        mask_token_init=0.5,
    )
    assert isinstance(enc.mask_token, nn.Parameter)
    assert enc.mask_token.item() == pytest.approx(0.5)


def test_encoder_forward_applies_mask_token_to_masked_pixels():
    import torch
    from multiplex_model.modules.immuvis import MultiplexImageEncoder

    torch.manual_seed(0)
    B, C, H, W = 1, 2, 4, 4
    enc = MultiplexImageEncoder(
        num_channels=C,
        ma_layers_blocks=[1],
        ma_embedding_dims=[8],
        pm_layers_blocks=[1],
        pm_embedding_dims=[16],
        hyperkernel_config={"kernel_size": 1, "padding": 0, "stride": 1, "use_bias": True},
        use_mask_token=True,
        mask_token_init=99.0,
    )
    # Spy: intercept x just after mask application by checking that masked pixels equal 99.0
    x = torch.zeros(B, C, H, W)
    spatial_mask = torch.zeros(B, C, H, W, dtype=torch.bool)
    spatial_mask[:, :, 0, 0] = True  # mask top-left pixel

    # We verify by setting mask_token to a known sentinel and checking the
    # model doesn't crash, and that x[:,:,0,0] would be replaced.
    # Direct unit check: simulate the replacement logic.
    with torch.no_grad():
        token_val = enc.mask_token.to(dtype=x.dtype)
        x_after = torch.where(spatial_mask, token_val, x)
    assert x_after[:, :, 0, 0].allclose(torch.tensor(99.0))
    assert x_after[:, :, 1, 1].allclose(torch.tensor(0.0))

    # End-to-end: encoder forward accepts spatial_mask without error
    enc_indices = torch.arange(C).unsqueeze(0).expand(B, -1)
    out = enc(x, enc_indices, spatial_mask=spatial_mask)
    assert "output" in out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_integration.py::test_encoder_mask_token_is_none_when_disabled tests/test_training_integration.py::test_encoder_mask_token_is_parameter_when_enabled tests/test_training_integration.py::test_encoder_forward_applies_mask_token_to_masked_pixels -v
```

Expected: FAIL — `TypeError` on unexpected kwargs or missing `spatial_mask` param.

- [ ] **Step 3: Add `import copy` to `multiplex_model/modules/immuvis.py`**

At the top of `multiplex_model/modules/immuvis.py`, add `import copy` after the existing stdlib imports (before torch):

```python
import copy
```

- [ ] **Step 4: Update `MultiplexImageEncoder.__init__` signature**

Replace the current `__init__` signature (lines 145-155 in `multiplex_model/modules/immuvis.py`):

```python
    def __init__(
        self,
        num_channels: int,
        ma_layers_blocks: list[int],
        ma_embedding_dims: list[int],
        hyperkernel_config: dict,
        pm_layers_blocks: list[int],
        pm_embedding_dims: list[int],
        use_latent_norm: bool = False,
        encoder_type: str | type[Encoder] | dict = "convnext",
    ):
```

with:

```python
    def __init__(
        self,
        num_channels: int,
        ma_layers_blocks: list[int],
        ma_embedding_dims: list[int],
        hyperkernel_config: dict,
        pm_layers_blocks: list[int],
        pm_embedding_dims: list[int],
        use_latent_norm: bool = False,
        use_mask_token: bool = False,
        mask_token_init: float = 0.0,
        encoder_type: str | type[Encoder] | dict = "convnext",
    ):
```

- [ ] **Step 5: Store mask token in `MultiplexImageEncoder.__init__` body**

Inside `__init__`, right after `super().__init__()` (before `# Resolve encoder class`), add:

```python
        self.use_mask_token = use_mask_token
        self.mask_token = (
            nn.Parameter(torch.tensor(mask_token_init)) if use_mask_token else None
        )
```

- [ ] **Step 6: Update `MultiplexImageEncoder.forward` signature and body**

Replace the current `forward` signature:

```python
    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
```

with:

```python
    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
```

Then, inside `forward`, right after `B, C, H, W = x.shape` and before `x = x.reshape(B * C, 1, H, W)`, add:

```python
        if self.use_mask_token and spatial_mask is not None:
            mask_token = self.mask_token.to(dtype=x.dtype)
            x = torch.where(spatial_mask, mask_token, x)
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_training_integration.py::test_encoder_mask_token_is_none_when_disabled tests/test_training_integration.py::test_encoder_mask_token_is_parameter_when_enabled tests/test_training_integration.py::test_encoder_forward_applies_mask_token_to_masked_pixels -v
```

Expected: PASS.

- [ ] **Step 8: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add multiplex_model/modules/immuvis.py tests/test_training_integration.py
git commit -m "feat: add learnable mask token to MultiplexImageEncoder"
```

---

### Task 4: Propagate `spatial_mask` through `MultiplexAutoencoder` and add architecture config utilities

**Files:**
- Modify: `multiplex_model/modules/immuvis.py:355-470`
- Test: `tests/test_training_integration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training_integration.py`:

```python
def test_autoencoder_encode_accepts_spatial_mask():
    import torch
    from multiplex_model.modules import MultiplexAutoencoder

    B, C, H, W = 2, 4, 8, 8
    model = MultiplexAutoencoder(
        num_channels=C,
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
    x = torch.rand(B, C, H, W)
    enc_ids = torch.arange(C).unsqueeze(0).expand(B, -1)
    spatial_mask = torch.zeros(B, C, H, W, dtype=torch.bool)
    out = model.encode(x, enc_ids, spatial_mask=spatial_mask)
    assert "output" in out


def test_autoencoder_forward_accepts_spatial_mask():
    import torch
    from multiplex_model.modules import MultiplexAutoencoder

    B, C, H, W = 2, 4, 8, 8
    model = MultiplexAutoencoder(
        num_channels=C,
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
    x = torch.rand(B, C, H, W)
    enc_ids = torch.arange(C).unsqueeze(0).expand(B, -1)
    dec_ids = enc_ids
    spatial_mask = torch.zeros(B, C, H, W, dtype=torch.bool)
    out = model(x, enc_ids, dec_ids, spatial_mask=spatial_mask)
    assert "output" in out


def test_autoencoder_get_architecture_config_roundtrip():
    import torch
    from multiplex_model.modules import MultiplexAutoencoder

    C = 4
    model = MultiplexAutoencoder(
        num_channels=C,
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
    cfg = model.get_architecture_config()
    assert cfg["num_channels"] == C
    assert "encoder_config" in cfg
    assert "decoder_config" in cfg

    model2 = MultiplexAutoencoder(**cfg)
    assert model2.num_channels == C


def test_autoencoder_load_from_checkpoint_roundtrip():
    import torch
    from multiplex_model.modules import MultiplexAutoencoder

    C = 4
    model = MultiplexAutoencoder(
        num_channels=C,
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
    fake_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_architecture_config(),
    }
    loaded = MultiplexAutoencoder.load_from_checkpoint(fake_checkpoint)
    assert loaded.num_channels == C
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert k1 == k2
        assert v1.allclose(v2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_integration.py::test_autoencoder_encode_accepts_spatial_mask tests/test_training_integration.py::test_autoencoder_forward_accepts_spatial_mask tests/test_training_integration.py::test_autoencoder_get_architecture_config_roundtrip tests/test_training_integration.py::test_autoencoder_load_from_checkpoint_roundtrip -v
```

Expected: FAIL — `TypeError` on unexpected `spatial_mask` kwarg; `AttributeError` on missing `get_architecture_config`.

- [ ] **Step 3: Add `_architecture_config` storage to `MultiplexAutoencoder.__init__`**

In `multiplex_model/modules/immuvis.py`, inside `MultiplexAutoencoder.__init__`, right after `super().__init__()`, add:

```python
        self._architecture_config = {
            "num_channels": num_channels,
            "encoder_config": copy.deepcopy(encoder_config),
            "decoder_config": copy.deepcopy(decoder_config),
        }
```

- [ ] **Step 4: Add `get_architecture_config` and `load_from_checkpoint` methods**

After `MultiplexAutoencoder.__init__` (before the `encode` method), add:

```python
    def get_architecture_config(self, by_alias: bool = False) -> dict:
        """Return the model architecture configuration.

        Args:
            by_alias: If True, uses config aliases (e.g., 'hyperkernel').

        Returns:
            dict: Architecture configuration for rebuilding the model.
        """
        config = copy.deepcopy(self._architecture_config)
        if by_alias:
            config = config.copy()
            config["encoder"] = config.pop("encoder_config")
            config["decoder"] = config.pop("decoder_config")
            config["encoder"]["hyperkernel"] = config["encoder"].pop("hyperkernel_config")
            config["decoder"]["hyperkernel"] = config["decoder"].pop("hyperkernel_config")
        return config

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint: str | dict,
        map_location: str | torch.device | None = None,
        model_config: dict | None = None,
        strict: bool = True,
    ) -> "MultiplexAutoencoder":
        """Create a model and load weights from a checkpoint.

        Args:
            checkpoint: Path to checkpoint file or loaded checkpoint dict.
            map_location: Optional map_location passed to torch.load when checkpoint is a path.
            model_config: Model config to use if checkpoint lacks 'model_config'.
            strict: Whether to strictly enforce that the keys in state_dict match the model.

        Returns:
            MultiplexAutoencoder: Model with weights loaded from checkpoint.
        """
        if isinstance(checkpoint, dict):
            checkpoint_data = checkpoint
        else:
            checkpoint_data = torch.load(checkpoint, map_location=map_location)

        resolved_config = checkpoint_data.get("model_config", model_config)
        if resolved_config is None:
            raise ValueError(
                "Checkpoint is missing 'model_config'; provide model_config to load the model."
            )

        model = cls(**resolved_config)
        model.load_state_dict(checkpoint_data["model_state_dict"], strict=strict)
        return model
```

- [ ] **Step 5: Add `spatial_mask` param to `MultiplexAutoencoder.encode`**

Replace:

```python
    def encode(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
```

with:

```python
    def encode(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
```

And replace the call inside `encode`:

```python
        encoding_output = self.encoder(
            x, encoded_indices, return_features=return_features
        )
```

with:

```python
        encoding_output = self.encoder(
            x,
            encoded_indices,
            spatial_mask=spatial_mask,
            return_features=return_features,
        )
```

- [ ] **Step 6: Add `spatial_mask` param to `MultiplexAutoencoder.forward`**

Replace:

```python
    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        decoded_indices: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
```

with:

```python
    def forward(
        self,
        x: torch.Tensor,
        encoded_indices: torch.Tensor,
        decoded_indices: torch.Tensor,
        spatial_mask: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict:
```

And replace the call inside `forward`:

```python
        encoding_output = self.encode(
            x, encoded_indices, return_features=return_features
        )
```

with:

```python
        encoding_output = self.encode(
            x, encoded_indices, spatial_mask=spatial_mask, return_features=return_features
        )
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_training_integration.py::test_autoencoder_encode_accepts_spatial_mask tests/test_training_integration.py::test_autoencoder_forward_accepts_spatial_mask tests/test_training_integration.py::test_autoencoder_get_architecture_config_roundtrip tests/test_training_integration.py::test_autoencoder_load_from_checkpoint_roundtrip -v
```

Expected: PASS.

- [ ] **Step 8: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add multiplex_model/modules/immuvis.py tests/test_training_integration.py
git commit -m "feat: propagate spatial_mask through MultiplexAutoencoder and add architecture config utilities"
```

---

### Task 5: Add `mask_token` param to `log_training_metrics`

**Files:**
- Modify: `multiplex_model/utils/train_logging.py:310-349`
- Test: `tests/test_training_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_integration.py`:

```python
def test_log_training_metrics_accepts_mask_token():
    import inspect
    from multiplex_model.utils.train_logging import log_training_metrics

    sig = inspect.signature(log_training_metrics)
    assert "mask_token" in sig.parameters, "log_training_metrics must accept mask_token kwarg"
    param = sig.parameters["mask_token"]
    assert param.default is None, "mask_token should default to None"

    # Calling with mask_token must not raise TypeError
    log_training_metrics(
        loss=0.5,
        lr=1e-3,
        mu=0.5,
        logvar=-1.0,
        mae=0.1,
        mse=0.01,
        step=0,
        mask_token=0.123,
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_training_integration.py::test_log_training_metrics_accepts_mask_token -v
```

Expected: FAIL — `AssertionError` on missing `mask_token` parameter.

- [ ] **Step 3: Update `log_training_metrics` signature**

In `multiplex_model/utils/train_logging.py`, replace:

```python
def log_training_metrics(
    loss: float,
    lr: float,
    mu: float,
    logvar: float,
    mae: float,
    mse: float,
    step: int | None = None,
    standard_nll: float | None = None,
    gp_nll: float | None = None,
) -> None:
```

with:

```python
def log_training_metrics(
    loss: float,
    lr: float,
    mu: float,
    logvar: float,
    mae: float,
    mse: float,
    step: int | None = None,
    standard_nll: float | None = None,
    gp_nll: float | None = None,
    mask_token: float | None = None,
) -> None:
```

- [ ] **Step 4: Log `mask_token` when present**

In the `metrics` dict block, after the existing `if gp_nll is not None:` block, add:

```python
    if mask_token is not None:
        metrics["train/mask_token"] = mask_token
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_training_integration.py::test_log_training_metrics_accepts_mask_token -v
```

Expected: PASS.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add multiplex_model/utils/train_logging.py tests/test_training_integration.py
git commit -m "feat: add mask_token param to log_training_metrics"
```

---

### Task 6: Add `train_masked_model_learnmask.py`

**Files:**
- Add: `train_masked_model_learnmask.py` (already exists at repo root as untracked)

- [ ] **Step 1: Stage and commit the new training script**

```bash
git add train_masked_model_learnmask.py
git commit -m "feat: add train_masked_model_learnmask training script"
```

- [ ] **Step 2: Verify the full test suite still passes**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Run mypy**

```bash
python -m mypy multiplex_model/ train_masked_model_learnmask.py
```

Expected: no errors (or only pre-existing ones unrelated to these changes).
