"""Integration tests for training loop, validation loop, and logging infrastructure.

These tests cover the scaffolding layer (masking, script functions, logging) that
the numerical unit tests in test_kronecker_marker.py do not exercise.
"""

import math
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pytest

# Make the training script importable as a module (functions only, __main__ is guarded)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_tiny_setup(grid_size=8, C_total=4):
    """Build a tiny model + GP module + loss function for integration tests."""
    from multiplex_model.modules import MultiplexAutoencoder
    from multiplex_model.modules.gp_covariance import KroneckerMarkerCovariance
    from multiplex_model.losses import HybridKroneckerMarkerGPNLLLoss

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
    hyperkernel_model_dim = 16 * 1 * 8

    gp_module = KroneckerMarkerCovariance(
        grid_size=grid_size,
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

    return model, gp_module, loss_fn


def _make_fake_dataloader(B=2, C=4, H=8, W=8, num_batches=3):
    """DataLoader yielding (img, channel_ids, panel_idx, img_path) like the real one."""

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            img = torch.rand(C, H, W)
            channel_ids = torch.arange(C)
            panel_idx = torch.tensor(0)
            img_path = f"fake/path/{idx}.tiff"
            return img, channel_ids, panel_idx, img_path

    return torch.utils.data.DataLoader(FakeDataset(B * num_batches), batch_size=B)


# ---------------------------------------------------------------------------
# Test 1: Validation loop smoke test
# ---------------------------------------------------------------------------

def test_validation_loop_runs():
    """test_masked_gp runs end-to-end without error and returns finite metrics."""
    from train_masked_model_gp import test_masked_gp

    torch.manual_seed(42)
    H = W = 8
    C = 4
    B = 2

    model, gp_module, loss_fn = _build_tiny_setup(grid_size=H, C_total=C)
    dataloader = _make_fake_dataloader(B=B, C=C, H=H, W=W, num_batches=3)
    marker_names_map = {i: f"marker_{i}" for i in range(C)}

    val_metrics = test_masked_gp(
        model=model,
        test_dataloader=dataloader,
        device="cpu",
        epoch=0,
        gp_covariance_module=gp_module,
        gp_loss_fn=loss_fn,
        marker_names_map=marker_names_map,
        num_plots=1,
        spatial_masking_ratio=0.5,
        fully_masked_channels_max_frac=0.25,
        mask_patch_size=2,
        use_gp_loss=True,
        use_marker_covariance=True,
    )

    for key in ("val_loss", "val_mae", "val_mse", "val_standard_nll", "val_gp_nll"):
        assert key in val_metrics, f"Missing key: {key}"
        assert math.isfinite(val_metrics[key]), f"{key} is not finite: {val_metrics[key]}"


# ---------------------------------------------------------------------------
# Test 2: Config parsing and module instantiation
# ---------------------------------------------------------------------------

def test_config_fields_and_module_instantiation():
    """TrainingConfig accepts new marker covariance fields; KroneckerMarkerCovariance
    instantiates correctly and its parameters are on the right device."""
    from multiplex_model.utils.configuration import TrainingConfig
    from multiplex_model.modules.gp_covariance import KroneckerMarkerCovariance

    # Verify new fields exist with correct defaults
    defaults = TrainingConfig.model_fields
    assert "use_marker_covariance" in defaults
    assert "marker_embed_dim" in defaults
    assert "marker_jitter" in defaults

    assert defaults["use_marker_covariance"].default is False
    assert defaults["marker_embed_dim"].default == 32
    assert defaults["marker_jitter"].default == pytest.approx(1e-2)

    # Instantiate the module and verify parameters are on device
    gp_module = KroneckerMarkerCovariance(
        grid_size=16,
        marker_embed_dim=32,
        hyperkernel_model_dim=128,
        kernel_jitter=1e-2,
        marker_jitter=1e-2,
        device="cpu",
    )
    gp_module = gp_module.to("cpu")

    for name, param in gp_module.named_parameters():
        assert param.device.type == "cpu", f"Parameter {name} is on {param.device}, expected cpu"

    # Verify the projection layer has the right shape
    assert gp_module.embedding_projection.in_features == 128
    assert gp_module.embedding_projection.out_features == 32


# ---------------------------------------------------------------------------
# Test 3: Logging function signature
# ---------------------------------------------------------------------------

def test_log_validation_images_accepts_name_suffix():
    """log_validation_images accepts name_suffix kwarg without TypeError."""
    from multiplex_model.utils.train_logging import log_validation_images

    fig, _ = plt.subplots(1, 1, figsize=(2, 2))
    # _experiment is None in tests so nothing is actually logged — just check no TypeError
    log_validation_images(
        fig=fig,
        panel_idx=0,
        img_path="fake/path.tiff",
        epoch=0,
        masked_channels_names="marker_0",
        img_idx=0,
        name_suffix="_sigma",
    )
    plt.close(fig)


def test_log_validation_images_default_no_suffix():
    """log_validation_images still works without name_suffix (backward compat)."""
    from multiplex_model.utils.train_logging import log_validation_images

    fig, _ = plt.subplots(1, 1, figsize=(2, 2))
    log_validation_images(
        fig=fig,
        panel_idx=0,
        img_path="fake/path.tiff",
        epoch=0,
        masked_channels_names="marker_0",
        img_idx=0,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 4: Training step with masking via script functions
# ---------------------------------------------------------------------------

def test_training_step_with_channel_and_spatial_masking():
    """Full training step: channel mask → spatial mask → forward → embed extract → loss → backward.

    Verifies that img, channel_ids, and marker_emb shapes are all consistent
    after apply_channel_masking reduces the channel set.
    """
    from multiplex_model.utils.masking import apply_channel_masking, apply_spatial_masking

    torch.manual_seed(42)
    H = W = 8
    C_total = 6
    B = 2

    model, gp_module, loss_fn = _build_tiny_setup(grid_size=H, C_total=C_total)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(gp_module.parameters()), lr=1e-3
    )

    img = torch.rand(B, C_total, H, W)
    channel_ids = torch.arange(C_total).unsqueeze(0).expand(B, -1).contiguous()

    # Channel masking — both img and channel_ids are reduced to the active subset
    img, channel_ids, masked_img, active_channel_ids = apply_channel_masking(
        img,
        channel_ids,
        min_channels_frac=0.5,
        fully_masked_channels_max_frac=0.25,
        apply_channel_subset_sampling=True,
    )

    # Spatial masking
    masked_img, _ = apply_spatial_masking(masked_img, spatial_masking_ratio=0.5, mask_patch_size=2)

    # Forward pass
    output = model(masked_img, active_channel_ids, channel_ids)["output"]
    mi, logvar = output.unbind(dim=-1)
    mi = torch.sigmoid(mi)
    logvar = torch.clamp(logvar, -15.0, 15.0)

    # Embedding extraction — channel_ids now matches the reduced img
    marker_emb = model.encoder.hyperkernel.hyperkernel_weights(channel_ids)

    C_active = img.shape[1]
    assert mi.shape == img.shape, f"mi {mi.shape} != img {img.shape}"
    assert marker_emb.shape[:2] == (B, C_active), (
        f"marker_emb {marker_emb.shape} inconsistent with img channel count {C_active}"
    )

    # Loss + backward
    loss, loss_dict = loss_fn(img, mi, logvar, marker_emb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.isfinite(), f"Loss is not finite: {loss.item()}"
    assert set(loss_dict.keys()) == {"standard_nll", "gp_nll", "total_loss"}
