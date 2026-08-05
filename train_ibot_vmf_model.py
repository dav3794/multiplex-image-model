"""Masked reconstruction + vMF iBOT self-distillation training.

Trains the MultiplexAutoencoder with the masked-reconstruction Beta-NLL objective,
augmented with a hyperspherical von Mises-Fisher (vMF) iBOT self-distillation loss and a
modified KoLeo regularizer. The discrete-prototype DINO loss is intentionally not used.

The vMF iBOT loss can be applied either on the global-average-pooled latent
(``ibot_mode="pooled"``, one target per image, analogous to the DINO loss) or on every
latent feature-map cell (``ibot_mode="dense"``), configurable via the config file.

Key points:
- Student sees channel-masked + spatially-masked views (identical masking logic to
  ``train_masked_model.py``); the EMA teacher sees the clean (channel-subset) views.
- The continuous vMF projection head lives inside the encoder and is EMA-copied to the
  teacher. A separate student-only predictor head produces the vMF parameters
  (mean direction + concentration).
- Reconstruction still uses Beta-NLL.
- Validation reuses ``test_masked`` from ``train_masked_model.py`` unchanged.
"""

import math
import os
import sys

import comet_ml  # noqa: F401
import torch
import torch.optim as optim
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import MultiplexDataset, PanelBatchSampler, TestCrop
from multiplex_model.losses import IBOTvMFLoss, VMFKoLeoLoss, beta_nll_loss
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    ClampWithGrad,
    IBOTTrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
)

# Reuse the exact validation logic from the masked training script.
from train_masked_model import test_masked


@torch.no_grad()
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float):
    """EMA-update the teacher parameters from the student and copy buffers."""
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def momentum_at(step: int, total_steps: int, base_momentum: float) -> float:
    """Cosine schedule increasing the teacher momentum from base to 1.0."""
    if total_steps <= 0:
        return base_momentum
    progress = min(step / total_steps, 1.0)
    return base_momentum + (1.0 - base_momentum) * 0.5 * (
        1.0 - math.cos(math.pi * progress)
    )


def build_views(
    img,
    channel_ids,
    num_views,
    min_channels_frac,
    fully_masked_channels_max_frac,
    spatial_masking_ratio,
    mask_patch_size,
):
    """Build ``num_views`` independently masked views for student/teacher.

    Returns a list of dicts with the clean (channel-subset) image and ids for the
    teacher, and the channel-masked + spatially-masked image and active ids for the
    student.
    """
    views = []
    for _ in range(num_views):
        clean_img, clean_ids, masked_img, active_ids = apply_channel_masking(
            img,
            channel_ids,
            min_channels_frac,
            fully_masked_channels_max_frac,
            apply_channel_subset_sampling=True,
        )
        masked_img, _ = apply_spatial_masking(
            masked_img, spatial_masking_ratio, mask_patch_size
        )
        views.append(
            {
                "clean_img": clean_img,
                "clean_ids": clean_ids,
                "masked_img": masked_img,
                "active_ids": active_ids,
            }
        )
    return views


def train_ibot(
    model,
    teacher,
    optimizer,
    scheduler,
    ibot_loss,
    koleo_loss,
    train_dataloader,
    val_dataloader,
    device,
    marker_names_map,
    config,
    start_epoch=0,
):
    """Train the model with masked reconstruction + vMF iBOT self-distillation."""
    model.train()
    teacher.eval()
    scaler = GradScaler()
    run_name = get_run_name()

    checkpoints_path = config.checkpoints_dir
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    grad_accum = config.gradient_accumulation_steps
    num_views = config.num_global_views
    dense = config.encoder_config.ibot_mode == "dense"
    pred_key = "ibot_patch_pred" if dense else "ibot_cls_pred"
    target_key = "ibot_patch" if dense else "ibot_cls"
    total_steps = len(train_dataloader) * config.epochs // grad_accum
    step = start_epoch * (len(train_dataloader) // grad_accum)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32)
            channel_ids = channel_ids.to(device, dtype=torch.long)

            views = build_views(
                img,
                channel_ids,
                num_views,
                config.min_channels_frac,
                config.fully_masked_channels_max_frac,
                config.spatial_masking_ratio,
                config.mask_patch_size,
            )

            student_preds = []
            recon_loss = 0.0
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                for view in views:
                    out = model(
                        view["masked_img"],
                        view["active_ids"],
                        view["clean_ids"],
                        return_ibot_pred=True,
                    )
                    mi, logvar = out["output"].unbind(dim=-1)
                    mi = torch.sigmoid(mi)
                    logvar = ClampWithGrad.apply(logvar, -15.0, 15.0)
                    recon_loss = recon_loss + beta_nll_loss(
                        view["clean_img"], mi, logvar, beta=config.beta
                    )
                    # Student vMF parameters (mean direction + concentration) from the
                    # in-model predictor head.
                    student_preds.append(out[pred_key])
                recon_loss = recon_loss / num_views

                with torch.no_grad():
                    teacher_targets = [
                        teacher.encode(
                            view["clean_img"],
                            view["clean_ids"],
                            return_ibot_proj=True,
                        )[target_key]
                        for view in views
                    ]

            ibot = sum(
                ibot_loss(student_preds[i], teacher_targets[i])
                for i in range(num_views)
            ) / num_views
            # KoLeo regularizes the student's predicted mean directions (mu), i.e. the
            # predictor output without its trailing concentration scalar.
            koleo_means = [p[..., :-1] for p in student_preds]
            if dense:
                # Flatten per-cell tokens into the sample dimension; VMFKoLeoLoss
                # subsamples via max_samples to bound the O(N^2) neighbour search.
                koleo_means = [m.reshape(-1, m.shape[-1]) for m in koleo_means]
            koleo_means = torch.stack(koleo_means, dim=0)
            koleo = koleo_loss(koleo_means)
            # Mean predicted log-concentration (last predictor output) for monitoring.
            mean_log_kappa = torch.stack(
                [p[..., -1].mean() for p in student_preds]
            ).mean()
            loss = (
                config.recon_loss_weight * recon_loss
                + config.ibot_loss_weight * ibot
                + config.koleo_loss_weight * koleo
            )

            scaler.scale(loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                momentum = momentum_at(step, total_steps, config.momentum_teacher)
                update_teacher(model, teacher, momentum)

                log_training_metrics(
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0],
                    mu=mi.mean().item(),
                    logvar=logvar.mean().item(),
                    mae=torch.abs(view["clean_img"] - mi).mean().item(),
                    mse=torch.square(view["clean_img"] - mi).mean().item(),
                    step=step,
                    extra_metrics={
                        "recon": recon_loss.item(),
                        "ibot": ibot.item(),
                        "koleo": koleo.item(),
                        "mean_log_kappa": mean_log_kappa.item(),
                        "teacher_momentum": momentum,
                    },
                )
                if step % 50 == 0:
                    print(
                        f"[step {step}] total={loss.item():.4f} recon={recon_loss.item():.4f} "
                        f"ibot={ibot.item():.4f} koleo={koleo.item():.4f} m={momentum:.4f}"
                    )
                step += 1

        test_masked(
            model,
            val_dataloader,
            device,
            epoch,
            spatial_masking_ratio=config.spatial_masking_ratio,
            fully_masked_channels_max_frac=config.fully_masked_channels_max_frac,
            mask_patch_size=config.mask_patch_size,
            marker_names_map=marker_names_map,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        }
        if hasattr(model, "get_architecture_config"):
            checkpoint["model_config"] = model.get_architecture_config()
        if (epoch + 1) % config.save_checkpoint_freq == 0:
            torch.save(
                checkpoint,
                f"{checkpoints_path}/checkpoint-{run_name}-epoch_{epoch}.pth",
            )
        torch.save(checkpoint, f"{checkpoints_path}/last_checkpoint-{run_name}.pth")

    final_model_path = f"{checkpoints_path}/final_model-{run_name}.pth"
    print(f"Training completed. Saving final model at {final_model_path}...")
    checkpoint = {"model_state_dict": model.state_dict()}
    if hasattr(model, "get_architecture_config"):
        checkpoint["model_config"] = model.get_architecture_config()
    torch.save(checkpoint, final_model_path)


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    config = IBOTTrainingConfig(**raw_config)

    if not config.encoder_config.ibot_head:
        raise ValueError(
            "vMF iBOT training requires encoder.ibot_head=True in the configuration."
        )

    device = config.device
    print(f"Using device: {device}")

    SIZE = config.input_image_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = config.panel_config
    TOKENIZER = config.tokenizer_config
    INV_TOKENIZER = {v: k for k, v in TOKENIZER.items()}

    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(SIZE),
            RandomHorizontalFlip(),
        ]
    )
    test_transform = TestCrop(SIZE[0])

    dataset_kwargs = config.data_config.model_dump()

    train_dataset = MultiplexDataset(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        **dataset_kwargs,
    )
    test_dataset = MultiplexDataset(
        panels_config=PANEL_CONFIG,
        split="test",
        marker_tokenizer=TOKENIZER,
        transform=test_transform,
        **dataset_kwargs,
    )

    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)
    test_batch_sampler = PanelBatchSampler(test_dataset, BATCH_SIZE, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    num_channels = len(TOKENIZER)
    model_config = {
        "num_channels": num_channels,
        "encoder_config": config.encoder_config.model_dump(),
        "decoder_config": config.decoder_config.model_dump(),
        "share_hyperkernel_coeff": config.share_hyperkernel_coeff,
    }

    ibot_head_cfg = config.encoder_config.ibot_head_config
    ibot_out_dim = ibot_head_cfg.out_dim if ibot_head_cfg is not None else 256

    start_epoch = 0
    checkpoint = None
    if config.resolve_checkpoint():
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        model = MultiplexAutoencoder.load_from_checkpoint(
            checkpoint,
            model_config=model_config,
        ).to(device)
        start_epoch = checkpoint.get("epoch", -1) + 1
    else:
        model = MultiplexAutoencoder(**model_config).to(device)

    # EMA teacher: a frozen copy of the student updated via momentum.
    teacher = MultiplexAutoencoder(**model_config).to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    if checkpoint is not None and "teacher_state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher_state_dict"])

    ibot_loss = IBOTvMFLoss(
        out_dim=ibot_out_dim,
        beta=config.ibot_beta,
        center_momentum=config.center_momentum,
    ).to(device)
    koleo_loss = VMFKoLeoLoss(max_samples=config.koleo_max_samples).to(device)

    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_warmup_steps,
        num_annealing_steps,
        final_lr=config.final_lr,
        peak_lr=config.peak_lr,
        type="cosine",
    )

    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    comet_config = config.model_dump()
    init_experiment(comet_config)

    train_ibot(
        model,
        teacher,
        optimizer,
        scheduler,
        ibot_loss,
        koleo_loss,
        train_dataloader,
        test_dataloader,
        device,
        marker_names_map=INV_TOKENIZER,
        config=config,
        start_epoch=start_epoch,
    )

    finish_experiment()
