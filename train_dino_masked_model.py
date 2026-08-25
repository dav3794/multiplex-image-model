"""Multi-crop DINO and decoder-level iBOT self-distillation training.

The EMA teacher sees one clean global crop. The student sees masked global views for
DINO and iBOT, plus clean local views for DINO. iBOT supervises masked decoder
representations before the frozen pixel projection layer.
"""

import math
import os
import sys

import comet_ml  # noqa: F401
import torch
import torch.optim as optim
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
)
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from multiplex_model.data import MultiplexDataset, PanelBatchSampler
from multiplex_model.losses import DINOLoss, IBOTLoss, KoLeoLoss
from multiplex_model.modules import MultiplexAutoencoder
from multiplex_model.utils import (
    DINOTrainingConfig,
    apply_channel_masking,
    apply_spatial_masking,
    cosine_schedule_at,
    finish_experiment,
    get_run_name,
    get_scheduler_with_warmup,
    init_experiment,
    log_training_metrics,
)

@torch.no_grad()
def update_teacher(student: torch.nn.Module, teacher: torch.nn.Module, momentum: float):
    """EMA-update the teacher parameters from the student and copy buffers."""
    student_parameters = list(student.parameters())
    teacher_parameters = list(teacher.parameters())
    torch._foreach_mul_(teacher_parameters, momentum)
    torch._foreach_add_(
        teacher_parameters, student_parameters, alpha=1.0 - momentum
    )
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)


def teacher_temp_at(
    epoch: int, warmup_teacher_temp: float, teacher_temp: float, warmup_epochs: int
) -> float:
    """Linearly warm up the teacher temperature over ``warmup_epochs``."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return teacher_temp
    frac = epoch / warmup_epochs
    return warmup_teacher_temp + frac * (teacher_temp - warmup_teacher_temp)


def momentum_at(step: int, total_steps: int, base_momentum: float) -> float:
    """Cosine schedule increasing the teacher momentum from base to 1.0."""
    if total_steps <= 0:
        return base_momentum
    progress = min(step / total_steps, 1.0)
    return base_momentum + (1.0 - base_momentum) * 0.5 * (
        1.0 - math.cos(math.pi * progress)
    )


def random_local_affine(img: torch.Tensor) -> torch.Tensor:
    """Apply an independent random rotation and reflection to a local crop."""
    angle = torch.empty(()).uniform_(-180.0, 180.0).item()
    img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
    if torch.rand(()).item() < 0.5:
        img = TF.hflip(img)
    if torch.rand(()).item() < 0.5:
        img = TF.vflip(img)
    return img


def build_ibot_mask(
    clean_ids: torch.Tensor,
    active_ids: torch.Tensor,
    spatial_mask: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Combine fully masked channels and per-channel spatial masks."""
    channel_matches = clean_ids.unsqueeze(2) == active_ids.unsqueeze(1)
    fully_masked_channels = ~channel_matches.any(dim=2)
    aligned_spatial_mask = (
        channel_matches.unsqueeze(-1).unsqueeze(-1)
        & spatial_mask.unsqueeze(1)
    ).any(dim=2)
    mask = aligned_spatial_mask | fully_masked_channels.unsqueeze(-1).unsqueeze(-1)
    return F.interpolate(mask.float(), size=output_size, mode="nearest").bool()


def build_views(
    img,
    channel_ids,
    num_global_views,
    num_local_views,
    local_crop_size,
    min_channels_frac,
    fully_masked_channels_max_frac,
    spatial_masking_ratio,
    mask_patch_size,
):
    """Build masked global and clean local student views of a clean teacher crop."""
    global_h, global_w = img.shape[-2:]
    local_h, local_w = local_crop_size
    if local_h > global_h or local_w > global_w:
        raise ValueError("local_crop_size must fit inside global_crop_size")

    repeated_img = img.repeat(num_global_views, 1, 1, 1)
    repeated_ids = channel_ids.repeat(num_global_views, 1)
    _, clean_ids, masked_img, active_ids = apply_channel_masking(
        repeated_img,
        repeated_ids,
        min_channels_frac,
        fully_masked_channels_max_frac,
        apply_channel_subset_sampling=True,
    )
    masked_img, spatial_mask = apply_spatial_masking(
        masked_img, spatial_masking_ratio, mask_patch_size
    )
    global_views = [
        {
            "clean_ids": view_clean_ids,
            "img": view_img,
            "active_ids": view_active_ids,
            "spatial_mask": view_spatial_mask,
        }
        for view_clean_ids, view_img, view_active_ids, view_spatial_mask in zip(
            clean_ids.chunk(num_global_views),
            masked_img.chunk(num_global_views),
            active_ids.chunk(num_global_views),
            spatial_mask.chunk(num_global_views),
        )
    ]

    local_views = []
    for _ in range(num_local_views):
        top = torch.randint(global_h - local_h + 1, ()).item()
        left = torch.randint(global_w - local_w + 1, ()).item()
        local_img = img[..., top : top + local_h, left : left + local_w]
        local_img = random_local_affine(local_img)
        local_views.append(
            {
                "img": local_img,
                "active_ids": channel_ids,
                "top": top,
                "left": left,
            }
        )
    return img, channel_ids, global_views, local_views


def train_dino(
    model,
    teacher,
    optimizer,
    scheduler,
    dino_loss,
    ibot_loss,
    koleo_loss,
    train_dataloader,
    device,
    config,
    start_epoch=0,
):
    """Train pooled encoder and masked decoder representations by distillation."""
    model.train()
    teacher.eval()
    scaler = GradScaler(enabled=False)
    run_name = get_run_name()

    checkpoints_path = config.checkpoints_dir
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"Created checkpoints directory at {checkpoints_path}")

    grad_accum = config.gradient_accumulation_steps
    total_steps = len(train_dataloader) * config.epochs // grad_accum
    step = start_epoch * (len(train_dataloader) // grad_accum)

    last_layers = [
        model.encoder.dino_head.last_layer,
        model.decoder.ibot_head.last_layer,
    ]

    for epoch in range(start_epoch, config.epochs):
        model.train()
        cur_teacher_temp = teacher_temp_at(
            epoch,
            config.warmup_teacher_temp,
            config.teacher_temp,
            config.warmup_teacher_temp_epochs,
        )
        for batch_idx, (img, channel_ids, panel_idx, img_path) in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            img = img.to(device, dtype=torch.float32, non_blocking=True)
            channel_ids = channel_ids.to(
                device, dtype=torch.long, non_blocking=True
            )

            clean_img, clean_ids, global_views, local_views = build_views(
                img,
                channel_ids,
                config.num_global_views,
                config.num_local_views,
                config.local_crop_size,
                config.min_channels_frac,
                config.fully_masked_channels_max_frac,
                config.spatial_masking_ratio,
                config.mask_patch_size,
            )

            student_cls = []
            student_ibot = []
            ibot_masks = []
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_size = img.shape[0]
                global_encoded = model.encode(
                    torch.cat([view["img"] for view in global_views]),
                    torch.cat([view["active_ids"] for view in global_views]),
                    return_dino_proj=True,
                )
                global_features = model.decoder.forward_features(
                    global_encoded["output"],
                    torch.cat([view["clean_ids"] for view in global_views]),
                )
                student_cls.extend(global_encoded["cls"].split(batch_size))
                for view, features in zip(
                    global_views, global_features.split(batch_size)
                ):
                    mask = build_ibot_mask(
                        view["clean_ids"],
                        view["active_ids"],
                        view["spatial_mask"],
                        features.shape[-2:],
                    )
                    ibot_masks.append(mask)

                combined_ibot_mask = torch.cat(ibot_masks)
                student_ibot = model.decoder.project_ibot(
                    global_features,
                    combined_ibot_mask,
                )
                ibot_token_counts = (
                    torch.stack([mask.sum() for mask in ibot_masks])
                    if len(ibot_masks) > 1
                    else None
                )

                if local_views:
                    local_encoded = model.encode(
                        torch.cat([view["img"] for view in local_views]),
                        torch.cat([view["active_ids"] for view in local_views]),
                        return_dino_proj=True,
                    )
                    student_cls.extend(local_encoded["cls"].split(batch_size))

                with torch.no_grad():
                    teacher_encoded = teacher.encode(
                        clean_img, clean_ids, return_dino_proj=True
                    )
                    teacher_cls = [teacher_encoded["cls"]]
                    teacher_features = teacher.decoder.forward_features(
                        teacher_encoded["output"].repeat(
                            config.num_global_views, 1, 1, 1
                        ),
                        torch.cat([view["clean_ids"] for view in global_views]),
                    )
                    teacher_ibot = teacher.decoder.project_ibot(
                        teacher_features,
                        combined_ibot_mask,
                    )

            dino = dino_loss(student_cls, teacher_cls, cur_teacher_temp)
            ibot = ibot_loss(
                student_ibot,
                teacher_ibot,
                cur_teacher_temp,
                view_token_counts=ibot_token_counts,
            )
            koleo = koleo_loss(torch.stack(student_cls, dim=0))
            loss = (
                config.dino_loss_weight * dino
                + config.ibot_loss_weight * ibot
                + config.koleo_loss_weight * koleo
            )

            scaler.scale(loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                if epoch < config.freeze_last_layer_epochs:
                    for layer in last_layers:
                        for p in layer.parameters():
                            if p.grad is not None:
                                p.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                weight_decay = cosine_schedule_at(
                    step,
                    total_steps,
                    config.weight_decay,
                    config.final_weight_decay,
                )
                for param_group in optimizer.param_groups:
                    param_group["weight_decay"] = weight_decay
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                momentum = momentum_at(step, total_steps, config.momentum_teacher)
                update_teacher(model, teacher, momentum)

                log_training_metrics(
                    loss=loss.item(),
                    lr=scheduler.get_last_lr()[0],
                    step=step,
                    extra_metrics={
                        "dino": dino.item(),
                        "ibot": ibot.item(),
                        "koleo": koleo.item(),
                        "teacher_momentum": momentum,
                        "weight_decay": weight_decay,
                    },
                )
                if step % 50 == 0:
                    print(
                        f"[step {step}] total={loss.item():.4f} dino={dino.item():.4f} "
                        f"ibot={ibot.item():.4f} koleo={koleo.item():.4f} m={momentum:.4f}"
                    )
                step += 1

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
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "epoch": config.epochs - 1,
    }
    if hasattr(model, "get_architecture_config"):
        checkpoint["model_config"] = model.get_architecture_config()
    torch.save(checkpoint, final_model_path)


if __name__ == "__main__":
    config_path = sys.argv[1]
    yaml = YAML(typ="safe")
    with open(config_path, "r") as f:
        raw_config = yaml.load(f)

    config = DINOTrainingConfig(**raw_config)

    if not config.encoder_config.dino_head:
        raise ValueError(
            "DINO training requires encoder.dino_head=True in the configuration."
        )
    if not config.decoder_config.ibot_head:
        raise ValueError(
            "DINO training requires decoder.ibot_head=True in the configuration."
        )

    device = config.device
    print(f"Using device: {device}")
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = False

    SIZE = config.global_crop_size
    BATCH_SIZE = config.batch_size
    NUM_WORKERS = config.num_workers

    PANEL_CONFIG = config.panel_config
    TOKENIZER = config.tokenizer_config

    train_transform = Compose(
        [
            RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
            RandomCrop(SIZE),
            RandomHorizontalFlip(),
        ]
    )
    dataset_kwargs = config.data_config.model_dump()

    train_dataset = MultiplexDataset(
        panels_config=PANEL_CONFIG,
        split="train",
        marker_tokenizer=TOKENIZER,
        transform=train_transform,
        skip_too_small=True,
        min_image_size=SIZE,
        **dataset_kwargs,
    )
    train_batch_sampler = PanelBatchSampler(train_dataset, BATCH_SIZE)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
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

    start_epoch = 0
    checkpoint = None
    if config.resolve_checkpoint():
        print(f"Loading model from checkpoint: {config.from_checkpoint}")
        checkpoint = torch.load(config.from_checkpoint, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain 'model_state_dict'.")
        model = MultiplexAutoencoder.load_from_checkpoint(
            checkpoint,
            model_config=model_config,
        ).to(device)
        start_epoch = checkpoint.get("epoch", -1) + 1
    else:
        model = MultiplexAutoencoder(**model_config).to(device)
    model.decoder.pred.requires_grad_(False)

    # EMA teacher: a frozen copy of the student updated via momentum.
    # Rebuilt from config instead of deepcopy, since weight_norm creates non-leaf
    # tensors that cannot be deepcopied (see pytorch/pytorch#103001).
    teacher = MultiplexAutoencoder(**model_config).to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    if checkpoint is not None and "teacher_state_dict" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher_state_dict"])
    if config.compile_encoder:
        print("Compiling student and teacher encoders...")
        model.encoder.compile(dynamic=True)
        teacher.encoder.compile(dynamic=True)

    dino_out_dim = model.encoder.dino_head.last_layer.weight_g.shape[0]
    dino_loss = DINOLoss(
        out_dim=dino_out_dim,
        student_temp=config.student_temp,
        center_momentum=config.center_momentum,
    ).to(device)
    ibot_out_dim = model.decoder.ibot_head.last_layer.weight_g.shape[0]
    ibot_loss = IBOTLoss(
        out_dim=ibot_out_dim,
        student_temp=config.student_temp,
        center_momentum=config.center_momentum,
    ).to(device)
    koleo_loss = KoLeoLoss().to(device)

    total_steps = (
        len(train_dataloader) * config.epochs // config.gradient_accumulation_steps
    )
    num_warmup_steps = int(total_steps * config.frac_warmup_steps)
    num_annealing_steps = total_steps - num_warmup_steps

    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.peak_lr,
        weight_decay=config.weight_decay,
        fused=str(device).startswith("cuda"),
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

    train_dino(
        model,
        teacher,
        optimizer,
        scheduler,
        dino_loss,
        ibot_loss,
        koleo_loss,
        train_dataloader,
        device,
        config=config,
        start_epoch=start_epoch,
    )

    finish_experiment()
