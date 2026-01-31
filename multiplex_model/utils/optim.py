from functools import partial
from math import cos, pi
from typing import Literal

import torch
from torch.optim.lr_scheduler import LambdaLR


class ClampWithGrad(torch.autograd.Function):
    """Custom autograd function for clamping with smooth gradients."""

    @staticmethod
    def forward(ctx, x, min_val=-15.0, max_val=15.0):
        ctx.save_for_backward(x)
        ctx.min_val, ctx.max_val = min_val, max_val
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        min_val, max_val = ctx.min_val, ctx.max_val

        grad_input = grad_output.clone()
        tanh_x = torch.tanh(x)
        mask = (x <= min_val) | (x >= max_val)

        # Smooth gradient outside the clamp region via tanh'
        smooth_grad = 1 - tanh_x.pow(2)
        grad_input[mask] = smooth_grad[mask] * grad_output[mask]

        # Optional: clip the grad norm to prevent blowup
        torch.nn.utils.clip_grad_norm_([grad_input], max_norm=1.0)

        return grad_input, None, None


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_annealing_steps: int,
    final_lr: float,
    peak_lr: float = 1.0,
    type: Literal["cosine", "linear"] = "cosine",
) -> LambdaLR:
    """Get cosine annealing scheduler with warmup, adapted from:
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104

    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        num_warmup_steps (int): Number of warmup steps
        num_annealing_steps (int): Number of cosine annealing steps
        final_lr (float): Minimum learning rate after annealing
        peak_lr (float): Peak learning rate
        type (Literal['cosine', 'linear']): Type of annealing schedule

    Returns:
        LambdaLR: Scheduler
    """
    final_lr_mult = final_lr / peak_lr

    def lr_lambda(current_step, type: Literal["cosine", "linear"] = "cosine"):
        if current_step < num_warmup_steps:
            return float(max(1, current_step)) / float(max(1, num_warmup_steps))
        elif current_step >= num_annealing_steps + num_warmup_steps:
            return final_lr_mult

        progress = (current_step - num_warmup_steps) / float(max(1, num_annealing_steps - num_warmup_steps))

        if type == "linear":
            return max(final_lr_mult, (1.0 - progress) * (1.0 - final_lr_mult) + final_lr_mult)

        return final_lr_mult + (1.0 - final_lr_mult) * 0.5 * (1.0 + cos(pi * progress))

    lr_lambda = partial(lr_lambda, type=type)
    return LambdaLR(optimizer, lr_lambda, -1)
