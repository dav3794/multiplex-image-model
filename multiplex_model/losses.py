import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(x, mi, logvar):
    return torch.mean((x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar)


def beta_nll_loss(x, mi, logvar, beta=1.0):
    sg_var_beta = logvar.detach().exp().pow(beta)
    nll = (x - mi) ** 2 / (torch.exp(logvar) + 1e-8) + logvar
    beta_nll = sg_var_beta * nll
    return torch.mean(beta_nll)


def RankMe(features):
    U, S, V = torch.linalg.svd(features)
    p = S / (S.sum() + 1e-7)
    entropy = -torch.sum(p * torch.log(p + 1e-7))
    rank_me = torch.exp(entropy)
    return rank_me


class DINOLoss(nn.Module):
    """DINO self-distillation loss with teacher output centering."""

    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        """Initialize the DINO loss.

        Args:
            out_dim (int): Number of prototypes (dimension of the head output).
            student_temp (float, optional): Softmax temperature for student outputs. Defaults to 0.1.
            center_momentum (float, optional): EMA momentum for the teacher output center. Defaults to 0.9.
        """
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_outputs: list[torch.Tensor],
        teacher_outputs: list[torch.Tensor],
        teacher_temp: float,
    ) -> torch.Tensor:
        """Compute the cross-view distillation loss.

        Args:
            student_outputs (list[Tensor]): List of student prototype logits, each [B, out_dim].
            teacher_outputs (list[Tensor]): List of teacher prototype logits, each [B, out_dim].
            teacher_temp (float): Current teacher temperature (usually warmed up over epochs).

        Returns:
            torch.Tensor: Scalar loss.
        """
        student = torch.stack(student_outputs, dim=0).float()  # [Vs, B, K]
        teacher = torch.stack(teacher_outputs, dim=0).float()  # [Vt, B, K]

        student_log = F.log_softmax(student / self.student_temp, dim=-1)
        teacher_prob = F.softmax(
            (teacher - self.center) / teacher_temp, dim=-1
        ).detach()

        # Cross-entropy over all teacher x student pairs (including diagonal),
        # averaged over pairs and batch.
        loss = -torch.einsum("tbk,sbk->tsb", teacher_prob, student_log).mean()

        self.update_center(teacher.reshape(-1, teacher.shape[-1]))
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """Update the teacher output center with an EMA of the batch mean."""
        batch_center = teacher_output.float().mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko differential-entropy regularizer encouraging uniform feature spread."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the KoLeo loss on a batch (or stack of batches) of features.

        Args:
            x (torch.Tensor): Features of shape [B, D] or a stack of views [V, B, D].
                Nearest neighbors are always computed within each [B, D] batch.

        Returns:
            torch.Tensor: Scalar loss.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)  # [1, B, D]

        x = F.normalize(x.float(), p=2, dim=-1, eps=self.eps)
        dots = torch.bmm(x, x.transpose(1, 2))  # [V, B, B]
        b = x.shape[1]
        diag = torch.eye(b, dtype=torch.bool, device=x.device)
        dots.masked_fill_(diag.unsqueeze(0), -2)  # mask self-similarity per view
        nearest = dots.max(dim=-1).values  # [V, B]
        distances = (2 - 2 * nearest).clamp(min=self.eps).sqrt()
        return -torch.log(distances + self.eps).mean()

