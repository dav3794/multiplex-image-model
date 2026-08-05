import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.optim import ClampWithGrad


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


class IBOTvMFLoss(nn.Module):
    """Hyperspherical von Mises-Fisher (vMF) iBOT loss with the beta-NLL stop-gradient trick.

    Replaces the discrete-prototype iBOT cross-entropy with a continuous directional
    negative log-likelihood on the unit hypersphere. The student predicts, per position,
    a mean direction ``mu`` (L2-normalized) and a concentration ``kappa`` (inverse
    variance); the EMA teacher provides a deterministic centered + L2-normalized target
    direction ``z_t``. The loss for each position is::

        L = -sg[kappa^{-beta}] * kappa * <mu, z_t> + log C_D(kappa)

    where ``log C_D(kappa) ~= kappa - (D - 1)/2 * log(kappa)`` is the high-dimensional
    asymptotic vMF log-normalizer (true Bessel functions overflow for large D). The
    ``sg[kappa^{-beta}]`` factor is the beta-NLL trick that detaches the uncertainty scale
    from the matching gradient, preventing concentration/uncertainty collapse.

    The student predicts ``log kappa`` directly (last scalar), so ``kappa = exp(log kappa)``
    stays positive without a softplus and its log is available for the normalizer for free.

    Works identically for the pooled case (inputs ``[B, D(+1)]``) and the dense per-cell
    case (inputs ``[B, N, D(+1)]``); the leading dimensions are flattened internally.
    """

    def __init__(
        self,
        out_dim: int,
        beta: float = 0.5,
        center_momentum: float = 0.9,
    ):
        """Initialize the vMF iBOT loss.

        Args:
            out_dim (int): Dimensionality D of the vMF projection space (hypersphere S^{D-1}).
            beta (float, optional): Beta-NLL exponent detaching the uncertainty scale from
                the matching gradient. Defaults to 0.5.
            center_momentum (float, optional): EMA momentum for the teacher centering
                vector (anti-collapse). Defaults to 0.9.
        """
        super().__init__()
        self.D = out_dim
        self.beta = beta
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_features: torch.Tensor) -> None:
        """EMA-update the teacher feature centroid from a flattened batch ``[N, D]``."""
        batch_center = teacher_features.float().mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )

    def log_vmf_normalizer(
        self, kappa: torch.Tensor, log_kappa: torch.Tensor
    ) -> torch.Tensor:
        """High-dimensional asymptotic approximation of ``log C_D(kappa)``."""
        return kappa - ((self.D - 1) / 2.0) * log_kappa

    def forward(
        self,
        student_preds: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the vMF iBOT loss.

        Args:
            student_preds (torch.Tensor): Student predictor output ``[..., D + 1]``
                (unnormalized mean direction followed by the raw log-concentration scalar).
            teacher_features (torch.Tensor): Teacher continuous projection ``[..., D]``.
            mask (torch.Tensor, optional): Boolean tensor with the same leading shape
                selecting the positions to score (e.g. masked cells). If None, all
                positions are used. Defaults to None.

        Returns:
            torch.Tensor: Scalar loss.
        """
        student_preds = student_preds.reshape(-1, student_preds.shape[-1])
        teacher_features = teacher_features.reshape(-1, teacher_features.shape[-1])
        if mask is not None:
            mask = mask.reshape(-1)
            student_preds = student_preds[mask]
            teacher_features = teacher_features[mask]

        with torch.no_grad():
            self.update_center(teacher_features)
            z_t = F.normalize(
                teacher_features.float() - self.center, p=2, dim=-1
            )

        mu_raw = student_preds[..., : self.D].float()
        log_kappa = student_preds[..., self.D :].float().squeeze(-1)
        mu_s = F.normalize(mu_raw, p=2, dim=-1)

        log_kappa = ClampWithGrad.apply(log_kappa, -15.0, 15.0)
        kappa = torch.exp(log_kappa)

        cos_sim = (mu_s * z_t).sum(dim=-1)
        log_norm = self.log_vmf_normalizer(kappa, log_kappa)
        beta_scale = kappa.detach().pow(-self.beta)  # sg[kappa^{-beta}]
        loss = -beta_scale * kappa * cos_sim + log_norm
        return loss.mean()


class VMFKoLeoLoss(nn.Module):
    """KoLeo differential-entropy regularizer adapted for the continuous vMF objective.

    Same nearest-neighbour differential-entropy estimator as ``KoLeoLoss`` (maximizing
    the distance to each sample's nearest neighbour on the unit hypersphere), but applied
    to the continuous vMF mean directions and with optional subsampling so it can also be
    used on dense per-cell token sets without an O(N^2) blow-up.
    """

    def __init__(self, eps: float = 1e-8, max_samples: int | None = None):
        """Initialize the modified KoLeo loss.

        Args:
            eps (float, optional): Numerical-stability epsilon. Defaults to 1e-8.
            max_samples (int, optional): If set, randomly subsample this many rows from the
                per-view batch dimension before computing nearest neighbours. Defaults to None.
        """
        super().__init__()
        self.eps = eps
        self.max_samples = max_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the modified KoLeo loss.

        Args:
            x (torch.Tensor): Features of shape [B, D] or a stack of views [V, B, D].
                Nearest neighbours are computed within each [B, D] slice.

        Returns:
            torch.Tensor: Scalar loss.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)  # [1, B, D]

        x = F.normalize(x.float(), p=2, dim=-1, eps=self.eps)

        b = x.shape[1]
        if self.max_samples is not None and b > self.max_samples:
            idx = torch.randperm(b, device=x.device)[: self.max_samples]
            x = x[:, idx, :]
            b = self.max_samples

        dots = torch.bmm(x, x.transpose(1, 2))  # [V, B, B]
        diag = torch.eye(b, dtype=torch.bool, device=x.device)
        dots.masked_fill_(diag.unsqueeze(0), -2)  # mask self-similarity per view
        nearest = dots.max(dim=-1).values  # [V, B]
        distances = (2 - 2 * nearest).clamp(min=self.eps).sqrt()
        return -torch.log(distances + self.eps).mean()
