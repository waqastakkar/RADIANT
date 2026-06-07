"""Loss functions for chem pre-training and downstream tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedLMLoss(nn.Module):
    """Cross-entropy at masked positions only.

    Expects ``logits: (B, S, V)`` and ``labels: (B, S)`` where labels are
    ``-100`` at non-masked positions (the standard ignore sentinel).
    """

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )


class RegressionLoss(nn.Module):
    """Regression loss for property prediction.

    ``kind="mse"`` preserves the original behavior. ``kind="huber"`` is
    often better for pChEMBL fine-tuning because curated activity tables still
    contain assay noise and occasional label outliers; it optimizes like MSE
    near the optimum but stops very large residuals from dominating updates.
    Set ``log_scale=True`` for IC50-like targets that are heavy-tailed -- the
    loss is then computed on ``log10(target + 1)``.
    """

    def __init__(
        self,
        *,
        log_scale: bool = False,
        kind: str = "mse",
        huber_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.log_scale = log_scale
        if kind not in {"mse", "huber", "smooth_l1"}:
            raise ValueError(f"unknown regression loss kind {kind!r}")
        if huber_beta <= 0:
            raise ValueError("huber_beta must be > 0")
        self.kind = kind
        self.huber_beta = huber_beta

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.log_scale:
            predictions = torch.log10(predictions.clamp_min(0) + 1.0)
            targets = torch.log10(targets.clamp_min(0) + 1.0)
        if self.kind == "mse":
            return F.mse_loss(predictions, targets)
        return F.smooth_l1_loss(predictions, targets, beta=self.huber_beta)


class ClassificationLoss(nn.Module):
    """Cross-entropy over class logits (multi-class)."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)


class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE over two batches of pooled embeddings.

    Inputs are ``(B, D)`` representations of the same molecules tokenized
    with two different SMILES randomizations. Positive pairs are the
    diagonal; negatives are all other batch entries. Returns the average of
    the two directional losses.
    """

    def __init__(self, *, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(view_a, dim=-1)
        b = F.normalize(view_b, dim=-1)
        logits_ab = (a @ b.T) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ab.T, labels)
        return 0.5 * (loss_ab + loss_ba)
