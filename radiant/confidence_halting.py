"""ConfidenceHalting: per-token continue/stop confidence head.

The head reads each token's hidden state at the end of every loop step and
emits a confidence in ``(0, 1)``. Two modes:

Training (default)
    The model unrolls a *fixed* number of loops; the confidence trace is
    observed but not enforced. This keeps gradients clean and the loss
    surface simple. A future, optional ponder-style regularizer can slot in
    on top via the returned trace.

Inference / dynamic-loop
    Cumulative confidence is tracked per token. Once a token's cumulative
    confidence exceeds ``confidence_threshold``, that token is considered
    "halted" -- its hidden state is frozen for subsequent steps. The whole
    forward returns the average compute depth (mean halt step + 1) for
    monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.norms import RMSNorm


@dataclass
class HaltingTrace:
    """Per-loop confidence + per-token halt step.

    ``confidences`` is a list of ``(B, S)`` tensors (one per loop step actually
    executed). ``halt_step`` is ``(B, S)`` integer tensor giving the loop
    index at which each token first exceeded the threshold (or
    ``n_loops_executed - 1`` if it never did). ``avg_depth`` is a Python float.
    """

    confidences: list[torch.Tensor] = field(default_factory=list)
    halt_step: torch.Tensor | None = None
    avg_depth: float | None = None

    def append(self, conf: torch.Tensor) -> None:
        self.confidences.append(conf)

    def finalize(self, threshold: float) -> None:
        """Compute ``halt_step`` and ``avg_depth`` from the collected confidences."""
        if not self.confidences:
            return
        # Stack to (T, B, S) and accumulate cumulative confidence (independent halt
        # decisions per token; we use cumulative to make the threshold comparable
        # across loop counts).
        stacked = torch.stack(self.confidences, dim=0)         # (T, B, S)
        cum = stacked.cumsum(dim=0)                             # (T, B, S)
        T = stacked.size(0)
        crossed = cum >= threshold                              # bool (T, B, S)
        # First True along T, or T-1 if never True.
        any_crossed = crossed.any(dim=0)                        # (B, S)
        first = crossed.float().argmax(dim=0)                   # (B, S), 0 if all False
        # Where never crossed, set to T-1.
        halt = torch.where(any_crossed, first, torch.full_like(first, T - 1))
        self.halt_step = halt
        # avg compute depth = mean halt index + 1 (1-based)
        self.avg_depth = float((halt.float() + 1.0).mean().item())


class ConfidenceHalting(nn.Module):
    """A single linear head over per-token hidden states producing confidence in (0,1)."""

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.head = nn.Linear(cfg.d_model, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, cfg.confidence_init_bias)
        self.threshold = cfg.confidence_threshold

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h: (B, S, D) -> (B, S)`` confidence in ``(0, 1)``."""
        return torch.sigmoid(self.head(self.norm(h))).squeeze(-1)


# ---------------------------------------------------------------------------
# PonderNet-style auxiliary loss
# ---------------------------------------------------------------------------

def halting_kl_loss(
    confidences: list[torch.Tensor],
    attention_mask: torch.Tensor | None,
    *,
    prior_lambda: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """KL divergence between the per-token halt distribution and a geometric prior.

    Following PonderNet (Banino et al., 2021), the per-step confidence
    ``c_t in (0, 1)`` is interpreted as the **conditional** halt
    probability "halt at step t given not yet halted". The unconditional
    halt distribution over steps is then

        p(t) = c_t * prod_{k<t} (1 - c_k)        for t = 0 .. T-2
        p(T-1) = prod_{k<T-1} (1 - c_k)          (mass forced into the last step)

    which sums to 1 by construction. We then compute
    ``KL(p || Geometric(prior_lambda))`` per token and average over valid
    (non-pad) tokens.

    A smaller ``prior_lambda`` shifts the prior toward deeper halting
    (uniform mass; 0.05 in the original paper). 0.2 is a reasonable
    default for our short loop budget (T <= 16): it puts ~70% of mass
    into the first 6 steps, leaving room for the head to push *complex*
    tokens further out.

    Parameters
    ----------
    confidences:
        list of ``(B, S)`` sigmoid outputs in (0, 1), one per executed loop step.
    attention_mask:
        ``(B, S)`` 0/1 mask; pad positions are excluded from the average.
        Pass ``None`` to average over all positions.
    prior_lambda:
        Geometric-prior hyperparameter, in (0, 1).
    eps:
        Numerical floor inside log/clamp.

    Returns
    -------
    Scalar tensor on the same device/dtype as the confidences.
    """
    if not confidences:
        return torch.tensor(0.0)
    T = len(confidences)
    if T < 2:
        # With a single step there is no distribution to regularize.
        return torch.zeros((), device=confidences[0].device, dtype=confidences[0].dtype)

    # (T, B, S) in (eps, 1 - eps); clamp keeps log() finite.
    conf = torch.stack(confidences, dim=0).clamp(eps, 1.0 - eps)
    log_conf = torch.log(conf)
    log_not_conf = torch.log1p(-conf)

    # cum_log_not[t] = sum_{k <= t} log(1 - conf_k)
    cum_log_not = torch.cumsum(log_not_conf, dim=0)
    # Shift right by one so position t holds sum_{k < t} log(1 - conf_k);
    # step 0 has no preceding "not halted" mass.
    shifted = torch.zeros_like(cum_log_not)
    shifted[1:] = cum_log_not[:-1]

    # log p(t) for t = 0 .. T-2: log c_t + sum_{k<t} log(1 - c_k)
    log_p = log_conf + shifted                                  # (T, B, S)
    # log p(T-1): everything that didn't halt earlier; absorb the tail.
    log_p = log_p.clone()
    log_p[-1] = cum_log_not[-2]                                  # (B, S)

    # Tail-absorbed geometric prior on steps 0..T-1: matches the structure
    # of the model distribution above (mass that "would have" gone past
    # T-1 is forced onto step T-1). Sums to 1 exactly by construction:
    #     prior[t]   = lambda * (1-lambda)^t       for t < T-1
    #     prior[T-1] = (1-lambda)^(T-1)
    log_1m_lambda = torch.log1p(torch.tensor(-prior_lambda, device=conf.device, dtype=conf.dtype))
    log_lambda = torch.log(torch.tensor(prior_lambda, device=conf.device, dtype=conf.dtype))
    steps = torch.arange(T, device=conf.device, dtype=conf.dtype)
    log_prior = log_lambda + steps * log_1m_lambda
    log_prior = log_prior.clone()
    log_prior[-1] = (T - 1) * log_1m_lambda                       # (T,)

    # KL(p || prior) = sum_t exp(log_p) * (log_p - log_prior)
    p = torch.exp(log_p)
    kl_per_token = (p * (log_p - log_prior.view(T, 1, 1))).sum(dim=0)   # (B, S)

    if attention_mask is not None:
        mask = attention_mask.bool()
        if not mask.any():
            return torch.zeros((), device=conf.device, dtype=conf.dtype)
        return kl_per_token[mask].mean()
    return kl_per_token.mean()


def _per_step_halt_probabilities(
    confidences: list[torch.Tensor], *, eps: float = 1e-6
) -> torch.Tensor:
    """Return ``(T, B, S)`` per-token unconditional halt-probability mass.

    Uses the same tail-absorbed model distribution as :func:`halting_kl_loss`,
    so the two losses are mutually consistent.
    """
    T = len(confidences)
    conf = torch.stack(confidences, dim=0).clamp(eps, 1.0 - eps)
    if T < 2:
        return conf  # degenerate; the single step takes all the mass

    log_conf = torch.log(conf)
    log_not_conf = torch.log1p(-conf)
    cum_log_not = torch.cumsum(log_not_conf, dim=0)
    shifted = torch.zeros_like(cum_log_not)
    shifted[1:] = cum_log_not[:-1]
    log_p = log_conf + shifted
    log_p = log_p.clone()
    log_p[-1] = cum_log_not[-2]
    return torch.exp(log_p)                                    # (T, B, S)


def pondernet_task_loss(
    per_step_predictions: list[torch.Tensor],
    target: torch.Tensor,
    confidences: list[torch.Tensor],
    *,
    attention_mask: torch.Tensor | None = None,
    task_kind: str = "regression",
) -> torch.Tensor:
    """PonderNet-style task loss: per-step task loss weighted by p_halt(t).

    Reference: Banino et al., "PonderNet: Learning to Ponder" (NeurIPS
    2021), eq. 4. Adapted for the chem regression head whose output is
    *one prediction per molecule* rather than per-token, which is the
    natural case here:

        L_pn = mean_b [ sum_t p_halt_mol(t)[b] * task_loss(y_t[b], y[b]) ]

    where ``p_halt_mol(t)`` is the per-molecule mean of the per-token
    halt distribution (averaging over valid tokens via
    ``attention_mask`` when provided).

    The whole point of this loss is to give the halting head a *task-
    correlated* gradient signal. Without it the head receives only the
    distribution-shape regularizer from :func:`halting_kl_loss` and
    collapses to a single constant scalar; with it the head learns to
    halt early when intermediate predictions are already good and late
    when they are still off.

    Parameters
    ----------
    per_step_predictions:
        ``T`` tensors, each shape ``(B, num_outputs)`` for regression
        (we squeeze a trailing dimension of size 1 internally) or
        ``(B, num_classes)`` for classification.
    target:
        Ground-truth labels: ``(B,)`` or ``(B, num_outputs)`` for
        regression; ``(B,)`` int64 for classification.
    confidences:
        ``T`` confidence tensors of shape ``(B, S)``, same source as
        :func:`halting_kl_loss`.
    attention_mask:
        ``(B, S)`` 0/1 mask used to compute the per-molecule halt
        distribution. ``None`` averages over all positions.
    task_kind:
        ``"regression"`` (MSE) or ``"classification"`` (cross-entropy).
    """
    import torch.nn.functional as F

    if not per_step_predictions or not confidences:
        return torch.tensor(0.0)
    T = len(per_step_predictions)
    if T != len(confidences):
        raise ValueError(
            f"length mismatch: {T} per-step preds vs {len(confidences)} confidence steps"
        )

    p_token = _per_step_halt_probabilities(confidences)            # (T, B, S)
    # Aggregate to per-molecule by averaging the halt-probability mass
    # over valid tokens. This keeps the distribution summed to 1 over t
    # (a convex combination of valid-probability distributions).
    if attention_mask is not None:
        mask = attention_mask.bool().to(p_token.dtype)              # (B, S)
        denom = mask.sum(dim=-1).clamp(min=1.0).unsqueeze(0)        # (1, B)
        p_mol = (p_token * mask.unsqueeze(0)).sum(dim=-1) / denom    # (T, B)
    else:
        p_mol = p_token.mean(dim=-1)                                # (T, B)

    # Stack predictions for vectorized per-step loss.
    preds = torch.stack(per_step_predictions, dim=0)               # (T, B, ...)

    if task_kind == "regression":
        if preds.dim() == 3 and preds.size(-1) == 1:
            preds = preds.squeeze(-1)                              # (T, B)
        target_b = target.float()
        if target_b.dim() > 1:
            target_b = target_b.squeeze(-1)
        per_step_loss = (preds - target_b.unsqueeze(0)) ** 2        # (T, B)
    elif task_kind == "classification":
        # preds: (T, B, C), target: (B,)
        T_, B, C = preds.shape
        flat = preds.reshape(T_ * B, C)
        rep_tgt = target.long().unsqueeze(0).expand(T_, -1).reshape(T_ * B)
        ce = F.cross_entropy(flat, rep_tgt, reduction="none").reshape(T_, B)
        per_step_loss = ce
    else:
        raise ValueError(f"unknown task_kind: {task_kind}")

    weighted = (p_mol * per_step_loss).sum(dim=0)                  # (B,)
    return weighted.mean()
