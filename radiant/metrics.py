"""Diagnostics for analyzing the recurrent loop's dynamics.

Three flavors:

* :class:`LoopMetrics` -- a lightweight, per-step accumulator collected
  *during* a forward pass when ``return_loop_metrics=True``.
* :func:`spectral_radius_estimate` -- offline power-iteration estimate of
  the recurrent map's largest singular value, useful for verifying
  recurrence stability after training.
* :func:`router_load_entropy`, :func:`halting_summary` -- post-hoc
  summaries of MoE routing and confidence-halting behavior.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn.functional as F


class LoopMetrics:
    """Records per-loop hidden-state statistics during a forward pass."""

    def __init__(self) -> None:
        self.norms: list[float] = []           # mean ||h_t||_2 across (B*S, D)
        self.cos_to_prev: list[float] = []     # cosine(h_t, h_{t-1})
        self.cos_to_first: list[float] = []    # cosine(h_t, h_0)
        self._prev: torch.Tensor | None = None
        self._first: torch.Tensor | None = None

    def record(self, t: int, h: torch.Tensor) -> None:
        with torch.no_grad():
            flat = h.detach().flatten(0, 1)             # (B*S, D)
            self.norms.append(float(flat.norm(dim=-1).mean().item()))
            if self._prev is not None:
                cos = F.cosine_similarity(flat, self._prev, dim=-1).mean()
                self.cos_to_prev.append(float(cos.item()))
            if self._first is None:
                self._first = flat
            else:
                cos0 = F.cosine_similarity(flat, self._first, dim=-1).mean()
                self.cos_to_first.append(float(cos0.item()))
            self._prev = flat

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "norms": list(self.norms),
            "cos_to_prev": list(self.cos_to_prev),
            "cos_to_first": list(self.cos_to_first),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LoopMetrics(steps={len(self.norms)}, "
            f"final_norm={self.norms[-1] if self.norms else None})"
        )


def router_load_entropy(aux_losses_or_routing: Iterable[torch.Tensor]) -> float | None:
    """Best-effort entropy summary over MoE auxiliary losses.

    The MoE block returns scalar auxiliary losses. We can't recover full
    routing distributions from them, but we can return their mean as a
    proxy diagnostic. Returns ``None`` if no values were provided.
    """
    values = [float(t.detach().mean().item()) for t in aux_losses_or_routing]
    if not values:
        return None
    return sum(values) / len(values)


def halting_summary(trace) -> dict[str, float]:
    """Summarize a :class:`HaltingTrace`. Trace must already be ``finalize()``d."""
    if trace is None or trace.halt_step is None:
        return {}
    halt = trace.halt_step.float()
    out = {
        "avg_depth": float(trace.avg_depth or (halt.mean().item() + 1.0)),
        "halt_step_mean": float(halt.mean().item()),
        "halt_step_std": float(halt.std(unbiased=False).item()),
        "halt_step_min": float(halt.min().item()),
        "halt_step_max": float(halt.max().item()),
    }
    # Entropy of the halt step distribution.
    if halt.numel() > 0:
        T = int(halt.max().item()) + 1
        if T > 1:
            counts = torch.bincount(halt.long().flatten(), minlength=T).float()
            probs = counts / counts.sum().clamp(min=1.0)
            ent = -(probs * (probs + 1e-12).log()).sum().item()
            out["halt_step_entropy"] = float(ent)
            out["halt_step_entropy_norm"] = float(ent / math.log(T))
    return out


@torch.no_grad()
def spectral_radius_estimate(
    refinement_core,
    h: torch.Tensor,
    e: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    t: int = 0,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    n_iter: int = 6,
    eps: float = 1e-3,
) -> float:
    """Power-iteration estimate of the recurrent map's spectral radius at ``(h, e)``.

    The "recurrent map" is the per-step transition implemented by
    ``refinement_core`` (StateAnchorUpdate + Core + optional IterationAdapter).
    We approximate its directional derivative via a finite-difference JVP and
    grow a unit vector along the dominant eigendirection for ``n_iter`` steps.
    Returns the (positive) estimated spectral radius.

    This is offline diagnostics: requires no autograd graph, runs under
    ``no_grad``. Use after training to certify recurrence stability.
    """
    anchor_e = refinement_core.state_anchor.precompute_anchor(e)

    def step(h_in: torch.Tensor) -> torch.Tensor:
        signal_t = refinement_core.iteration_signal(t, device=h_in.device, dtype=h_in.dtype)
        pre = refinement_core.state_anchor.pre_core(h_in, signal_t)
        core_out = refinement_core._apply_core(pre, rope_cos, rope_sin, attn_mask, is_causal)[0]
        if refinement_core.iteration_adapter is not None:
            core_out = refinement_core.iteration_adapter(core_out, t)
        return refinement_core.state_anchor.update(h_in, core_out, anchor_e, t)

    base = step(h)
    v = torch.randn_like(h)
    v = v / v.norm().clamp(min=1e-9)

    sigma = 0.0
    for _ in range(n_iter):
        perturbed = step(h + eps * v)
        jv = (perturbed - base) / eps
        sigma_t = jv.norm()
        if sigma_t.item() == 0:
            break
        v = jv / sigma_t
        sigma = float(sigma_t.item())
    return sigma
