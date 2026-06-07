"""StateAnchorUpdate: the recurrence transition.

For each loop step ``t`` with current hidden state ``h``, original encoded
input ``e``, and per-loop signal ``s_t``::

    pre_core   = norm(h + s_t)
    core_out   = Core(pre_core)              # the shared transformer stack
    h_next     = h + beta_t * core_out + gamma_t * Anchor(e)

* ``beta_t`` is a learned per-loop scalar bounded to ``(0, 1)`` via sigmoid.
  Initialized small (``state_gate_init_scale``) so early iterations are
  near-identity in ``h``.
* ``gamma_t`` is a learned per-loop scalar bounded to ``(0, 1)`` similarly.
* ``Anchor`` is a single linear projection of ``e``, computed *once* per
  forward pass (constant across the loop) and zero-initialized so at start
  of training the anchor contributes nothing.

This module owns the gates, the pre-core RMSNorm, and the Anchor projection.
The Core itself (a stack of TransformerBlocks) is owned by
:class:`IterativeRefinementCore` and passed in via ``compute_core_output``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.norms import RMSNorm


def _gate_logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class StateAnchorUpdate(nn.Module):
    """Holds the bounded loop scales, the pre-Core norm, and the Anchor projection."""

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        init_logit = _gate_logit(cfg.state_gate_init_scale)
        self.beta_logits = nn.Parameter(torch.full((cfg.max_loops,), init_logit))
        self.gamma_logits = nn.Parameter(torch.full((cfg.max_loops,), init_logit))
        self.max_loops = cfg.max_loops
        self.use_state_anchor = cfg.use_state_anchor

        self.pre_core_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        # Anchor(e). Zero-init so the second residual path contributes nothing
        # at init and the recurrence starts as plain weight-shared depth.
        self.anchor_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.zeros_(self.anchor_proj.weight)

    def precompute_anchor(self, e: torch.Tensor) -> torch.Tensor:
        """Compute ``Anchor(e)`` once at the start of the loop."""
        if not self.use_state_anchor:
            return torch.zeros_like(e)
        return self.anchor_proj(e)

    def beta(self, t: int) -> torch.Tensor:
        idx = min(t, self.max_loops - 1)
        return torch.sigmoid(self.beta_logits[idx])

    def gamma(self, t: int) -> torch.Tensor:
        if not self.use_state_anchor:
            return self.gamma_logits.new_zeros(())
        idx = min(t, self.max_loops - 1)
        return torch.sigmoid(self.gamma_logits[idx])

    def pre_core(self, h: torch.Tensor, signal_t: torch.Tensor) -> torch.Tensor:
        """Compute ``norm(h + s_t)`` for the current loop step."""
        # ``signal_t`` may be ``(d_model,)`` or already broadcast.
        return self.pre_core_norm(h + signal_t)

    def update(
        self,
        h: torch.Tensor,
        core_out: torch.Tensor,
        anchor_e: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Compute ``h + beta_t * core_out + gamma_t * Anchor(e)``."""
        return h + self.beta(t) * core_out + self.gamma(t) * anchor_e
