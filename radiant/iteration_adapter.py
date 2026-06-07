"""IterationAdapter: depth-adaptive residual modulation.

The shared recurrent Core uses the same parameters at every loop. When a
problem benefits from slightly different processing at different depths, the
IterationAdapter provides a tiny, residual, loop-conditioned correction:

    h_out = h + scale_t * MLP_bottleneck(norm(h))

The bottleneck weights are shared across loops; only ``scale_t`` (a per-loop,
per-channel vector) varies with the loop index. ``scale_t`` is zero-init so
the adapter is the identity at start-of-training and only contributes when
the model finds it useful.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from radiant.config import RadiantConfig
from radiant.norms import RMSNorm
from radiant.utils import init_linear_


class IterationAdapter(nn.Module):
    """Residual bottleneck MLP with loop-conditioned per-channel scale."""

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        bottleneck = cfg.iteration_adapter_bottleneck
        self.max_loops = cfg.max_loops

        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.down = nn.Linear(cfg.d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, cfg.d_model, bias=False)
        init_linear_(self.down, std=cfg.initializer_range)
        # Up is zero-init so the whole adapter starts as the identity.
        nn.init.zeros_(self.up.weight)

        # Per-loop, per-channel scale; zero-init.
        self.scale = nn.Parameter(torch.zeros(cfg.max_loops, cfg.d_model))

    def forward(self, h: torch.Tensor, t: int) -> torch.Tensor:
        if t < 0:
            raise ValueError(f"loop index must be >= 0, got {t}")
        t_clamped = min(t, self.max_loops - 1)
        delta = self.up(F.silu(self.down(self.norm(h))))   # (B, S, D)
        return h + self.scale[t_clamped] * delta            # broadcast over (B, S, D)
