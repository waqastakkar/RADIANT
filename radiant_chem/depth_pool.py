"""Depth-adaptive pooling — RADIANT's unique advantage.

Uses the per-step halting confidences from the PonderNet mechanism to
produce a depth-weighted molecule representation.  Instead of only using
the final hidden state, we weight each refinement step's hidden state by
its halting probability, letting complex molecules incorporate deeper
layer representations.

This is something neither Morgan RF nor MolFormer can do: the
representation itself adapts to molecular complexity.

Architecture::

    h_1, h_2, ..., h_T = intermediate hidden states from refinement core
    p_1, p_2, ..., p_T = halting probabilities (from confidence head)

    h_depth = sum_t( p_t * pool(h_t) )       # depth-weighted pooled repr

    final = LayerNorm(h_depth) + pool(h_T)    # residual with final state

When halting is disabled or no intermediates are available, falls back
to standard pooling of the final hidden state.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.norms import RMSNorm


class DepthAdaptivePool(nn.Module):
    """Depth-weighted pooling over intermediate refinement states.

    Combines per-step hidden states weighted by halting probabilities
    with a residual connection to the final hidden state.

    Parameters
    ----------
    cfg : RadiantConfig
        Model configuration (for d_model, rms_norm_eps).
    gate_init : float
        Initial value for the learnable gate that balances depth-weighted
        vs final-state contributions. 0.0 = start with final-only,
        positive = start with some depth contribution.
    pool_fn : callable, optional
        A pooling function ``(h: (B,S,D), mask: (B,S)|None) -> (B,D)``
        used for the intermediate hidden states. When provided, this
        should match the model's main pooling strategy (e.g. attention
        pooling) so that intermediate and final representations live in
        the same space. Falls back to masked mean pooling when ``None``.
    """

    def __init__(
        self,
        cfg: RadiantConfig,
        gate_init: float = 0.0,
        pool_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        # Learnable scalar gate: sigmoid(gate) blends depth vs final
        self.gate = nn.Parameter(torch.tensor(gate_init))
        self._pool_fn = pool_fn

    def forward(
        self,
        final_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        intermediate_hiddens: list[torch.Tensor] | None = None,
        halting_probs: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        final_hidden : (B, S, D)
            Last hidden state from exit decoder.
        attention_mask : (B, S) or None
            1 = real token, 0 = pad.
        intermediate_hiddens : list of (B, S, D), length T
            Per-step hidden states from refinement core.
        halting_probs : list of (B, S), length T
            Per-step halting probabilities from confidence head.

        Returns
        -------
        (B, D) pooled representation.
        """
        # Pool the final hidden state (always available)
        h_final = self._pool(self.norm(final_hidden), attention_mask)

        # If no intermediates or halting probs, fall back to final-only
        if (
            intermediate_hiddens is None
            or halting_probs is None
            or len(intermediate_hiddens) == 0
            or len(halting_probs) == 0
        ):
            return h_final

        # Depth-weighted pooling: weight each step's pooled repr by p_halt
        T = min(len(intermediate_hiddens), len(halting_probs))
        depth_parts = []
        for t in range(T):
            h_t = intermediate_hiddens[t]           # (B, S, D)
            p_t = halting_probs[t]                   # (B, S)
            # Pool each step's hidden state using the same strategy as
            # the model's main pool (attention/mean/first) so the
            # intermediate representations are in the same space.
            pooled_t = self._pool(h_t, attention_mask)  # (B, D)
            # Weight by mean halting probability across tokens
            if attention_mask is not None:
                p_mean = (p_t * attention_mask.float()).sum(dim=1) / attention_mask.float().sum(dim=1).clamp(min=1e-6)
            else:
                p_mean = p_t.mean(dim=1)             # (B,)
            depth_parts.append(pooled_t * p_mean.unsqueeze(-1))  # (B, D)

        h_depth = torch.stack(depth_parts, dim=0).sum(dim=0)     # (B, D)
        h_depth = self.norm(h_depth)

        # Gated combination: alpha * h_depth + (1 - alpha) * h_final
        alpha = torch.sigmoid(self.gate)
        return alpha * h_depth + (1.0 - alpha) * h_final

    def _pool(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Pool (B,S,D) -> (B,D) using the model's pooling strategy."""
        if self._pool_fn is not None:
            return self._pool_fn(h, mask)
        return self._mean_pool(h, mask)

    @staticmethod
    def _mean_pool(
        h: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        m = mask.to(dtype=h.dtype).unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
