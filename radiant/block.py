"""TransformerBlock: the single building block reused by Prelude, RecurrentCore, Coda.

Pre-norm style: ``x = x + Attn(Norm(x))``, then ``x = x + FFN(Norm(x))``.
The FFN can be dense (SwiGLU) or sparse (MoE); when MoE, the block forwards
the auxiliary load-balancing loss through its return value.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.attention import GQAAttention
from radiant.config import RadiantConfig
from radiant.feedforward import MoEFeedForward, SwiGLUFeedForward
from radiant.norms import RMSNorm


class TransformerBlock(nn.Module):
    """One pre-norm GQA + FFN block. ``moe`` controls the FFN type."""

    def __init__(self, cfg: RadiantConfig, *, moe: bool = False) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)
        self.uses_moe = moe and cfg.use_moe
        self.ffn: SwiGLUFeedForward | MoEFeedForward
        if self.uses_moe:
            self.ffn = MoEFeedForward(cfg)
        else:
            self.ffn = SwiGLUFeedForward(cfg)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns ``(hidden_states, aux_loss_or_None)``."""
        h = x + self.resid_dropout(
            self.attn(self.norm_attn(x), rope_cos, rope_sin, attn_mask, is_causal)
        )
        if self.uses_moe:
            ff_out, aux = self.ffn(self.norm_ffn(h))
            h = h + self.resid_dropout(ff_out)
            return h, aux
        h = h + self.resid_dropout(self.ffn(self.norm_ffn(h)))
        return h, None
