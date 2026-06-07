"""Grouped-query attention with RoPE.

GQA reduces KV memory by sharing K/V heads across groups of query heads.
With ``n_query_heads = G * n_kv_heads``, each KV head is consumed by ``G``
query heads. Attention itself is ``F.scaled_dot_product_attention`` so we
benefit from any fused kernel the platform provides.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from radiant.config import RadiantConfig
from radiant.positional import apply_rope
from radiant.utils import init_linear_


class GQAAttention(nn.Module):
    """Grouped-query attention.

    Args:
        cfg: model configuration.
    """

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.n_q = cfg.n_query_heads
        self.n_kv = cfg.n_kv_heads
        self.kv_groups = cfg.kv_groups
        self.head_dim = cfg.head_dim
        self.attn_dropout_p = cfg.attention_dropout

        q_total = self.n_q * self.head_dim
        kv_total = self.n_kv * self.head_dim

        self.q_proj = nn.Linear(cfg.d_model, q_total, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_total, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_total, bias=False)
        self.o_proj = nn.Linear(q_total, cfg.d_model, bias=False)

        for m in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            init_linear_(m, std=cfg.initializer_range)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:         (B, S, D) input hidden states.
            rope_cos:  (S, head_dim/2) RoPE cosine table.
            rope_sin:  (S, head_dim/2) RoPE sine table.
            attn_mask: optional boolean (B, 1, S, S) mask passed to SDPA. When
                       provided, ``is_causal`` is ignored (mask must already
                       encode causality if desired).
            is_causal: whether to use SDPA's fused causal kernel. Ignored if
                       ``attn_mask`` is given.

        Returns:
            (B, S, D) output.
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_q, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv, self.head_dim)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Expand K, V from n_kv heads to n_q heads via repeat_interleave.
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=2)
            v = v.repeat_interleave(self.kv_groups, dim=2)

        # SDPA wants (B, H, S, D).
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        dropout_p = self.attn_dropout_p if self.training else 0.0
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=is_causal
            )

        out = out.transpose(1, 2).contiguous().view(B, S, self.n_q * self.head_dim)
        return self.o_proj(out)
