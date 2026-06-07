"""Output heads.

The core RADIANT model owns an LMHead (next-token logits). The chem
variant adds property-prediction heads on top of pooled hidden states.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.norms import RMSNorm
from radiant.utils import init_linear_


class LMHead(nn.Module):
    """Linear projection from hidden state to vocab logits, optionally tied."""

    def __init__(self, cfg: RadiantConfig, *, tied_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tied_weight is not None:
            # Share storage with the embedding table.
            self.proj.weight = tied_weight
        else:
            init_linear_(self.proj, std=cfg.initializer_range)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


class PoolingHead(nn.Module):
    """Pool a sequence to a single vector, masking pads.

    Returns ``(B, D)`` regardless of input ``(B, S, D)``. If no mask is
    provided, every token contributes equally.

    Supported kinds:

    * ``"mean"``      -- masked mean pooling (default, parameter-free).
    * ``"first"``     -- take the first token (CLS-style).
    * ``"attention"`` -- learnable multi-head attention pooling: a single
      query vector attends to all token hidden states and produces a
      weighted summary. This lets the model learn which atoms/tokens
      matter for the downstream task rather than weighting all equally.
    """

    def __init__(self, cfg: RadiantConfig, *, kind: str = "mean") -> None:
        super().__init__()
        if kind not in ("mean", "first", "attention"):
            raise ValueError(f"PoolingHead kind must be 'mean', 'first', or 'attention', got {kind!r}")
        self.kind = kind
        self.norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

        if kind == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=cfg.d_model,
                num_heads=cfg.n_query_heads,
                dropout=cfg.attention_dropout,
                batch_first=True,
            )

    def forward(
        self, h: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm(h)
        if self.kind == "first":
            return h[:, 0]
        if self.kind == "attention":
            B = h.size(0)
            q = self.query.expand(B, -1, -1)                       # (B, 1, D)
            # nn.MHA expects key_padding_mask True = ignore
            kpm = (~attention_mask.bool()) if attention_mask is not None else None
            pooled, _ = self.attn(q, h, h, key_padding_mask=kpm)   # (B, 1, D)
            return pooled.squeeze(1)                                # (B, D)
        # mean pooling
        if attention_mask is None:
            return h.mean(dim=1)
        m = attention_mask.to(dtype=h.dtype).unsqueeze(-1)        # (B, S, 1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)


class RegressionHead(nn.Module):
    """Regression head over a pooled vector. ``num_outputs`` may be > 1."""

    def __init__(
        self,
        cfg: RadiantConfig,
        *,
        num_outputs: int = 1,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = PoolingHead(cfg)
        if hidden_dim > 0:
            self.hidden = nn.Sequential(
                RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps),
                nn.Linear(cfg.d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.proj = nn.Linear(hidden_dim, num_outputs)
            init_linear_(self.hidden[1], std=cfg.initializer_range)
        else:
            self.hidden = nn.Identity()
            self.proj = nn.Linear(cfg.d_model, num_outputs)
        init_linear_(self.proj, std=cfg.initializer_range)

    def forward(
        self, h: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Accept both (B, S, D) unpooled and (B, D) pre-pooled inputs.
        # When a pre-pooled vector is provided (e.g. from attention or
        # depth-adaptive pooling), skip the internal mean-pool.
        if h.ndim == 3:
            h = self.pool(h, attention_mask)
        return self.proj(self.hidden(h))


class ClassificationHead(nn.Module):
    """Linear classification head over a pooled vector."""

    def __init__(self, cfg: RadiantConfig, *, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.pool = PoolingHead(cfg)
        self.proj = nn.Linear(cfg.d_model, num_classes)
        init_linear_(self.proj, std=cfg.initializer_range)

    def forward(
        self, h: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if h.ndim == 3:
            h = self.pool(h, attention_mask)
        return self.proj(h)
