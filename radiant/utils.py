"""Small shared helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


def init_linear_(module: nn.Linear, std: float = 0.02) -> None:
    """Truncated-normal init (clipped at 2 sigma) for linear weights; zero bias."""
    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_embedding_(module: nn.Embedding, std: float = 0.02) -> None:
    nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Boolean (S, S) mask: True where attention is allowed (i <= j)."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def expand_attention_mask(
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device | str | None = None,
    causal: bool = True,
) -> torch.Tensor | None:
    """Build the additive mask passed to ``F.scaled_dot_product_attention``.

    Args:
        attention_mask: optional ``(B, S)`` 1/0 mask where 0 marks pad tokens.
        batch_size:     B.
        seq_len:        S.
        device:         destination device.
        causal:         if True, AND the result with a causal (lower-triangular) mask.

    Returns:
        A boolean ``(B, 1, S, S)`` mask suitable for SDPA's ``attn_mask`` arg
        (True = keep, False = mask out), or ``None`` if no masking is needed
        (in which case the caller can rely on SDPA's ``is_causal=True`` flag).
    """
    if attention_mask is None and not causal:
        return None
    if attention_mask is None:
        # Caller will use is_causal=True instead; signal "no per-batch mask".
        return None

    keep = attention_mask.to(dtype=torch.bool, device=device)  # (B, S)
    # Mask padded keys, not padded queries. If an entire padded query row is
    # masked out, PyTorch SDPA returns NaNs for that row; those NaNs later
    # poison residual states and masked pooling because NaN * 0 is still NaN.
    # Let pad queries attend to real tokens and ignore them downstream.
    pad_mask = keep.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, seq_len)
    if causal:
        cm = causal_mask(seq_len, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,S,S)
        return pad_mask & cm
    return pad_mask


def num_params(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module."""
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def split_param_groups(
    module: nn.Module,
    weight_decay: float = 0.1,
    no_decay_keywords: Iterable[str] = ("bias", "norm", "embed", "alpha_logit"),
) -> list[dict]:
    """Split parameters into decayed / no-decay groups for AdamW.

    Anything containing one of ``no_decay_keywords`` (case-insensitive) in its
    fully-qualified name is placed in the no-decay group. This is the convention
    used by most modern transformer training recipes.
    """
    decay, no_decay = [], []
    keywords = tuple(k.lower() for k in no_decay_keywords)
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        lower = name.lower()
        if p.ndim < 2 or any(k in lower for k in keywords):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def gate_logit_for(p: float) -> float:
    """Inverse sigmoid: solve sigmoid(x) = p."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"gate_logit_for expects p in (0,1), got {p}")
    return math.log(p / (1.0 - p))
