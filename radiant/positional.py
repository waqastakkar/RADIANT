"""Rotary positional embedding (RoPE).

The cache holds (cos, sin) tables of shape ``(max_seq_len, head_dim // 2)``.
``apply_rope`` rotates query/key vectors in-place style by the per-position
phases, leaving values unchanged. We keep the implementation deliberately
plain torch (no fused kernels) so it remains hackable; on CUDA, the surrounding
SDPA kernel dominates anyway.
"""

from __future__ import annotations

import math

import torch


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) tables for positions ``[0, max_seq_len)``.

    Returns tensors of shape ``(max_seq_len, head_dim // 2)``. They are real
    tensors; rotation is performed by the standard "split into pairs and
    rotate" trick in :func:`apply_rope`.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    half = head_dim // 2
    inv_freq = torch.exp(
        -math.log(theta) * torch.arange(0, half, dtype=dtype, device=device) / half
    )
    pos = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(pos, inv_freq)  # (S, half)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation to ``x``.

    Args:
        x:    ``(B, S, H, D)`` (queries or keys).
        cos:  ``(S, D // 2)`` from :func:`build_rope_cache`.
        sin:  ``(S, D // 2)`` from :func:`build_rope_cache`.

    Returns:
        Rotated tensor of the same shape and dtype as ``x``.
    """
    if x.dim() != 4:
        raise ValueError(f"apply_rope expects 4D (B,S,H,D), got shape {tuple(x.shape)}")
    seq_len = x.size(1)
    if cos.size(0) < seq_len:
        raise ValueError(
            f"RoPE cache has {cos.size(0)} positions but input has {seq_len}"
        )
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)

    # Pair up the last dim into (a, b) pairs of size 2.
    x_pairs = x.float().unflatten(-1, (-1, 2))  # (B, S, H, D/2, 2)
    a, b = x_pairs.unbind(-1)
    rot_a = a * cos - b * sin
    rot_b = a * sin + b * cos
    out = torch.stack((rot_a, rot_b), dim=-1).flatten(-2)
    return out.to(x.dtype)
