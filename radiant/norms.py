"""Normalization layers used throughout RADIANT.

Only RMSNorm is exported; LayerNorm is available from ``torch.nn`` if needed.
RMSNorm is preferred because it has slightly fewer parameters, is numerically
well-behaved at fp16/bf16, and is what most modern transformer recipes assume.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019).

    y = x / sqrt(mean(x^2) + eps) * weight

    The mean and bias terms of standard LayerNorm are dropped. The learnable
    ``weight`` (initialized to 1) provides per-channel rescaling.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability under bf16/fp16 inputs.
        dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
