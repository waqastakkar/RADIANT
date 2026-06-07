"""Per-loop conditioning signal injected into the recurrent state.

The signal mixes a sinusoidal vector (extrapolates to any non-negative loop
index) with an optional learned per-loop token, then routes the combined
vector through a small gated linear adapter. Both the gate and adapter are
zero-initialized so the signal contributes nothing at start-of-training and
the model is free to use as much or as little loop conditioning as it needs.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.utils import init_embedding_, init_linear_


class IterationSignal(nn.Module):
    """Returns a ``(d_model,)`` per-loop bias to add into the recurrent state."""

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.kind = cfg.iteration_signal_kind
        self.d_model = cfg.d_model
        self.max_loops = cfg.max_loops

        if self.kind in ("learned", "both"):
            self.learned = nn.Embedding(cfg.max_loops, cfg.d_model)
            init_embedding_(self.learned, std=0.01)

        if self.kind != "none":
            # Gated adapter: tanh(gate) * Linear(signal). Zero-init both for a
            # clean identity-recurrence start.
            self.adapter = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            nn.init.zeros_(self.adapter.weight)
            self.gate = nn.Parameter(torch.zeros(cfg.d_model))

        # Pre-computed sinusoidal frequency basis. Stored as a buffer so it
        # follows ``.to(device)``.
        if self.kind in ("sinusoidal", "both"):
            half = cfg.d_model // 2
            inv_freq = torch.exp(
                -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32) / max(half, 1)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _sinusoidal(self, t: int) -> torch.Tensor:
        # Computed analytically each call so any non-negative ``t`` is valid,
        # including values past max_loops.
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype
        signal = torch.zeros(self.d_model, device=device, dtype=dtype)
        if self.d_model % 2 == 0:
            angle = float(t) * self.inv_freq
            signal[0::2] = torch.sin(angle)
            signal[1::2] = torch.cos(angle)
        else:
            angle = float(t) * self.inv_freq
            signal[: 2 * angle.numel() : 2] = torch.sin(angle)
            signal[1 : 2 * angle.numel() : 2] = torch.cos(angle)
        return signal

    def forward(self, t: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute the loop-``t`` conditioning vector.

        Args:
            t:      non-negative loop index.
            device: target device.
            dtype:  target floating dtype.

        Returns:
            Tensor of shape ``(d_model,)``; zeros when ``kind == 'none'``.
        """
        if t < 0:
            raise ValueError(f"loop index must be >= 0, got {t}")

        if self.kind == "none":
            return torch.zeros(self.d_model, device=device, dtype=dtype)

        signal = torch.zeros(self.d_model, device=device, dtype=dtype)
        if self.kind in ("sinusoidal", "both"):
            signal = signal + self._sinusoidal(t).to(device=device, dtype=dtype)
        if self.kind in ("learned", "both"):
            t_clamped = min(t, self.max_loops - 1)
            signal = signal + self.learned.weight[t_clamped].to(dtype=dtype)

        gate = torch.tanh(self.gate)
        return gate * self.adapter(signal)
