"""ExitDecoder: once-only post-loop transformer stack + final RMSNorm.

Receives the refined hidden state ``h`` from the recurrent core and applies
``n_exit_blocks`` regular transformer blocks to produce the representation
that downstream task heads will consume.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.block import TransformerBlock
from radiant.config import RadiantConfig
from radiant.norms import RMSNorm


class ExitDecoder(nn.Module):
    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        moe_here = cfg.use_moe and cfg.moe_placement in ("stem_exit", "all")
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, moe=moe_here) for _ in range(cfg.n_exit_blocks)]
        )
        self.final_norm = RMSNorm(cfg.d_model, eps=cfg.rms_norm_eps)

    def forward(
        self,
        h: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            h, aux = block(h, rope_cos, rope_sin, attn_mask, is_causal)
            if aux is not None:
                aux_losses.append(aux)
        return self.final_norm(h), aux_losses
