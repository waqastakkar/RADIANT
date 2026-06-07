"""StemEncoder: token embedding + a small once-only transformer stack.

Owns the token embedding table (whose weight is optionally tied to the LM
head) and ``n_stem_blocks`` regular pre-norm transformer blocks. Outputs
the embedded representation ``e`` that becomes both the initial hidden
state and the persistent input anchor for the recurrent loop.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.block import TransformerBlock
from radiant.config import RadiantConfig
from radiant.utils import init_embedding_


class StemEncoder(nn.Module):
    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        init_embedding_(self.token_embed, std=cfg.initializer_range)

        # Stem may use MoE if configured (placement "stem_exit" or "all").
        moe_here = cfg.use_moe and cfg.moe_placement in ("stem_exit", "all")
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, moe=moe_here) for _ in range(cfg.n_stem_blocks)]
        )
        self.embed_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.embed_dropout(self.token_embed(input_ids))
        aux_losses: list[torch.Tensor] = []
        for block in self.blocks:
            h, aux = block(h, rope_cos, rope_sin, attn_mask, is_causal)
            if aux is not None:
                aux_losses.append(aux)
        return h, aux_losses
