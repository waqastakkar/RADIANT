"""Feed-forward sub-layers: SwiGLU and (optionally) routed MoE.

SwiGLU [Shazeer 2020] is the modern default: two parallel projections gated
by SiLU, then a down-projection. The MoE variant routes each token through
``n_active_experts`` of ``n_experts`` SwiGLU experts and emits a switch-style
auxiliary load-balancing loss alongside the output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from radiant.config import RadiantConfig
from radiant.utils import init_linear_


class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN: ``W_down( silu(W_gate(x)) * W_up(x) )``."""

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        for m in (self.gate_proj, self.up_proj, self.down_proj):
            init_linear_(m, std=cfg.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of SwiGLU experts.

    For each token, the router computes scores over ``n_experts``; we keep
    the top ``n_active_experts``, softmax-renormalize their weights, and sum
    their expert outputs. Returns ``(out, aux_loss)`` where ``aux_loss`` is
    a switch-style load balancing penalty.
    """

    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.n_experts = cfg.n_experts
        self.k = cfg.n_active_experts
        self.aux_weight = cfg.moe_aux_loss_weight

        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        init_linear_(self.router, std=cfg.initializer_range)

        # Each expert is its own SwiGLU.
        self.experts = nn.ModuleList(
            [SwiGLUFeedForward(cfg) for _ in range(cfg.n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)
        router_logits = self.router(x_flat)            # (BS, E)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_w, topk_idx = router_probs.topk(self.k, dim=-1)  # (BS, k)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x_flat)
        # Per-expert dispatch. ``where`` returns indices of tokens that touched expert e.
        for e in range(self.n_experts):
            sel_mask = (topk_idx == e)                       # (BS, k)
            if not sel_mask.any():
                continue
            tok_indices = sel_mask.any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
            if tok_indices.numel() == 0:
                continue
            # Sum the (possibly two) routing weights this expert received per token.
            weight_per_token = (topk_w * sel_mask.float()).sum(dim=-1)[tok_indices]
            expert_out = self.experts[e](x_flat[tok_indices].unsqueeze(0)).squeeze(0)
            out[tok_indices] += weight_per_token.unsqueeze(-1) * expert_out

        out = out.view(B, S, D)

        # Switch-style load balancing loss:
        #   aux = E * sum_e ( fraction_routed_e * mean_router_prob_e )
        # encouraging both quantities to be uniform (both equal 1/E).
        with torch.no_grad():
            # fraction_routed[e] = (#tokens routing any of their k slots to e) / BS
            fraction_routed = (topk_idx == torch.arange(
                self.n_experts, device=topk_idx.device
            ).view(self.n_experts, 1, 1)).any(dim=-1).float().mean(dim=-1)
        importance = router_probs.mean(dim=0)
        aux_loss = self.aux_weight * self.n_experts * (fraction_routed * importance).sum()
        return out, aux_loss
