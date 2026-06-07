"""RadiantModel: the top-level recurrent-depth transformer.

Forward sequence::

    e, _    = StemEncoder(input_ids)
    h       = e
    h, ...  = IterativeRefinementCore(h, e, n_loops)
    h, _    = ExitDecoder(h)
    logits  = LMHead(h)

The same instance handles arbitrary loop counts at runtime: pass
``n_loops`` to :py:meth:`forward` to override the default. With sinusoidal
loop conditioning the model can extrapolate to loop counts beyond
``cfg.max_loops``; with the learned variant, indices past ``max_loops``
are clamped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from radiant.config import RadiantConfig
from radiant.confidence_halting import HaltingTrace
from radiant.exit_decoder import ExitDecoder
from radiant.heads import LMHead
from radiant.metrics import LoopMetrics
from radiant.positional import build_rope_cache
from radiant.refinement_core import IterativeRefinementCore
from radiant.stem_encoder import StemEncoder
from radiant.utils import expand_attention_mask


@dataclass
class RadiantOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor
    aux_losses: list[torch.Tensor] = field(default_factory=list)
    halting: HaltingTrace | None = None
    loop_metrics: LoopMetrics | None = None
    n_loops_executed: int = 0
    # Per-step *pre-exit-decoder* hidden states, one per executed loop
    # step (i.e. the output of the refinement loop at step t before the
    # exit decoder is applied). Populated only when
    # ``forward(..., return_intermediate_hidden=True)``; used by
    # PonderNet-style per-step task supervision. Running the exit
    # decoder on each intermediate is memory-prohibitive (~2 GB extra
    # activations for n_loops=4 at typical chem batch sizes), so the
    # task head is applied directly to these pre-exit states.
    intermediate_hidden_states: list[torch.Tensor] | None = None

    @property
    def aux_loss(self) -> torch.Tensor | None:
        if not self.aux_losses:
            return None
        return torch.stack(self.aux_losses).sum()


class RadiantModel(nn.Module):
    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stem = StemEncoder(cfg)
        self.refinement = IterativeRefinementCore(cfg)
        self.exit = ExitDecoder(cfg)

        tied = self.stem.token_embed.weight if cfg.tie_word_embeddings else None
        self.lm_head = LMHead(cfg, tied_weight=tied)

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, theta=cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    # ------------------------------------------------------------------
    def num_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def num_recurrent_params(self) -> int:
        """Parameters that participate in the shared loop. Useful for compute-equiv baselines."""
        return sum(p.numel() for p in self.refinement.parameters())

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: int | None = None,
        attention_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        return_loop_metrics: bool = False,
        return_intermediate_hidden: bool = False,
    ) -> RadiantOutput:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be (B, S), got shape {tuple(input_ids.shape)}"
            )
        B, S = input_ids.shape
        if S > self.cfg.max_seq_len:
            raise ValueError(
                f"input length {S} exceeds max_seq_len ({self.cfg.max_seq_len})"
            )

        n_loops = n_loops if n_loops is not None else self.cfg.n_loops_train
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")
        if n_loops > self.cfg.max_loops and self.cfg.iteration_signal_kind in ("learned", "both"):
            # Learned per-loop tokens cannot meaningfully extend; raise so the
            # user notices. Sinusoidal-only or "none" extrapolates by design.
            raise ValueError(
                f"n_loops ({n_loops}) > max_loops ({self.cfg.max_loops}) requires "
                "iteration_signal_kind in {'sinusoidal','none'} for safe extrapolation"
            )

        rope_cos = self.rope_cos[:S]
        rope_sin = self.rope_sin[:S]

        attn_mask = expand_attention_mask(
            attention_mask, batch_size=B, seq_len=S, device=input_ids.device, causal=is_causal
        )
        # If we built an explicit per-batch mask, don't double-apply causality.
        sdpa_is_causal = is_causal and attn_mask is None

        e, aux_stem = self.stem(input_ids, rope_cos, rope_sin, attn_mask, sdpa_is_causal)
        h = e

        h, aux_loop, halting, loop_metrics, n_executed, intermediate_h = self.refinement(
            h, e, n_loops, rope_cos, rope_sin, attn_mask, sdpa_is_causal,
            return_loop_metrics=return_loop_metrics,
            token_attention_mask=attention_mask,
            return_intermediate_hidden=return_intermediate_hidden,
        )

        h, aux_exit = self.exit(h, rope_cos, rope_sin, attn_mask, sdpa_is_causal)

        # PonderNet supervision asks: "could we halt at step t and predict
        # well from h_t?" We answer that with the *pre-exit* hidden state
        # at each step. Running the exit decoder once per loop step was
        # the original implementation but blew up activation memory on
        # cells with long sequences (4x exit-decoder activations stored
        # for backprop). The task head is small + already pools its
        # input, so pre-exit hidden states give a perfectly serviceable
        # per-step prediction at a fraction of the memory cost.
        logits = self.lm_head(h)

        aux_losses = aux_stem + aux_loop + aux_exit
        return RadiantOutput(
            logits=logits,
            last_hidden_state=h,
            aux_losses=aux_losses,
            halting=halting,
            loop_metrics=loop_metrics,
            n_loops_executed=n_executed,
            intermediate_hidden_states=intermediate_h,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        n_loops: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Naive autoregressive sampling. No KV cache: every step recomputes
        the full prefix. Adequate for examples and tests; not for production
        long-context generation.
        """
        self.eval()
        for _ in range(max_new_tokens):
            seq = input_ids[:, -self.cfg.max_seq_len:]
            out = self.forward(seq, n_loops=n_loops)
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                topv, _ = logits.topk(min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topv[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break
        return input_ids
