"""IterativeRefinementCore: the loop executor.

Orchestrates one full recurrent pass:

* owns the shared transformer Core (a stack of ``n_refinement_blocks``
  TransformerBlocks);
* owns :class:`StateAnchorUpdate`, :class:`IterationSignal`, optional
  :class:`IterationAdapter` and :class:`ConfidenceHalting`;
* loops, applying the StateAnchorUpdate transition once per step;
* during inference, supports per-token dynamic halting (token freezes its
  hidden state once cumulative confidence exceeds the threshold).

The Core's parameters are shared across every loop iteration -- that weight
sharing is the point of the architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from radiant.block import TransformerBlock
from radiant.config import RadiantConfig
from radiant.confidence_halting import ConfidenceHalting, HaltingTrace, halting_kl_loss
from radiant.iteration_adapter import IterationAdapter
from radiant.iteration_signal import IterationSignal
from radiant.metrics import LoopMetrics
from radiant.state_anchor import StateAnchorUpdate


class IterativeRefinementCore(nn.Module):
    def __init__(self, cfg: RadiantConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.max_loops = cfg.max_loops

        moe_in_loop = cfg.use_moe and cfg.moe_in_loop_experimental
        self.core_blocks = nn.ModuleList(
            [TransformerBlock(cfg, moe=moe_in_loop) for _ in range(cfg.n_refinement_blocks)]
        )

        self.iteration_signal = IterationSignal(cfg)
        self.state_anchor = StateAnchorUpdate(cfg)
        self.iteration_adapter: IterationAdapter | None = (
            IterationAdapter(cfg) if cfg.use_iteration_adapter else None
        )
        self.halting: ConfidenceHalting | None = (
            ConfidenceHalting(cfg) if cfg.use_confidence_halting else None
        )

    # ------------------------------------------------------------------
    def _apply_core(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the shared Core block stack on ``x``."""
        aux_losses: list[torch.Tensor] = []
        h = x
        for block in self.core_blocks:
            h, aux = block(h, rope_cos, rope_sin, attn_mask, is_causal)
            if aux is not None:
                aux_losses.append(aux)
        return h, aux_losses

    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        n_loops: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = True,
        return_loop_metrics: bool = False,
        token_attention_mask: torch.Tensor | None = None,
        return_intermediate_hidden: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        HaltingTrace | None,
        LoopMetrics | None,
        int,
        list[torch.Tensor] | None,
    ]:
        """Run the recurrent loop for ``n_loops`` steps.

        Returns ``(h_final, aux_losses, halting_trace, loop_metrics, n_executed)``.
        ``n_executed`` may be less than ``n_loops`` if dynamic halting fired
        for *all* tokens during inference.
        """
        if n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {n_loops}")

        anchor_e = self.state_anchor.precompute_anchor(e)
        aux_losses: list[torch.Tensor] = []
        loop_metrics = LoopMetrics() if return_loop_metrics else None
        halting_trace: HaltingTrace | None = HaltingTrace() if self.halting is not None else None
        intermediate_hidden: list[torch.Tensor] | None = [] if return_intermediate_hidden else None

        # Dynamic halting bookkeeping (inference only).
        dynamic_halt = self.halting is not None and not self.training
        if dynamic_halt:
            cum_conf = torch.zeros(h.size(0), h.size(1), device=h.device, dtype=h.dtype)
            halted_mask = torch.zeros_like(cum_conf, dtype=torch.bool)
            frozen_h = h.clone()
        else:
            cum_conf = halted_mask = frozen_h = None

        n_executed = 0
        for t in range(n_loops):
            signal_t = self.iteration_signal(t, device=h.device, dtype=h.dtype)
            pre_core = self.state_anchor.pre_core(h, signal_t)
            core_out, core_aux = self._apply_core(
                pre_core, rope_cos, rope_sin, attn_mask, is_causal
            )
            aux_losses.extend(core_aux)
            if self.iteration_adapter is not None:
                core_out = self.iteration_adapter(core_out, t)
            h_new = self.state_anchor.update(h, core_out, anchor_e, t)

            if self.halting is not None:
                conf = self.halting(h_new)
                # Keep grad on the trace when we're training *and* the
                # auxiliary halting loss is enabled. Otherwise the head
                # would receive no gradient signal and stay at its random
                # init. Detach at inference (or when the loss is off) so
                # downstream summarization is allocation-free.
                trace_grad_needed = (
                    self.cfg.halting_loss_weight > 0
                    and self.training
                    and self.halting is not None
                )
                halting_trace.append(conf if trace_grad_needed else conf.detach())
                if dynamic_halt:
                    # Only accumulate for tokens that haven't halted yet.
                    masked_conf = torch.where(halted_mask, torch.zeros_like(conf), conf)
                    cum_conf = cum_conf + masked_conf
                    newly = (cum_conf >= self.halting.threshold) & (~halted_mask)
                    if newly.any():
                        frozen_h = torch.where(
                            newly.unsqueeze(-1), h_new, frozen_h
                        )
                        halted_mask = halted_mask | newly
                    # Halted tokens get their frozen state back.
                    h_new = torch.where(halted_mask.unsqueeze(-1), frozen_h, h_new)

            h = h_new
            n_executed = t + 1

            if intermediate_hidden is not None:
                intermediate_hidden.append(h)

            if loop_metrics is not None:
                loop_metrics.record(t, h)

            if dynamic_halt and halted_mask.all():
                break

        if halting_trace is not None and self.halting is not None:
            # PonderNet-style auxiliary loss. Only fires during training and
            # only if the user opted in via `halting_loss_weight > 0`.
            # Finalization (computing halt_step / avg_depth) needs detached
            # confidences, so we run the aux loss *before* finalize() while
            # `halting_trace.confidences` still carries grad.
            if (
                self.cfg.halting_loss_weight > 0
                and self.training
                and halting_trace.confidences
                and halting_trace.confidences[0].requires_grad
            ):
                aux = halting_kl_loss(
                    halting_trace.confidences,
                    token_attention_mask,
                    prior_lambda=self.cfg.halting_prior_lambda,
                )
                aux_losses.append(self.cfg.halting_loss_weight * aux)
            halting_trace.finalize(self.halting.threshold)

        return h, aux_losses, halting_trace, loop_metrics, n_executed, intermediate_hidden
