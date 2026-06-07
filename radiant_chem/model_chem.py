"""RadiantChemModel: chem-specific wrapper around RadiantModel.

Adds:
  * one task head per registered :class:`TaskSpec` (regression /
    classification);
  * a pooled-embedding helper for contrastive learning;
  * a path that returns LM logits for masked-token pre-training (delegates
    to the core's tied LM head -- nothing chem-specific).

Architecture features (backward-compatible — disabled by default):
  * **Attention pooling**: learnable query-based pooling that focuses
    on pharmacophore-relevant atoms instead of treating all equally.
  * **Fingerprint augmentation** (ablation only): concatenates Morgan
    FP (2048 bits) with the pooled embedding before the task head.
  * **Depth-adaptive pooling**: weights intermediate hidden states by
    halting probabilities, producing complexity-aware representations.

The chem wrapper does not modify the core architecture in any way; it only
adds heads and provides task-specific forward methods that thread through
the core's ``forward``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from radiant import (
    ClassificationHead,
    RadiantModel,
    RadiantOutput,
    PoolingHead,
    RegressionHead,
)
from radiant_chem.config import RadiantChemConfig
from radiant_chem.tasks import TaskRegistry, TaskSpec


@dataclass
class ChemForwardOutput:
    base: RadiantOutput
    task_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    pooled: torch.Tensor | None = None
    # Per-step task outputs, one entry per task name; each is a list of
    # per-loop-step task predictions matching ``base.intermediate_hidden_states``.
    # Populated only when ``forward(..., return_per_step_task=True)``.
    per_step_task_outputs: dict[str, list[torch.Tensor]] | None = None


# ======================================================================
# Fingerprint-augmented regression head (ablation only)
# ======================================================================

class FingerprintAugmentedHead(nn.Module):
    """Regression head that fuses RADIANT pooled embedding + Morgan FP.

    Architecture::

        radiant_pooled  (B, D)          morgan_fp (B, fp_dim)
                 │                              │
                 └──────── concat ──────────────┘
                              │
                        LayerNorm(D + fp_dim)
                              │
                        Linear(D + fp_dim, D)
                              │
                           SiLU
                              │
                         Dropout
                              │
                        Linear(D, 1)
                              │
                         pChEMBL

    The head learns to combine structural fingerprint features with
    contextual RADIANT representations.  This gives the model the same
    substructure knowledge RF gets for free while adding learned
    molecular context on top.
    """

    def __init__(
        self,
        d_model: int,
        fp_dim: int,
        num_outputs: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = d_model + fp_dim
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_outputs),
        )
        # Small init for the output layer
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, pooled: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pooled : (B, D) from RADIANT pooling
        fp     : (B, fp_dim) Morgan fingerprint

        Returns
        -------
        (B, num_outputs)
        """
        h = torch.cat([pooled, fp], dim=-1)   # (B, D + fp_dim)
        return self.mlp(self.norm(h))


class RadiantChemModel(nn.Module):
    """Wraps :class:`RadiantModel` and adds molecular task heads."""

    def __init__(
        self,
        cfg: RadiantChemConfig,
        tasks: TaskRegistry | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.core = RadiantModel(cfg.base)
        self.pool = PoolingHead(cfg.base, kind=cfg.pooling_kind)
        self.tasks = tasks if tasks is not None else TaskRegistry()
        self.task_heads = nn.ModuleDict()

        # Depth-adaptive pooling — pass the main pool so intermediates
        # are pooled with the same strategy (attention/mean/first) as
        # the final hidden state. This keeps both branches in the same
        # representation space for the gated combination.
        self.depth_pool = None
        if cfg.use_depth_adaptive_pool:
            from radiant_chem.depth_pool import DepthAdaptivePool
            self.depth_pool = DepthAdaptivePool(
                cfg.base, gate_init=cfg.depth_pool_gate_init,
                pool_fn=self.pool,
            )

        # Fingerprint projection (lazy — only used when fp is provided)
        self.fp_dim = cfg.fingerprint_dim
        self.fp_proj = None
        if cfg.fingerprint_dim > 0:
            # Project FP to d_model space for fusion (used when
            # FingerprintAugmentedHead is not available, e.g. contrastive)
            self.fp_proj = nn.Sequential(
                nn.Linear(cfg.fingerprint_dim, cfg.base.d_model),
                nn.SiLU(),
            )

        for spec in self.tasks:
            self._build_head(spec)

    # ------------------------------------------------------------------
    def _build_head(self, spec: TaskSpec) -> None:
        if spec.kind == "regression":
            if self.cfg.fingerprint_dim > 0:
                # Fingerprint-augmented head (ablation only)
                head: nn.Module = FingerprintAugmentedHead(
                    d_model=self.cfg.base.d_model,
                    fp_dim=self.cfg.fingerprint_dim,
                    num_outputs=spec.num_outputs,
                    dropout=self.cfg.regression_head_dropout,
                )
            else:
                head = RegressionHead(
                    self.cfg.base,
                    num_outputs=spec.num_outputs,
                    hidden_dim=self.cfg.regression_head_hidden_dim,
                    dropout=self.cfg.regression_head_dropout,
                )
        else:
            head = ClassificationHead(self.cfg.base, num_classes=spec.num_outputs)
        self.task_heads[spec.name] = head

    def add_task(self, spec: TaskSpec) -> None:
        self.tasks.register(spec)
        self._build_head(spec)

    # ------------------------------------------------------------------
    def num_params(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: int | None = None,
        attention_mask: torch.Tensor | None = None,
        return_loop_metrics: bool = False,
        is_causal: bool = False,  # chem MLM is bidirectional by default
        run_tasks: bool = True,
        return_pooled: bool = False,
        return_per_step_task: bool = False,
        fingerprints: torch.Tensor | None = None,   # (B, fp_dim) — ablation only
    ) -> ChemForwardOutput:
        # Need intermediates if depth-adaptive pooling or per-step task
        need_intermediates = return_per_step_task or (
            self.depth_pool is not None and (return_pooled or run_tasks)
        )

        base_out = self.core(
            input_ids,
            n_loops=n_loops,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_loop_metrics=return_loop_metrics,
            return_intermediate_hidden=need_intermediates,
        )

        pooled = None
        if return_pooled or run_tasks:
            # Depth-adaptive pooling
            if self.depth_pool is not None:
                halting_probs = None
                if base_out.halting is not None and base_out.halting.confidences:
                    halting_probs = base_out.halting.confidences
                pooled = self.depth_pool(
                    final_hidden=base_out.last_hidden_state,
                    attention_mask=attention_mask,
                    intermediate_hiddens=base_out.intermediate_hidden_states,
                    halting_probs=halting_probs,
                )
            else:
                pooled = self.pool(base_out.last_hidden_state, attention_mask)

        task_outputs: dict[str, torch.Tensor] = {}
        if run_tasks:
            for spec in self.tasks:
                head = self.task_heads[spec.name]
                if isinstance(head, FingerprintAugmentedHead):
                    # Fingerprint-augmented head needs pooled + fp
                    if pooled is None:
                        pooled = self.pool(base_out.last_hidden_state, attention_mask)
                    if fingerprints is None:
                        raise ValueError(
                            f"Task {spec.name!r} uses FingerprintAugmentedHead but "
                            "no fingerprints were provided to forward()"
                        )
                    task_outputs[spec.name] = head(pooled, fingerprints)
                else:
                    # Pass pre-pooled (B, D) when available (from attention
                    # or depth-adaptive pooling); head detects ndim==2 and
                    # skips its internal mean-pool.
                    if pooled is not None:
                        task_outputs[spec.name] = head(pooled)
                    else:
                        task_outputs[spec.name] = head(
                            base_out.last_hidden_state, attention_mask
                        )

        per_step_task_outputs: dict[str, list[torch.Tensor]] | None = None
        if return_per_step_task and base_out.intermediate_hidden_states is not None:
            per_step_task_outputs = {}
            for spec in self.tasks:
                head = self.task_heads[spec.name]
                if isinstance(head, FingerprintAugmentedHead):
                    # Per-step with FP: pool each intermediate, concat fp
                    per_step_task_outputs[spec.name] = [
                        head(
                            self.pool(h_step, attention_mask),
                            fingerprints,
                        )
                        for h_step in base_out.intermediate_hidden_states
                    ]
                else:
                    per_step_task_outputs[spec.name] = [
                        head(h_step, attention_mask)
                        for h_step in base_out.intermediate_hidden_states
                    ]

        return ChemForwardOutput(
            base=base_out,
            task_outputs=task_outputs,
            pooled=pooled,
            per_step_task_outputs=per_step_task_outputs,
        )

    # ------------------------------------------------------------------
    def embed_pooled(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_loops: int | None = None,
    ) -> torch.Tensor:
        """Return ``(B, d_model)`` pooled molecule embeddings for downstream use."""
        out = self.forward(
            input_ids,
            n_loops=n_loops,
            attention_mask=attention_mask,
            run_tasks=False,
            return_pooled=True,
            is_causal=False,
        )
        assert out.pooled is not None
        return out.pooled

    def forward_mlm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_loops: int | None = None,
    ) -> torch.Tensor:
        """Return ``(B, S, V)`` LM logits for masked-token training."""
        return self.core(
            input_ids,
            n_loops=n_loops,
            attention_mask=attention_mask,
            is_causal=False,
            return_loop_metrics=False,
        ).logits
