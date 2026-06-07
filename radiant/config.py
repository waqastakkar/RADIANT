"""Typed, frozen configuration for RADIANT.

Every architectural switch is named here. The dataclass is immutable
(``frozen=True``) so models and configs stay in lockstep, and dict/JSON
round-trip means a config can be serialized next to a checkpoint and the
exact model definition recovered later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal


IterationSignalKind = Literal["none", "sinusoidal", "learned", "both"]
MoEPlacement = Literal["none", "stem_exit", "all"]


@dataclass(frozen=True, slots=True)
class RadiantConfig:
    """Frozen architecture configuration."""

    # --- Vocabulary -----------------------------------------------------
    vocab_size: int = 1024
    pad_token_id: int = 0

    # --- Hidden dimensions ---------------------------------------------
    d_model: int = 256
    n_query_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 1024

    # --- Sequence ------------------------------------------------------
    max_seq_len: int = 512

    # --- Block stage sizes (matches StemEncoder / IterativeRefinementCore / ExitDecoder)
    n_stem_blocks: int = 2
    n_refinement_blocks: int = 2  # blocks INSIDE the shared loop Core
    n_exit_blocks: int = 2

    # --- Loop control --------------------------------------------------
    n_loops_train: int = 4
    min_loops: int = 1
    max_loops: int = 16

    # --- IterationSignal (per-loop conditioning) -----------------------
    iteration_signal_kind: IterationSignalKind = "both"

    # --- StateAnchorUpdate (the recurrence) ----------------------------
    use_state_anchor: bool = True
    # Initial sigmoid value for both beta_t (Core scale) and gamma_t
    # (anchor scale). Small => loop starts stable.
    state_gate_init_scale: float = 0.1

    # --- IterationAdapter (loop-conditioned per-step modulation) -------
    use_iteration_adapter: bool = True
    iteration_adapter_bottleneck_ratio: float = 0.25  # bottleneck = ratio * d_model

    # --- ConfidenceHalting --------------------------------------------
    use_confidence_halting: bool = False
    confidence_threshold: float = 0.99
    confidence_init_bias: float = -2.0
    # PonderNet-style auxiliary loss that trains the halting head. The
    # halting head outputs a per-token, per-step "halt now" probability;
    # the auxiliary loss is KL(halt_distribution || geometric(prior)).
    # Setting `halting_loss_weight = 0` (the default) trains a model with
    # halting *machinery* present but no learning signal -- the head will
    # stay at its random init and halt_step won't correlate with anything.
    # The published C-claims (C1/C4/C5) require `halting_loss_weight > 0`.
    halting_loss_weight: float = 0.0
    halting_prior_lambda: float = 0.2     # geometric prior parameter; lower => deeper halting
    # Training-time knobs that let the halting head train without
    # fighting the regression head:
    #   * `halting_loss_warmup_epochs`: epochs over which the effective
    #     halting loss weight linearly ramps from 0 to
    #     `halting_loss_weight`. 0 disables the ramp. 10 is a sane
    #     default with 50-100-epoch fine-tunes: the regression converges
    #     in the first phase, then halting trains on a stable backbone.
    #   * `halting_head_lr_mult`: multiplicative factor on the optimizer's
    #     base LR for the halting head parameters only. 10.0 means the
    #     halting head trains 10x faster than the rest of the model,
    #     so the head doesn't need a large halting_loss_weight to
    #     develop input dependence -- it explores faster on its own.
    halting_loss_warmup_epochs: int = 0
    halting_head_lr_mult: float = 1.0

    # --- Mixture of experts -------------------------------------------
    use_moe: bool = False
    moe_placement: MoEPlacement = "stem_exit"  # never inside loop by default
    moe_in_loop_experimental: bool = False     # explicit opt-in flag
    n_experts: int = 8
    n_active_experts: int = 2
    moe_aux_loss_weight: float = 0.01

    # --- Regularization -----------------------------------------------
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # --- Positional ---------------------------------------------------
    rope_theta: float = 10000.0

    # --- Output -------------------------------------------------------
    tie_word_embeddings: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.n_query_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_query_heads ({self.n_query_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if not (0 < self.state_gate_init_scale < 1):
            raise ValueError(
                f"state_gate_init_scale must be in (0, 1), got {self.state_gate_init_scale}"
            )
        if not (1 <= self.min_loops <= self.n_loops_train <= self.max_loops):
            raise ValueError(
                f"Loop counts must satisfy 1 <= min_loops ({self.min_loops}) "
                f"<= n_loops_train ({self.n_loops_train}) "
                f"<= max_loops ({self.max_loops})"
            )
        if self.use_moe and not (1 <= self.n_active_experts <= self.n_experts):
            raise ValueError(
                f"n_active_experts ({self.n_active_experts}) must be in "
                f"[1, n_experts={self.n_experts}]"
            )
        if self.use_confidence_halting and not (0 < self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}"
            )
        if self.halting_loss_weight < 0:
            raise ValueError(
                f"halting_loss_weight must be >= 0, got {self.halting_loss_weight}"
            )
        if self.halting_loss_weight > 0 and not self.use_confidence_halting:
            raise ValueError(
                "halting_loss_weight > 0 requires use_confidence_halting=True"
            )
        if not (0 < self.halting_prior_lambda < 1):
            raise ValueError(
                f"halting_prior_lambda must be in (0, 1), got {self.halting_prior_lambda}"
            )
        if self.halting_loss_warmup_epochs < 0:
            raise ValueError(
                f"halting_loss_warmup_epochs must be >= 0, got {self.halting_loss_warmup_epochs}"
            )
        if self.halting_head_lr_mult <= 0:
            raise ValueError(
                f"halting_head_lr_mult must be > 0, got {self.halting_head_lr_mult}"
            )
        if not (0 < self.iteration_adapter_bottleneck_ratio <= 1):
            raise ValueError(
                "iteration_adapter_bottleneck_ratio must be in (0, 1], got "
                f"{self.iteration_adapter_bottleneck_ratio}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even (RoPE constraint), got {self.head_dim}")
        if self.moe_in_loop_experimental and not self.use_moe:
            raise ValueError(
                "moe_in_loop_experimental requires use_moe=True"
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RadiantConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown RadiantConfig fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "RadiantConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def replace(self, **overrides: Any) -> "RadiantConfig":
        return replace(self, **overrides)

    @property
    def kv_groups(self) -> int:
        return self.n_query_heads // self.n_kv_heads

    @property
    def iteration_adapter_bottleneck(self) -> int:
        return max(int(self.d_model * self.iteration_adapter_bottleneck_ratio), 8)
