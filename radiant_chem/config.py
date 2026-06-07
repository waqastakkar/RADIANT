"""Configuration for RADIANT-Chem.

Composes a :class:`RadiantConfig` for the architecture with a small set of
chem-specific fields covering the masked language objective, contrastive
representation learning, pooling, and tokenizer choice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

from radiant.config import RadiantConfig


TokenizerKind = Literal["smiles", "selfies"]
PoolingKind = Literal["mean", "first", "attention"]


@dataclass(frozen=True, slots=True)
class RadiantChemConfig:
    base: RadiantConfig = field(default_factory=RadiantConfig)

    # --- Tokenization ---------------------------------------------------
    tokenizer_kind: TokenizerKind = "smiles"

    # --- Masked language modeling ---------------------------------------
    mlm_mask_prob: float = 0.15
    mlm_replace_random_prob: float = 0.10  # of the masked positions
    mlm_keep_orig_prob: float = 0.10       # of the masked positions

    # --- Contrastive representation learning ----------------------------
    contrastive_temperature: float = 0.07

    # --- Pooling --------------------------------------------------------
    pooling_kind: PoolingKind = "mean"

    # --- Downstream property heads -------------------------------------
    # Default 0 preserves the original linear regression head. Fine-tuning
    # scripts can opt into a small MLP head without changing the pretrained
    # RADIANT core checkpoint.
    regression_head_hidden_dim: int = 0
    regression_head_dropout: float = 0.0

    # --- Fingerprint augmentation (ablation only) ------------------------
    # NOT part of the main RADIANT architecture.  Kept for ablation
    # experiments showing learned vs engineered feature complementarity.
    # The main model relies purely on learned representations.
    fingerprint_dim: int = 0        # 0 = disabled (default); 2048 for ablation
    fingerprint_radius: int = 2     # Morgan FP radius (ablation only)

    # --- Depth-adaptive pooling ----------------------------------------
    # When True, uses halting probabilities to weight intermediate hidden
    # states, producing a depth-aware molecular representation.
    use_depth_adaptive_pool: bool = False
    depth_pool_gate_init: float = 0.0

    def __post_init__(self) -> None:
        if not (0 < self.mlm_mask_prob < 1):
            raise ValueError(f"mlm_mask_prob must be in (0,1), got {self.mlm_mask_prob}")
        if self.mlm_replace_random_prob + self.mlm_keep_orig_prob > 1.0:
            raise ValueError("mlm replace+keep probs sum must be <= 1.0")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be > 0")
        if self.regression_head_hidden_dim < 0:
            raise ValueError("regression_head_hidden_dim must be >= 0")
        if not (0.0 <= self.regression_head_dropout < 1.0):
            raise ValueError("regression_head_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict() already serializes the nested dataclass; nothing else to do.
        return d

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RadiantChemConfig":
        d = dict(data)
        if "base" in d and isinstance(d["base"], dict):
            d["base"] = RadiantConfig.from_dict(d["base"])
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown RadiantChemConfig fields: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str | Path) -> "RadiantChemConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def replace(self, **overrides: Any) -> "RadiantChemConfig":
        return replace(self, **overrides)
