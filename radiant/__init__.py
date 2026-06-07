"""RADIANT: an original recurrent-depth transformer.

High-level data flow::

    input_ids
        |
        v
    StemEncoder              # once
        |  e
        v
    IterativeRefinementCore  # loops; weight-shared transformer Core,
        |                    # StateAnchorUpdate per step, optional
        |  h                 # IterationAdapter and ConfidenceHalting
        v
    ExitDecoder              # once
        |
        v
    LMHead / task heads      # logits

Each box is a small, single-purpose module. Imports here re-export the
public API so most callers only need ``from radiant import ...``.
"""

from radiant.config import RadiantConfig
from radiant.model import RadiantModel, RadiantOutput
from radiant.stem_encoder import StemEncoder
from radiant.exit_decoder import ExitDecoder
from radiant.refinement_core import IterativeRefinementCore
from radiant.state_anchor import StateAnchorUpdate
from radiant.iteration_signal import IterationSignal
from radiant.iteration_adapter import IterationAdapter
from radiant.confidence_halting import ConfidenceHalting, HaltingTrace
from radiant.attention import GQAAttention
from radiant.feedforward import SwiGLUFeedForward, MoEFeedForward
from radiant.block import TransformerBlock
from radiant.norms import RMSNorm
from radiant.positional import build_rope_cache, apply_rope
from radiant.heads import LMHead, RegressionHead, ClassificationHead, PoolingHead
from radiant.metrics import (
    LoopMetrics,
    spectral_radius_estimate,
    router_load_entropy,
    halting_summary,
)
from radiant.presets import tiny_config, small_config, base_config

__version__ = "0.1.0"

__all__ = [
    "RadiantConfig",
    "RadiantModel",
    "RadiantOutput",
    "StemEncoder",
    "ExitDecoder",
    "IterativeRefinementCore",
    "StateAnchorUpdate",
    "IterationSignal",
    "IterationAdapter",
    "ConfidenceHalting",
    "HaltingTrace",
    "GQAAttention",
    "SwiGLUFeedForward",
    "MoEFeedForward",
    "TransformerBlock",
    "RMSNorm",
    "build_rope_cache",
    "apply_rope",
    "LMHead",
    "RegressionHead",
    "ClassificationHead",
    "PoolingHead",
    "LoopMetrics",
    "spectral_radius_estimate",
    "router_load_entropy",
    "halting_summary",
    "tiny_config",
    "small_config",
    "base_config",
]
