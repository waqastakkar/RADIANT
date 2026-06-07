"""Generic training scaffold for RADIANT models.

Minimal, dependency-free Trainer with hooks for callbacks. Loop counts are
sampled per step from a ``LoopSchedule`` so the same model can be trained
under fixed, uniform, or curriculum loop-depth schedules without changes
to the training loop.
"""

from training.trainer import Trainer
from training.callbacks import (
    Callback,
    LossLogger,
    EarlyStopping,
    MetricsRecorder,
)
from training.loop_schedule import (
    LoopSchedule,
    FixedLoopSchedule,
    UniformLoopSchedule,
    CurriculumLoopSchedule,
)
from training.analysis import collect_loop_metrics, summarize_loop_dynamics

__all__ = [
    "Trainer",
    "Callback",
    "LossLogger",
    "EarlyStopping",
    "MetricsRecorder",
    "LoopSchedule",
    "FixedLoopSchedule",
    "UniformLoopSchedule",
    "CurriculumLoopSchedule",
    "collect_loop_metrics",
    "summarize_loop_dynamics",
]
