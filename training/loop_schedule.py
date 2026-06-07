"""Per-step ``n_loops`` schedules for training and evaluation.

Pluggable so the same Trainer can run fixed-loop, uniformly-sampled, or
curriculum-style runs without code changes -- only a config swap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


class LoopSchedule:
    """Base class. Subclasses implement :py:meth:`sample`."""

    def sample(self, step: int) -> int:
        raise NotImplementedError


@dataclass
class FixedLoopSchedule(LoopSchedule):
    n: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")

    def sample(self, step: int) -> int:
        return self.n


@dataclass
class UniformLoopSchedule(LoopSchedule):
    """Uniform-integer between ``low`` and ``high`` (inclusive)."""

    low: int
    high: int
    seed: int = 0

    def __post_init__(self) -> None:
        if self.low < 1 or self.high < self.low:
            raise ValueError(f"need 1 <= low <= high, got low={self.low}, high={self.high}")
        self._rng = random.Random(self.seed)

    def sample(self, step: int) -> int:
        return self._rng.randint(self.low, self.high)


@dataclass
class CurriculumLoopSchedule(LoopSchedule):
    """Linearly anneal n_loops from ``start`` to ``end`` over ``n_steps`` steps.

    After ``n_steps`` the schedule clamps at ``end``. Useful when you want
    to train a model up gradually to deeper recurrence.
    """

    start: int
    end: int
    n_steps: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < 1:
            raise ValueError("start and end must be >= 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")

    def sample(self, step: int) -> int:
        frac = min(1.0, max(0.0, step / float(self.n_steps)))
        return int(round(self.start + frac * (self.end - self.start)))
