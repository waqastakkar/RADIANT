"""Task-spec dataclasses and a small registry.

A :class:`TaskSpec` describes a downstream task (regression / classification)
in terms of the column it reads from a CSV and the head it attaches to
:class:`RadiantChemModel`. The :class:`TaskRegistry` lets multiple tasks
coexist on the same model so a single forward can produce several heads'
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

TaskKind = Literal["regression", "classification"]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    kind: TaskKind
    target_column: str
    num_outputs: int = 1
    log_scale: bool = False  # for regression tasks

    def __post_init__(self) -> None:
        if self.kind == "classification" and self.num_outputs < 2:
            raise ValueError(
                f"classification task {self.name!r} needs num_outputs >= 2"
            )
        if self.kind == "regression" and self.num_outputs < 1:
            raise ValueError(
                f"regression task {self.name!r} needs num_outputs >= 1"
            )


class TaskRegistry:
    """A small ordered map from task name to :class:`TaskSpec`."""

    def __init__(self, tasks: list[TaskSpec] | None = None) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        for t in tasks or []:
            self.register(t)

    def register(self, task: TaskSpec) -> None:
        if task.name in self._tasks:
            raise KeyError(f"Task {task.name!r} already registered")
        self._tasks[task.name] = task

    def get(self, name: str) -> TaskSpec:
        return self._tasks[name]

    def names(self) -> list[str]:
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, name: str) -> bool:
        return name in self._tasks

    def __iter__(self):
        return iter(self._tasks.values())
