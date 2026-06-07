"""Training callbacks.

Each callback subclasses :class:`Callback` and overrides whichever methods
it needs. The Trainer invokes them at well-defined points so callers can
slot in logging, early stopping, metric recording, learning-rate schedulers,
etc., without changing the trainer itself.
"""

from __future__ import annotations

import math
import time
from typing import Any


class Callback:
    """No-op base. Override the hooks you care about."""

    def on_train_begin(self, trainer: "Trainer") -> None: ...
    def on_train_end(self, trainer: "Trainer") -> None: ...
    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None: ...
    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict[str, Any]) -> None: ...
    def on_step_end(self, trainer: "Trainer", step: int, logs: dict[str, Any]) -> None: ...
    def on_eval_end(self, trainer: "Trainer", logs: dict[str, Any]) -> None: ...


class LossLogger(Callback):
    """Print step / epoch losses to stdout at a configurable cadence."""

    def __init__(self, every_n_steps: int = 50) -> None:
        self.every = max(1, every_n_steps)
        self._t_start = 0.0

    def on_train_begin(self, trainer):
        self._t_start = time.time()

    def on_step_end(self, trainer, step, logs):
        if step % self.every == 0:
            extra = ""
            if "n_loops" in logs:
                extra += f" loops={logs['n_loops']}"
            if "aux" in logs:
                extra += f" aux={logs['aux']:.4f}"
            print(f"[step {step:>6d}] loss={logs.get('loss', float('nan')):.4f}{extra}", flush=True)

    def on_epoch_end(self, trainer, epoch, logs):
        elapsed = time.time() - self._t_start
        msg = f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in logs.items())
        msg += f" t={elapsed:.1f}s"
        print(msg, flush=True)


class EarlyStopping(Callback):
    """Halt training when a monitored metric stops improving for ``patience`` epochs."""

    def __init__(self, monitor: str = "val_loss", patience: int = 3, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.bad_epochs = 0

    def on_epoch_end(self, trainer, epoch, logs):
        if self.monitor not in logs:
            return
        v = logs[self.monitor]
        improved = v < self.best if self.mode == "min" else v > self.best
        if improved:
            self.best = v
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs > self.patience:
                trainer.should_stop = True


class MetricsRecorder(Callback):
    """Captures all per-step / per-epoch ``logs`` dicts for offline analysis."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.epochs: list[dict[str, Any]] = []
        self.eval_runs: list[dict[str, Any]] = []

    def on_step_end(self, trainer, step, logs):
        self.steps.append({"step": step, **logs})

    def on_epoch_end(self, trainer, epoch, logs):
        self.epochs.append({"epoch": epoch, **logs})

    def on_eval_end(self, trainer, logs):
        self.eval_runs.append(dict(logs))
