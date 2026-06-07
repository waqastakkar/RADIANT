"""A small, dependency-free Trainer for RADIANT.

Goals:
* Stay short and readable -- not a Lightning replacement.
* Loop counts are sampled per step from a :class:`LoopSchedule`, so the
  *same* training loop runs fixed-loop, uniform, or curriculum experiments.
* Callbacks are the only extension point. Plug logging or early stopping
  via :class:`Callback` subclasses.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.callbacks import Callback
from training.loop_schedule import FixedLoopSchedule, LoopSchedule


LossFn = Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor]


class Trainer:
    """A minimal training driver.

    Args:
        model:        The torch module being trained (RADIANT or chem variant).
        optimizer:    Configured optimizer.
        loss_fn:      ``(model_output, batch_dict) -> scalar loss``. ``model_output``
                      is whatever ``model.forward(...)`` returned.
        loop_schedule: How to pick ``n_loops`` per step. Defaults to fixed at
                      ``model.cfg.n_loops_train`` if available.
        callbacks:    List of :class:`Callback` instances.
        device:       Where to move the model and batches.
        grad_clip:    Optional max-norm gradient clipping value.
        forward_kwargs: Extra kwargs forwarded to ``model(**batch, **forward_kwargs)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFn,
        *,
        loop_schedule: LoopSchedule | None = None,
        callbacks: Sequence[Callback] = (),
        device: torch.device | str = "cpu",
        grad_clip: float | None = 1.0,
        forward_kwargs: dict | None = None,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = list(callbacks)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.forward_kwargs = dict(forward_kwargs or {})

        if loop_schedule is None:
            n = getattr(getattr(model, "cfg", None), "n_loops_train", None)
            n = n or getattr(getattr(getattr(model, "cfg", None), "base", None), "n_loops_train", 4)
            loop_schedule = FixedLoopSchedule(n)
        self.loop_schedule = loop_schedule

        self.global_step = 0
        self.should_stop = False

    # ------------------------------------------------------------------
    def _batch_to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        epochs: int = 1,
    ) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(self)
        try:
            for epoch in range(epochs):
                if self.should_stop:
                    break
                for cb in self.callbacks:
                    cb.on_epoch_begin(self, epoch)
                epoch_loss = 0.0
                n_seen = 0
                self.model.train()
                for batch in train_loader:
                    if self.should_stop:
                        break
                    n_loops = self.loop_schedule.sample(self.global_step)
                    batch = self._batch_to_device(batch)
                    loss, logs = self._step(batch, n_loops)
                    epoch_loss += loss.item() * self._batch_size(batch)
                    n_seen += self._batch_size(batch)
                    for cb in self.callbacks:
                        cb.on_step_end(self, self.global_step, logs)
                    self.global_step += 1

                logs = {"train_loss": epoch_loss / max(n_seen, 1)}
                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader)
                    logs.update({f"val_{k}": v for k, v in val_metrics.items()})
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, logs)
        finally:
            for cb in self.callbacks:
                cb.on_train_end(self)

    # ------------------------------------------------------------------
    def _step(self, batch: dict, n_loops: int) -> tuple[torch.Tensor, dict]:
        self.optimizer.zero_grad(set_to_none=True)
        out = self._forward(batch, n_loops)
        loss = self.loss_fn(out, batch)
        # Add MoE / aux losses if the model exposes them.
        aux_total = self._aux_loss(out)
        if aux_total is not None:
            loss = loss + aux_total
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        logs = {"loss": float(loss.detach().item()), "n_loops": n_loops}
        if aux_total is not None:
            logs["aux"] = float(aux_total.detach().item())
        return loss, logs

    def _forward(self, batch: dict, n_loops: int):
        # Discover argument names ``forward`` accepts. This lets the same
        # trainer drive both RadiantModel and RadiantChemModel.
        kwargs = dict(self.forward_kwargs)
        kwargs["n_loops"] = n_loops
        for k in ("attention_mask", "is_causal"):
            if k in batch:
                kwargs[k] = batch[k]
        return self.model(batch["input_ids"], **kwargs)

    @staticmethod
    def _aux_loss(out) -> torch.Tensor | None:
        # RadiantOutput has .aux_loss; ChemForwardOutput nests it under .base.
        for path in (("aux_loss",), ("base", "aux_loss")):
            obj = out
            ok = True
            for p in path:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    ok = False
                    break
            if ok and obj is not None:
                return obj
        return None

    @staticmethod
    def _batch_size(batch: dict) -> int:
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return int(v.size(0))
        return 1

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, *, n_loops: int | None = None) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_seen = 0
        n_eval = n_loops if n_loops is not None else self.loop_schedule.sample(self.global_step)
        for batch in loader:
            batch = self._batch_to_device(batch)
            out = self._forward(batch, n_eval)
            loss = self.loss_fn(out, batch)
            aux = self._aux_loss(out)
            if aux is not None:
                loss = loss + aux
            B = self._batch_size(batch)
            total_loss += loss.item() * B
            n_seen += B
        logs = {"loss": total_loss / max(n_seen, 1), "n_loops": n_eval}
        for cb in self.callbacks:
            cb.on_eval_end(self, logs)
        return logs
