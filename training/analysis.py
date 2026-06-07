"""Offline loop-dynamics analysis utilities.

Used by example scripts and downstream notebooks to summarize what the
recurrent loop is doing on real data after training.
"""

from __future__ import annotations

import statistics
from typing import Iterable

import torch


@torch.no_grad()
def collect_loop_metrics(
    model,
    loader: Iterable[dict],
    *,
    n_loops: int,
    device: torch.device | str = "cpu",
) -> dict[str, list[float]]:
    """Run ``model`` over ``loader`` and stitch together every batch's per-loop norms.

    Returns a dict with keys ``"norms_per_loop"`` (a list of length ``n_loops``,
    each entry the average across all batches of that loop's hidden-state
    norm) and ``"cos_to_prev_per_loop"`` (length ``n_loops - 1``).
    """
    device = torch.device(device)
    model.eval().to(device)
    norms: list[list[float]] = [[] for _ in range(n_loops)]
    cos: list[list[float]] = [[] for _ in range(n_loops - 1)]

    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch.get("attention_mask")
        if isinstance(attn, torch.Tensor):
            attn = attn.to(device)
        out = _forward(model, ids, n_loops=n_loops, attention_mask=attn, return_loop_metrics=True)
        m = _loop_metrics(out)
        for t, v in enumerate(m.norms):
            if t < n_loops:
                norms[t].append(v)
        for t, v in enumerate(m.cos_to_prev):
            if t < n_loops - 1:
                cos[t].append(v)

    return {
        "norms_per_loop": [statistics.mean(xs) if xs else float("nan") for xs in norms],
        "cos_to_prev_per_loop": [statistics.mean(xs) if xs else float("nan") for xs in cos],
    }


def summarize_loop_dynamics(metrics: dict[str, list[float]]) -> dict[str, float]:
    """Crunch the result of :func:`collect_loop_metrics` into a few scalars."""
    norms = metrics.get("norms_per_loop", [])
    cos = metrics.get("cos_to_prev_per_loop", [])
    out: dict[str, float] = {}
    if norms:
        out["norm_first"] = norms[0]
        out["norm_last"] = norms[-1]
        out["norm_growth_ratio"] = norms[-1] / norms[0] if norms[0] != 0 else float("nan")
    if cos:
        out["cos_first"] = cos[0]
        out["cos_last"] = cos[-1]
        out["cos_min"] = min(cos)
    return out


def _forward(model, input_ids, **kwargs):
    """Call either RadiantModel directly or a chem wrapper."""
    if hasattr(model, "core"):
        out = model(input_ids, **kwargs)
        return out.base if hasattr(out, "base") else out
    return model(input_ids, **kwargs)


def _loop_metrics(out):
    return getattr(out, "loop_metrics", None) or out.base.loop_metrics
