"""Print per-loop hidden-state dynamics for a tiny model on synthetic data.

Demonstrates how to use ``return_loop_metrics=True`` and the helpers in
``training.analysis``. Useful as a starting point for paper figures.

Usage::

    python examples/analyze_loop_dynamics.py
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from radiant import RadiantModel, halting_summary, tiny_config
from training.analysis import collect_loop_metrics, summarize_loop_dynamics


class _RandomBatches(Dataset):
    def __init__(self, n: int = 16, seq_len: int = 12, vocab_size: int = 32):
        self.x = torch.randint(0, vocab_size, (n, seq_len))

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.x[idx]}


def main() -> None:
    cfg = tiny_config(iteration_signal_kind="sinusoidal", use_confidence_halting=True)
    model = RadiantModel(cfg).eval()

    ds = _RandomBatches(n=16, seq_len=12, vocab_size=cfg.vocab_size)
    loader = DataLoader(ds, batch_size=4)

    # Single-batch path: print everything.
    x = next(iter(loader))["input_ids"]
    with torch.no_grad():
        out = model(x, n_loops=cfg.max_loops, return_loop_metrics=True)
    print("=== single-batch loop metrics ===")
    print(f"per-loop |h| (mean over tokens):  {[round(v, 3) for v in out.loop_metrics.norms]}")
    print(f"cos(h_t, h_{{t-1}}):              {[round(v, 4) for v in out.loop_metrics.cos_to_prev]}")
    print(f"cos(h_t, h_0):                   {[round(v, 4) for v in out.loop_metrics.cos_to_first]}")

    if out.halting is not None:
        print("=== confidence halting summary ===")
        for k, v in halting_summary(out.halting).items():
            print(f"  {k:30s} = {v:.4f}")

    # Multi-batch path: average across the loader.
    print("=== averaged across loader ===")
    metrics = collect_loop_metrics(model, loader, n_loops=cfg.max_loops)
    print(f"avg per-loop |h|:           {[round(v, 3) for v in metrics['norms_per_loop']]}")
    print(f"avg cos(h_t, h_{{t-1}}):    {[round(v, 4) for v in metrics['cos_to_prev_per_loop']]}")
    summary = summarize_loop_dynamics(metrics)
    print("scalar summary:")
    for k, v in summary.items():
        print(f"  {k:30s} = {v:.4f}")


if __name__ == "__main__":
    main()
