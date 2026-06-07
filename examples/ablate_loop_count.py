"""Ablation: same checkpoint evaluated at multiple loop counts.

Trains a small RADIANT briefly, then evaluates it at n_loops in
{1, 2, 4, 8, max_loops}. Demonstrates that the same model can be queried
at any depth, which is the cornerstone of the "controllable compute"
property.

Usage::

    python examples/ablate_loop_count.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radiant import RadiantConfig, RadiantModel
from training import FixedLoopSchedule, MetricsRecorder, Trainer


class _Synth(Dataset):
    """Predict the previous token -- learnable via shifted attention."""

    def __init__(self, n: int = 256, seq_len: int = 16, vocab_size: int = 8):
        self.x = torch.randint(0, vocab_size, (n, seq_len))
        self.y = torch.cat([torch.zeros((n, 1), dtype=torch.long), self.x[:, :-1]], dim=1)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"input_ids": self.x[idx], "labels": self.y[idx]}


def lm_loss(out, batch):
    return F.cross_entropy(
        out.logits.reshape(-1, out.logits.size(-1)),
        batch["labels"].reshape(-1),
    )


def main() -> None:
    cfg = RadiantConfig(
        vocab_size=8,
        d_model=64,
        n_query_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
        max_seq_len=32,
        n_stem_blocks=1, n_refinement_blocks=1, n_exit_blocks=1,
        n_loops_train=4, min_loops=1, max_loops=16,
        iteration_signal_kind="sinusoidal",  # extrapolation-friendly
    )
    model = RadiantModel(cfg)
    train_ds = _Synth(n=256, seq_len=16, vocab_size=cfg.vocab_size)
    val_ds = _Synth(n=64, seq_len=16, vocab_size=cfg.vocab_size)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print(f"params: {model.num_params():,}")
    print("=== brief training ===")
    trainer = Trainer(
        model, opt, lm_loss,
        loop_schedule=FixedLoopSchedule(4),
        callbacks=[MetricsRecorder()],
        grad_clip=1.0,
    )
    trainer.fit(train_loader, val_loader=val_loader, epochs=2)

    # Sweep eval depth.
    print("=== eval depth sweep ===")
    rows: list[tuple[int, float]] = []
    for n_loops in (1, 2, 4, 8, cfg.max_loops, cfg.max_loops + 4):
        m = trainer.evaluate(val_loader, n_loops=n_loops)
        rows.append((n_loops, m["loss"]))
        print(f"  n_loops={n_loops:>3}  val_loss={m['loss']:.4f}")

    # Print loss-vs-depth curve.
    print("\nloss-vs-depth (CSV):\nn_loops,val_loss")
    for n, l in rows:
        print(f"{n},{l:.6f}")


if __name__ == "__main__":
    main()
