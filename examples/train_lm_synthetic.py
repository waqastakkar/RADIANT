"""Tiny synthetic-task LM training.

Demonstrates the full Trainer + LoopSchedule pipeline on a deliberately
solvable synthetic task: a 4-token alphabet where the target sequence is
the input shifted by one position with the last token replaced by token 0.

A few hundred steps drives the loss well below the random baseline of
``log(4) ~= 1.386``. Runs on CPU in well under a minute.

Usage::

    python examples/train_lm_synthetic.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radiant import RadiantConfig, RadiantModel
from training import (
    FixedLoopSchedule,
    LossLogger,
    MetricsRecorder,
    Trainer,
)


class ShiftLMDataset(Dataset):
    """A trivially learnable LM task: target[t] = input[t-1].

    With attention + residual stream the model can learn the
    "shift right" copy in a few hundred steps; the validation loss
    should fall well below the random baseline of ``log(vocab_size)``.
    """

    def __init__(self, n: int = 256, seq_len: int = 16, vocab_size: int = 8):
        self.x = torch.randint(0, vocab_size, (n, seq_len))
        # target[t] = input[t-1]; first position predicts a sentinel (0).
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
        n_loops_train=3, min_loops=1, max_loops=6,
    )
    model = RadiantModel(cfg)
    print(f"params: {model.num_params():,}")

    train_ds = ShiftLMDataset(n=256, seq_len=16, vocab_size=cfg.vocab_size)
    val_ds = ShiftLMDataset(n=64, seq_len=16, vocab_size=cfg.vocab_size)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rec = MetricsRecorder()
    trainer = Trainer(
        model, opt, lm_loss,
        loop_schedule=FixedLoopSchedule(3),
        callbacks=[LossLogger(every_n_steps=20), rec],
        grad_clip=1.0,
    )
    trainer.fit(train_loader, val_loader=val_loader, epochs=3)

    if rec.epochs:
        last = rec.epochs[-1]
        print(f"final train_loss={last['train_loss']:.4f} val_loss={last.get('val_loss', float('nan')):.4f}")


if __name__ == "__main__":
    main()
