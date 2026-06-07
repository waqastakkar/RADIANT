"""ChEMBL-shaped property regression with RADIANT-Chem.

Reads a CSV (default: examples/data/sample_chembl.csv), tokenizes SMILES,
splits scaffold-style, and trains a regression head on the indicated
column. Tiny by default so this runs on CPU in under a minute; bump
config sizes for serious work.

Usage::

    python examples/train_chem_property.py
    python examples/train_chem_property.py --csv my.csv --target pIC50
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from radiant import RadiantConfig
from radiant_chem import (
    ChemblCsvDataset,
    RadiantChemConfig,
    RadiantChemModel,
    SmilesTokenizer,
    scaffold_split,
)
from radiant_chem.objectives import RegressionLoss
from radiant_chem.tasks import TaskRegistry, TaskSpec
from training import FixedLoopSchedule, LossLogger, MetricsRecorder, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(Path(__file__).parent / "data" / "sample_chembl.csv"))
    p.add_argument("--smiles-column", type=str, default="smiles")
    p.add_argument("--target", type=str, default="logP_demo")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-loops", type=int, default=3)
    p.add_argument("--max-len", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build vocab from full CSV first.
    smiles_all: list[str] = []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get(args.smiles_column):
                smiles_all.append(row[args.smiles_column])
    tok = SmilesTokenizer.from_corpus(smiles_all)
    print(f"corpus size={len(smiles_all)} vocab={tok.vocab_size}")

    base = RadiantConfig(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
        d_model=96,
        n_query_heads=4, n_kv_heads=2, head_dim=24, d_ff=192,
        max_seq_len=args.max_len,
        n_stem_blocks=1, n_refinement_blocks=1, n_exit_blocks=1,
        n_loops_train=args.n_loops, min_loops=1, max_loops=max(8, args.n_loops + 2),
    )
    cfg = RadiantChemConfig(base=base)
    tasks = TaskRegistry([TaskSpec(name=args.target, kind="regression",
                                   target_column=args.target, num_outputs=1)])
    model = RadiantChemModel(cfg, tasks)
    print(f"params: {model.num_params():,}")

    full = ChemblCsvDataset(
        args.csv, tok,
        smiles_column=args.smiles_column,
        target_columns=[args.target],
        max_len=args.max_len,
    )
    train_idx, val_idx, test_idx = scaffold_split(full.smiles, ratios=(0.7, 0.15, 0.15), seed=0)
    train_loader = DataLoader(Subset(full, train_idx), batch_size=args.batch_size,
                              shuffle=True, collate_fn=full.collate)
    val_loader = DataLoader(Subset(full, val_idx), batch_size=args.batch_size,
                            shuffle=False, collate_fn=full.collate)

    reg = RegressionLoss()

    def loss_fn(out, batch):
        pred = out.task_outputs[args.target].squeeze(-1)
        return reg(pred, batch["targets"][:, 0])

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    rec = MetricsRecorder()
    trainer = Trainer(
        model, opt, loss_fn,
        loop_schedule=FixedLoopSchedule(args.n_loops),
        callbacks=[LossLogger(every_n_steps=10), rec],
        grad_clip=1.0,
        forward_kwargs={"is_causal": False},
    )
    trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)

    if rec.epochs:
        last = rec.epochs[-1]
        print(f"final train_loss={last.get('train_loss', float('nan')):.4f} "
              f"val_loss={last.get('val_loss', float('nan')):.4f}")


if __name__ == "__main__":
    main()
