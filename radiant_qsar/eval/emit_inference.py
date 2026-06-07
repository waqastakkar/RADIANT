"""Re-emit predictions.csv on an existing trained cell with halting extras.

Use this when you've already trained a RADIANT cell (so a ``best.pt``
exists) but its ``predictions.csv`` was written *before* the
fine-tune driver started emitting halting extras. It reconstructs the
deterministic test split for that (target, split, seed) triple, loads
the checkpoint, and re-runs the test forward pass -- emitting:

* the canonical 7 columns (``idx, inchikey14, target_chembl_id,
  split_kind, smiles, true_pchembl, pred_pchembl``)
* the four standard halting extras (``halt_step, effective_depth,
  confidence_var, tokens``)
* optionally ``per_atom_halt`` (a JSON-encoded per-atom heat map,
  requires rdkit; enabled with ``--emit-per-atom``)

This is a pure inference pass -- no training. On the default
75M-parameter cell it runs in a couple of minutes on a single GPU.

Two CLI styles
--------------

1. **Auto-discovery from a cell directory** (recommended for an
   existing panel layout)::

     python -m radiant_qsar.eval.emit_inference \\
         --cell-dir runs/panel_75m/radiant/CHEMBL203/scaffold \\
         --activities data/processed/v1/activities.parquet \\
         --vocab      data/processed/v1/smiles_vocab.json \\
         --config     configs/radiant_75m.json

   ``--cell-dir`` infers ``target_chembl_id`` from the parent dir,
   ``split_kind`` from the leaf, and ``--checkpoint`` from
   ``cell-dir/best.pt``. The original training invocation is parsed
   from ``cell.log`` to recover ``--seed`` and ``--n-loops-train``.

2. **Explicit** (when re-emitting on a checkpoint that doesn't live in
   a per-cell directory)::

     python -m radiant_qsar.eval.emit_inference \\
         --checkpoint runs/.../best.pt \\
         --config     configs/radiant_75m.json \\
         --vocab      data/processed/v1/smiles_vocab.json \\
         --activities data/processed/v1/activities.parquet \\
         --target     CHEMBL203 \\
         --split      scaffold \\
         --seed       1337 \\
         --out        runs/.../emitted_predictions.csv

The default output path is ``<cell-dir>/predictions.csv`` (overwriting
the existing one); pass ``--out`` to write somewhere else. The script
prints a warning and refuses to overwrite if the destination already
contains halting-extras columns, so you don't lose a richer file by
accident.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell-dir auto-discovery
# ---------------------------------------------------------------------------

@dataclass
class CellSpec:
    cell_dir: Path
    target: str
    split: str
    checkpoint: Path
    seed: int = 1337
    n_loops_train: int = 8

    @classmethod
    def from_dir(cls, cell_dir: Path) -> "CellSpec":
        cell_dir = Path(cell_dir).resolve()
        if not cell_dir.exists():
            raise SystemExit(f"cell-dir not found: {cell_dir}")

        split = cell_dir.name
        target = cell_dir.parent.name
        ckpt = cell_dir / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")

        spec = cls(cell_dir=cell_dir, target=target, split=split, checkpoint=ckpt)

        log_path = cell_dir / "cell.log"
        if log_path.exists():
            spec._parse_log(log_path.read_text(encoding="utf-8", errors="ignore"))
        return spec

    def _parse_log(self, text: str) -> None:
        # cell.log starts with a `$ python -m ... --flag value --flag value ...` line.
        m = re.search(r"--seed\s+(\d+)", text)
        if m:
            self.seed = int(m.group(1))
        m = re.search(r"--n-loops-train\s+(\d+)", text)
        if m:
            self.n_loops_train = int(m.group(1))
        m = re.search(r"--target(?:-chembl-id)?\s+(\S+)", text)
        if m:
            self.target = m.group(1)
        m = re.search(r"--split(?:-kind)?\s+(\S+)", text)
        if m:
            self.split = m.group(1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    *,
    activities: Path,
    target: str,
    split_kind: str,
    seed: int,
    config: Path,
    vocab: Path,
    checkpoint: Path,
    n_loops: int,
    device: str,
    batch_size: int = 32,
    splits_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    activity_cliff_sim: float = 0.9,
    activity_cliff_delta: float = 1.0,
    emit_per_atom: bool = False,
) -> dict[str, Any]:
    """Run the test-split forward pass and return a row-aligned dict."""
    import torch
    from torch.utils.data import DataLoader, Subset

    from radiant import RadiantConfig
    from radiant_chem import (
        RadiantChemConfig,
        RadiantChemModel,
        SmilesTokenizer,
    )
    from radiant_chem.tasks import TaskRegistry, TaskSpec
    from radiant_qsar.eval.halting_extras import HaltingExtrasAccumulator
    from radiant_qsar.finetune.single_task import (
        SingleTaskTrainArgs,
        _ActivityDataset,
        _collate,
        _select_target,
        _split,
    )

    sub = _select_target(activities, target)
    train_idx, val_idx, test_idx = _split(
        sub, split_kind, splits_ratios, seed,
        sim=activity_cliff_sim, delta=activity_cliff_delta,
    )
    logger.info("[%s/%s seed=%d] test rows=%d", target, split_kind, seed, len(test_idx))

    tokenizer = SmilesTokenizer.load(vocab)
    base_cfg = RadiantConfig.from_json(config).replace(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        n_loops_train=n_loops,
    )
    chem_cfg = RadiantChemConfig(base=base_cfg)
    tasks = TaskRegistry([TaskSpec("pchembl", "regression", "pchembl", num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    smi = sub["standard_smiles"].tolist()
    pch = sub["pchembl"].astype(float).tolist()
    ds = _ActivityDataset(smi, pch, tokenizer, max_len=base_cfg.max_seq_len)
    pad = tokenizer.pad_id
    loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: _collate(b, pad),
    )

    halting_acc = HaltingExtrasAccumulator(include_per_atom=emit_per_atom)
    pred, true = [], []
    test_smiles = [smi[i] for i in test_idx]
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tgt = batch["targets"]
            out = model(ids, attention_mask=attn, n_loops=n_loops, is_causal=False)
            pred.extend(out.task_outputs["pchembl"].squeeze(-1).cpu().tolist())
            true.extend(tgt.tolist())
            B = ids.shape[0]
            batch_smiles = test_smiles[cursor:cursor + B] if emit_per_atom else None
            cursor += B
            halting_acc.add(
                halting=out.base.halting,
                input_ids=ids,
                attention_mask=attn,
                pad_id=pad,
                id_to_token=tokenizer.id_to_token,
                smiles_batch=batch_smiles,
            )

    inchikeys = sub["inchikey14"].iloc[test_idx].tolist()
    return {
        "idx": list(test_idx),
        "inchikey14": inchikeys,
        "smiles": test_smiles,
        "true_pchembl": true,
        "pred_pchembl": pred,
        "extras": halting_acc.finalize(),
    }


# ---------------------------------------------------------------------------
# Output / safety
# ---------------------------------------------------------------------------

def _existing_has_halting_columns(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    return "effective_depth" in header or "halt_step" in header


def write_emitted(
    out_path: Path,
    *,
    target: str,
    split_kind: str,
    result: dict[str, Any],
    overwrite: bool,
    backup_suffix: str = ".pre_emit.bak",
) -> Path:
    from radiant_qsar.eval.predictions import write_predictions

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        if _existing_has_halting_columns(out_path) and not overwrite:
            raise SystemExit(
                f"{out_path} already has halting columns; pass --overwrite to replace."
            )
        backup = out_path.with_suffix(out_path.suffix + backup_suffix)
        shutil.copy2(out_path, backup)
        logger.info("backed up existing %s -> %s", out_path, backup)

    written = write_predictions(
        out_path.parent,
        indices=result["idx"],
        inchikeys=result["inchikey14"],
        smiles=result["smiles"],
        true_pchembl=result["true_pchembl"],
        pred_pchembl=result["pred_pchembl"],
        target_chembl_id=target,
        split_kind=split_kind,
        extra_columns=result["extras"],
        filename=out_path.name,
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-emit predictions.csv with halting extras")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell-dir", type=Path,
                     help="Auto-discover target/split/seed/checkpoint from a panel cell directory")
    src.add_argument("--checkpoint", type=Path,
                     help="Path to a trained best.pt (use with --target/--split/--seed)")

    p.add_argument("--activities", required=True, type=Path,
                   help="data/processed/v1/activities.parquet")
    p.add_argument("--vocab", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="RadiantConfig json -- MUST match what the cell was trained with")

    p.add_argument("--target", type=str, help="(explicit mode) target ChEMBL id")
    p.add_argument("--split", type=str, help="(explicit mode) split kind")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-loops", type=int, default=8)

    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path,
                   help="Output predictions.csv; defaults to <cell-dir>/predictions.csv")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing predictions.csv that already has halting columns")
    p.add_argument("--allow-empty-halting", action="store_true",
                   help="Allow overwriting when the checkpoint has no halting head "
                        "(use_confidence_halting=False). Off by default to avoid silently "
                        "losing the original predictions.")
    p.add_argument("--emit-per-atom", action="store_true",
                   help="Also emit per_atom_halt (requires rdkit; ~3x slower per batch)")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()

    if args.cell_dir is not None:
        spec = CellSpec.from_dir(args.cell_dir)
        target = spec.target
        split_kind = spec.split
        seed = spec.seed
        n_loops = spec.n_loops_train if args.n_loops == 8 else args.n_loops
        checkpoint = spec.checkpoint
        out_path = args.out or (spec.cell_dir / "predictions.csv")
        logger.info("cell-dir mode: target=%s split=%s seed=%d n_loops=%d ckpt=%s",
                    target, split_kind, seed, n_loops, checkpoint)
    else:
        if not (args.target and args.split and args.out):
            raise SystemExit("explicit mode requires --target, --split, --out")
        target = args.target
        split_kind = args.split
        seed = args.seed
        n_loops = args.n_loops
        checkpoint = args.checkpoint
        out_path = args.out

    result = run_inference(
        activities=args.activities,
        target=target,
        split_kind=split_kind,
        seed=seed,
        config=args.config,
        vocab=args.vocab,
        checkpoint=checkpoint,
        n_loops=n_loops,
        device=args.device,
        batch_size=args.batch_size,
        emit_per_atom=args.emit_per_atom,
    )

    eff = result["extras"]["effective_depth"]
    eff_arr = np.array([v for v in eff if v == v], dtype=float)  # drop NaN
    halting_present = eff_arr.size > 0

    # Refuse to overwrite a real predictions.csv with one that has no halting
    # signal -- that loses information rather than adding any. The user has
    # to ask for it explicitly via --allow-empty-halting.
    if not halting_present and out_path.exists() and not args.allow_empty_halting:
        logger.error(
            "checkpoint has no halting head (use_confidence_halting=False in the config),\n"
            "  so all halting extras would be NaN. Refusing to overwrite %s.\n"
            "  To proceed anyway, pass --allow-empty-halting. To get a *real* halting\n"
            "  trace, set use_confidence_halting=true in the config and re-fine-tune.",
            out_path,
        )
        return 2

    written = write_emitted(
        out_path,
        target=target,
        split_kind=split_kind,
        result=result,
        overwrite=args.overwrite,
    )
    if halting_present:
        logger.info("wrote %s  n=%d  mean effective_depth=%.2f  range=[%.2f, %.2f]",
                    written, len(result["idx"]), float(eff_arr.mean()),
                    float(eff_arr.min()), float(eff_arr.max()))
    else:
        logger.warning(
            "wrote %s  n=%d  but effective_depth was empty (halting disabled). "
            "G.1 / G.4 / G.5 / label-permutation cannot run on this checkpoint.",
            written, len(result["idx"]),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
