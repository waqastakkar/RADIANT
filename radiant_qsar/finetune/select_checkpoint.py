"""Select a validated RADIANT checkpoint before screening.

The screening stage should never guess which fine-tuned model to use.
This gate reads either ``panel_results.csv`` or the per-cell ``result.json``
files from a sweep, ranks validated cells by a chosen validation metric, and
writes a small manifest consumed by downstream screening scripts.

By default this is intentionally RADIANT-only. Morgan/RF remains available as
a baseline in the panel, but it is not used to choose a production scoring
checkpoint for the main model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _metric_value(rec: dict[str, Any], metric: str) -> float | None:
    candidates = [
        metric,
        f"val_{metric}",
        f"best_val_{metric}",
    ]
    for key in candidates:
        if key in rec:
            try:
                return float(rec[key])
            except Exception:
                return None
    block = rec.get("val")
    if isinstance(block, dict) and metric in block:
        try:
            return float(block[metric])
        except Exception:
            return None
    return None


def _scan_result_json(panel_root: Path, model: str, target: str | None, split: str | None, metric: str) -> list[dict]:
    rows: list[dict] = []
    root = panel_root / model
    for result_path in sorted(root.glob("*/*/result.json")):
        cell = result_path.parent
        cell_target = cell.parent.name
        cell_split = cell.name
        if target and cell_target != target:
            continue
        if split and cell_split != split:
            continue
        rec = json.loads(result_path.read_text(encoding="utf-8"))
        value = _metric_value(rec, metric)
        ckpt = cell / rec.get("model_path", "best.pt")
        if value is None or not ckpt.exists():
            continue
        rows.append({
            "model": model,
            "target_chembl_id": cell_target,
            "split": cell_split,
            f"val_{metric}": value,
            "checkpoint_path": str(ckpt),
            "result_path": str(result_path),
            "predictions_path": str(cell / rec.get("predictions_path", "predictions.csv")),
            "chem_config_path": str(cell / "chem_config.json"),
            "task_name": "pchembl",
        })
    return rows


def _read_panel_results(panel_root: Path, model: str, target: str | None, split: str | None, metric: str) -> list[dict]:
    csv_path = panel_root / "panel_results.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "model" in df.columns:
        df = df[df["model"] == model]
    if target and "target_chembl_id" in df.columns:
        df = df[df["target_chembl_id"] == target]
    if split and "split" in df.columns:
        df = df[df["split"] == split]
    metric_col = f"val_{metric}"
    if metric_col not in df.columns:
        return []
    rows: list[dict] = []
    for _, row in df.iterrows():
        cell = panel_root / model / str(row["target_chembl_id"]) / str(row["split"])
        ckpt = cell / "best.pt"
        if not ckpt.exists() or pd.isna(row[metric_col]):
            continue
        rows.append({
            "model": model,
            "target_chembl_id": str(row["target_chembl_id"]),
            "split": str(row["split"]),
            metric_col: float(row[metric_col]),
            "checkpoint_path": str(ckpt),
            "result_path": str(cell / "result.json"),
            "predictions_path": str(cell / "predictions.csv"),
            "chem_config_path": str(cell / "chem_config.json"),
            "task_name": "pchembl",
        })
    return rows


def select_checkpoint(
    panel_root: Path,
    *,
    out: Path,
    model: str = "radiant",
    target: str | None = None,
    split: str | None = "scaffold",
    metric: str = "pearson",
    maximize: bool = True,
    vocab: Path | None = None,
) -> dict:
    rows = _read_panel_results(panel_root, model, target, split, metric)
    if not rows:
        rows = _scan_result_json(panel_root, model, target, split, metric)
    if not rows:
        scope = "/".join(x for x in (model, target or "*", split or "*") if x)
        raise SystemExit(
            f"No selectable {scope} checkpoints under {panel_root}. "
            "Run the RADIANT panel first and make sure result.json + best.pt exist."
        )

    metric_col = f"val_{metric}"
    rows = sorted(rows, key=lambda r: r[metric_col], reverse=maximize)
    selected = rows[0]
    selected["selection"] = {
        "panel_root": str(panel_root),
        "model_filter": model,
        "target_filter": target,
        "split_filter": split,
        "metric": metric_col,
        "maximize": maximize,
        "n_candidates": len(rows),
    }
    if vocab is not None:
        selected["vocab_path"] = str(vocab)

    ckpt_path = Path(selected["checkpoint_path"])
    cfg_path = Path(selected["chem_config_path"])
    if not cfg_path.exists():
        raise SystemExit(
            f"Selected checkpoint is missing {cfg_path}. "
            "Re-run fine-tuning after the checkpoint config export patch."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    return selected


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="radiant", choices=("radiant",))
    p.add_argument("--target", default=None)
    p.add_argument("--split", default="scaffold")
    p.add_argument("--metric", default="pearson")
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--vocab", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse()
    selected = select_checkpoint(
        args.panel_root,
        out=args.out,
        model=args.model,
        target=args.target,
        split=args.split,
        metric=args.metric,
        maximize=not args.minimize,
        vocab=args.vocab,
    )
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
