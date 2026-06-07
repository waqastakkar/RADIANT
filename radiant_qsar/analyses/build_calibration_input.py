"""Build the long calibration CSV consumed by Phase G.3.

Sources:
* RADIANT ``confidence_var`` in predictions.csv -> ``radiant_halt_var``.
* ``loop_sweep/predictions_nloops*.csv`` -> ``radiant_mc_loops``.
* repeated model directories such as ``radiant_seed1`` -> ensemble rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import discover_predictions
from radiant_qsar.analyses.g3_calibration import mc_loops_to_sigma


BASE_COLUMNS = ["inchikey14", "target_chembl_id", "split_kind", "smiles", "true_pchembl", "pred_pchembl", "sigma_pchembl", "model"]


def _halt_var_frame(predictions_csv: Path) -> pd.DataFrame | None:
    df = pd.read_csv(predictions_csv)
    if "confidence_var" not in df.columns:
        return None
    sigma = np.sqrt(np.clip(pd.to_numeric(df["confidence_var"], errors="coerce").to_numpy(float), 1e-8, None))
    out = df.copy()
    out["sigma_pchembl"] = sigma
    out["model"] = "radiant_halt_var"
    return out[[c for c in BASE_COLUMNS if c in out.columns]]


def _loop_sweep_frame(cell_dir: Path) -> pd.DataFrame | None:
    loop_dir = cell_dir / "loop_sweep"
    if not loop_dir.exists():
        return None
    try:
        out = mc_loops_to_sigma(loop_dir)
    except FileNotFoundError:
        return None
    out["model"] = "radiant_mc_loops"
    return out[[c for c in BASE_COLUMNS if c in out.columns]]


def _ensemble_frames(panel_root: Path, ensemble_prefix: str, split: str | None) -> list[pd.DataFrame]:
    models = [p.name for p in panel_root.iterdir() if p.is_dir() and p.name.startswith(ensemble_prefix)]
    if len(models) < 2:
        return []
    discovered = discover_predictions(panel_root)
    discovered = discovered[discovered["model"].isin(models)]
    if split:
        discovered = discovered[discovered["split"] == split]
    frames: list[pd.DataFrame] = []
    for (target, cell_split), grp in discovered.groupby(["target", "split"]):
        per_seed = []
        for _, row in grp.iterrows():
            df = pd.read_csv(row["path"])
            df["seed_model"] = row["model"]
            per_seed.append(df)
        if len(per_seed) < 2:
            continue
        full = pd.concat(per_seed, ignore_index=True)
        agg = full.groupby(["inchikey14", "target_chembl_id", "split_kind"]).agg(
            pred_pchembl=("pred_pchembl", "mean"),
            sigma_pchembl=("pred_pchembl", "std"),
            true_pchembl=("true_pchembl", "first"),
            smiles=("smiles", "first"),
        ).reset_index()
        agg["sigma_pchembl"] = agg["sigma_pchembl"].fillna(0.0)
        agg["model"] = "radiant_ensemble"
        frames.append(agg[BASE_COLUMNS])
    return frames


def build_calibration_input(
    panel_root: Path,
    out: Path,
    *,
    model: str = "radiant",
    split: str | None = "scaffold",
    include_halt_var: bool = True,
    include_mc_loops: bool = True,
    ensemble_prefix: str | None = None,
) -> pd.DataFrame:
    panel = discover_predictions(panel_root)
    panel = panel[panel["model"] == model]
    if split:
        panel = panel[panel["split"] == split]

    frames: list[pd.DataFrame] = []
    for _, row in panel.iterrows():
        pred_path = Path(row["path"])
        if include_halt_var:
            frame = _halt_var_frame(pred_path)
            if frame is not None:
                frames.append(frame)
        if include_mc_loops:
            frame = _loop_sweep_frame(pred_path.parent)
            if frame is not None:
                frames.append(frame)
    if ensemble_prefix:
        frames.extend(_ensemble_frames(panel_root, ensemble_prefix, split))

    if not frames:
        raise SystemExit(
            f"No calibration sources found under {panel_root}. "
            "Need confidence_var columns, loop_sweep predictions, or ensemble directories."
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[BASE_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="radiant")
    p.add_argument("--split", default="scaffold")
    p.add_argument("--no-halt-var", action="store_true")
    p.add_argument("--no-mc-loops", action="store_true")
    p.add_argument("--ensemble-prefix", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse()
    df = build_calibration_input(
        args.panel_root,
        args.out,
        model=args.model,
        split=args.split,
        include_halt_var=not args.no_halt_var,
        include_mc_loops=not args.no_mc_loops,
        ensemble_prefix=args.ensemble_prefix,
    )
    print(f"wrote {args.out} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
