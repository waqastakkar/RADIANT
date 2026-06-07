"""Phase G -- Halting ON vs OFF comparison.

The default inference path with `n_loops <= max_loops` lets the halting
head decide *when to stop*. Running G.4's loop sweep at a FIXED `n_loops`
effectively disables that decision -- every molecule processes for
exactly K loops. Comparing default predictions to the loop-sweep K-loops
predictions therefore answers "did halting help, hurt, or do nothing?"

This module is pure post-processing on:
* runs/panel/radiant/<TARGET>/scaffold/predictions.csv         (halting ON)
* runs/panel/radiant/<TARGET>/scaffold/loop_sweep/predictions_nloops{K}.csv
  for K in {1, 2, 4, 8, 12, 16}                                (halting OFF)

For each cell we compute MAE under each setting and report:
* MAE per K
* Δ(MAE_K - MAE_halting_on) -- positive = fixed-depth is worse
* per-cell bar chart
* aggregate MAE-vs-K curve

NOTE: in the current checkpoint the halting head collapsed to depth=2
(documented in g1_depth_distribution.png), so we expect halting-ON to be
indistinguishable from fixed K=2. This module makes that explicit with
numbers reviewers can audit.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


def _cell_mae(csv_path: Path) -> float | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "true_pchembl" not in df.columns or "pred_pchembl" not in df.columns:
        return None
    return float((df["pred_pchembl"] - df["true_pchembl"]).abs().mean())


def run(*, panel_root: Path | str, out_dir: Path | str,
        lf_model_dir: str = "radiant", split: str = "scaffold",
        loops: tuple[int, ...] = (1, 2, 4, 8, 12, 16)) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    paths = AnalysisPaths(Path(out_dir), "g_halting_toggle")

    cell_dirs = sorted(
        d for d in (panel_root / lf_model_dir).glob(f"*/{split}")
        if (d / "predictions.csv").exists()
    )
    if not cell_dirs:
        raise FileNotFoundError(
            f"no cells at {panel_root / lf_model_dir}/*/{split}/")

    rows: list[dict] = []
    for cell_dir in cell_dirs:
        target = cell_dir.parent.name
        halt_on_mae = _cell_mae(cell_dir / "predictions.csv")
        if halt_on_mae is None:
            continue
        rec: dict = {"target": target, "halting_on_mae": halt_on_mae}
        loop_dir = cell_dir / "loop_sweep"
        for k in loops:
            mae_k = _cell_mae(loop_dir / f"predictions_nloops{k}.csv")
            if mae_k is not None:
                rec[f"mae_K{k}"] = mae_k
                rec[f"delta_K{k}"] = mae_k - halt_on_mae
        rows.append(rec)
    if not rows:
        raise RuntimeError(
            "no cells with both default predictions.csv and loop_sweep/. "
            "Did G.4 test-time loop sweep run?")
    per_cell = pd.DataFrame(rows)
    save_table(per_cell, paths, "halting_toggle_per_cell")

    # Aggregate: mean MAE vs K + halting-on baseline
    summary_rows = [{"setting": "halting_on", "mean_mae": float(per_cell["halting_on_mae"].mean()),
                     "median_mae": float(per_cell["halting_on_mae"].median()),
                     "std_mae": float(per_cell["halting_on_mae"].std()),
                     "n_cells": int(per_cell["halting_on_mae"].notna().sum())}]
    for k in loops:
        col = f"mae_K{k}"
        if col not in per_cell.columns:
            continue
        summary_rows.append({"setting": f"halting_off_K{k}",
                             "mean_mae": float(per_cell[col].mean()),
                             "median_mae": float(per_cell[col].median()),
                             "std_mae": float(per_cell[col].std()),
                             "n_cells": int(per_cell[col].notna().sum())})
    summary = pd.DataFrame(summary_rows)
    save_table(summary, paths, "halting_toggle_summary")

    _plot_mae_vs_k(per_cell, loops, paths)
    _plot_delta_bars(per_cell, loops, paths)

    headline = "halting toggle produced no data"
    if not summary.empty:
        baseline = float(summary.loc[summary["setting"] == "halting_on", "mean_mae"].iloc[0])
        off_rows = summary[summary["setting"] != "halting_on"]
        if not off_rows.empty:
            best_off = off_rows.loc[off_rows["mean_mae"].idxmin()]
            delta = float(best_off["mean_mae"]) - baseline
            if delta > 0.005:
                verdict = "Halting helps."
            elif delta < -0.005:
                verdict = ("Fixed depth BEATS halting -- the trained "
                           "halting head is actively hurting inference MAE.")
            else:
                verdict = ("Halting equivalent to fixed depth (consistent "
                           "with the collapsed halt head documented in G.1).")
            headline = (
                f"Default (halting ON) mean MAE = {baseline:.3f} across "
                f"{len(per_cell)} cells. Best fixed-depth setting: "
                f"{best_off['setting']} with mean MAE = {best_off['mean_mae']:.3f} "
                f"(Δ = {delta:+.3f}). {verdict}")

    write_summary_md(
        paths,
        title="Halting ON vs OFF (fixed-depth) comparison",
        claim=("Does the halting head help at inference, or could the model be "
               "replaced by a fixed-loop forward pass with no loss in MAE? "
               "Reads existing predictions.csv (halting ON) and G.4's loop "
               "sweep outputs (halting OFF at each fixed K)."),
        headline=headline,
        details={
            "Loop counts compared": ", ".join(str(k) for k in loops),
            "Note": ("The trained halting head collapsed (depth=2 across all "
                     "molecules; see g1_depth_distribution.png). Expect "
                     "halting-on MAE ≈ fixed K=2 MAE."),
            "Cells": str(len(per_cell)),
        },
        tables_referenced=[
            "halting_toggle_per_cell.csv",
            "halting_toggle_summary.csv",
        ],
        figures_referenced=[
            "g_halting_toggle_mae_vs_k.png",
            "g_halting_toggle_delta_bars.png",
        ],
    )
    return {"paths": paths, "per_cell": per_cell, "summary": summary}


def _plot_mae_vs_k(per_cell: pd.DataFrame, loops, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    rows = []
    for k in loops:
        col = f"mae_K{k}"
        if col not in per_cell.columns:
            continue
        rows.append({"K": k,
                     "mean": float(per_cell[col].mean()),
                     "q25": float(per_cell[col].quantile(0.25)),
                     "q75": float(per_cell[col].quantile(0.75))})
    if not rows:
        return
    df = pd.DataFrame(rows)
    baseline = float(per_cell["halting_on_mae"].mean())
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.6, 3.0))
    color = nature_palette(1)[0]
    ax.plot(df["K"], df["mean"], "-o", color=color, lw=1.6, ms=5,
            label="halting OFF (fixed depth)")
    ax.fill_between(df["K"], df["q25"], df["q75"], color=color,
                    alpha=0.18, lw=0)
    ax.axhline(baseline, color="#cc3333", ls="--", lw=1.2,
               label=f"halting ON (default) = {baseline:.3f}")
    ax.set_xlabel("Fixed n_loops (halting OFF)", fontweight="bold")
    ax.set_ylabel("Mean test MAE (IQR band)", fontweight="bold")
    ax.set_title("MAE vs fixed depth, with halting-ON baseline",
                 fontweight="bold", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(df["K"])
    ax.set_xticklabels(df["K"].astype(int).astype(str))
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_halting_toggle_mae_vs_k")
    plt.close(fig)


def _plot_delta_bars(per_cell: pd.DataFrame, loops, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    rows = []
    for k in loops:
        col = f"delta_K{k}"
        if col not in per_cell.columns:
            continue
        rows.append({"K": int(k),
                     "mean_delta": float(per_cell[col].mean()),
                     "median_delta": float(per_cell[col].median())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    colors = ["#cc3333" if v > 0 else "#2ca02c" for v in df["mean_delta"]]
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 2.8))
    ax.bar(df["K"].astype(str), df["mean_delta"],
           color=colors, edgecolor="none")
    ax.axhline(0, color="black", lw=0.6)
    for i, v in enumerate(df["mean_delta"]):
        ax.text(i, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7, fontweight="bold")
    ax.set_xlabel("Fixed n_loops K (halting OFF)", fontweight="bold")
    ax.set_ylabel("Mean Δ MAE  (K - halting_on)", fontweight="bold")
    ax.set_title("Cost of disabling halting (positive = worse)",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, "g_halting_toggle_delta_bars")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--lf-model-dir", default="radiant")
    p.add_argument("--split", default="scaffold")
    p.add_argument("--loops", nargs="*", type=int,
                   default=[1, 2, 4, 8, 12, 16])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        lf_model_dir=args.lf_model_dir, split=args.split,
        loops=tuple(args.loops))


if __name__ == "__main__":
    main()
