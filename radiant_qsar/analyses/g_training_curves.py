"""Phase G — Training & validation curves (RADIANT only).

For every (target, split) cell of the RADIANT panel, reads
``result.json["history"]`` and plots:

* train loss per epoch
* val MAE / val Pearson per epoch
* a per-target grid (20 small multiples) for val MAE, with one line per split

Baselines (chemberta / molformer / morgan_rf / gin) do not save an
epoch-level history; this analysis therefore covers the RADIANT cells
only, which is exactly what's needed for the "training & test scores
from start to 20 targets" figure requested for the manuscript.

Outputs
-------
figures/
    g_training_curves_val_mae_grid.{png,svg}
    g_training_curves_val_pearson_grid.{png,svg}
    g_training_curves_train_loss_grid.{png,svg}
    g_training_curves_aggregate.{png,svg}
tables/
    training_curves_per_epoch.csv   (long form: model,target,split,epoch,...)
    training_curves_best_epoch.csv  (one row per cell, summary of best epoch)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    NATURE_PALETTE,
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


SPLIT_ORDER = ("random", "scaffold", "time", "cluster", "activity_cliff")


def _load_histories(panel_root: Path, model: str) -> pd.DataFrame:
    """Discover every result.json under panel_root/<model>/* and return long history."""
    rows: list[dict] = []
    n_cells = 0
    n_with_history = 0
    for rj in sorted((panel_root / model).rglob("result.json")):
        parts = rj.relative_to(panel_root).parts
        if len(parts) < 4:
            continue
        n_cells += 1
        target = parts[1]
        split = parts[2]
        try:
            rec = json.loads(rj.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("could not parse %s: %s", rj, exc)
            continue
        history = rec.get("history") or rec.get("train_history") or []
        if not history:
            continue
        n_with_history += 1
        for h in history:
            row = {"model": model, "target": target, "split": split, **h}
            rows.append(row)
    logger.info("model=%s: %d cells found, %d had non-empty history", model, n_cells, n_with_history)
    return pd.DataFrame(rows)


def _best_epoch_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each cell, return the epoch with the lowest val_mae."""
    if df.empty:
        return df
    out: list[dict] = []
    for (model, target, split), grp in df.groupby(["model", "target", "split"]):
        if "val_mae" not in grp.columns or grp["val_mae"].dropna().empty:
            continue
        best = grp.loc[grp["val_mae"].idxmin()]
        rec = {"model": model, "target": target, "split": split,
               "best_epoch": int(best.get("epoch", -1)),
               "n_epochs_seen": int(grp["epoch"].max() + 1) if "epoch" in grp else len(grp)}
        for k in ("train_loss", "val_mae", "val_rmse", "val_r2",
                  "val_pearson", "val_spearman"):
            if k in best.index:
                rec[k] = float(best[k])
        out.append(rec)
    return pd.DataFrame(out)


def _plot_grid(df: pd.DataFrame, *, ycol: str, ylabel: str, paths: AnalysisPaths,
               stem: str, title: str) -> None:
    """20-cell grid: one panel per target, one line per split."""
    import matplotlib.pyplot as plt

    if df.empty or ycol not in df.columns or "epoch" not in df.columns:
        logger.warning("skipping %s: column %r missing or empty df", stem, ycol)
        return
    targets = sorted(df["target"].unique())
    splits = [s for s in SPLIT_ORDER if s in df["split"].unique()]
    if not targets or not splits:
        return
    colors = {s: c for s, c in zip(splits, nature_palette(len(splits)))}

    ncols = 5
    nrows = (len(targets) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * nrows / ncols * 0.65),
                             squeeze=False, sharex=True)
    for i, target in enumerate(targets):
        ax = axes[i // ncols][i % ncols]
        sub_t = df[df["target"] == target]
        for split in splits:
            sub = sub_t[sub_t["split"] == split].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub[ycol], color=colors[split], lw=0.9, label=split)
        ax.set_title(target, fontsize=7, fontweight="bold")
        ax.tick_params(labelsize=6)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=7)
        if i // ncols == nrows - 1:
            ax.set_xlabel("epoch", fontsize=7)

    for k in range(len(targets), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    # Legend in the last (now-hidden) cell, or below the grid
    handles = [plt.Line2D([0], [0], color=colors[s], lw=1.5, label=s) for s in splits]
    fig.legend(handles=handles, loc="lower center", ncol=len(splits),
               frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_aggregate(df: pd.DataFrame, paths: AnalysisPaths) -> None:
    """Median val MAE & val Pearson curves aggregated across 20 targets, per split."""
    import matplotlib.pyplot as plt
    if df.empty or "val_mae" not in df.columns or "val_pearson" not in df.columns:
        return
    splits = [s for s in SPLIT_ORDER if s in df["split"].unique()]
    colors = {s: c for s, c in zip(splits, nature_palette(len(splits)))}

    fig, axes = plt.subplots(1, 2, figsize=(NC_DOUBLE_COL, 2.6))
    for split in splits:
        sub = df[df["split"] == split]
        agg = sub.groupby("epoch").agg(
            val_mae_med=("val_mae", "median"),
            val_mae_q25=("val_mae", lambda v: np.nanpercentile(v, 25)),
            val_mae_q75=("val_mae", lambda v: np.nanpercentile(v, 75)),
            val_pe_med=("val_pearson", "median"),
            val_pe_q25=("val_pearson", lambda v: np.nanpercentile(v, 25)),
            val_pe_q75=("val_pearson", lambda v: np.nanpercentile(v, 75)),
        ).reset_index()
        axes[0].plot(agg["epoch"], agg["val_mae_med"], color=colors[split],
                     lw=1.4, label=split)
        axes[0].fill_between(agg["epoch"], agg["val_mae_q25"], agg["val_mae_q75"],
                             color=colors[split], alpha=0.15, lw=0)
        axes[1].plot(agg["epoch"], agg["val_pe_med"], color=colors[split],
                     lw=1.4, label=split)
        axes[1].fill_between(agg["epoch"], agg["val_pe_q25"], agg["val_pe_q75"],
                             color=colors[split], alpha=0.15, lw=0)
    axes[0].set_xlabel("epoch", fontweight="bold")
    axes[0].set_ylabel("Val MAE (median ± IQR across 20 targets)", fontweight="bold")
    axes[1].set_xlabel("epoch", fontweight="bold")
    axes[1].set_ylabel("Val Pearson r (median ± IQR)", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("RADIANT training dynamics across 20 panel targets",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, paths, "g_training_curves_aggregate")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        model: str = "radiant") -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_training_curves")
    panel_root = Path(panel_root)

    long_df = _load_histories(panel_root, model)
    if long_df.empty:
        raise FileNotFoundError(
            f"No epoch history found under {panel_root / model}. "
            f"Only the RADIANT runs save per-epoch history; "
            f"baselines (chemberta/molformer/morgan_rf/gin) only save a final summary."
        )

    save_table(long_df, paths, "training_curves_per_epoch")
    best_df = _best_epoch_table(long_df)
    save_table(best_df, paths, "training_curves_best_epoch")

    _plot_grid(long_df, ycol="val_mae", ylabel="val MAE",
               paths=paths, stem="g_training_curves_val_mae_grid",
               title=f"{model}: validation MAE per epoch  (20 targets x 5 splits)")
    _plot_grid(long_df, ycol="val_pearson", ylabel="val Pearson r",
               paths=paths, stem="g_training_curves_val_pearson_grid",
               title=f"{model}: validation Pearson r per epoch")
    _plot_grid(long_df, ycol="train_loss", ylabel="train loss",
               paths=paths, stem="g_training_curves_train_loss_grid",
               title=f"{model}: training loss per epoch")
    _plot_aggregate(long_df, paths)

    n_cells = best_df.shape[0]
    mean_best_mae = float(best_df["val_mae"].mean()) if "val_mae" in best_df.columns else float("nan")
    mean_best_pe = float(best_df["val_pearson"].mean()) if "val_pearson" in best_df.columns else float("nan")
    mean_best_epoch = float(best_df["best_epoch"].mean()) if "best_epoch" in best_df.columns else float("nan")

    write_summary_md(
        paths,
        title=f"Training curves ({model})",
        claim=(f"Per-epoch training and validation trajectories across all "
               f"{model} fine-tuning cells (20 targets x 5 splits)."),
        headline=(f"{n_cells} cells with per-epoch history; "
                  f"mean best-epoch val MAE = {mean_best_mae:.3f}, "
                  f"val Pearson = {mean_best_pe:.3f}, "
                  f"reached at mean epoch {mean_best_epoch:.1f}."),
        details={
            "Model": model,
            "Cells with history": str(n_cells),
            "Splits covered": ", ".join(sorted(long_df["split"].unique())),
        },
        tables_referenced=[
            "training_curves_per_epoch.csv",
            "training_curves_best_epoch.csv",
        ],
        figures_referenced=[
            "g_training_curves_val_mae_grid.png",
            "g_training_curves_val_pearson_grid.png",
            "g_training_curves_train_loss_grid.png",
            "g_training_curves_aggregate.png",
        ],
    )
    return {"paths": paths, "long": long_df, "best": best_df}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="radiant")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir, model=args.model)


if __name__ == "__main__":
    main()
