"""Phase G — Hard-split-only summary (excludes random).

Reviewers care more about chemical generalization than analogue
memorization, so this module re-emits the headline benchmark restricted
to the four "hard" splits:

    scaffold, time, cluster, activity_cliff

Outputs per metric (MAE, R2, Pearson, Spearman):

* mean and median value per model (lower / higher is better)
* mean rank per model
* win rate vs RADIANT (fraction of hard-split cells where model_X beats radiant)
* a bar chart and a heatmap with the four split columns

The win-rate bar in particular is intended as the main "RADIANT wins
under distribution shift" figure for the manuscript.
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


HARD_SPLITS = ("scaffold", "time", "cluster", "activity_cliff")
METRICS = (
    ("MAE", "mae", True),
    ("R2", "r2", False),
    ("Pearson", "pearson", False),
    ("Spearman", "spearman", False),
)


def _win_rate_vs_reference(df: pd.DataFrame, *, metric_col: str, lower_better: bool,
                           reference: str) -> pd.Series:
    """Per model: fraction of (target, split) cells where model beats ``reference``."""
    p = df.pivot_table(index=["target", "split"], columns="model",
                       values=metric_col, aggfunc="mean")
    if reference not in p.columns:
        logger.warning("reference model %r not in panel; skipping win rate", reference)
        return pd.Series(dtype=float)
    ref = p[reference]
    if lower_better:
        wins = p.lt(ref, axis=0)  # better = smaller
    else:
        wins = p.gt(ref, axis=0)
    return wins.mean(axis=0).drop(reference)


def _avg_rank(df: pd.DataFrame, *, metric_col: str, lower_better: bool) -> pd.Series:
    """Avg rank per model across (target, split) cells."""
    out = df.copy()
    out["rank"] = out.groupby(["target", "split"])[metric_col].rank(
        method="min", ascending=lower_better)
    return out.groupby("model")["rank"].mean().sort_values()


def _plot_metric_bar(means: pd.DataFrame, paths: AnalysisPaths,
                     stem: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt
    if means.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 2.8))
    colors = nature_palette(len(means))
    ax.bar(means.index, means.values, color=colors, edgecolor="none")
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("model", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_per_split_heatmap(matrix: pd.DataFrame, paths: AnalysisPaths,
                            stem: str, title: str,
                            lower_better: bool, fmt: str = ".3f") -> None:
    import matplotlib.pyplot as plt
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.3, 0.45 * len(matrix) + 1))
    cmap = "RdYlGn_r" if lower_better else "RdYlGn"
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title, fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_win_rate_bar(win_rates: pd.Series, paths: AnalysisPaths,
                       reference: str) -> None:
    import matplotlib.pyplot as plt
    if win_rates.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 2.8))
    colors = ["#d62728" if v >= 0.5 else "#2ca02c" for v in win_rates.values]
    # Note: win_rates is "X beats reference"; if reference is best, all bars are < 0.5 (good for reference).
    ax.bar(win_rates.index, win_rates.values, color=colors, edgecolor="none")
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    for i, v in enumerate(win_rates.values):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    ax.set_ylabel(f"P( model beats {reference} )", fontweight="bold")
    ax.set_xlabel("model", fontweight="bold")
    ax.set_title(f"Hard-split win rate vs {reference}", fontweight="bold", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylim(0, max(1.0, float(win_rates.max()) * 1.15))
    fig.tight_layout()
    save_figure(fig, paths, "g_hard_splits_winrate_vs_reference")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        g0_cell_metrics: Path | str | None = None,
        reference_model: str = "radiant") -> dict:
    publication_style()
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_hard_splits")

    if g0_cell_metrics is None:
        g0_cell_metrics = out_dir / "g0_validation_metrics" / "tables" / "g0_cell_metrics.csv"
    g0_cell_metrics = Path(g0_cell_metrics)
    if not g0_cell_metrics.exists():
        raise FileNotFoundError(
            f"g0_cell_metrics.csv not found at {g0_cell_metrics}; run G.0 first.")

    df = pd.read_csv(g0_cell_metrics)
    df = df[df["split"].isin(HARD_SPLITS)].copy()
    if df.empty:
        raise FileNotFoundError("No hard-split cells found in g0_cell_metrics.csv.")

    means_per_model: list[dict] = []
    ranks_per_model: dict[str, pd.Series] = {}
    figures: list[str] = []
    tables: list[str] = []

    for name, col, lower in METRICS:
        if col not in df.columns:
            continue
        # Per-model summary (mean, median over all hard-split cells)
        summary = df.groupby("model")[col].agg(["mean", "median", "std", "count"])
        per_split = df.groupby(["model", "split"])[col].mean().unstack("split").reindex(
            columns=[s for s in HARD_SPLITS if s in df["split"].unique()])
        # Order rows by avg rank (best first)
        avg_rank = _avg_rank(df, metric_col=col, lower_better=lower)
        summary = summary.reindex(avg_rank.index)
        per_split = per_split.reindex(avg_rank.index)
        summary["avg_rank_hard"] = avg_rank.values
        save_table(summary.reset_index(), paths, f"hard_splits_summary_{name}")
        save_table(per_split.reset_index(), paths, f"hard_splits_per_split_mean_{name}")
        tables.extend([f"hard_splits_summary_{name}.csv",
                       f"hard_splits_per_split_mean_{name}.csv"])

        ranks_per_model[name] = avg_rank
        for m, v in summary["mean"].items():
            means_per_model.append({"metric": name, "model": m, "mean": float(v),
                                    "median": float(summary.loc[m, "median"]),
                                    "avg_rank": float(summary.loc[m, "avg_rank_hard"]),
                                    "n_cells": int(summary.loc[m, "count"])})

        _plot_metric_bar(summary["mean"], paths,
                         stem=f"g_hard_splits_mean_{name.lower()}",
                         ylabel=f"hard-split mean {name}", title=f"Hard-split mean {name}")
        _plot_per_split_heatmap(per_split, paths,
                                stem=f"g_hard_splits_per_split_{name.lower()}",
                                title=f"{name} by hard split", lower_better=lower)
        figures.extend([f"g_hard_splits_mean_{name.lower()}.png",
                        f"g_hard_splits_per_split_{name.lower()}.png"])

    # Cross-metric long table
    long_df = pd.DataFrame(means_per_model)
    save_table(long_df, paths, "hard_splits_overall")
    tables.insert(0, "hard_splits_overall.csv")

    # Win rate vs RADIANT for the primary metric (MAE)
    win_rates_mae = _win_rate_vs_reference(df, metric_col="mae", lower_better=True,
                                           reference=reference_model)
    if not win_rates_mae.empty:
        save_table(win_rates_mae.reset_index(name="win_rate_vs_radiant_mae"),
                   paths, "hard_splits_winrate_mae_vs_reference")
        tables.append("hard_splits_winrate_mae_vs_reference.csv")
        _plot_win_rate_bar(win_rates_mae, paths, reference=reference_model)
        figures.append("g_hard_splits_winrate_vs_reference.png")

    # Headline
    headline = "no metrics computed"
    if "MAE" in ranks_per_model:
        top_model = ranks_per_model["MAE"].index[0]
        top_rank = ranks_per_model["MAE"].iloc[0]
        n_cells = int(df.groupby(["target", "split"]).ngroups)
        headline = (f"On {n_cells} hard-split cells "
                    f"({', '.join(HARD_SPLITS)}), best avg rank on MAE is "
                    f"{top_model} ({top_rank:.2f}).")

    write_summary_md(
        paths,
        title="Hard-split-only benchmark (random excluded)",
        claim=("Headline metrics restricted to chemically meaningful splits "
               "(scaffold, time, cluster, activity_cliff). Random split tends "
               "to favour analogue memorization."),
        headline=headline,
        details={
            "Hard splits": ", ".join(HARD_SPLITS),
            "Reference model for win rate": reference_model,
            "Models": ", ".join(sorted(df["model"].unique())),
        },
        tables_referenced=tables,
        figures_referenced=figures,
    )
    return {"paths": paths, "long": long_df, "ranks": ranks_per_model}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--g0-cell-metrics", type=Path, default=None)
    p.add_argument("--reference-model", default="radiant")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        g0_cell_metrics=args.g0_cell_metrics,
        reference_model=args.reference_model)


if __name__ == "__main__":
    main()
