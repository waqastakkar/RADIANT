"""Phase G -- Per-split pairwise win-rate matrices.

Five K x K heatmaps (one per split: random / scaffold / time / cluster /
activity_cliff). Cell (i, j) = fraction of (target) cells where model i
beats model j on test MAE. Each heatmap directly answers "under split S,
which model pairs flip and by how much?"

Reads ``g0_validation_metrics/tables/g0_cell_metrics.csv`` so it requires
nothing beyond G.0 output.

Outputs
-------
tables/
    winrate_per_split_<split>.csv   -- one K x K matrix per split
    winrate_summary.csv             -- one row per (split, winner, loser, rate, n)
figures/
    g_winrate_<split>.{png,svg}     -- annotated heatmap per split
    g_winrate_grid.{png,svg}        -- 5-panel grid (one heatmap per split)
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
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


SPLITS = ("random", "scaffold", "time", "cluster", "activity_cliff")


def _win_matrix(df_split: pd.DataFrame, *, metric_col: str,
                lower_better: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (rate_matrix, n_matrix). rate[i, j] = P(model i beats model j on metric)."""
    # pivot to (target x model) matrix of metric values
    p = df_split.pivot_table(index="target", columns="model",
                             values=metric_col, aggfunc="mean")
    p = p.dropna(axis=0, how="any")
    models = list(p.columns)
    k = len(models)
    rate = np.zeros((k, k), dtype=float)
    n_mat = np.zeros((k, k), dtype=int)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                rate[i, j] = float("nan")
                n_mat[i, j] = 0
                continue
            if lower_better:
                wins = (p[mi] < p[mj]).sum()
            else:
                wins = (p[mi] > p[mj]).sum()
            rate[i, j] = wins / len(p)
            n_mat[i, j] = len(p)
    rmat = pd.DataFrame(rate, index=models, columns=models)
    nmat = pd.DataFrame(n_mat, index=models, columns=models)
    return rmat, nmat


def _plot_one_heatmap(rmat: pd.DataFrame, paths: AnalysisPaths,
                      stem: str, title: str) -> None:
    import matplotlib.pyplot as plt
    if rmat.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, NC_SINGLE_COL * 1.4))
    im = ax.imshow(rmat.values, cmap="RdBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(rmat.shape[1]))
    ax.set_xticklabels(rmat.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(rmat.shape[0]))
    ax.set_yticklabels(rmat.index, fontsize=8, fontweight="bold")
    for i in range(rmat.shape[0]):
        for j in range(rmat.shape[1]):
            v = rmat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v - 0.5) > 0.30 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7,
                 label="P(row beats column)")
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_xlabel("loser", fontweight="bold")
    ax.set_ylabel("winner", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_grid(rate_per_split: dict[str, pd.DataFrame],
               paths: AnalysisPaths) -> None:
    """Single grid figure with one mini-heatmap per split."""
    import matplotlib.pyplot as plt
    if not rate_per_split:
        return
    splits = [s for s in SPLITS if s in rate_per_split]
    if not splits:
        return
    ncols = min(3, len(splits))
    nrows = (len(splits) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NC_DOUBLE_COL,
                                      NC_DOUBLE_COL * nrows / ncols * 0.95),
                             squeeze=False)
    im = None
    for idx, s in enumerate(splits):
        ax = axes[idx // ncols][idx % ncols]
        rmat = rate_per_split[s]
        if rmat.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(rmat.values, cmap="RdBu", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(rmat.shape[1]))
        ax.set_xticklabels(rmat.columns, rotation=25, ha="right", fontsize=6)
        ax.set_yticks(range(rmat.shape[0]))
        ax.set_yticklabels(rmat.index, fontsize=7, fontweight="bold")
        for i in range(rmat.shape[0]):
            for j in range(rmat.shape[1]):
                v = rmat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=6,
                            color="white" if abs(v - 0.5) > 0.30 else "black")
        ax.set_title(s, fontweight="bold", fontsize=9)
    for k in range(len(splits), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6,
                     label="P(row beats column)")
    fig.suptitle("Per-split pairwise win-rate matrices (MAE)",
                 fontweight="bold", fontsize=11, y=1.02)
    save_figure(fig, paths, "g_winrate_grid")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        g0_cell_metrics: Path | str | None = None,
        metric_col: str = "mae", lower_better: bool = True) -> dict:
    publication_style()
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_per_split_winrate")
    if g0_cell_metrics is None:
        g0_cell_metrics = out_dir / "g0_validation_metrics" / "tables" / "g0_cell_metrics.csv"
    g0_cell_metrics = Path(g0_cell_metrics)
    if not g0_cell_metrics.exists():
        raise FileNotFoundError(
            f"g0_cell_metrics.csv not found at {g0_cell_metrics}; run G.0 first.")

    df = pd.read_csv(g0_cell_metrics)
    if metric_col not in df.columns:
        raise ValueError(f"{metric_col!r} not in g0_cell_metrics")

    rate_per_split: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    for s in SPLITS:
        sub = df[df["split"] == s]
        if sub.empty:
            continue
        rmat, nmat = _win_matrix(sub, metric_col=metric_col,
                                 lower_better=lower_better)
        rate_per_split[s] = rmat
        save_table(rmat.reset_index().rename(columns={"index": "model"}),
                   paths, f"winrate_per_split_{s}")
        # long form
        for mi in rmat.index:
            for mj in rmat.columns:
                if mi == mj:
                    continue
                summary_rows.append({
                    "split": s, "winner": mi, "loser": mj,
                    "rate": float(rmat.loc[mi, mj]),
                    "n": int(nmat.loc[mi, mj]),
                })
        _plot_one_heatmap(rmat, paths, stem=f"g_winrate_{s}",
                          title=f"Win rate on {metric_col.upper()} -- {s}")

    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, paths, "winrate_summary")
    _plot_grid(rate_per_split, paths)

    headline = "no per-split win matrices produced"
    if summary_df.shape[0]:
        # Find rows where radiant is the winner with >=70% rate on hard splits
        hard = summary_df[(summary_df["winner"] == "radiant") &
                          (summary_df["split"].isin({"scaffold", "time",
                                                     "cluster", "activity_cliff"}))]
        if not hard.empty:
            best = hard.sort_values("rate", ascending=False).head(3)
            wins_text = "; ".join(
                f"vs {r['loser']} on {r['split']}: {r['rate']:.0%}"
                for _, r in best.iterrows())
            headline = f"RADIANT top wins: {wins_text}"
        else:
            headline = "RADIANT did not dominate any hard split; see CSVs."

    write_summary_md(
        paths,
        title="Per-split pairwise win-rate matrices",
        claim=("For every split type, how often does model A beat model B "
               "on test MAE (per-target)? Reveals where RADIANT actually wins."),
        headline=headline,
        details={
            "Splits evaluated": ", ".join(rate_per_split.keys()),
            "Metric": metric_col,
        },
        tables_referenced=(
            ["winrate_summary.csv"]
            + [f"winrate_per_split_{s}.csv" for s in rate_per_split]
        ),
        figures_referenced=(
            ["g_winrate_grid.png"]
            + [f"g_winrate_{s}.png" for s in rate_per_split]
        ),
    )
    return {"paths": paths, "rate_per_split": rate_per_split,
            "summary": summary_df}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--g0-cell-metrics", type=Path, default=None)
    p.add_argument("--metric", default="mae")
    p.add_argument("--higher-better", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        g0_cell_metrics=args.g0_cell_metrics,
        metric_col=args.metric,
        lower_better=not args.higher_better)


if __name__ == "__main__":
    main()
