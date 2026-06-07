"""Phase G — Friedman omnibus + Nemenyi pairwise post-hoc tests.

For each metric (MAE / R2 / Pearson / Spearman) we build a matrix where:

    rows    = (target, split) cells -- the "datasets" of Demsar 2006
    columns = models

and run:

* the Friedman omnibus test (are *any* models different?)
* the Nemenyi pairwise post-hoc test (which pairs differ?)

The Nemenyi p-value uses the Studentized-range distribution
(``scipy.stats.studentized_range``); no external dependency on
``scikit-posthocs`` is required.

Outputs
-------
tables/
    friedman_omnibus.csv             -- p-value per metric
    nemenyi_pvalues_<metric>.csv     -- KxK p-value matrix
    nemenyi_significant_pairs.csv    -- one row per (metric, pair_a, pair_b)
                                        with p < alpha
figures/
    g_stat_tests_nemenyi_<metric>.{png,svg}   -- p-value heatmap per metric
    g_stat_tests_friedman_bar.{png,svg}       -- omnibus p across metrics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    NC_SINGLE_COL,
    NC_DOUBLE_COL,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


METRICS = (
    ("MAE", "mae", True),
    ("R2", "r2", False),
    ("Pearson", "pearson", False),
    ("Spearman", "spearman", False),
)


def _pivot_models_by_cell(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Return a matrix indexed by (target, split), columns are models."""
    p = df.pivot_table(index=["target", "split"], columns="model",
                       values=metric_col, aggfunc="mean")
    return p.dropna(axis=0, how="any")


def _friedman(matrix: pd.DataFrame, lower_better: bool) -> tuple[float, float]:
    """Returns (chi2 stat, two-tailed p-value)."""
    from scipy.stats import friedmanchisquare
    arrs = [matrix[c].to_numpy() for c in matrix.columns]
    if lower_better:
        pass  # friedmanchisquare doesn't care about direction
    stat, p = friedmanchisquare(*arrs)
    return float(stat), float(p)


def _nemenyi_pvalues(matrix: pd.DataFrame, lower_better: bool) -> pd.DataFrame:
    """Pairwise Nemenyi p-values via the Studentized-range distribution.

    p_ij = 1 - F_q( |R_i - R_j| * sqrt(6N / (k(k+1))) * sqrt(2), k, inf )
    where R_i is mean rank of model i across N datasets, k is # models.
    """
    from scipy.stats import studentized_range, rankdata

    k = matrix.shape[1]
    N = matrix.shape[0]
    # Rank within each row (dataset). For lower-better metrics we rank ascending;
    # for higher-better we rank the negative so rank 1 is best.
    arr = matrix.to_numpy(dtype=float)
    if not lower_better:
        arr = -arr
    ranks = np.apply_along_axis(rankdata, 1, arr)
    mean_ranks = ranks.mean(axis=0)

    pvals = np.ones((k, k), dtype=float)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            q = diff / se * np.sqrt(2.0)  # back to Studentized range scale
            p = 1.0 - studentized_range.cdf(q, k, np.inf)
            pvals[i, j] = pvals[j, i] = float(p)

    return pd.DataFrame(pvals, index=matrix.columns, columns=matrix.columns), pd.Series(mean_ranks, index=matrix.columns)


def _plot_pmat(pmat: pd.DataFrame, paths: AnalysisPaths, stem: str, title: str,
               alpha: float = 0.05) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if pmat.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.3, NC_SINGLE_COL * 1.3))
    data = pmat.values.copy()
    # Avoid log(0)
    data = np.clip(data, 1e-12, 1.0)
    im = ax.imshow(data, cmap="viridis_r", norm=LogNorm(vmin=1e-6, vmax=1.0))
    ax.set_xticks(range(pmat.shape[1]))
    ax.set_xticklabels(pmat.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(pmat.shape[0]))
    ax.set_yticklabels(pmat.index, fontsize=8, fontweight="bold")
    # Annotate
    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):
            if i == j:
                txt = "-"
            else:
                p = pmat.values[i, j]
                txt = f"{p:.2g}"
                if p < alpha:
                    txt += "*"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] < 0.1 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Nemenyi p-value (log)")
    ax.set_title(f"{title}\n* = p < {alpha}", fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_friedman_bar(omnibus_df: pd.DataFrame, paths: AnalysisPaths,
                       alpha: float = 0.05) -> None:
    import matplotlib.pyplot as plt
    if omnibus_df.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.4))
    p = omnibus_df["p"].astype(float).clip(lower=1e-12).values
    colors = ["#2ca02c" if v < alpha else "#d62728" for v in p]
    ax.barh(omnibus_df["metric"], -np.log10(p), color=colors, edgecolor="none")
    ax.axvline(-np.log10(alpha), color="black", ls="--", lw=0.8, label=f"alpha = {alpha}")
    ax.set_xlabel(r"$-\log_{10}(p)$ -- Friedman omnibus", fontweight="bold")
    ax.set_title("Friedman test: are models distinguishable across cells?",
                 fontweight="bold", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, paths, "g_stat_tests_friedman_bar")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        g0_cell_metrics: Path | str | None = None,
        alpha: float = 0.05) -> dict:
    publication_style()
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_stat_tests")

    if g0_cell_metrics is None:
        g0_cell_metrics = out_dir / "g0_validation_metrics" / "tables" / "g0_cell_metrics.csv"
    g0_cell_metrics = Path(g0_cell_metrics)
    if not g0_cell_metrics.exists():
        raise FileNotFoundError(
            f"g0_cell_metrics.csv not found at {g0_cell_metrics}; run G.0 first.")
    df = pd.read_csv(g0_cell_metrics)

    omnibus: list[dict] = []
    sig_pairs: list[dict] = []
    for name, col, lower in METRICS:
        if col not in df.columns:
            continue
        matrix = _pivot_models_by_cell(df, col)
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            logger.warning("Friedman requires >=3 datasets and >=3 models; skipping %s", name)
            continue
        stat, p = _friedman(matrix, lower_better=lower)
        omnibus.append({"metric": name, "n_cells": int(matrix.shape[0]),
                        "k_models": int(matrix.shape[1]), "friedman_chi2": stat,
                        "p": p, "significant": p < alpha})
        pmat, mean_ranks = _nemenyi_pvalues(matrix, lower_better=lower)
        save_table(pmat.reset_index().rename(columns={"index": "model"}),
                   paths, f"nemenyi_pvalues_{name}")
        save_table(mean_ranks.reset_index(name="mean_rank").rename(columns={"index": "model"}),
                   paths, f"nemenyi_mean_ranks_{name}")
        _plot_pmat(pmat, paths, f"g_stat_tests_nemenyi_{name.lower()}",
                   title=f"Nemenyi pairwise p-values ({name})", alpha=alpha)
        # Significant pairs (lower triangle)
        for i, mi in enumerate(pmat.index):
            for j, mj in enumerate(pmat.columns):
                if j <= i:
                    continue
                p_ij = float(pmat.iloc[i, j])
                if p_ij < alpha:
                    sig_pairs.append({"metric": name, "model_a": mi, "model_b": mj,
                                      "p": p_ij,
                                      "mean_rank_a": float(mean_ranks[mi]),
                                      "mean_rank_b": float(mean_ranks[mj])})

    omnibus_df = pd.DataFrame(omnibus)
    save_table(omnibus_df, paths, "friedman_omnibus")
    sig_df = pd.DataFrame(sig_pairs).sort_values(["metric", "p"]) if sig_pairs else pd.DataFrame()
    save_table(sig_df, paths, "nemenyi_significant_pairs")
    _plot_friedman_bar(omnibus_df, paths, alpha=alpha)

    if omnibus_df.empty:
        headline = "Not enough data to run Friedman."
    else:
        n_sig = int(omnibus_df["significant"].sum())
        headline = (f"Friedman omnibus significant in {n_sig} of "
                    f"{len(omnibus_df)} metrics (alpha={alpha}). "
                    f"{len(sig_df)} model pairs are significantly different "
                    f"under Nemenyi post-hoc.")

    write_summary_md(
        paths,
        title="Friedman + Nemenyi statistical tests",
        claim=("Friedman omnibus tests whether any model differs across "
               "(target, split) cells; Nemenyi post-hoc identifies which pairs."),
        headline=headline,
        details={
            "Significance threshold": f"alpha = {alpha}",
            "Metrics evaluated": ", ".join(name for name, *_ in METRICS),
        },
        tables_referenced=(
            ["friedman_omnibus.csv", "nemenyi_significant_pairs.csv"]
            + [f"nemenyi_pvalues_{n}.csv" for n, *_ in METRICS]
            + [f"nemenyi_mean_ranks_{n}.csv" for n, *_ in METRICS]
        ),
        figures_referenced=(
            ["g_stat_tests_friedman_bar.png"]
            + [f"g_stat_tests_nemenyi_{n.lower()}.png" for n, *_ in METRICS]
        ),
    )
    return {"paths": paths, "friedman": omnibus_df, "significant_pairs": sig_df}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--g0-cell-metrics", type=Path, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        g0_cell_metrics=args.g0_cell_metrics, alpha=args.alpha)


if __name__ == "__main__":
    main()
