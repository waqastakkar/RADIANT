"""Pairwise model benchmarking & win analysis.

For each pair of models in the panel, compute on the *same* test
molecules:

* Win rate (fraction of molecules where model A has a smaller absolute
  error than model B).
* Mean and median ΔPearson, ΔMAE, ΔRMSE.
* Margin distribution categorized as
  ``very_small (<0.1σ) / small (<0.3σ) / moderate (<1σ) / large``
  where σ is the within-target SD of the per-molecule absolute error.
* Aggregations across splits, targets, and complexity bins.

Outputs
-------
* ``pairwise_wins_per_split.csv``         — long format, one row per (target, split, model_a, model_b).
* ``pairwise_aggregated.csv``             — wide format across all splits/targets.
* ``g_winrate_heatmap.{png,svg}``         — win-rate matrix model_a vs model_b.
* ``g_margin_categories.{png,svg}``       — stacked bar of margin categories.
* ``g_wins_per_complexity_bin.{png,svg}`` — bar chart, win rate per complexity bin.
* ``summary.md``.

Input
-----
A panel-root directory laid out as ``<model>/<target>/<split>/predictions.csv``.
A descriptors Parquet is required for complexity binning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    COMPLEXITY_DESCRIPTORS,
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    NATURE_PALETTE,
    absolute_error,
    bootstrap_paired_diff,
    complexity_bins,
    discover_predictions,
    holm_correction,
    join_descriptors,
    load_predictions,
    nature_palette,
    publication_style,
    regression_metrics,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


MARGIN_LABELS = ("very_small", "small", "moderate", "large")
MARGIN_BOUNDS = (0.1, 0.3, 1.0)  # in units of σ(|error|) within the target


def _categorize_margin(margin: float, sigma: float) -> str:
    if not np.isfinite(margin) or sigma <= 0:
        return "very_small"
    z = abs(margin) / sigma
    if z < MARGIN_BOUNDS[0]:
        return "very_small"
    if z < MARGIN_BOUNDS[1]:
        return "small"
    if z < MARGIN_BOUNDS[2]:
        return "moderate"
    return "large"


# ---------------------------------------------------------------------------
# Pairwise comparisons
# ---------------------------------------------------------------------------

def _wide_predictions_for_pair(
    panel: pd.DataFrame,
    *,
    target: str,
    split: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame | None:
    pa = panel[(panel["target"] == target) & (panel["split"] == split) & (panel["model"] == model_a)]
    pb = panel[(panel["target"] == target) & (panel["split"] == split) & (panel["model"] == model_b)]
    if pa.empty or pb.empty:
        return None
    a = load_predictions(pa["path"].iloc[0]).rename(
        columns={"pred_pchembl": f"pred_{model_a}"}
    )[["inchikey14", "true_pchembl", "smiles", f"pred_{model_a}"]]
    b = load_predictions(pb["path"].iloc[0]).rename(
        columns={"pred_pchembl": f"pred_{model_b}"}
    )[["inchikey14", f"pred_{model_b}"]]
    merged = a.merge(b, on="inchikey14", how="inner")
    if merged.empty:
        return None
    return merged


def pair_metrics(
    panel: pd.DataFrame,
    *,
    target: str,
    split: str,
    model_a: str,
    model_b: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict | None:
    """Compute paired metrics for a single (target, split, model_a, model_b)."""
    merged = _wide_predictions_for_pair(
        panel, target=target, split=split, model_a=model_a, model_b=model_b
    )
    if merged is None:
        return None

    err_a = np.abs(merged[f"pred_{model_a}"].to_numpy() - merged["true_pchembl"].to_numpy())
    err_b = np.abs(merged[f"pred_{model_b}"].to_numpy() - merged["true_pchembl"].to_numpy())

    win_a = float(np.mean(err_a < err_b))
    win_b = float(np.mean(err_b < err_a))
    tie = float(np.mean(err_a == err_b))

    # within-target sigma for margin categorization
    sigma = float(np.std(np.concatenate([err_a, err_b])))
    margins = err_b - err_a  # positive => A better
    cats = {lab: 0 for lab in MARGIN_LABELS}
    for m in margins:
        cats[_categorize_margin(m, sigma)] += 1
    total = sum(cats.values()) or 1
    cat_frac = {f"frac_{k}": v / total for k, v in cats.items()}

    boot_mae = bootstrap_paired_diff(err_a, err_b, statistic="mean",
                                     n_bootstrap=n_bootstrap, seed=seed)
    boot_med = bootstrap_paired_diff(err_a, err_b, statistic="median",
                                     n_bootstrap=n_bootstrap, seed=seed + 1)

    # Pearson / RMSE per model
    a_df = merged.rename(columns={f"pred_{model_a}": "pred_pchembl"})[["pred_pchembl", "true_pchembl"]]
    b_df = merged.rename(columns={f"pred_{model_b}": "pred_pchembl"})[["pred_pchembl", "true_pchembl"]]
    m_a = regression_metrics(a_df)
    m_b = regression_metrics(b_df)

    return {
        "target": target,
        "split": split,
        "model_a": model_a,
        "model_b": model_b,
        "n": int(len(merged)),
        "winrate_a": win_a,
        "winrate_b": win_b,
        "tie": tie,
        "mean_delta_mae": boot_mae["mean_diff"],   # positive => A worse (higher error)
        "mean_delta_mae_ci_lo": boot_mae["ci"][0],
        "mean_delta_mae_ci_hi": boot_mae["ci"][1],
        "p_mae": boot_mae["p_value"],
        "median_delta_mae": boot_med["mean_diff"],
        "delta_pearson": m_a["pearson"] - m_b["pearson"],
        "delta_rmse": m_a["rmse"] - m_b["rmse"],
        "mae_a": m_a["mae"], "mae_b": m_b["mae"],
        "pearson_a": m_a["pearson"], "pearson_b": m_b["pearson"],
        **cat_frac,
    }


def all_pairwise(
    panel_root: Path | str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    panel = discover_predictions(panel_root)
    rows: list[dict] = []
    for (target, split), grp in panel.groupby(["target", "split"]):
        models = grp["model"].tolist()
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                res = pair_metrics(panel, target=target, split=split,
                                   model_a=ma, model_b=mb,
                                   n_bootstrap=n_bootstrap, seed=seed)
                if res is not None:
                    rows.append(res)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_mae_holm"] = holm_correction(df["p_mae"].fillna(1.0).to_numpy())
    return df


# ---------------------------------------------------------------------------
# Aggregation across splits / targets / bins
# ---------------------------------------------------------------------------

def aggregate(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Mean winrate / delta MAE etc across (target, split) per model pair."""
    if pairs_df.empty:
        return pairs_df
    agg_cols = ["winrate_a", "winrate_b", "tie", "mean_delta_mae",
                "median_delta_mae", "delta_pearson", "delta_rmse"]
    out = (pairs_df.groupby(["model_a", "model_b"])[agg_cols]
           .agg(["mean", "median"]).reset_index())
    out.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c
                   for c in out.columns]
    return out


def per_complexity_bin(
    panel_root: Path | str,
    descriptors: pd.DataFrame,
    *,
    bin_descriptor: str = "BertzCT",
    n_bins: int = 4,
) -> pd.DataFrame:
    """For each model pair, compute win rate per complexity bin."""
    panel = discover_predictions(panel_root)
    rows: list[dict] = []
    for (target, split), grp in panel.groupby(["target", "split"]):
        models = grp["model"].tolist()
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                merged = _wide_predictions_for_pair(panel, target=target, split=split,
                                                   model_a=ma, model_b=mb)
                if merged is None or bin_descriptor not in descriptors.columns:
                    continue
                merged = join_descriptors(merged, descriptors)
                if bin_descriptor not in merged.columns:
                    continue
                merged["__bin__"] = complexity_bins(merged[bin_descriptor], n_bins=n_bins)
                for b, sub in merged.groupby("__bin__"):
                    if sub.empty:
                        continue
                    err_a = np.abs(sub[f"pred_{ma}"].to_numpy() - sub["true_pchembl"].to_numpy())
                    err_b = np.abs(sub[f"pred_{mb}"].to_numpy() - sub["true_pchembl"].to_numpy())
                    rows.append({
                        "target": target, "split": split,
                        "model_a": ma, "model_b": mb,
                        "bin": str(b), "n": int(len(sub)),
                        "winrate_a": float(np.mean(err_a < err_b)),
                        "winrate_b": float(np.mean(err_b < err_a)),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_winrate_heatmap(agg_df: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if agg_df.empty:
        return []
    models = sorted(set(agg_df["model_a"]).union(agg_df["model_b"]))
    mat = np.full((len(models), len(models)), np.nan)
    midx = {m: i for i, m in enumerate(models)}
    for _, r in agg_df.iterrows():
        i, j = midx[r["model_a"]], midx[r["model_b"]]
        mat[i, j] = r["winrate_a_mean"]
        mat[j, i] = r["winrate_b_mean"]
    np.fill_diagonal(mat, 0.5)

    # Nature-friendly diverging palette: white at 0.5, NPG red vs teal
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nc_winrate",
        [NATURE_PALETTE[0], "#FFFFFF", NATURE_PALETTE[2]],   # vermilion → white → teal
    )
    cell_px = max(NC_SINGLE_COL / len(models), 0.55)
    sz = cell_px * len(models) + 0.8
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right", fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontweight="bold")
    for i in range(len(models)):
        for j in range(len(models)):
            if np.isfinite(mat[i, j]):
                tc = "white" if abs(mat[i, j] - 0.5) > 0.25 else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, fontweight="bold", color=tc)
    ax.set_title("Win rate  (row beats column)", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Win rate", fontweight="bold", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    out = save_figure(fig, paths, "g_winrate_heatmap")
    plt.close(fig)
    return out


def plot_margin_categories(pairs_df: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    if pairs_df.empty:
        return []
    cols = [f"frac_{lab}" for lab in MARGIN_LABELS]
    grouped = pairs_df.groupby(["model_a", "model_b"])[cols].mean().reset_index()
    grouped["pair"] = grouped["model_a"] + " vs " + grouped["model_b"]
    grouped = grouped.sort_values("frac_large", ascending=False)

    # 4-tone Nature palette for margin severity
    margin_colors = ["#D9D9D9", NATURE_PALETTE[4], NATURE_PALETTE[2], NATURE_PALETTE[3]]
    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL, 0.32 * max(3, len(grouped)) + 0.9))
    bottom = np.zeros(len(grouped))
    for col, color in zip(cols, margin_colors):
        ax.barh(grouped["pair"], grouped[col], left=bottom,
                label=col.replace("frac_", ""), color=color, edgecolor="none")
        bottom += grouped[col].to_numpy()
    ax.set_xlabel("Fraction of test molecules", fontweight="bold")
    ax.set_title("Win-margin categories per model pair", fontweight="bold")
    ax.axvline(0.5, color="#000000", lw=0.6, ls="--")
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    out = save_figure(fig, paths, "g_margin_categories")
    plt.close(fig)
    return out


def plot_wins_per_bin(bin_df: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    if bin_df.empty:
        return []
    agg = bin_df.groupby(["model_a", "model_b", "bin"])["winrate_a"].mean().reset_index()
    pairs_unique = (agg["model_a"] + " vs " + agg["model_b"]).unique()
    bins = sorted(agg["bin"].unique())
    colors = nature_palette(len(bins))
    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL, max(2.5, 0.42 * len(pairs_unique) + 0.8)))
    x = np.arange(len(pairs_unique))
    width = 0.8 / max(1, len(bins))
    for i, (b, c) in enumerate(zip(bins, colors)):
        sub = agg[agg["bin"] == b].copy()
        sub["pair"] = sub["model_a"] + " vs " + sub["model_b"]
        sub = sub.set_index("pair").reindex(pairs_unique)
        ax.bar(x + i * width, sub["winrate_a"].fillna(0.5),
               width=width, label=b, color=c, edgecolor="none")
    ax.set_xticks(x + (len(bins) - 1) * width / 2)
    ax.set_xticklabels(pairs_unique, rotation=40, ha="right", fontsize=6, fontweight="bold")
    ax.set_ylabel("Win rate (A)", fontweight="bold")
    ax.set_title("Win rate per complexity bin", fontweight="bold")
    ax.axhline(0.5, color="#000000", lw=0.6, ls="--")
    ax.legend(title="Complexity bin", title_fontsize=6, fontsize=6, frameon=False)
    fig.tight_layout()
    out = save_figure(fig, paths, "g_wins_per_complexity_bin")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _read_descriptors(p: Path | str | None) -> pd.DataFrame:
    if p is None:
        return pd.DataFrame()
    p = Path(p)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def run(
    panel_root: Path | str,
    out_dir: Path | str,
    *,
    descriptors_path: Path | str | None = None,
    bin_descriptor: str = "BertzCT",
    n_bins: int = 4,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g_pairwise_wins")

    pairs = all_pairwise(panel_root, n_bootstrap=n_bootstrap, seed=seed)
    save_table(pairs, paths, "pairwise_wins_per_split")

    agg = aggregate(pairs)
    save_table(agg, paths, "pairwise_aggregated")

    figs: list[Path] = []
    figs += plot_winrate_heatmap(agg, paths)
    figs += plot_margin_categories(pairs, paths)

    descriptors = _read_descriptors(descriptors_path)
    bin_df = pd.DataFrame()
    if not descriptors.empty:
        bin_df = per_complexity_bin(panel_root, descriptors,
                                    bin_descriptor=bin_descriptor, n_bins=n_bins)
        if not bin_df.empty:
            save_table(bin_df, paths, "pairwise_wins_per_complexity_bin")
            figs += plot_wins_per_bin(bin_df, paths)

    if not pairs.empty:
        loop_rows = pairs[pairs["model_a"].str.contains("radiant") | pairs["model_b"].str.contains("radiant")]
        loop_wins = loop_rows["winrate_a"].where(loop_rows["model_a"].str.contains("radiant"),
                                                  loop_rows["winrate_b"]).mean()
        verdict = f"RADIANT wins on {loop_wins:.1%} of test molecules averaged over all pair-matches."
    else:
        verdict = "No pairwise comparisons could be built (panel root empty?)."

    write_summary_md(
        paths,
        title="Pairwise model benchmarking & win analysis",
        claim="Cross-cutting: head-to-head wins, margins, and per-bin behavior across all model pairs.",
        headline=verdict,
        details={
            "Pairs compared": str(len(pairs)),
            "Bootstrap resamples": str(n_bootstrap),
            "Margin cuts (in σ)": str(MARGIN_BOUNDS),
            "Complexity-bin descriptor": bin_descriptor if not descriptors.empty else "(not supplied)",
        },
        tables_referenced=[
            "pairwise_wins_per_split.csv",
            "pairwise_aggregated.csv",
            *(["pairwise_wins_per_complexity_bin.csv"] if not bin_df.empty else []),
        ],
        figures_referenced=[p.name for p in figs],
    )

    return {"pairs": pairs, "aggregated": agg, "per_bin": bin_df, "paths": paths}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G pairwise model wins")
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--descriptors", type=Path, default=None)
    p.add_argument("--bin-descriptor", default="BertzCT")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(
        panel_root=args.panel_root,
        out_dir=args.out_dir,
        descriptors_path=args.descriptors,
        bin_descriptor=args.bin_descriptor,
        n_bins=args.n_bins,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
