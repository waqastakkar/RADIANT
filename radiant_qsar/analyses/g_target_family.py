"""Phase G -- Per-target-family analysis.

Joins g0_validation_metrics with target_class from panel.json and emits:

* mean MAE / R2 / Pearson / Spearman per (model, target_class)
* a grouped-bar chart per metric (one bar per model x family)
* a per-family ranking + best-model table
* a per-family x split heatmap (MAE only)

Why this matters: reviewers want to know whether RADIANT helps for
kinases but not GPCRs, etc. Same pre-computed data (g0_cell_metrics.csv),
no inference, no training.

Outputs
-------
tables/
    family_metric_summary.csv         -- one row per (model, family, metric)
    family_winners.csv                -- per family, who is best on each metric
figures/
    g_family_mae_bar.{png,svg}        -- grouped bar (model x family) on MAE
    g_family_pearson_bar.{png,svg}    -- same on Pearson
    g_family_split_heatmap.{png,svg}  -- family x split MAE heatmap (best model only)
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
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    nature_palette,
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


def _load_class_map(panel_root: Path) -> dict[str, str]:
    for cand in [Path("data/processed/v1/panel.json"),
                 panel_root / "panel.json",
                 panel_root.parent / "panel.json"]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                entries = data.get("entries", data if isinstance(data, list) else [])
                return {e["target_chembl_id"]: e.get("target_class", "unknown")
                        for e in entries if "target_chembl_id" in e}
            except Exception:
                continue
    # fallback: panel_results.csv
    pr = panel_root / "panel_results.csv"
    if pr.exists():
        try:
            d = pd.read_csv(pr, usecols=["target_chembl_id", "target_class"]).drop_duplicates()
            return dict(zip(d["target_chembl_id"], d["target_class"]))
        except Exception:
            pass
    return {}


def _plot_grouped_bar(summary: pd.DataFrame, *, metric: str,
                      lower_better: bool, paths: AnalysisPaths,
                      stem: str) -> None:
    import matplotlib.pyplot as plt
    sub = summary[summary["metric"] == metric]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="family", columns="model", values="mean",
                            aggfunc="mean")
    if pivot.empty:
        return
    families = list(pivot.index)
    models = list(pivot.columns)
    x = np.arange(len(families))
    w = 0.8 / max(len(models), 1)
    colors = nature_palette(len(models))

    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL * 0.85, 3.4))
    for i, m in enumerate(models):
        vals = pivot[m].to_numpy(dtype=float)
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
               color=colors[i], edgecolor="none", label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(f"Mean test {metric}", fontweight="bold")
    arrow = "↓" if lower_better else "↑"
    ax.set_title(f"{metric} per target family  ({arrow})",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, frameon=False,
              loc="upper right" if lower_better else "lower right")
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_family_split_heatmap(df: pd.DataFrame, paths: AnalysisPaths,
                               best_model: str = "radiant") -> None:
    import matplotlib.pyplot as plt
    sub = df[df["model"] == best_model]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="target_class", columns="split",
                            values="mae", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 0.4 * len(pivot) + 1.4))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.7, label="mean test MAE")
    ax.set_title(f"{best_model}: MAE per family x split",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, "g_family_split_heatmap")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        g0_cell_metrics: Path | str | None = None) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_target_family")

    if g0_cell_metrics is None:
        g0_cell_metrics = out_dir / "g0_validation_metrics" / "tables" / "g0_cell_metrics.csv"
    g0_cell_metrics = Path(g0_cell_metrics)
    if not g0_cell_metrics.exists():
        raise FileNotFoundError(
            f"g0_cell_metrics.csv not found at {g0_cell_metrics}; run G.0 first.")

    df = pd.read_csv(g0_cell_metrics)
    class_map = _load_class_map(panel_root)
    if not class_map:
        raise FileNotFoundError(
            "target_class map not found. Looked in data/processed/v1/panel.json, "
            f"{panel_root}/panel.json, {panel_root}/panel_results.csv.")
    df["target_class"] = df["target"].map(class_map).fillna("unknown")

    # Per-(model, family, metric) summary
    summary_rows: list[dict] = []
    for name, col, lower in METRICS:
        if col not in df.columns:
            continue
        g = df.groupby(["model", "target_class"])[col].agg(
            mean="mean", median="median", std="std", n="count").reset_index()
        g["metric"] = name
        g = g.rename(columns={"target_class": "family"})
        summary_rows.append(g)
    summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    save_table(summary, paths, "family_metric_summary")

    # Per-family winners (best model on each metric)
    winners: list[dict] = []
    if not summary.empty:
        for (metric, family), sub in summary.groupby(["metric", "family"]):
            lower = next(l for n, _, l in METRICS if n == metric)
            best_idx = sub["mean"].idxmin() if lower else sub["mean"].idxmax()
            row = sub.loc[best_idx]
            winners.append({"family": family, "metric": metric,
                            "winner": row["model"],
                            "value": float(row["mean"]),
                            "n_cells": int(row["n"])})
    winners_df = pd.DataFrame(winners)
    save_table(winners_df, paths, "family_winners")

    # Figures
    _plot_grouped_bar(summary, metric="MAE", lower_better=True,
                      paths=paths, stem="g_family_mae_bar")
    _plot_grouped_bar(summary, metric="Pearson", lower_better=False,
                      paths=paths, stem="g_family_pearson_bar")
    _plot_family_split_heatmap(df, paths, best_model="radiant")

    # Headline: how many families RADIANT wins on MAE
    headline = "no winners table produced"
    if not winners_df.empty:
        mae_winners = winners_df[winners_df["metric"] == "MAE"]
        radiant_wins = mae_winners[mae_winners["winner"] == "radiant"]["family"].tolist()
        all_fam = mae_winners["family"].unique()
        if radiant_wins:
            headline = (f"On test MAE, RADIANT wins {len(radiant_wins)} of "
                        f"{len(all_fam)} target families: "
                        f"{', '.join(sorted(radiant_wins))}.")
        else:
            headline = (f"RADIANT does not win the per-family MAE on any of "
                        f"{len(all_fam)} families ({', '.join(sorted(all_fam))}).")

    write_summary_md(
        paths,
        title="Per target-family analysis",
        claim=("Per (model x target-family) metrics from G.0 cell metrics. "
               "Shows whether RADIANT's edge is uniform across families or "
               "concentrated in a subset (kinases, GPCRs, etc.)."),
        headline=headline,
        details={
            "Families found": ", ".join(sorted(df["target_class"].unique())),
            "Cells per family": ", ".join(
                f"{k}={v}" for k, v in df["target_class"].value_counts().items()),
        },
        tables_referenced=[
            "family_metric_summary.csv",
            "family_winners.csv",
        ],
        figures_referenced=[
            "g_family_mae_bar.png",
            "g_family_pearson_bar.png",
            "g_family_split_heatmap.png",
        ],
    )
    return {"paths": paths, "summary": summary, "winners": winners_df}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--g0-cell-metrics", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        g0_cell_metrics=args.g0_cell_metrics)


if __name__ == "__main__":
    main()
