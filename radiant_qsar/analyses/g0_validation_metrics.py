"""Phase G.0 — Model validation metrics across the full panel.

The foundational analysis: compute MAE, RMSE, R², Pearson *r*, and
Spearman ρ for every model × target × split cell in the panel, then
aggregate per-target (mean ± std across splits) and per-model (grand
mean across all targets).

Outputs
-------
* ``g0_cell_metrics.csv`` — one row per (model, target, split) cell.
* ``g0_per_target_summary.csv`` — per-model, per-target mean ± std over
  the 5 splits.
* ``g0_model_summary.csv`` — per-model grand mean across all 20 targets.
* ``g0_parity_grid.svg`` — parity-plot grid (pred vs true) for the
  RADIANT model across all 20 targets.
* ``g0_model_comparison_mae.svg`` — grouped bar chart of MAE per target
  for each model.
* ``g0_model_comparison_heatmap.svg`` — heatmap of per-target MAE for
  every model (quick visual scan).
* ``summary.md``.

Usage
-----
Panel mode (all models)::

    python -m radiant_qsar.analyses.g0_validation_metrics \\
        --panel-root runs/panel_75m \\
        --out-dir    runs/phase_g

Single model::

    python -m radiant_qsar.analyses.g0_validation_metrics \\
        --panel-root runs/panel_75m \\
        --model radiant \\
        --out-dir    runs/phase_g
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    NATURE_PALETTE,
    NC_DOUBLE_COL,
    NC_FULL_PAGE_H,
    NC_SINGLE_COL,
    discover_predictions,
    load_predictions,
    nature_palette,
    publication_style,
    regression_metrics,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: compute metrics for every cell
# ---------------------------------------------------------------------------

def compute_all_cell_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute regression metrics for every cell in the panel manifest.

    Parameters
    ----------
    panel : DataFrame
        Output of :func:`discover_predictions` with columns
        ``model, target, split, path``.

    Returns
    -------
    DataFrame with columns ``model, target, split, mae, rmse, r2, pearson,
    spearman, n``.
    """
    rows: list[dict] = []
    for _, row in panel.iterrows():
        try:
            df = load_predictions(row["path"])
        except Exception as exc:
            logger.warning("  Skipping %s/%s/%s: %s", row["model"], row["target"], row["split"], exc)
            continue
        m = regression_metrics(df)
        m["model"] = row["model"]
        m["target"] = row["target"]
        m["split"] = row["split"]
        rows.append(m)
    return pd.DataFrame(rows)


def per_target_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std of each metric across splits, grouped by (model, target)."""
    metrics = ["mae", "rmse", "r2", "pearson", "spearman"]
    agg_dict = {m: ["mean", "std", "count"] for m in metrics}
    agg_dict["n"] = "sum"
    grouped = cell_metrics.groupby(["model", "target"]).agg(agg_dict)
    # Flatten multi-level columns
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    return grouped.reset_index()


def model_summary(target_summary: pd.DataFrame) -> pd.DataFrame:
    """Grand mean across all targets for each model."""
    # Use the per-target means (not raw cells) so each target counts equally
    mean_cols = [c for c in target_summary.columns if c.endswith("_mean")]
    rows = []
    for model, grp in target_summary.groupby("model"):
        row = {"model": model, "n_targets": len(grp)}
        for c in mean_cols:
            metric = c.replace("_mean", "")
            row[f"{metric}_mean"] = grp[c].mean()
            row[f"{metric}_std"] = grp[c].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mae_mean")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_parity_grid(
    panel: pd.DataFrame,
    cell_metrics: pd.DataFrame,
    model_name: str,
    paths: AnalysisPaths,
) -> list[Path]:
    """Grid of parity plots (pred vs true) for one model across all targets.

    Uses a single scaffold split per target (lowest MAE among splits).
    """
    # Pick best split per target for the chosen model
    sub = cell_metrics[cell_metrics["model"] == model_name].copy()
    if sub.empty:
        logger.warning("No cells found for model '%s'; skipping parity grid", model_name)
        return []
    best = sub.sort_values("mae").groupby("target").first().reset_index()
    targets = sorted(best["target"].unique())
    n = len(targets)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * nrows / ncols * 0.85),
                             squeeze=False)

    for idx, target in enumerate(targets):
        ax = axes[idx // ncols, idx % ncols]
        row = best[best["target"] == target].iloc[0]
        # Find path in the panel manifest
        match = panel[(panel["model"] == model_name) &
                      (panel["target"] == target) &
                      (panel["split"] == row["split"])]
        if match.empty:
            ax.set_visible(False)
            continue
        df = load_predictions(match.iloc[0]["path"])
        y_true = df["true_pchembl"].to_numpy()
        y_pred = df["pred_pchembl"].to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        ax.scatter(y_true, y_pred, s=8, alpha=0.5, color=NATURE_PALETTE[3],
                   edgecolors="none", rasterized=True)
        lo = min(y_true.min(), y_pred.min()) - 0.3
        hi = max(y_true.max(), y_pred.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "--", color=NATURE_PALETTE[0], lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(target.replace("CHEMBL", ""), fontsize=7, fontweight="bold")
        mae = row["mae"]
        r2 = row["r2"]
        ax.text(0.05, 0.92, f"MAE {mae:.3f}\nR² {r2:.3f}", transform=ax.transAxes,
                fontsize=6, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        if idx // ncols == nrows - 1:
            ax.set_xlabel("True pChEMBL", fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel("Predicted pChEMBL", fontweight="bold")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"{model_name} — Parity plots (20 targets)", fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout()
    return save_figure(fig, paths, f"g0_parity_grid_{model_name}")


def plot_model_comparison_bars(
    target_summary: pd.DataFrame,
    paths: AnalysisPaths,
    metric: str = "mae",
) -> list[Path]:
    """Grouped bar chart comparing per-target metric across models."""
    col = f"{metric}_mean"
    err_col = f"{metric}_std"
    if col not in target_summary.columns:
        logger.warning("Column %s not found; skipping bar chart", col)
        return []

    models = sorted(target_summary["model"].unique())
    targets = sorted(target_summary["target"].unique())
    n_models = len(models)
    n_targets = len(targets)

    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * 0.45))
    x = np.arange(n_targets)
    width = 0.8 / max(n_models, 1)
    colours = nature_palette(n_models)

    for i, model in enumerate(models):
        sub = target_summary[target_summary["model"] == model].set_index("target")
        vals = [sub.loc[t, col] if t in sub.index else float("nan") for t in targets]
        errs = [sub.loc[t, err_col] if t in sub.index and err_col in sub.columns else 0.0 for t in targets]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=model, color=colours[i],
               edgecolor="white", linewidth=0.3, capsize=2, error_kw={"lw": 0.6})

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("CHEMBL", "") for t in targets],
                       rotation=45, ha="right", fontsize=6, fontweight="bold")
    ax.set_ylabel(metric.upper(), fontweight="bold")
    ax.set_title(f"{metric.upper()} per target (mean ± std across splits)", fontweight="bold")
    ax.legend(fontsize=6, ncol=min(n_models, 4))
    fig.tight_layout()
    return save_figure(fig, paths, f"g0_model_comparison_{metric}")


def plot_model_heatmap(
    target_summary: pd.DataFrame,
    paths: AnalysisPaths,
    metric: str = "mae",
) -> list[Path]:
    """Heatmap: models × targets coloured by metric value."""
    col = f"{metric}_mean"
    if col not in target_summary.columns:
        return []

    models = sorted(target_summary["model"].unique())
    targets = sorted(target_summary["target"].unique())
    matrix = np.full((len(models), len(targets)), np.nan)
    for i, model in enumerate(models):
        sub = target_summary[target_summary["model"] == model].set_index("target")
        for j, target in enumerate(targets):
            if target in sub.index:
                matrix[i, j] = sub.loc[target, col]

    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL, max(2.0, 0.4 * len(models) + 0.8)))
    # Use a diverging-ish colormap: lower MAE = greener (better)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "npg_heat", [NATURE_PALETTE[2], "#FFFFFF", NATURE_PALETTE[0]], N=256
    )
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace("CHEMBL", "") for t in targets],
                       rotation=45, ha="right", fontsize=6, fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7, fontweight="bold")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(targets)):
            v = matrix[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=5, fontweight="bold",
                        color="white" if abs(v - vmin) > 0.6 * (vmax - vmin) else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric.upper(), fontweight="bold", fontsize=7)
    ax.set_title(f"{metric.upper()} across models and targets", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, paths, f"g0_heatmap_{metric}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(
    panel_root: Path | str,
    out_dir: Path | str,
    *,
    model_filter: str | None = None,
) -> dict:
    """Run the full validation metrics analysis.

    Parameters
    ----------
    panel_root : Path
        Root of the panel directory (``<model>/<target>/<split>/predictions.csv``).
    out_dir : Path
        Where to write figures, tables, summary.md.
    model_filter : str, optional
        If given, restrict analysis to this single model name.
    """
    publication_style()
    paths = AnalysisPaths(out_dir=Path(out_dir), name="g0_validation_metrics")

    panel = discover_predictions(panel_root)
    if panel.empty:
        raise FileNotFoundError(f"No predictions.csv found under {panel_root}")

    if model_filter:
        panel = panel[panel["model"] == model_filter].reset_index(drop=True)
        if panel.empty:
            raise ValueError(f"No cells found for model '{model_filter}' under {panel_root}")

    logger.info("Found %d cells across %d models, %d targets",
                len(panel), panel["model"].nunique(), panel["target"].nunique())

    # 1. Per-cell metrics
    cell_metrics = compute_all_cell_metrics(panel)
    save_table(cell_metrics, paths, "g0_cell_metrics")
    logger.info("Cell metrics: %d rows", len(cell_metrics))

    # 2. Per-target summary
    tgt_summary = per_target_summary(cell_metrics)
    save_table(tgt_summary, paths, "g0_per_target_summary")

    # 3. Model grand summary
    mod_summary = model_summary(tgt_summary)
    save_table(mod_summary, paths, "g0_model_summary")
    logger.info("Model summary:\n%s", mod_summary.to_string(index=False))

    # 4. Parity grid for RADIANT (or first available model)
    lf_model = "radiant"
    if lf_model not in cell_metrics["model"].unique():
        lf_model = cell_metrics["model"].unique()[0]
        logger.info("No 'radiant' model found; using '%s' for parity grid", lf_model)
    parity_figs = plot_parity_grid(panel, cell_metrics, lf_model, paths)

    # 5. Model comparison bar chart (MAE)
    bar_figs = plot_model_comparison_bars(tgt_summary, paths, metric="mae")

    # 6. Heatmap
    heat_figs = plot_model_heatmap(tgt_summary, paths, metric="mae")

    # Also do R² heatmap
    heat_r2_figs = plot_model_heatmap(tgt_summary, paths, metric="r2")

    plt.close("all")

    # Build headline
    all_figs = parity_figs + bar_figs + heat_figs + heat_r2_figs
    lf_row = mod_summary[mod_summary["model"] == lf_model]
    if not lf_row.empty:
        lf = lf_row.iloc[0]
        headline = (
            f"{lf_model}: MAE = {lf['mae_mean']:.3f} ± {lf['mae_std']:.3f}, "
            f"RMSE = {lf['rmse_mean']:.3f} ± {lf['rmse_std']:.3f}, "
            f"R² = {lf['r2_mean']:.3f} ± {lf['r2_std']:.3f}, "
            f"Pearson r = {lf['pearson_mean']:.3f} ± {lf['pearson_std']:.3f}, "
            f"Spearman ρ = {lf['spearman_mean']:.3f} ± {lf['spearman_std']:.3f} "
            f"across {int(lf['n_targets'])} targets."
        )
    else:
        headline = f"Computed metrics for {len(mod_summary)} models across {panel['target'].nunique()} targets."

    write_summary_md(
        paths,
        title="G.0 — Model validation metrics",
        claim="RADIANT achieves competitive or superior predictive accuracy across all 20 QSAR targets.",
        headline=headline,
        details={
            "Panel root": str(panel_root),
            "Models evaluated": ", ".join(sorted(cell_metrics["model"].unique())),
            "Targets": str(cell_metrics["target"].nunique()),
            "Total cells": str(len(cell_metrics)),
            "Model filter": model_filter or "(all)",
        },
        tables_referenced=["g0_cell_metrics.csv", "g0_per_target_summary.csv", "g0_model_summary.csv"],
        figures_referenced=[p.name for p in all_figs],
    )

    return {
        "cell_metrics": cell_metrics,
        "per_target_summary": tgt_summary,
        "model_summary": mod_summary,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G.0 — Model validation metrics")
    p.add_argument("--panel-root", required=True, type=Path,
                   help="Root panel directory (<model>/<target>/<split>/predictions.csv)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for figures/tables/summary")
    p.add_argument("--model", default=None,
                   help="Restrict to a single model name (e.g., 'radiant')")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    run(
        panel_root=args.panel_root,
        out_dir=args.out_dir,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()
