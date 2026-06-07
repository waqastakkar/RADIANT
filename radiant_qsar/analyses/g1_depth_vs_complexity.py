"""Phase G.1 — RADIANT depth analysis per molecule (Sub-claim C1).

Per-molecule ``soft_effective_depth`` (continuous E[t × p_halt(t)] per
token, preferred) or ``effective_depth`` (mean halt step + 1) is
correlated with chemistry-grounded complexity descriptors:

    MW, NumRotatableBonds, NumRings, FractionCSP3, BertzCT, SAscore_proxy

Two modes
---------
Single-cell mode (``--predictions``)
    Correlates depth with descriptors for one target/split cell.
    Outputs per-descriptor Spearman ρ + CI, CV regression R², figures.

Panel mode (``--panel-root``)
    Discovers **all** ``radiant/*/*/predictions.csv`` under the panel
    root, runs per-cell correlations, and aggregates across the full
    100-cell panel. Produces a per-cell table and a forest-plot style
    bar chart of mean ρ per descriptor. The headline statistic is the
    median |ρ| across cells for the best descriptor, which is more
    representative than a single cell.

Inputs
------
* ``predictions_path``: RADIANT predictions.csv with ``effective_depth``
  or ``halt_step``.
* ``descriptors_path``: Parquet (or CSV) with ``inchikey14`` + descriptors.

Usage
-----
Single cell::

    python -m radiant_qsar.analyses.g1_depth_vs_complexity \\
        --predictions runs/panel_75m/radiant/CHEMBL203/scaffold/predictions.csv \\
        --descriptors data/processed/v1/descriptors.parquet \\
        --out-dir runs/phase_g

Panel (all 100 cells)::

    python -m radiant_qsar.analyses.g1_depth_vs_complexity \\
        --panel-root runs/panel_75m \\
        --descriptors data/processed/v1/descriptors.parquet \\
        --out-dir runs/phase_g
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    COMPLEXITY_DESCRIPTORS,
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    NATURE_PALETTE,
    AnalysisPaths,
    cv_regression,
    join_descriptors,
    load_predictions,
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    spearman_pearson,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# Order matters: soft_effective_depth (continuous E[t*p_halt(t)]) carries
# strictly more information than the binary halt_step + 1, because the
# threshold-crossing logic flattens close-but-not-equal cum-conf curves.
# We prefer it when available; fall back to the binary version for
# backward compatibility with older predictions.csv files.
DEPTH_COLUMN_CANDIDATES = ("soft_effective_depth", "effective_depth", "avg_depth", "depth")


def _resolve_depth_column(df: pd.DataFrame) -> str:
    """Pick the most-specific available depth column, falling back to halt_step+1.

    Prefers the *continuous* soft_effective_depth over the binary
    effective_depth: the former captures sub-threshold variation the
    latter discards.
    """
    for c in DEPTH_COLUMN_CANDIDATES:
        if c in df.columns:
            # Skip columns that are all-NaN (a checkpoint without halting).
            if df[c].notna().any():
                return c
    if "halt_step" in df.columns:
        df["effective_depth"] = df["halt_step"].astype(float) + 1.0
        return "effective_depth"
    raise ValueError(
        f"No depth column found. Expected one of {DEPTH_COLUMN_CANDIDATES} or 'halt_step'. "
        f"Available: {list(df.columns)}"
    )


def _read_descriptors(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def correlate_depth_descriptors(
    df: pd.DataFrame,
    *,
    depth_col: str,
    descriptors: tuple[str, ...] = COMPLEXITY_DESCRIPTORS,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Spearman/Pearson per descriptor against effective depth."""
    rows: list[dict] = []
    depth = df[depth_col].to_numpy(dtype=float)
    for d in descriptors:
        if d not in df.columns:
            logger.warning("descriptor '%s' missing; skipping", d)
            continue
        stats = spearman_pearson(
            df[d].to_numpy(dtype=float), depth, n_bootstrap=n_bootstrap, seed=seed
        )
        rows.append({
            "descriptor": d,
            "n": stats["n"],
            "spearman_rho": stats["spearman"],
            "spearman_ci_lo": stats["spearman_ci"][0],
            "spearman_ci_hi": stats["spearman_ci"][1],
            "pearson_r": stats["pearson"],
            "pearson_ci_lo": stats["pearson_ci"][0],
            "pearson_ci_hi": stats["pearson_ci"][1],
        })
    return pd.DataFrame(rows)


def regress_depth_on_descriptors(
    df: pd.DataFrame,
    *,
    depth_col: str,
    descriptors: tuple[str, ...] = COMPLEXITY_DESCRIPTORS,
    n_splits: int = 5,
    seed: int = 0,
    model: str = "ridge",
) -> dict:
    """CV regression of depth on descriptor vector. Returns R² and importance."""
    cols = [c for c in descriptors if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    y = df[depth_col].to_numpy(dtype=float)
    return cv_regression(X, y, feature_names=cols, n_splits=n_splits, seed=seed, model=model)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_depth_vs_descriptors(
    df: pd.DataFrame,
    *,
    depth_col: str,
    descriptors: tuple[str, ...],
    paths: AnalysisPaths,
    corr_table: pd.DataFrame,
) -> list[Path]:
    """Scatter grid: depth vs each descriptor with running-median overlay."""
    import matplotlib.pyplot as plt

    colors = nature_palette()
    ds = [d for d in descriptors if d in df.columns]
    ncols = 3
    nrows = (len(ds) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * nrows / ncols * 0.85),
        squeeze=False,
    )
    corr_lookup = {r["descriptor"]: r for _, r in corr_table.iterrows()}

    for idx, d in enumerate(ds):
        ax = axes[idx // ncols][idx % ncols]
        x = df[d].to_numpy(dtype=float)
        y = df[depth_col].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=5, alpha=0.30, linewidths=0,
                   color=colors[idx % len(colors)])
        try:
            bins = np.quantile(x[mask], np.linspace(0, 1, 11))
            bin_idx = np.digitize(x[mask], bins[1:-1])
            med_x = np.array([np.median(x[mask][bin_idx == k]) for k in range(10) if (bin_idx == k).any()])
            med_y = np.array([np.median(y[mask][bin_idx == k]) for k in range(10) if (bin_idx == k).any()])
            ax.plot(med_x, med_y, "-", lw=1.5, color="#000000")
        except Exception:
            pass
        r = corr_lookup.get(d, {})
        rho = r.get("spearman_rho", float("nan"))
        ax.set_title(f"{d}  ρ = {rho:.2f}", fontweight="bold")
        ax.set_xlabel(d, fontweight="bold")
        ax.set_ylabel("Effective depth", fontweight="bold")

    for k in range(len(ds), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.suptitle("Effective halting depth vs molecular complexity descriptors",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = save_figure(fig, paths, "g1_depth_vs_descriptors")
    plt.close(fig)
    return out


def plot_feature_importance(importance: dict, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    items = sorted(importance.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = nature_palette(len(names))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 0.30 * max(3, len(names)) + 0.8))
    ax.barh(names, vals, color=colors, edgecolor="none")
    ax.set_xlabel("Importance", fontweight="bold")
    ax.set_title("Descriptor importance for predicting\neffective halting depth",
                 fontweight="bold")
    fig.tight_layout()
    out = save_figure(fig, paths, "g1_feature_importance")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    predictions_path: Path | str,
    descriptors_path: Path | str,
    out_dir: Path | str,
    *,
    n_bootstrap: int = 1000,
    n_cv_splits: int = 5,
    seed: int = 0,
    regression_model: str = "ridge",
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g1_depth_vs_complexity")

    preds = load_predictions(predictions_path)
    depth_col = _resolve_depth_column(preds)
    desc = _read_descriptors(Path(descriptors_path))
    joined = join_descriptors(preds, desc)

    corr_table = correlate_depth_descriptors(
        joined, depth_col=depth_col, n_bootstrap=n_bootstrap, seed=seed
    )
    save_table(corr_table, paths, "g1_depth_descriptor_correlations")

    reg = regress_depth_on_descriptors(
        joined,
        depth_col=depth_col,
        n_splits=n_cv_splits,
        seed=seed,
        model=regression_model,
    )
    reg_table = pd.DataFrame([{
        "r2_mean": reg["r2_mean"], "r2_std": reg["r2_std"], "n": reg["n"],
    } | {f"fold{i+1}_r2": v for i, v in enumerate(reg["fold_r2"])}])
    save_table(reg_table, paths, "g1_cv_regression_summary")

    imp_df = pd.DataFrame(
        [{"descriptor": k, "importance": v} for k, v in reg["feature_importance"].items()]
    ).sort_values("importance", ascending=False)
    save_table(imp_df, paths, "g1_feature_importance")

    figs_a = plot_depth_vs_descriptors(
        joined, depth_col=depth_col, descriptors=COMPLEXITY_DESCRIPTORS,
        paths=paths, corr_table=corr_table,
    )
    figs_b = plot_feature_importance(reg["feature_importance"], paths)

    # Headline: maximum |ρ| across descriptors and CV R²
    max_rho = corr_table["spearman_rho"].abs().max() if not corr_table.empty else float("nan")
    best_desc = (
        corr_table.iloc[corr_table["spearman_rho"].abs().idxmax()]["descriptor"]
        if not corr_table.empty else "n/a"
    )
    if np.isnan(max_rho):
        verdict = "indeterminate (no descriptors available)"
    elif max_rho >= 0.4:
        verdict = "STRONG support for C1"
    elif max_rho >= 0.2:
        verdict = "WEAK but consistent support for C1"
    else:
        verdict = "C1 FALSIFIED at this checkpoint"

    write_summary_md(
        paths,
        title="G.1 — Effective halting depth vs molecular complexity",
        claim="C1: per-token halting depth correlates with chemistry-grounded complexity descriptors.",
        headline=f"{verdict}: max |ρ|={max_rho:.3f} on {best_desc}; CV R²={reg['r2_mean']:.3f}±{reg['r2_std']:.3f} (n={reg['n']}).",
        details={
            "Descriptors evaluated": ", ".join(COMPLEXITY_DESCRIPTORS),
            "Bootstrap resamples": str(n_bootstrap),
            "CV splits": str(n_cv_splits),
            "Regression model": regression_model,
        },
        tables_referenced=[
            "g1_depth_descriptor_correlations.csv",
            "g1_cv_regression_summary.csv",
            "g1_feature_importance.csv",
        ],
        figures_referenced=[p.name for p in figs_a + figs_b],
    )

    return {
        "correlations": corr_table,
        "cv_regression": reg,
        "verdict": verdict,
        "paths": paths,
    }


def run_panel(
    panel_root: Path | str,
    descriptors_path: Path | str,
    out_dir: Path | str,
    *,
    lf_model_dir: str = "radiant",
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    """Aggregate G.1 correlations across every LF cell in the panel.

    Discovers ``<panel_root>/<lf_model_dir>/*/*/predictions.csv``, runs
    per-cell Spearman ρ (no CV regression — too slow across 100 cells),
    and produces aggregate statistics + a forest-plot figure.
    """
    import matplotlib.pyplot as plt

    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, name="g1_depth_vs_complexity")

    desc = _read_descriptors(Path(descriptors_path))

    cell_files = sorted(
        (panel_root / lf_model_dir).glob("*/*/predictions.csv")
    )
    if not cell_files:
        raise FileNotFoundError(
            f"No predictions.csv found under {panel_root / lf_model_dir}. "
            "Check that fine-tuning completed and paths are correct."
        )
    logger.info("Panel mode: found %d LF cells", len(cell_files))

    per_cell_rows: list[dict] = []
    all_rho: dict[str, list[float]] = {d: [] for d in COMPLEXITY_DESCRIPTORS}

    for cell_path in cell_files:
        parts = cell_path.parts
        # …/radiant/<target>/<split>/predictions.csv
        split_name = parts[-2]
        target_name = parts[-3]
        try:
            preds = load_predictions(cell_path)
            depth_col = _resolve_depth_column(preds)
            joined = join_descriptors(preds, desc)
            if len(joined) < 10:
                logger.warning("Cell %s/%s: only %d molecules after join; skipping",
                               target_name, split_name, len(joined))
                continue
            corr = correlate_depth_descriptors(
                joined, depth_col=depth_col,
                n_bootstrap=n_bootstrap, seed=seed,
            )
            row: dict = {"target": target_name, "split": split_name, "depth_col": depth_col,
                         "n": len(joined)}
            for _, r in corr.iterrows():
                d = r["descriptor"]
                rho = r["spearman_rho"]
                row[f"{d}_rho"] = rho
                all_rho.setdefault(d, []).append(rho)
            # headline: best |ρ| across descriptors
            rho_vals = corr["spearman_rho"].abs()
            if rho_vals.notna().any():
                best_idx = rho_vals.idxmax()
                row["best_rho"] = corr.iloc[best_idx]["spearman_rho"]
                row["best_desc"] = corr.iloc[best_idx]["descriptor"]
            per_cell_rows.append(row)
        except Exception as exc:
            logger.warning("Cell %s/%s failed: %s", target_name, split_name, exc)

    if not per_cell_rows:
        raise RuntimeError("All panel cells failed; cannot produce G.1 panel report.")

    cell_df = pd.DataFrame(per_cell_rows)
    save_table(cell_df, paths, "g1_panel_per_cell_correlations")

    # Aggregate: median + IQR per descriptor
    agg_rows = []
    for d in COMPLEXITY_DESCRIPTORS:
        col = f"{d}_rho"
        if col not in cell_df.columns:
            continue
        vals = cell_df[col].dropna().to_numpy(float)
        if len(vals) == 0:
            continue
        agg_rows.append({
            "descriptor": d,
            "n_cells": len(vals),
            "median_rho": float(np.median(vals)),
            "mean_rho": float(np.mean(vals)),
            "std_rho": float(np.std(vals)),
            "q25_rho": float(np.percentile(vals, 25)),
            "q75_rho": float(np.percentile(vals, 75)),
        })
    if agg_rows:
        agg_df = pd.DataFrame(agg_rows).sort_values("median_rho", key=abs, ascending=False)
    else:
        # Every cell produced all-NaN correlations -- typically because the
        # halting head collapsed and the depth column is constant. Keep an
        # empty (but properly-typed) frame so the rest of the function can
        # still emit a diagnostic figure / summary instead of crashing.
        agg_df = pd.DataFrame(columns=["descriptor", "n_cells", "median_rho",
                                       "mean_rho", "std_rho", "q25_rho", "q75_rho"])
    save_table(agg_df, paths, "g1_panel_aggregate_correlations")

    # Always emit a depth-distribution diagnostic figure. When the halting
    # head collapsed (depth is constant) this is the only meaningful figure
    # we can produce; when it varies, it's a useful sanity check.
    _plot_depth_distribution(cell_files, paths)

    # Forest plot: median ρ per descriptor, IQR as error bars
    figs: list[Path] = []
    if not agg_df.empty:
        _pos_col = NATURE_PALETTE[2]   # teal green for positive ρ
        _neg_col = NATURE_PALETTE[0]   # vermilion for negative ρ
        fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, max(2.2, 0.42 * len(agg_df))))
        descs = agg_df["descriptor"].tolist()
        med = agg_df["median_rho"].to_numpy()
        err_lo = (agg_df["median_rho"] - agg_df["q25_rho"]).to_numpy()
        err_hi = (agg_df["q75_rho"] - agg_df["median_rho"]).to_numpy()
        bar_colors = [_pos_col if v >= 0 else _neg_col for v in med]
        ax.barh(descs, med, xerr=[err_lo, err_hi], color=bar_colors,
                capsize=3, ecolor="#333333", error_kw={"lw": 0.8}, edgecolor="none")
        ax.axvline(0, color="#000000", lw=0.8, ls="--")
        ax.set_xlabel("Spearman ρ  (median, IQR bars)", fontweight="bold")
        ax.set_title(f"Halting depth vs complexity\n({len(per_cell_rows)} panel cells)",
                     fontweight="bold")
        fig.tight_layout()
        figs = save_figure(fig, paths, "g1_panel_forest_plot")
        plt.close(fig)

    # Per-split violin across targets (skips internally when no signal)
    _plot_panel_split_violin(cell_df, paths)

    # Headline metric
    if not agg_df.empty:
        best_row = agg_df.iloc[0]
        max_med_rho = abs(best_row["median_rho"])
        best_desc = best_row["descriptor"]
        n_cells = int(best_row["n_cells"])
        if max_med_rho >= 0.3:
            verdict = "STRONG support for C1 across panel"
        elif max_med_rho >= 0.15:
            verdict = "MODERATE support for C1 across panel"
        else:
            verdict = "WEAK / no panel-level C1 support"
    else:
        # Depth column was constant in every cell -- halting head collapsed.
        # Surface that explicitly so the diagnostic figure has a clear story.
        max_med_rho = float("nan")
        best_desc = "n/a"
        n_cells = 0
        verdict = ("HALTING COLLAPSED: depth column is constant across every "
                   "molecule; ρ is undefined. See g1_depth_distribution.png.")
        # Build a stub row so the summary code below has something to write.
        best_row = {"q25_rho": float("nan"), "q75_rho": float("nan"),
                    "mean_rho": float("nan"), "std_rho": float("nan")}

    write_summary_md(
        paths,
        title="G.1 — Effective halting depth vs molecular complexity (panel)",
        claim="C1: per-token halting depth correlates with chemistry-grounded complexity descriptors across all 100 panel cells.",
        headline=(
            f"{verdict}: median |ρ|={max_med_rho:.3f} on {best_desc} "
            f"across {n_cells} LF cells (IQR [{best_row['q25_rho']:.3f}, {best_row['q75_rho']:.3f}])."
        ),
        details={
            "Panel cells processed": str(len(per_cell_rows)),
            "Bootstrap resamples per cell": str(n_bootstrap),
            "Best descriptor (median |ρ|)": best_desc,
            "Mean ρ (best desc)": f"{best_row['mean_rho']:.3f} ± {best_row['std_rho']:.3f}",
        },
        tables_referenced=[
            "g1_panel_per_cell_correlations.csv",
            "g1_panel_aggregate_correlations.csv",
        ],
        figures_referenced=(
            [p.name for p in figs] +
            [p.name for p in (paths.figures.glob("g1_depth_distribution.*")
                              if paths.figures.exists() else [])]
        ),
    )

    return {
        "per_cell": cell_df,
        "aggregate": agg_df,
        "verdict": verdict,
        "paths": paths,
    }


def _plot_depth_distribution(cell_files, paths: AnalysisPaths) -> None:
    """Diagnostic figure: histograms of effective_depth / soft_effective_depth.

    Always emitted (even when the halting head collapses to a single value)
    because it is the most direct visual evidence of *what* the model is
    doing with its loop budget per molecule.
    """
    import matplotlib.pyplot as plt

    eff: list[float] = []
    soft: list[float] = []
    for cell_path in cell_files:
        try:
            df = pd.read_csv(cell_path)
        except Exception:
            continue
        if "effective_depth" in df.columns:
            eff.extend(pd.to_numeric(df["effective_depth"], errors="coerce").dropna().tolist())
        if "soft_effective_depth" in df.columns:
            soft.extend(pd.to_numeric(df["soft_effective_depth"], errors="coerce").dropna().tolist())
    if not eff and not soft:
        return

    colors = nature_palette(2)
    fig, axes = plt.subplots(1, 2, figsize=(NC_DOUBLE_COL, 2.6), sharey=True)

    def _hist(ax, vals, color, label):
        if not vals:
            ax.text(0.5, 0.5, "no values", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label, fontweight="bold")
            return
        arr = np.asarray(vals, dtype=float)
        nuniq = len(np.unique(np.round(arr, 4)))
        if nuniq <= 1:
            ax.axvline(float(arr[0]), color=color, lw=2.0)
            ax.set_title(f"{label} (COLLAPSED to {arr[0]:.3f})", fontweight="bold")
        else:
            ax.hist(arr, bins=40, color=color, edgecolor="#000000", linewidth=0.4)
            ax.set_title(f"{label} (n_unique={nuniq})", fontweight="bold")
        ax.set_xlabel(label, fontweight="bold")

    _hist(axes[0], eff, colors[0], "effective_depth")
    _hist(axes[1], soft, colors[1], "soft_effective_depth")
    axes[0].set_ylabel("Molecule count", fontweight="bold")
    fig.suptitle(f"Halt-depth distribution across {len(cell_files)} panel cells",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, paths, "g1_depth_distribution")
    plt.close(fig)


def _plot_panel_split_violin(cell_df: pd.DataFrame, paths: AnalysisPaths) -> None:
    """Box-per-split for BertzCT_rho — shows if one split type drives signal."""
    import matplotlib.pyplot as plt

    col = "BertzCT_rho"
    if col not in cell_df.columns or "split" not in cell_df.columns:
        return
    splits = sorted(cell_df["split"].unique())
    data = [cell_df.loc[cell_df["split"] == s, col].dropna().to_numpy() for s in splits]
    if all(len(d) == 0 for d in data):
        return
    colors = nature_palette(len(splits))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.8))
    bp = ax.boxplot(data, labels=splits, patch_artist=True, widths=0.5,
                    medianprops={"color": "#000000", "lw": 1.2},
                    whiskerprops={"lw": 0.8}, capprops={"lw": 0.8},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#000000")
        patch.set_linewidth(0.6)
    ax.axhline(0, color="#000000", lw=0.8, ls="--")
    ax.set_ylabel("Spearman ρ  (BertzCT vs depth)", fontweight="bold")
    ax.set_title("ρ by split type across 20 targets", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, paths, "g1_panel_split_boxplot")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path, help="Single-cell predictions.csv")
    mode.add_argument("--panel-root", type=Path,
                      help="Panel root dir; discovers all radiant/*/predictions.csv")
    p.add_argument("--descriptors", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--n-cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regression-model", choices=["ridge", "rf"], default="ridge")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    if args.panel_root:
        run_panel(
            panel_root=args.panel_root,
            descriptors_path=args.descriptors,
            out_dir=args.out_dir,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    else:
        run(
            predictions_path=args.predictions,
            descriptors_path=args.descriptors,
            out_dir=args.out_dir,
            n_bootstrap=args.n_bootstrap,
            n_cv_splits=args.n_cv_splits,
            seed=args.seed,
            regression_model=args.regression_model,
        )


if __name__ == "__main__":
    main()
