"""Compute parity & cross-cutting controls (Phase G.6).

* **FLOPs / params annotation** per model from a panel-root manifest
  (``params_flops.csv``) or auto-counted for RADIANT checkpoints.
* **Paired bootstrap CIs (10 K resamples)** on Pearson, MAE, RMSE for
  each (model, target, split). Holm-corrected p-values over the family
  of RADIANT-vs-baseline comparisons.
* **Label-permutation sanity check**: re-runs the per-descriptor
  Spearman of effective_depth vs complexity using a *shuffled*
  label assignment; reported alongside the real correlation to
  demonstrate the C1 signal is not a data artefact.

Inputs
------
* ``panel-root`` — same layout as :mod:`g_pairwise_wins`.
* ``--params-flops`` — optional CSV with columns ``model, params, flops_per_forward``;
  if omitted we try to read each model dir for a ``params_flops.json``.
* ``--predictions`` (for label-permutation): a RADIANT predictions CSV
  with ``effective_depth`` and joined to descriptors.

Outputs
-------
* ``compute_table.csv``       — params, FLOPs, FLOPs-per-prediction.
* ``bootstrap_cis.csv``       — Pearson/MAE/RMSE per (model, target, split) with 95% CIs.
* ``radiant_vs_baselines.csv`` — paired tests with Holm-corrected p-values.
* ``g_compute_vs_metric.{png,svg}`` — scatter of MAE vs FLOPs.
* ``g_label_permutation.{png,svg}`` — depth-vs-descriptor ρ, real vs shuffled.
* ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    bootstrap_paired_diff,
    discover_predictions,
    holm_correction,
    join_descriptors,
    load_predictions,
    nature_palette,
    publication_style,
    regression_metrics,
    save_figure,
    save_table,
    spearman_pearson,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compute table (params / FLOPs)
# ---------------------------------------------------------------------------

def _try_load_params_flops_json(model_dir: Path) -> dict | None:
    for cand in ("params_flops.json", "compute.json"):
        p = model_dir / cand
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def build_compute_table(panel_root: Path, manifest_csv: Path | None) -> pd.DataFrame:
    """Either read a manifest CSV or auto-collect per-model compute.json."""
    if manifest_csv is not None and Path(manifest_csv).exists():
        df = pd.read_csv(manifest_csv)
        if "flops_per_forward" not in df.columns and "flops" in df.columns:
            df = df.rename(columns={"flops": "flops_per_forward"})
        return df

    rows: list[dict] = []
    for model_dir in sorted(p for p in panel_root.iterdir() if p.is_dir()):
        info = _try_load_params_flops_json(model_dir) or {}
        rows.append({
            "model": model_dir.name,
            "params": info.get("params", np.nan),
            "flops_per_forward": info.get("flops_per_forward", info.get("flops", np.nan)),
            "n_loops": info.get("n_loops", np.nan),
            "has_halting": bool(info.get("has_halting", False)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-(model, target, split) bootstrap CIs
# ---------------------------------------------------------------------------

def bootstrap_metric(values_true: np.ndarray, values_pred: np.ndarray, *,
                     metric: str, n_bootstrap: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap CI for one of mae / rmse / pearson / spearman."""
    from scipy import stats as scistats

    mask = np.isfinite(values_true) & np.isfinite(values_pred)
    yt, yp = values_true[mask], values_pred[mask]
    n = yt.size
    if n < 2:
        return {"value": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": n}

    def calc(yt_, yp_):
        if metric == "mae":
            return float(np.mean(np.abs(yt_ - yp_)))
        if metric == "rmse":
            return float(np.sqrt(np.mean((yt_ - yp_) ** 2)))
        if metric == "pearson":
            return float(scistats.pearsonr(yt_, yp_)[0])
        if metric == "spearman":
            return float(scistats.spearmanr(yt_, yp_).correlation)
        raise ValueError(metric)

    obs = calc(yt, yp)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[b] = calc(yt[idx], yp[idx])
    return {
        "value": obs,
        "ci_lo": float(np.nanpercentile(boots, 2.5)),
        "ci_hi": float(np.nanpercentile(boots, 97.5)),
        "n": int(n),
    }


def bootstrap_table(
    panel_root: Path | str,
    *,
    metrics: Sequence[str] = ("mae", "rmse", "pearson"),
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    panel = discover_predictions(panel_root)
    rows: list[dict] = []
    for _, r in panel.iterrows():
        df = load_predictions(r["path"])
        row = {"model": r["model"], "target": r["target"], "split": r["split"], "n": len(df)}
        for m in metrics:
            res = bootstrap_metric(df["true_pchembl"].to_numpy(), df["pred_pchembl"].to_numpy(),
                                   metric=m, n_bootstrap=n_bootstrap, seed=seed)
            row[m] = res["value"]
            row[f"{m}_ci_lo"] = res["ci_lo"]
            row[f"{m}_ci_hi"] = res["ci_hi"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired RADIANT vs baseline tests with Holm correction
# ---------------------------------------------------------------------------

def radiant_vs_baselines(
    panel_root: Path | str,
    *,
    radiant_pattern: str = "radiant",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    panel = discover_predictions(panel_root)
    rows: list[dict] = []
    for (target, split), grp in panel.groupby(["target", "split"]):
        loop_models = [m for m in grp["model"] if radiant_pattern in m]
        other_models = [m for m in grp["model"] if radiant_pattern not in m]
        if not loop_models or not other_models:
            continue
        for lf in loop_models:
            lf_df = load_predictions(grp[grp["model"] == lf]["path"].iloc[0])
            for bl in other_models:
                bl_df = load_predictions(grp[grp["model"] == bl]["path"].iloc[0])
                merged = lf_df[["inchikey14", "true_pchembl", "pred_pchembl"]].rename(
                    columns={"pred_pchembl": "pred_lf"}
                ).merge(
                    bl_df[["inchikey14", "pred_pchembl"]].rename(columns={"pred_pchembl": "pred_bl"}),
                    on="inchikey14", how="inner",
                )
                if merged.empty:
                    continue
                err_lf = np.abs(merged["pred_lf"].to_numpy() - merged["true_pchembl"].to_numpy())
                err_bl = np.abs(merged["pred_bl"].to_numpy() - merged["true_pchembl"].to_numpy())
                boot = bootstrap_paired_diff(err_lf, err_bl, statistic="mean",
                                             n_bootstrap=n_bootstrap, seed=seed)
                rows.append({
                    "target": target, "split": split, "radiant_model": lf, "baseline": bl,
                    "n": int(len(merged)),
                    "mean_delta_mae": boot["mean_diff"],
                    "ci_lo": boot["ci"][0], "ci_hi": boot["ci"][1],
                    "p_value": boot["p_value"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["p_value_holm"] = holm_correction(df["p_value"].fillna(1.0).to_numpy())
    return df


# ---------------------------------------------------------------------------
# Label-permutation sanity check
# ---------------------------------------------------------------------------

def label_permutation_check(
    predictions_source: Path | str,
    descriptors_path: Path | str,
    *,
    n_perm: int = 200,
    seed: int = 0,
    lf_model_dir: str = "radiant",
    split: str = "scaffold",
) -> pd.DataFrame:
    """Shuffle effective_depth -> molecule assignment N times across all panel cells.

    ``predictions_source`` is treated as a panel root if it is a directory
    containing ``<lf_model_dir>/*/`` subdirectories; otherwise it is
    treated as a single predictions.csv file (legacy single-cell mode).

    Pools all LF cells for the real ρ estimate so the null comparison is
    representative of the full 20-target study, not a single cell.
    """
    desc = (pd.read_parquet(descriptors_path)
            if Path(descriptors_path).suffix.lower() == ".parquet"
            else pd.read_csv(descriptors_path))

    source = Path(predictions_source)
    # Panel mode: source is a directory; gather all LF cells
    lf_root = source / lf_model_dir
    if source.is_dir() and lf_root.is_dir():
        cell_csvs = sorted(lf_root.glob(f"*/{split}/predictions.csv"))
        if not cell_csvs:
            raise FileNotFoundError(
                f"No predictions.csv under {lf_root}/*/{split}/"
            )
        logger.info("label-perm check: pooling %d LF cells", len(cell_csvs))
        frames = []
        for p in cell_csvs:
            try:
                df = load_predictions(p)
                frames.append(df)
            except Exception as exc:
                logger.warning("skipping %s: %s", p, exc)
        preds = pd.concat(frames, ignore_index=True)
    else:
        preds = load_predictions(source)

    if "soft_effective_depth" in preds.columns and preds["soft_effective_depth"].notna().any():
        depth_col = "soft_effective_depth"
    elif "effective_depth" in preds.columns:
        depth_col = "effective_depth"
    elif "halt_step" in preds.columns:
        preds = preds.copy()
        preds["effective_depth"] = preds["halt_step"].astype(float) + 1.0
        depth_col = "effective_depth"
    else:
        raise ValueError("predictions need soft_effective_depth, effective_depth, or halt_step")

    joined = join_descriptors(preds, desc)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for d in COMPLEXITY_DESCRIPTORS:
        if d not in joined.columns:
            continue
        real = spearman_pearson(joined[d].to_numpy(dtype=float),
                                joined[depth_col].to_numpy(dtype=float),
                                n_bootstrap=200)
        shuffled_rho = np.empty(n_perm)
        depth = joined[depth_col].to_numpy(dtype=float)
        for k in range(n_perm):
            permuted = rng.permutation(depth)
            shuffled_rho[k] = spearman_pearson(joined[d].to_numpy(dtype=float),
                                               permuted, n_bootstrap=0)["spearman"]
        shuffled_rho = shuffled_rho[np.isfinite(shuffled_rho)]
        rows.append({
            "descriptor": d,
            "real_rho": real["spearman"],
            "shuffled_mean": float(np.mean(shuffled_rho)) if shuffled_rho.size else float("nan"),
            "shuffled_p2_5": float(np.percentile(shuffled_rho, 2.5)) if shuffled_rho.size else float("nan"),
            "shuffled_p97_5": float(np.percentile(shuffled_rho, 97.5)) if shuffled_rho.size else float("nan"),
            "permutations": int(shuffled_rho.size),
            "n_molecules": int(len(joined)),
            "depth_col": depth_col,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_compute_vs_metric(
    bootstrap_df: pd.DataFrame, compute_df: pd.DataFrame, paths: AnalysisPaths,
    *, metric: str = "mae",
) -> list[Path]:
    import matplotlib.pyplot as plt

    if bootstrap_df.empty or compute_df.empty:
        return []
    agg = bootstrap_df.groupby("model")[metric].mean().reset_index()
    merged = agg.merge(compute_df, on="model", how="left")
    has_flops = "flops_per_forward" in merged.columns and merged["flops_per_forward"].notna().any()
    xcol = "flops_per_forward" if has_flops else "params"
    if xcol not in merged.columns or merged[xcol].dropna().empty:
        logger.warning("no compute column to plot against; skipping compute-vs-metric figure")
        return []
    xlabel = "FLOPs / forward  (log scale)" if has_flops else "Parameters  (log scale)"
    colors = nature_palette(len(merged))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.8))
    for (_, r), c in zip(merged.iterrows(), colors):
        xv, yv = r.get(xcol, np.nan), r.get(metric, np.nan)
        if np.isfinite(xv) and np.isfinite(yv):
            ax.scatter(xv, yv, s=40, color=c, zorder=3)
            ax.annotate(r["model"], (xv, yv), fontsize=5, fontweight="bold",
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(f"Mean {metric.upper()}  (pChEMBL)", fontweight="bold")
    ax.set_title("Compute–accuracy Pareto", fontweight="bold")
    fig.tight_layout()
    out = save_figure(fig, paths, "g_compute_vs_metric")
    plt.close(fig)
    return out


def plot_label_perm(perm_df: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    if perm_df.empty:
        return []
    x = np.arange(len(perm_df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.8))
    ax.bar(x,       perm_df["real_rho"],      width=w, color=NATURE_PALETTE[2],
           label="Real ρ", edgecolor="none")
    ax.bar(x + w,   perm_df["shuffled_mean"], width=w, color="#BBBBBB",
           label="Shuffled mean", edgecolor="none")
    err_lo = perm_df["shuffled_mean"] - perm_df["shuffled_p2_5"]
    err_hi = perm_df["shuffled_p97_5"] - perm_df["shuffled_mean"]
    ax.errorbar(x + w, perm_df["shuffled_mean"], yerr=[err_lo, err_hi],
                fmt="none", ecolor="#333333", capsize=2.5, elinewidth=0.8)
    ax.axhline(0, color="#000000", lw=0.6)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(perm_df["descriptor"], rotation=30, ha="right", fontsize=6,
                       fontweight="bold")
    ax.set_ylabel("Spearman ρ  (depth vs descriptor)", fontweight="bold")
    ax.set_title("Label-permutation sanity check", fontweight="bold")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = save_figure(fig, paths, "g_label_permutation")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    panel_root: Path | str,
    out_dir: Path | str,
    *,
    params_flops_csv: Path | str | None = None,
    perm_predictions: Path | str | None = None,
    perm_descriptors: Path | str | None = None,
    n_bootstrap: int = 10_000,
    n_perm: int = 200,
    seed: int = 0,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g_compute_parity")
    panel_root = Path(panel_root)

    compute_df = build_compute_table(panel_root,
                                     Path(params_flops_csv) if params_flops_csv else None)
    save_table(compute_df, paths, "compute_table")

    boot_df = bootstrap_table(panel_root, n_bootstrap=n_bootstrap, seed=seed)
    save_table(boot_df, paths, "bootstrap_cis")

    paired_df = radiant_vs_baselines(panel_root, n_bootstrap=n_bootstrap, seed=seed)
    save_table(paired_df, paths, "radiant_vs_baselines")

    perm_df = pd.DataFrame()
    # Use panel_root for the perm check when no explicit single-cell file given.
    perm_source = perm_predictions if perm_predictions is not None else panel_root
    perm_desc = perm_descriptors if perm_descriptors is not None else None
    if perm_desc is not None:
        perm_df = label_permutation_check(perm_source, perm_desc,
                                          n_perm=n_perm, seed=seed)
        save_table(perm_df, paths, "label_permutation")

    figs: list[Path] = []
    figs += plot_compute_vs_metric(boot_df, compute_df, paths, metric="mae")
    figs += plot_label_perm(perm_df, paths)

    if not paired_df.empty:
        sig = paired_df[paired_df["p_value_holm"] < 0.05]
        verdict = (
            f"{len(sig)}/{len(paired_df)} RADIANT-vs-baseline comparisons reach "
            f"Holm-corrected significance (p<0.05); "
            f"mean ΔMAE = {paired_df['mean_delta_mae'].mean():.3f} (negative = LF better)."
        )
    else:
        verdict = "No RADIANT-vs-baseline pairs found in panel."

    write_summary_md(
        paths,
        title="Compute parity & cross-cutting controls",
        claim="Cross-cutting: compute annotation, paired bootstrap CIs, Holm correction, label-permutation sanity.",
        headline=verdict,
        details={
            "Bootstrap resamples": str(n_bootstrap),
            "Label-perm shuffles": str(n_perm),
            "Compute manifest source": str(params_flops_csv) if params_flops_csv else "auto (per-model compute.json)",
        },
        tables_referenced=[
            "compute_table.csv",
            "bootstrap_cis.csv",
            "radiant_vs_baselines.csv",
            *(["label_permutation.csv"] if not perm_df.empty else []),
        ],
        figures_referenced=[p.name for p in figs],
    )

    return {
        "compute": compute_df,
        "bootstrap": boot_df,
        "paired": paired_df,
        "label_perm": perm_df,
        "paths": paths,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G compute parity & controls")
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--params-flops-csv", type=Path, default=None)
    p.add_argument("--perm-predictions", type=Path, default=None,
                   help="Optional: single LF predictions CSV for label-perm. "
                        "Omit to pool all LF scaffold cells from --panel-root automatically.")
    p.add_argument("--perm-descriptors", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=10_000)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(
        panel_root=args.panel_root,
        out_dir=args.out_dir,
        params_flops_csv=args.params_flops_csv,
        perm_predictions=args.perm_predictions,
        perm_descriptors=args.perm_descriptors,
        n_bootstrap=args.n_bootstrap,
        n_perm=args.n_perm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
