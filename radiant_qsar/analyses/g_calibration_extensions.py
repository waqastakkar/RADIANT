"""Phase G -- Calibration extensions: ECE, reliability diagram, top-k MAE.

Standalone module that augments the existing g3_calibration with the
classical "regression calibration" diagnostics reviewers expect:

* Reliability diagram (10 quantile bins of |error|) -- per model
* Expected Calibration Error (ECE-like) for absolute error vs predicted-uncertainty proxy
* Top-k% confidence-filtered MAE curve (using AD-Tanimoto as proxy since
  the trained checkpoint's confidence_var is degenerate)
* Predicted-vs-observed parity overlay for all models on the same axes
* Sharpness diagnostic: spread of pred residuals per cell

Outputs
-------
tables/
    calibration_ece.csv
    calibration_reliability.csv
    calibration_topk.csv
figures/
    g_cal_ext_reliability.{png,svg}
    g_cal_ext_ece_bar.{png,svg}
    g_cal_ext_topk_mae.{png,svg}
    g_cal_ext_parity_overlay.{png,svg}
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
    discover_predictions,
    load_predictions,
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


def _load_all_test(panel_root: Path, splits: tuple[str, ...] | None) -> pd.DataFrame:
    manifest = discover_predictions(panel_root)
    if splits:
        manifest = manifest[manifest["split"].isin(splits)]
    rows = []
    for _, item in manifest.iterrows():
        try:
            p = load_predictions(item["path"])
            p["model"] = item["model"]
            p["target"] = item["target"]
            p["panel_split"] = item["split"]
            p["abs_err"] = (p["pred_pchembl"] - p["true_pchembl"]).abs()
            rows.append(p[["model", "target", "panel_split", "true_pchembl",
                           "pred_pchembl", "abs_err"]])
        except Exception as exc:
            logger.warning("skip %s: %s", item["path"], exc)
    if not rows:
        raise FileNotFoundError(f"no predictions under {panel_root}")
    return pd.concat(rows, ignore_index=True)


def _reliability_diagram(per_mol: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    """For each model, bin test molecules by predicted value quantile and compare
    mean predicted to mean observed inside each bin.

    This is the QSAR-flavoured analog of the classification reliability diagram:
    a perfectly-calibrated regressor sits on the y = x line.
    """
    import matplotlib.pyplot as plt

    out_rows = []
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, NC_SINGLE_COL * 1.4))
    models = sorted(per_mol["model"].unique())
    colors = nature_palette(len(models))
    lo = float(per_mol[["true_pchembl", "pred_pchembl"]].min().min())
    hi = float(per_mol[["true_pchembl", "pred_pchembl"]].max().max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y = x")

    for c, m in zip(colors, models):
        sub = per_mol[per_mol["model"] == m]
        if sub.empty:
            continue
        qbins = pd.qcut(sub["pred_pchembl"], q=10, duplicates="drop")
        agg = sub.groupby(qbins, observed=True).agg(
            mean_pred=("pred_pchembl", "mean"),
            mean_true=("true_pchembl", "mean"),
            n=("pred_pchembl", "count"),
        ).reset_index(drop=True)
        ax.plot(agg["mean_pred"], agg["mean_true"], "-o",
                color=c, lw=1.4, ms=4, label=m)
        for _, r in agg.iterrows():
            out_rows.append({"model": m, "bin_mean_pred": float(r["mean_pred"]),
                             "bin_mean_true": float(r["mean_true"]),
                             "n": int(r["n"])})
    ax.set_xlabel("Mean predicted pChEMBL in bin", fontweight="bold")
    ax.set_ylabel("Mean observed pChEMBL in bin", fontweight="bold")
    ax.set_title("Regression reliability diagram\n(10 quantile bins of prediction)",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_cal_ext_reliability")
    plt.close(fig)
    return pd.DataFrame(out_rows)


def _ece(per_mol: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    """Expected Calibration Error: mean over 10 quantile-of-prediction bins of
    |bin_mean_pred - bin_mean_true|. Lower is better.
    """
    import matplotlib.pyplot as plt

    rows = []
    for m, sub in per_mol.groupby("model"):
        if sub.empty:
            continue
        qbins = pd.qcut(sub["pred_pchembl"], q=10, duplicates="drop")
        agg = sub.groupby(qbins, observed=True).agg(
            mean_pred=("pred_pchembl", "mean"),
            mean_true=("true_pchembl", "mean"),
            n=("pred_pchembl", "count"),
        )
        ece = float((agg["mean_pred"] - agg["mean_true"]).abs()
                    .mul(agg["n"], axis=0).sum() / agg["n"].sum())
        rows.append({"model": m, "ECE": ece, "n_bins": int(len(agg)),
                     "n_total": int(agg["n"].sum())})
    df = pd.DataFrame(rows).sort_values("ECE")
    if df.empty:
        return df

    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 2.8))
    colors = nature_palette(len(df))
    ax.bar(df["model"], df["ECE"], color=colors, edgecolor="none")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(i, r["ECE"], f"{r['ECE']:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="bold")
    ax.set_ylabel("ECE (lower = better calibrated)", fontweight="bold")
    ax.set_title("Expected Calibration Error\n(quantile-bin mean |pred - true|)",
                 fontweight="bold", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    save_figure(fig, paths, "g_cal_ext_ece_bar")
    plt.close(fig)
    return df


def _topk_mae(per_mol: pd.DataFrame, *,
              ad_csv: Path | None,
              paths: AnalysisPaths) -> pd.DataFrame:
    """Top-k% confidence-filtered MAE curve.

    Uses Tanimoto-NN-to-train from the AD module (high similarity -> high
    confidence). Falls back to predicted-magnitude proximity to the test
    median if AD output is missing.
    """
    import matplotlib.pyplot as plt

    if ad_csv is not None and Path(ad_csv).exists():
        ad = pd.read_csv(ad_csv)[["model", "target", "split",
                                  "inchikey14", "max_tanimoto_to_train"]]
        df = per_mol.merge(ad, left_on=["model", "target", "panel_split"],
                           right_on=["model", "target", "split"],
                           how="left")
        df = df.dropna(subset=["max_tanimoto_to_train"])
        score_col = "max_tanimoto_to_train"
        score_asc = False  # high sim -> high confidence
        score_label = "Tanimoto-NN to train"
    else:
        # fallback: use 1 - |z-score of pred| as confidence
        df = per_mol.copy()
        df["score"] = -np.abs(
            (df["pred_pchembl"] - df["pred_pchembl"].median())
            / df["pred_pchembl"].std(ddof=0))
        score_col = "score"
        score_asc = False
        score_label = "fallback (pred z-score)"

    rows = []
    ks = np.arange(0.05, 1.001, 0.05)
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.6, 3.0))
    models = sorted(df["model"].unique())
    colors = nature_palette(len(models))
    for c, m in zip(colors, models):
        sub = df[df["model"] == m].dropna(subset=[score_col, "abs_err"])
        if sub.empty:
            continue
        sub_sorted = sub.sort_values(score_col, ascending=score_asc)
        n = len(sub_sorted)
        xs, ys = [], []
        for k in ks:
            keep = max(int(n * k), 1)
            mae_k = float(sub_sorted.head(keep)["abs_err"].mean())
            rows.append({"model": m, "retention": float(k),
                         "n_kept": int(keep), "mae": mae_k,
                         "score_signal": score_label})
            xs.append(float(k) * 100)
            ys.append(mae_k)
        ax.plot(xs, ys, "-o", color=c, lw=1.4, ms=4, label=m)
    ax.set_xlabel(f"Retention (%) -- ranked by {score_label}", fontweight="bold")
    ax.set_ylabel("MAE on retained subset (pChEMBL)", fontweight="bold")
    ax.set_title("Top-k confidence-filtered MAE", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_cal_ext_topk_mae")
    plt.close(fig)
    return pd.DataFrame(rows)


def _parity_overlay(per_mol: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt

    if per_mol.empty:
        return
    models = sorted(per_mol["model"].unique())
    ncols = min(3, len(models))
    nrows = (len(models) + ncols - 1) // ncols
    colors = nature_palette(len(models))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(NC_DOUBLE_COL,
                                      NC_DOUBLE_COL * nrows / ncols * 0.85),
                             squeeze=False)
    lo = float(per_mol[["true_pchembl", "pred_pchembl"]].min().min())
    hi = float(per_mol[["true_pchembl", "pred_pchembl"]].max().max())
    for i, m in enumerate(models):
        ax = axes[i // ncols][i % ncols]
        sub = per_mol[per_mol["model"] == m]
        if sub.empty:
            ax.set_axis_off()
            continue
        ax.scatter(sub["true_pchembl"], sub["pred_pchembl"],
                   s=4, alpha=0.15, color=colors[i], linewidths=0)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{m}  (n={len(sub):,})", fontweight="bold", fontsize=9)
        ax.set_xlabel("Observed pChEMBL", fontweight="bold", fontsize=8)
        ax.set_ylabel("Predicted pChEMBL", fontweight="bold", fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.4)
    for k in range(len(models), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle("Parity plots (all test molecules pooled across cells)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, paths, "g_cal_ext_parity_overlay")
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        splits: tuple[str, ...] | None = None) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_calibration_extensions")

    per_mol = _load_all_test(panel_root, splits)

    ece_df = _ece(per_mol, paths)
    save_table(ece_df, paths, "calibration_ece")
    rel_df = _reliability_diagram(per_mol, paths)
    save_table(rel_df, paths, "calibration_reliability")

    ad_csv = (out_dir / "g_applicability_domain" / "tables" /
              "ad_per_molecule.csv")
    if not ad_csv.exists():
        ad_csv = None
    topk_df = _topk_mae(per_mol, ad_csv=ad_csv, paths=paths)
    save_table(topk_df, paths, "calibration_topk")

    _parity_overlay(per_mol, paths)

    # Headline
    headline = "calibration extensions produced no output."
    if not ece_df.empty:
        best = ece_df.iloc[0]
        worst = ece_df.iloc[-1]
        headline = (f"Best-calibrated model: {best['model']} "
                    f"(ECE = {best['ECE']:.3f}). Worst: {worst['model']} "
                    f"(ECE = {worst['ECE']:.3f}).")

    write_summary_md(
        paths,
        title="Calibration extensions (ECE, reliability, top-k filter)",
        claim=("Classical regression-calibration diagnostics: quantile-bin "
               "mean prediction vs observation (reliability diagram), "
               "ECE per model, and a confidence-filtered MAE curve."),
        headline=headline,
        details={
            "Bins per reliability curve": "10 (quantile-of-prediction)",
            "Confidence signal for top-k": ("Tanimoto-NN to train (from g_applicability_domain)"
                                            if ad_csv else "fallback: pred z-score"),
            "Splits": ", ".join(splits) if splits else "all",
        },
        tables_referenced=[
            "calibration_ece.csv",
            "calibration_reliability.csv",
            "calibration_topk.csv",
        ],
        figures_referenced=[
            "g_cal_ext_ece_bar.png",
            "g_cal_ext_reliability.png",
            "g_cal_ext_topk_mae.png",
            "g_cal_ext_parity_overlay.png",
        ],
    )
    return {"paths": paths, "ece": ece_df, "reliability": rel_df, "topk": topk_df}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--splits", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        splits=tuple(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
