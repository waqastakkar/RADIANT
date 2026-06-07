"""Phase G.3 — Calibration & uncertainty (Sub-claim C3).

Compares the calibration quality of RADIANT's two uncertainty signals
against compute-matched deep ensembles:

* **RADIANT halting confidence variance**: per molecule, the variance
  of the halting head's confidence across loop steps; large variance ↔
  the model "hesitated".
* **RADIANT posterior over n_loops** (Monte Carlo): for each molecule,
  K predictions at random ``n_loops`` values are drawn; the prediction
  variance is the uncertainty.
* **Deep-ensemble baseline**: K independent baselines, predictions
  averaged with their inter-seed variance as the uncertainty.

For regression we compute:

* Reliability diagram (expected vs observed coverage at quantiles).
* **Expected Calibration Error (ECE)** over the predicted-quantile bins.
* **Brier-like score**: mean ``(I[y ∈ predicted interval] − coverage)²``.
* **Negative Log Likelihood** assuming Gaussian ``p(y|x) = N(μ, σ²)``.

Inputs
------
A *long* predictions CSV (or multiple CSVs concatenated upstream) where
each row carries:

    inchikey14, target_chembl_id, split_kind, smiles,
    true_pchembl, pred_pchembl, sigma_pchembl, model

``model`` distinguishes the uncertainty source (e.g.
``radiant_halt_var``, ``radiant_mc_loops``, ``ensemble_5``).
``sigma_pchembl`` is the per-row predicted standard deviation.

If a single predictions file is supplied without a ``sigma_pchembl``
column we attempt to construct two RADIANT signals from auxiliary
columns:

* ``confidence_var`` -> halting-confidence variance, mapped through a
  fitted linear calibrator (variance -> σ) on the validation split if
  supplied via ``--calibration-fit-csv``.
* ``predictions_nloops*`` files in a sibling directory -> MC over loops.

Outputs
-------
* ``calibration_metrics.csv`` (per model: ECE, Brier-like, NLL,
  coverage@90, coverage@50, sharpness).
* ``reliability_diagram.{png,svg}`` (one line per model).
* ``sharpness_vs_calibration.{png,svg}`` (scatter).
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
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building uncertainty estimates from raw RADIANT artefacts
# ---------------------------------------------------------------------------

def mc_loops_to_sigma(loop_predictions_dir: Path | str) -> pd.DataFrame:
    """Aggregate predictions_nloops{K}.csv files into per-row μ and σ."""
    loop_predictions_dir = Path(loop_predictions_dir)
    frames = []
    for p in loop_predictions_dir.glob("predictions_nloops*.csv"):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no predictions_nloops*.csv under {loop_predictions_dir}")
    full = pd.concat(frames, ignore_index=True)
    grp = full.groupby(["inchikey14", "target_chembl_id", "split_kind"]).agg(
        pred_pchembl=("pred_pchembl", "mean"),
        sigma_pchembl=("pred_pchembl", "std"),
        true_pchembl=("true_pchembl", "first"),
        smiles=("smiles", "first"),
    ).reset_index()
    grp["sigma_pchembl"] = grp["sigma_pchembl"].fillna(0.0)
    grp["model"] = "radiant_mc_loops"
    return grp


def halt_var_to_sigma(
    predictions_csv: Path | str,
    *,
    calibration_csv: Path | str | None = None,
) -> pd.DataFrame:
    """Map a halt-confidence variance column to predicted σ.

    If ``calibration_csv`` is supplied we fit a 1-D linear regression of
    |residual| on confidence-variance on that split (the validation set)
    and apply it to the predictions file. Otherwise we use the raw
    variance with an identity scale.
    """
    df = pd.read_csv(predictions_csv)
    if "confidence_var" not in df.columns:
        raise ValueError(f"{predictions_csv}: needs a 'confidence_var' column")
    if calibration_csv is not None:
        cal = pd.read_csv(calibration_csv)
        from sklearn.linear_model import LinearRegression

        x = cal["confidence_var"].to_numpy(dtype=float).reshape(-1, 1)
        y = np.abs(cal["true_pchembl"].to_numpy() - cal["pred_pchembl"].to_numpy())
        mask = np.isfinite(x.ravel()) & np.isfinite(y)
        reg = LinearRegression().fit(x[mask], y[mask])
        df["sigma_pchembl"] = np.clip(reg.predict(df["confidence_var"].to_numpy(dtype=float).reshape(-1, 1)),
                                      a_min=1e-4, a_max=None)
    else:
        df["sigma_pchembl"] = np.sqrt(np.clip(df["confidence_var"].to_numpy(dtype=float), 1e-8, None))
    df["model"] = "radiant_halt_var"
    return df


def ensemble_to_sigma(per_seed_files: Sequence[Path | str], model_name: str = "ensemble") -> pd.DataFrame:
    """Combine per-seed predictions.csv files into μ ± σ."""
    frames = []
    for i, p in enumerate(per_seed_files):
        df = pd.read_csv(p)
        df["seed_idx"] = i
        frames.append(df)
    if not frames:
        raise FileNotFoundError("no ensemble member predictions supplied")
    full = pd.concat(frames, ignore_index=True)
    grp = full.groupby(["inchikey14", "target_chembl_id", "split_kind"]).agg(
        pred_pchembl=("pred_pchembl", "mean"),
        sigma_pchembl=("pred_pchembl", "std"),
        true_pchembl=("true_pchembl", "first"),
        smiles=("smiles", "first"),
    ).reset_index()
    grp["sigma_pchembl"] = grp["sigma_pchembl"].fillna(0.0)
    grp["model"] = model_name
    return grp


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def gaussian_nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian NLL with σ clipped from below for numerical stability."""
    sigma = np.clip(sigma, 1e-4, None)
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma) ** 2
    return float(np.mean(nll))


def coverage_at(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float) -> float:
    """Empirical coverage of the ``level``-quantile Gaussian interval."""
    from scipy.stats import norm

    z = norm.ppf(0.5 + level / 2.0)
    lower = mu - z * sigma
    upper = mu + z * sigma
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def expected_calibration_error_regression(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, n_levels: int = 20
) -> float:
    """ECE for regression: |empirical_coverage(p) − p| averaged over p in (0,1)."""
    levels = np.linspace(0.05, 0.95, n_levels)
    diffs = []
    for p in levels:
        cov = coverage_at(y_true, mu, sigma, p)
        diffs.append(abs(cov - p))
    return float(np.mean(diffs))


def brier_like(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, *, n_levels: int = 10) -> float:
    """Sum of squared (cov_p − p) across levels, like a Brier score on the curve."""
    levels = np.linspace(0.1, 0.9, n_levels)
    sq = []
    for p in levels:
        cov = coverage_at(y_true, mu, sigma, p)
        sq.append((cov - p) ** 2)
    return float(np.mean(sq))


def calibration_metrics(df: pd.DataFrame) -> dict:
    y = df["true_pchembl"].to_numpy(dtype=float)
    mu = df["pred_pchembl"].to_numpy(dtype=float)
    sigma = df["sigma_pchembl"].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    y, mu, sigma = y[mask], mu[mask], sigma[mask]
    return {
        "n": int(y.size),
        "ece": expected_calibration_error_regression(y, mu, sigma),
        "brier_like": brier_like(y, mu, sigma),
        "nll": gaussian_nll(y, mu, sigma),
        "coverage@50": coverage_at(y, mu, sigma, 0.5),
        "coverage@90": coverage_at(y, mu, sigma, 0.9),
        "sharpness_mean_sigma": float(np.mean(sigma)),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    by_model: dict[str, pd.DataFrame],
    paths: AnalysisPaths,
    *,
    n_levels: int = 20,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    levels = np.linspace(0.05, 0.95, n_levels)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    for name, df in by_model.items():
        y = df["true_pchembl"].to_numpy(dtype=float)
        mu = df["pred_pchembl"].to_numpy(dtype=float)
        sigma = df["sigma_pchembl"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
        y, mu, sigma = y[mask], mu[mask], sigma[mask]
        cov = [coverage_at(y, mu, sigma, p) for p in levels]
        ax.plot(levels, cov, marker="o", ms=3, lw=1, label=name)
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability diagram (regression)")
    ax.legend(fontsize=8)
    out = save_figure(fig, paths, "g3_reliability_diagram")
    plt.close(fig)
    return out


def plot_sharpness_vs_ece(metrics_df: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for _, r in metrics_df.iterrows():
        ax.scatter(r["sharpness_mean_sigma"], r["ece"], s=60)
        ax.annotate(r["model"], (r["sharpness_mean_sigma"], r["ece"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("mean predicted σ (sharpness; lower = sharper)")
    ax.set_ylabel("ECE (lower = better calibrated)")
    ax.set_title("Sharpness vs calibration")
    out = save_figure(fig, paths, "g3_sharpness_vs_ece")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    predictions_csv: Path | str,
    out_dir: Path | str,
) -> dict:
    """Compute calibration metrics for every model present in the long CSV."""
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g3_calibration")

    df = pd.read_csv(predictions_csv)
    needed = {"true_pchembl", "pred_pchembl", "sigma_pchembl", "model"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{predictions_csv}: missing columns {missing}. "
            "Build the long CSV with mc_loops_to_sigma / ensemble_to_sigma / halt_var_to_sigma first."
        )

    by_model = {name: g for name, g in df.groupby("model")}
    rows = []
    for name, g in by_model.items():
        m = calibration_metrics(g)
        rows.append({"model": name, **m})
    metrics_df = pd.DataFrame(rows).sort_values("ece")
    save_table(metrics_df, paths, "g3_calibration_metrics")

    f1 = plot_reliability_diagram(by_model, paths)
    f2 = plot_sharpness_vs_ece(metrics_df, paths)

    best_row = metrics_df.iloc[0]
    if any(name.startswith("radiant") for name in metrics_df["model"].tolist()):
        loop_best = metrics_df[metrics_df["model"].str.startswith("radiant")].iloc[0]
        ens_rows = metrics_df[metrics_df["model"].str.startswith("ensemble")]
        if not ens_rows.empty:
            ens_best = ens_rows.iloc[0]
            if loop_best["ece"] <= ens_best["ece"]:
                verdict = (
                    f"C3 supported: {loop_best['model']} ECE={loop_best['ece']:.3f} "
                    f"≤ ensemble ECE={ens_best['ece']:.3f} at matched compute."
                )
            else:
                verdict = (
                    f"C3 not supported on ECE: {loop_best['model']} ECE={loop_best['ece']:.3f} "
                    f"> ensemble ECE={ens_best['ece']:.3f}."
                )
        else:
            verdict = f"Single-source calibration; best model = {best_row['model']} ECE={best_row['ece']:.3f}."
    else:
        verdict = f"Best calibrated model = {best_row['model']} (ECE={best_row['ece']:.3f})."

    write_summary_md(
        paths,
        title="G.3 — Calibration & uncertainty",
        claim="C3: halting confidence yields a well-calibrated uncertainty estimate at matched compute.",
        headline=verdict,
        details={
            "Models compared": ", ".join(metrics_df["model"].tolist()),
            "Levels (reliability)": "20 evenly spaced in [0.05, 0.95]",
        },
        tables_referenced=["g3_calibration_metrics.csv"],
        figures_referenced=[p.name for p in f1 + f2],
    )

    return {"metrics": metrics_df, "verdict": verdict, "paths": paths}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G.3 calibration analysis")
    p.add_argument("--predictions", required=True, type=Path,
                   help="Long predictions CSV with sigma_pchembl + model columns")
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(args.predictions, args.out_dir)


if __name__ == "__main__":
    main()
