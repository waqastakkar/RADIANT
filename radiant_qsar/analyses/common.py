"""Shared utilities for Phase G analyses.

Everything that is not specific to a single sub-claim lives here:

* :class:`AnalysisPaths` — typed container for the directory layout each
  analysis writes into.
* :func:`load_predictions` / :func:`discover_predictions` — uniform reader
  for the canonical ``predictions.csv`` schema written by
  :mod:`radiant_qsar.eval.predictions`.
* Statistical helpers (Spearman / Pearson with bootstrap CIs, paired
  bootstrap, Holm correction).
* Plot styling and save helpers for publication-ready figures.

All helpers are framework-agnostic (numpy + pandas + scipy + matplotlib)
so the analyses do not pull in PyTorch unless they need to invoke a
model directly.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


CANONICAL_PRED_COLUMNS: tuple[str, ...] = (
    "idx",
    "inchikey14",
    "target_chembl_id",
    "split_kind",
    "smiles",
    "true_pchembl",
    "pred_pchembl",
)


COMPLEXITY_DESCRIPTORS: tuple[str, ...] = (
    "MolWt",
    "NumRotatableBonds",
    "NumRings",
    "FractionCSP3",
    "BertzCT",
    "SAscore_proxy",
)


@dataclass
class AnalysisPaths:
    """Standard directory layout for Phase G outputs.

    A single analysis ``A`` writes to ``out_dir/A/{figures,tables,summary.md}``.
    """

    out_dir: Path
    name: str
    figure_formats: tuple[str, ...] = ("png", "svg")

    @property
    def root(self) -> Path:
        p = self.out_dir / self.name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures(self) -> Path:
        p = self.root / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def tables(self) -> Path:
        p = self.root / "tables"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def summary_md(self) -> Path:
        return self.root / "summary.md"


# ---------------------------------------------------------------------------
# Nature Communications colour palette & sizing constants
# ---------------------------------------------------------------------------

# NPG (Nature Publishing Group) 10-colour palette — identical to
# ggsci::scale_colour_npg / RColorBrewer "Set1" variant used by Nature journals.
NATURE_PALETTE: tuple[str, ...] = (
    "#E64B35",  # vermilion red
    "#4DBBD5",  # sky blue
    "#00A087",  # teal green
    "#3C5488",  # navy blue
    "#F39B7F",  # salmon orange
    "#8491B4",  # slate purple
    "#91D1C2",  # mint
    "#DC0000",  # crimson
    "#7E6148",  # warm brown
    "#B09C85",  # tan
)

# Nature Communications column widths in inches (88 mm single, 183 mm double).
NC_SINGLE_COL: float = 3.46   # 88 mm
NC_DOUBLE_COL: float = 7.20   # 183 mm
NC_FULL_PAGE_H: float = 9.45  # 240 mm max height


def nature_palette(n: int | None = None) -> list[str]:
    """Return the first *n* NPG colours (cycles if n > 10)."""
    if n is None:
        return list(NATURE_PALETTE)
    return [NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(n)]


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

def publication_style() -> None:
    """Apply Nature Communications figure style globally.

    Fonts: Times New Roman bold (falls back to platform 'serif' if TNR is
    not installed).  Colour cycle: NPG 10-colour palette.  Sizes follow
    the NC author guidelines (88 mm single column, 300 dpi, ≥ 7 pt text).
    Idempotent — safe to call multiple times.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Resolve Times New Roman or best available Times-compatible serif.
    # STIXGeneral is bundled with matplotlib and is the publication-grade
    # Times substitute (it's what matplotlib uses for serif math, and it
    # has explicit bold weight=700). Most Linux scientific Python installs
    # won't have actual Times New Roman or Liberation Serif unless the
    # user installed msttcorefonts / fonts-liberation system packages.
    _TNR_CANDIDATES = [
        "Times New Roman",      # Microsoft core fonts (if installed)
        "Times",                # PostScript Times (if installed)
        "Liberation Serif",     # fonts-liberation system package
        "Nimbus Roman",         # gsfonts
        "STIXGeneral",          # bundled with matplotlib — RECOMMENDED fallback
        "STIX Two Text",
        "DejaVu Serif",         # last resort -- not Times-metric
    ]
    _resolved_font = "serif"
    for _name in _TNR_CANDIDATES:
        try:
            if font_manager.findfont(_name, fallback_to_default=False):
                _resolved_font = _name
                break
        except Exception:
            continue

    mpl.rcParams.update({
        # --- Font ---
        "font.family":           _resolved_font,
        "font.weight":           "bold",
        "font.size":             8,
        "axes.titlesize":        9,
        "axes.titleweight":      "bold",
        "axes.labelsize":        8,
        "axes.labelweight":      "bold",
        "xtick.labelsize":       7,
        "ytick.labelsize":       7,
        "legend.fontsize":       7,
        "legend.title_fontsize": 7,

        # --- Colour cycle ---
        "axes.prop_cycle": mpl.cycler("color", list(NATURE_PALETTE)),

        # --- Lines & markers ---
        "lines.linewidth":       1.2,
        "lines.markersize":      4,
        "patch.linewidth":       0.6,

        # --- Axes frame ---
        "axes.linewidth":        0.8,
        "axes.edgecolor":        "#000000",
        "axes.facecolor":        "#FFFFFF",
        "axes.spines.top":       False,
        "axes.spines.right":     False,

        # --- Grid: subtle, not obtrusive ---
        "axes.grid":             True,
        "grid.color":            "#E5E5E5",
        "grid.linewidth":        0.5,
        "grid.linestyle":        "-",

        # --- Ticks ---
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        "xtick.major.width":     0.8,
        "ytick.major.width":     0.8,
        "xtick.major.size":      3,
        "ytick.major.size":      3,
        "xtick.minor.visible":   False,
        "ytick.minor.visible":   False,

        # --- Legend ---
        "legend.frameon":        False,
        "legend.handlelength":   1.2,
        "legend.handletextpad":  0.4,
        "legend.borderpad":      0.3,

        # --- Save ---
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "savefig.facecolor":     "white",
        "savefig.transparent":   False,

        # Embed fonts in PDF/PS (required by Nature).
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
    })
    _ = plt  # silence unused-import


def save_figure(fig, paths: AnalysisPaths, stem: str) -> list[Path]:
    """Save figure as PNG (300 dpi) and SVG at ``paths.figures/stem.*``."""
    out = []
    for fmt in paths.figure_formats:
        p = paths.figures / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)
        out.append(p)
    return out


def save_table(df: pd.DataFrame, paths: AnalysisPaths, stem: str) -> tuple[Path, Path]:
    """Save table as both CSV and TSV under ``paths.tables/stem.*``."""
    csv_p = paths.tables / f"{stem}.csv"
    tsv_p = paths.tables / f"{stem}.tsv"
    df.to_csv(csv_p, index=False)
    df.to_csv(tsv_p, index=False, sep="\t")
    return csv_p, tsv_p


# ---------------------------------------------------------------------------
# Predictions IO
# ---------------------------------------------------------------------------

def discover_predictions(panel_root: Path | str) -> pd.DataFrame:
    """Walk ``panel_root`` and return a manifest of all predictions.csv files.

    Expected layout (the panel sweep produces this)::

        panel_root/<model>/<target>/<split>/predictions.csv

    Models with non-standard layouts (e.g., a single ``predictions.csv``
    directly under the model dir) are also picked up — every leaf
    ``predictions.csv`` is discovered.

    Returns columns: ``model, target, split, path``.
    """
    panel_root = Path(panel_root)
    rows: list[dict] = []
    for p in panel_root.rglob("predictions.csv"):
        rel = p.relative_to(panel_root).parts
        if len(rel) < 2:
            continue
        model = rel[0]
        target = rel[1] if len(rel) >= 3 else "default"
        split = rel[2] if len(rel) >= 4 else "default"
        if len(rel) >= 4:
            target = rel[-3]
            split = rel[-2]
        rows.append({"model": model, "target": target, "split": split, "path": str(p)})
    cols = ["model", "target", "split", "path"]
    if not rows:
        # No predictions.csv anywhere under panel_root. Return an empty frame
        # with the expected columns so downstream sort/group ops degrade
        # gracefully instead of raising a cryptic ``KeyError: 'model'``.
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["model", "target", "split"]).reset_index(drop=True)


def load_predictions(path: Path | str) -> pd.DataFrame:
    """Load one canonical predictions.csv. Validates leading columns exist."""
    df = pd.read_csv(path)
    missing = [c for c in CANONICAL_PRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: predictions.csv missing canonical columns {missing}; got {list(df.columns)}"
        )
    df["true_pchembl"] = df["true_pchembl"].astype(float)
    df["pred_pchembl"] = df["pred_pchembl"].astype(float)
    return df


def join_descriptors(
    preds: pd.DataFrame,
    descriptors: pd.DataFrame,
    *,
    on: str = "inchikey14",
) -> pd.DataFrame:
    """Left-join predictions against a per-compound descriptors table."""
    if on not in descriptors.columns:
        raise ValueError(f"join_descriptors: '{on}' not in descriptors columns {list(descriptors.columns)}")
    if on not in preds.columns:
        raise ValueError(f"join_descriptors: '{on}' not in predictions columns {list(preds.columns)}")
    return preds.merge(descriptors, on=on, how="left", suffixes=("", "_desc"))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def spearman_pearson(
    x: Sequence[float], y: Sequence[float], *, n_bootstrap: int = 1000, seed: int = 0
) -> dict:
    """Spearman ρ and Pearson r with bootstrap 95% CIs.

    Returns ``{spearman, pearson, spearman_ci, pearson_ci, n}``.
    NaNs in either vector drop the row.
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    if n < 4:
        return {"spearman": float("nan"), "pearson": float("nan"),
                "spearman_ci": (float("nan"), float("nan")),
                "pearson_ci": (float("nan"), float("nan")),
                "n": n}

    sp = stats.spearmanr(x, y).correlation
    pe = stats.pearsonr(x, y)[0]

    rng = np.random.default_rng(seed)
    sps = np.empty(n_bootstrap)
    pes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sps[b] = stats.spearmanr(x[idx], y[idx]).correlation
        pes[b] = stats.pearsonr(x[idx], y[idx])[0]
    sp_ci = (float(np.nanpercentile(sps, 2.5)), float(np.nanpercentile(sps, 97.5)))
    pe_ci = (float(np.nanpercentile(pes, 2.5)), float(np.nanpercentile(pes, 97.5)))
    return {
        "spearman": float(sp),
        "pearson": float(pe),
        "spearman_ci": sp_ci,
        "pearson_ci": pe_ci,
        "n": n,
    }


def bootstrap_paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    statistic: str = "mean",
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap on ``a - b``.

    Used for win/CI analysis where ``a`` and ``b`` are per-molecule
    metric residuals (e.g., absolute error of two models on the same
    test set). Returns mean diff, 95% CI, and a two-sided p-value from
    the proportion of resamples crossing zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"bootstrap_paired_diff: shape mismatch {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 2:
        return {"mean_diff": float("nan"), "ci": (float("nan"), float("nan")),
                "p_value": float("nan"), "n": n}

    diff = a - b
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(n_bootstrap)
    if statistic == "mean":
        agg = np.mean
    elif statistic == "median":
        agg = np.median
    else:
        raise ValueError(f"unknown statistic: {statistic}")

    obs = float(agg(diff))
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        stats_boot[i] = agg(diff[idx])
    lo, hi = float(np.percentile(stats_boot, 2.5)), float(np.percentile(stats_boot, 97.5))

    # Two-sided p: 2 * min(P(stat >= 0), P(stat <= 0)) under the null
    # of zero difference, approximated by recentering the bootstrap
    # distribution.
    centered = stats_boot - obs
    p = 2.0 * min(
        float(np.mean(centered >= abs(obs))),
        float(np.mean(centered <= -abs(obs))),
    )
    p = max(min(p, 1.0), 1.0 / n_bootstrap)

    return {"mean_diff": obs, "ci": (lo, hi), "p_value": p, "n": n}


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down correction. Returns adjusted p in input order."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


# ---------------------------------------------------------------------------
# Complexity binning (shared by G1/G2/G4)
# ---------------------------------------------------------------------------

def complexity_bins(
    values: Sequence[float],
    *,
    n_bins: int = 5,
    labels: Sequence[str] | None = None,
) -> pd.Categorical:
    """Quantile-bin a complexity score; returns a pandas Categorical."""
    if labels is None:
        labels = [f"Q{i+1}" for i in range(n_bins)]
    series = pd.Series(values, dtype=float)
    cats = pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")
    return cats


# ---------------------------------------------------------------------------
# Markdown summary helpers
# ---------------------------------------------------------------------------

def write_summary_md(
    paths: AnalysisPaths,
    *,
    title: str,
    claim: str,
    headline: str,
    details: Mapping[str, str],
    tables_referenced: Sequence[str] = (),
    figures_referenced: Sequence[str] = (),
) -> Path:
    """Write a per-analysis Markdown blurb suitable for paper assembly.

    The format is intentionally rigid so :func:`assemble_phase_g_report`
    in :mod:`run_phase_g` can concatenate the individual files into one
    section.
    """
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Sub-claim:** {claim}")
    lines.append("")
    lines.append(f"**Headline result:** {headline}")
    lines.append("")
    lines.append("## Details")
    lines.append("")
    for k, v in details.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    if figures_referenced:
        lines.append("## Figures")
        for f in figures_referenced:
            lines.append(f"- `{f}`")
        lines.append("")
    if tables_referenced:
        lines.append("## Tables")
        for t in tables_referenced:
            lines.append(f"- `{t}`")
        lines.append("")

    paths.summary_md.write_text("\n".join(lines), encoding="utf-8")
    return paths.summary_md


# ---------------------------------------------------------------------------
# Regression helpers used by G1 (descriptor -> depth) and G4 (CV)
# ---------------------------------------------------------------------------

def cv_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Sequence[str],
    n_splits: int = 5,
    seed: int = 0,
    model: str = "ridge",
) -> dict:
    """Cross-validated regression with R² and feature importance.

    For ``ridge`` we return absolute standardized coefficients; for
    ``rf`` we return mean-decrease-impurity importance.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] < n_splits * 2:
        return {
            "r2_mean": float("nan"), "r2_std": float("nan"),
            "fold_r2": [], "feature_importance": dict(zip(feature_names, [float("nan")] * len(feature_names))),
            "n": int(X.shape[0]),
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_r2: list[float] = []
    importances = np.zeros(X.shape[1], dtype=float)
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        if model == "ridge":
            est = Ridge(alpha=1.0, random_state=seed)
        elif model == "rf":
            est = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        else:
            raise ValueError(f"unknown model {model}")
        est.fit(Xtr, y[tr])
        pred = est.predict(Xte)
        fold_r2.append(float(r2_score(y[te], pred)))
        if model == "ridge":
            importances += np.abs(est.coef_)
        else:
            importances += est.feature_importances_

    importances /= n_splits
    imp_dict = {name: float(v) for name, v in zip(feature_names, importances)}
    return {
        "r2_mean": float(np.mean(fold_r2)),
        "r2_std": float(np.std(fold_r2)),
        "fold_r2": fold_r2,
        "feature_importance": imp_dict,
        "n": int(X.shape[0]),
    }


def absolute_error(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df["pred_pchembl"].to_numpy() - df["true_pchembl"].to_numpy())


def squared_error(df: pd.DataFrame) -> np.ndarray:
    return (df["pred_pchembl"].to_numpy() - df["true_pchembl"].to_numpy()) ** 2


def regression_metrics(df: pd.DataFrame) -> dict:
    """MAE / RMSE / R² / Pearson / Spearman from a predictions frame."""
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = df["true_pchembl"].to_numpy()
    y_pred = df["pred_pchembl"].to_numpy()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size < 2:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"), "n": int(y_true.size)}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman": float(stats.spearmanr(y_true, y_pred).correlation),
        "n": int(y_true.size),
    }
