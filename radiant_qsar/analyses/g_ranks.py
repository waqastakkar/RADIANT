"""Phase G — Average rank + critical-difference diagram across models.

For each metric in {MAE (lower is better), R2, Pearson, Spearman (higher is
better)} we rank the models within every (target, split) cell and average
across cells. We also compute:

* per-split ranks (one column per split)
* per-target-class ranks (joined via panel_results.csv or panel.json)
* a Demsar critical-difference (CD) diagram (Demsar 2006) for the omnibus
  metric (val MAE), showing which model groups are statistically
  indistinguishable.

Critical-difference: ``CD = q_alpha * sqrt(k(k+1) / (6N))`` where ``k`` is
the number of models and ``N`` is the number of (target, split) cells.
``q_alpha`` is the Studentized-range upper quantile divided by sqrt(2).

Source of metrics: ``runs/phase_g/g0_validation_metrics/tables/g0_cell_metrics.csv``
(written by G.0 and covering all 5 models x 100 cells).
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
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


# (display_name, column_name_in_g0_cell_metrics, lower_is_better)
METRICS = (
    ("MAE", "mae", True),
    ("R2", "r2", False),
    ("Pearson", "pearson", False),
    ("Spearman", "spearman", False),
)


def _rank_within_cells(df: pd.DataFrame, *, metric_col: str,
                       lower_better: bool) -> pd.DataFrame:
    """Add a `rank` column ranking models within each (target, split) cell."""
    out = df.copy()
    method = "min"
    if lower_better:
        out["rank"] = out.groupby(["target", "split"])[metric_col].rank(method=method, ascending=True)
    else:
        out["rank"] = out.groupby(["target", "split"])[metric_col].rank(method=method, ascending=False)
    return out


def _avg_rank_by_model(ranked: pd.DataFrame) -> pd.Series:
    return ranked.groupby("model")["rank"].mean().sort_values()


def _per_split_rank(ranked: pd.DataFrame) -> pd.DataFrame:
    tbl = ranked.groupby(["model", "split"])["rank"].mean().unstack("split")
    tbl["overall"] = ranked.groupby("model")["rank"].mean()
    return tbl.sort_values("overall")


def _per_class_rank(ranked: pd.DataFrame, class_map: dict[str, str] | None) -> pd.DataFrame:
    if not class_map:
        return pd.DataFrame()
    df = ranked.copy()
    df["target_class"] = df["target"].map(class_map).fillna("unknown")
    return df.groupby(["model", "target_class"])["rank"].mean().unstack("target_class")


def _load_class_map(panel_root: Path) -> dict[str, str] | None:
    """Try to recover target -> target_class from panel.json (sweep source) or
    panel_results.csv. Returns None on failure."""
    candidates = [
        Path("data/processed/v1/panel.json"),
        panel_root / "panel.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                entries = data.get("entries", data if isinstance(data, list) else [])
                return {e["target_chembl_id"]: e.get("target_class", "unknown")
                        for e in entries if "target_chembl_id" in e}
            except Exception:
                continue
    pr = panel_root / "panel_results.csv"
    if pr.exists():
        try:
            d = pd.read_csv(pr, usecols=["target_chembl_id", "target_class"]).drop_duplicates()
            return dict(zip(d["target_chembl_id"], d["target_class"]))
        except Exception:
            pass
    return None


def _critical_difference(n_cells: int, k_models: int, alpha: float = 0.05) -> float:
    """Demsar 2006 critical difference for Nemenyi at significance ``alpha``.

    CD = q_alpha * sqrt(k(k+1) / (6N))   with q_alpha = q_studentized / sqrt(2).
    Uses scipy.stats.studentized_range when available; falls back to a
    hardcoded q_alpha=0.05 lookup for k up to 10.
    """
    try:
        from scipy.stats import studentized_range
        # quantile at 1-alpha for q (Studentized range), df = inf
        q = studentized_range.ppf(1 - alpha, k_models, np.inf)
        q_alpha = q / np.sqrt(2.0)
    except Exception:
        # Demsar 2006 Table 5: q_alpha at alpha=0.05 for k models
        LOOKUP = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                  7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q_alpha = LOOKUP.get(k_models, 2.728)
    return float(q_alpha * np.sqrt(k_models * (k_models + 1) / (6.0 * max(n_cells, 1))))


def _plot_cd_diagram(avg_rank: pd.Series, cd: float, paths: AnalysisPaths,
                     stem: str, title: str) -> None:
    """Demsar critical-difference diagram.

    Horizontal axis = rank; each model is a tick on the axis. Models within
    CD of one another are connected by a thick bar above the axis.
    """
    import matplotlib.pyplot as plt

    models = avg_rank.index.tolist()
    ranks = avg_rank.values
    k = len(models)
    if k == 0:
        return

    lo = max(1.0, float(ranks.min()) - 0.3)
    hi = float(ranks.max()) + 0.3
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 0.35 * k + 1.3))

    # Number line
    ax.hlines(0, lo, hi, color="black", lw=1.0)
    for x in np.arange(np.ceil(lo), np.floor(hi) + 0.5, 1.0):
        ax.vlines(x, -0.02, 0.02, color="black", lw=0.8)
        ax.text(x, 0.06, f"{int(x)}", ha="center", va="bottom", fontsize=7)

    # Place model labels on left or right of the axis
    sorted_idx = np.argsort(ranks)
    half = k // 2
    colors = nature_palette(k)
    yspan = -0.05
    for rank_pos, idx in enumerate(sorted_idx):
        m = models[idx]
        r = ranks[idx]
        side = "left" if rank_pos < half else "right"
        x_label = lo - (hi - lo) * 0.08 if side == "left" else hi + (hi - lo) * 0.08
        # Connector
        ax.plot([r, r], [0, yspan * (1 + (rank_pos % half) * 0.5)],
                color=colors[idx], lw=0.8)
        ax.plot([r, x_label], [yspan * (1 + (rank_pos % half) * 0.5)] * 2,
                color=colors[idx], lw=0.8)
        ax.text(x_label, yspan * (1 + (rank_pos % half) * 0.5),
                f"{m}\n({r:.2f})",
                ha="right" if side == "left" else "left",
                va="center", fontsize=8, fontweight="bold", color=colors[idx])

    # CD groups: connect models within CD of one another by a thick bar above
    order = np.argsort(ranks)
    sorted_models = [models[i] for i in order]
    sorted_ranks = ranks[order]
    y_bar = 0.18
    bar_step = 0.06
    groups: list[tuple[int, int]] = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and sorted_ranks[j + 1] - sorted_ranks[i] <= cd:
            j += 1
        if j > i:
            groups.append((i, j))
            i = j + 1
        else:
            i += 1
    for gi, (a, b) in enumerate(groups):
        y = y_bar + gi * bar_step
        ax.hlines(y, sorted_ranks[a], sorted_ranks[b], color="#cc3333", lw=2.5)

    ax.set_xlim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.18)
    ax.set_ylim(yspan * (1 + (half + 0.5) * 0.5) - 0.05, 0.18 + bar_step * (len(groups) + 1))
    ax.axis("off")
    ax.set_title(f"{title}\nCD (alpha=0.05) = {cd:.3f}", fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def _plot_split_heatmap(per_split: pd.DataFrame, paths: AnalysisPaths,
                        stem: str, title: str) -> None:
    import matplotlib.pyplot as plt

    if per_split.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.3, 0.45 * len(per_split) + 1))
    data = per_split.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(per_split.shape[1]))
    ax.set_xticklabels(per_split.columns, rotation=25, ha="right", fontsize=7)
    ax.set_yticks(range(per_split.shape[0]))
    ax.set_yticklabels(per_split.index, fontsize=8, fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v - data.mean()) < 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.7, label="avg rank")
    ax.set_title(title, fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, stem)
    plt.close(fig)


def run(*, panel_root: Path | str, out_dir: Path | str,
        g0_cell_metrics: Path | str | None = None) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_ranks")

    if g0_cell_metrics is None:
        g0_cell_metrics = out_dir / "g0_validation_metrics" / "tables" / "g0_cell_metrics.csv"
    g0_cell_metrics = Path(g0_cell_metrics)
    if not g0_cell_metrics.exists():
        raise FileNotFoundError(
            f"g0_cell_metrics.csv not found at {g0_cell_metrics}; "
            f"run G.0 validation metrics first."
        )
    df = pd.read_csv(g0_cell_metrics)

    class_map = _load_class_map(panel_root)
    if class_map is None:
        logger.warning("target_class map not found; per-class ranks will be skipped")

    summary_overall: dict[str, pd.Series] = {}
    summary_per_split: dict[str, pd.DataFrame] = {}
    cds: dict[str, float] = {}
    long_rows: list[dict] = []

    for name, col, lower in METRICS:
        if col not in df.columns:
            logger.warning("metric column %r missing; skipping", col)
            continue
        ranked = _rank_within_cells(df, metric_col=col, lower_better=lower)
        overall = _avg_rank_by_model(ranked)
        per_split = _per_split_rank(ranked)
        per_class = _per_class_rank(ranked, class_map) if class_map else pd.DataFrame()

        save_table(per_split.reset_index().rename(columns={"index": "model"}),
                   paths, f"ranks_per_split_{name}")
        if not per_class.empty:
            save_table(per_class.reset_index().rename(columns={"index": "model"}),
                       paths, f"ranks_per_target_class_{name}")
        summary_overall[name] = overall
        summary_per_split[name] = per_split
        n_cells = ranked.groupby(["target", "split"]).ngroups
        cd = _critical_difference(n_cells, k_models=overall.shape[0])
        cds[name] = cd

        _plot_cd_diagram(overall, cd, paths,
                         stem=f"g_ranks_cd_{name.lower()}",
                         title=f"Critical-difference diagram ({name})")
        _plot_split_heatmap(per_split, paths,
                            stem=f"g_ranks_split_heatmap_{name.lower()}",
                            title=f"Average rank ({name}) by split  -- lower is better")

        for m, r in overall.items():
            long_rows.append({"metric": name, "model": m, "avg_rank": float(r),
                              "n_cells": int(n_cells), "cd_alpha_05": float(cd)})

    overall_df = pd.DataFrame(long_rows).sort_values(["metric", "avg_rank"])
    save_table(overall_df, paths, "ranks_overall")

    # Headline: best model on MAE
    headline = "no metrics computed"
    if "MAE" in summary_overall:
        winner = summary_overall["MAE"].index[0]
        winner_rank = summary_overall["MAE"].iloc[0]
        headline = (f"Best avg rank on MAE: {winner} ({winner_rank:.2f}). "
                    f"CD (alpha=0.05) = {cds.get('MAE', float('nan')):.3f}; "
                    f"any two models whose mean ranks differ by less than CD "
                    f"are statistically indistinguishable.")

    write_summary_md(
        paths,
        title="Average rank + critical-difference diagram",
        claim=("Models are ranked within every (target, split) cell on MAE / R2 / "
               "Pearson / Spearman and averaged across cells. The Demsar CD "
               "diagram clusters models that are not significantly separated."),
        headline=headline,
        details={
            "Cells per metric": str(df.groupby(["target", "split"]).ngroups),
            "Models compared": ", ".join(sorted(df["model"].unique())),
            "Metrics evaluated": ", ".join(name for name, *_ in METRICS),
        },
        tables_referenced=(
            ["ranks_overall.csv"]
            + [f"ranks_per_split_{n}.csv" for n, *_ in METRICS]
            + ([f"ranks_per_target_class_{n}.csv" for n, *_ in METRICS] if class_map else [])
        ),
        figures_referenced=(
            [f"g_ranks_cd_{n.lower()}.png" for n, *_ in METRICS]
            + [f"g_ranks_split_heatmap_{n.lower()}.png" for n, *_ in METRICS]
        ),
    )
    return {"paths": paths, "overall": overall_df, "cds": cds}


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
