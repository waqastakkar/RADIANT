"""R-group chemistry ablation comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import AnalysisPaths, publication_style, save_figure, save_table, write_summary_md


METRIC_DIRECTIONS = {
    "mae": -1.0,
    "rmse": -1.0,
    "r2": 1.0,
    "pearson": 1.0,
    "spearman": 1.0,
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for src, dst in [("target_chembl_id", "target"), ("split_kind", "split"), ("model_name", "model")]:
        if src in out.columns and dst not in out.columns:
            rename[src] = dst
    out = out.rename(columns=rename)
    required = {"model", "target", "split"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"panel_results missing columns: {sorted(missing)}")
    return out


def run(
    *,
    panel_results: Path | str,
    out_dir: Path | str,
    primary: str = "radiant",
    ablations: tuple[str, ...] = ("radiant_no_stage1_rgroup", "radiant_no_stage2_rgroup", "radiant_no_rgroup"),
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_rgroup_ablation")
    df = _normalise_columns(pd.read_csv(panel_results))
    metrics = [m for m in METRIC_DIRECTIONS if m in df.columns]
    if not metrics:
        raise ValueError("panel_results must contain at least one metric column: mae/rmse/r2/pearson/spearman")

    rows = []
    base = df[df["model"] == primary]
    for ablation in ablations:
        other = df[df["model"] == ablation]
        merged = base.merge(other, on=["target", "split"], suffixes=("_primary", "_ablation"))
        for metric in metrics:
            direction = METRIC_DIRECTIONS[metric]
            for _, r in merged.iterrows():
                raw_delta = float(r[f"{metric}_primary"] - r[f"{metric}_ablation"])
                rows.append({
                    "primary": primary,
                    "ablation": ablation,
                    "target": r["target"],
                    "split": r["split"],
                    "metric": metric,
                    "primary_value": float(r[f"{metric}_primary"]),
                    "ablation_value": float(r[f"{metric}_ablation"]),
                    "raw_delta": raw_delta,
                    "improvement_vs_ablation": raw_delta * direction,
                })
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        comparison = pd.DataFrame(columns=[
            "primary", "ablation", "target", "split", "metric",
            "primary_value", "ablation_value", "raw_delta", "improvement_vs_ablation",
        ])
    summary = (
        comparison.groupby(["ablation", "metric"], as_index=False)
        .agg(
            n_cells=("improvement_vs_ablation", "size"),
            mean_improvement=("improvement_vs_ablation", "mean"),
            median_improvement=("improvement_vs_ablation", "median"),
            win_rate=("improvement_vs_ablation", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        if not comparison.empty else pd.DataFrame(columns=["ablation", "metric", "n_cells", "mean_improvement", "median_improvement", "win_rate"])
    )
    save_table(comparison, paths, "rgroup_ablation_cells")
    save_table(summary, paths, "rgroup_ablation_summary")

    if not summary.empty:
        import matplotlib.pyplot as plt

        plot = summary[summary["metric"].isin(["mae", "pearson", "spearman"])].copy()
        if not plot.empty:
            labels = (plot["ablation"] + "\n" + plot["metric"]).to_list()
            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            ax.bar(np.arange(len(plot)), plot["mean_improvement"].to_numpy())
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(np.arange(len(plot)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Improvement vs ablation")
            ax.set_title("R-group chemistry ablation")
            save_figure(fig, paths, "rgroup_ablation_improvement")
            plt.close(fig)

    headline = "No matching R-group ablation cells were found."
    if not summary.empty:
        best = summary.sort_values("mean_improvement", ascending=False).iloc[0]
        headline = (
            f"Largest mean improvement is vs {best['ablation']} on {best['metric']}: "
            f"{best['mean_improvement']:.4f}."
        )
    write_summary_md(
        paths,
        title="R-group Chemistry Ablation",
        claim="Removing Stage-1 or Stage-2 R-group chemistry should reduce benchmark performance if the chemistry-aware objectives matter.",
        headline=headline,
        details={"Primary": primary, "Ablations": ", ".join(ablations)},
        tables_referenced=["rgroup_ablation_cells.csv", "rgroup_ablation_summary.csv"],
        figures_referenced=["rgroup_ablation_improvement.png"] if not summary.empty else [],
    )
    return {"paths": paths, "comparison": comparison, "summary": summary}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R-group chemistry ablation comparison")
    p.add_argument("--panel-results", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--primary", default="radiant")
    p.add_argument("--ablations", nargs="+", default=["radiant_no_stage1_rgroup", "radiant_no_stage2_rgroup", "radiant_no_rgroup"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        panel_results=args.panel_results,
        out_dir=args.out_dir,
        primary=args.primary,
        ablations=tuple(args.ablations),
    )


if __name__ == "__main__":
    main()
