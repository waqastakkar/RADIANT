"""Reviewer-facing statistical tests from ``panel_results.csv``.

Compares a primary model against every available comparator over matched
target/split cells. Reports paired deltas, Wilcoxon signed-rank p-values,
Holm-corrected p-values, and win/tie/loss counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import holm_correction


def _direction(metric: str) -> int:
    return -1 if metric.lower() in {"mae", "rmse", "mse", "nll", "ece"} else 1


def run(
    panel_results: Path,
    out_dir: Path,
    *,
    primary: str = "radiant",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    metrics = metrics or ["test_mae", "test_rmse", "test_pearson", "test_spearman"]
    df = pd.read_csv(panel_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_cols = ["target_chembl_id", "split"]
    rows: list[dict] = []

    models = sorted(m for m in df["model"].dropna().unique() if m != primary)
    for metric in metrics:
        if metric not in df.columns:
            continue
        short_metric = metric.replace("test_", "")
        sign = _direction(short_metric)
        for model in models:
            a = df[df["model"] == primary][key_cols + [metric]].rename(columns={metric: "primary"})
            b = df[df["model"] == model][key_cols + [metric]].rename(columns={metric: "baseline"})
            merged = a.merge(b, on=key_cols, how="inner").dropna()
            if merged.empty:
                continue
            delta = sign * (merged["primary"].to_numpy(float) - merged["baseline"].to_numpy(float))
            wins = int((delta > 0).sum())
            ties = int((delta == 0).sum())
            losses = int((delta < 0).sum())
            try:
                p = float(wilcoxon(delta, alternative="greater", zero_method="zsplit").pvalue)
            except Exception:
                p = float("nan")
            rows.append({
                "metric": metric,
                "primary": primary,
                "baseline": model,
                "n_matched_cells": int(len(merged)),
                "mean_signed_improvement": float(np.mean(delta)),
                "median_signed_improvement": float(np.median(delta)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_fraction": wins / max(len(merged), 1),
                "wilcoxon_p_greater": p,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_p"] = holm_correction(out["wilcoxon_p_greater"].fillna(1.0).to_numpy())
        out["significant_holm_0_05"] = out["holm_p"] < 0.05
    out.to_csv(out_dir / "statistical_significance.csv", index=False)

    lines = ["# Statistical Significance\n"]
    if out.empty:
        lines.append("No matched model comparisons were available.\n")
    else:
        for _, r in out.sort_values(["metric", "baseline"]).iterrows():
            lines.append(
                f"- {r['metric']}: {primary} vs {r['baseline']} over {int(r['n_matched_cells'])} cells; "
                f"wins/ties/losses={int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}; "
                f"median signed improvement={r['median_signed_improvement']:.4g}; Holm p={r['holm_p']:.3g}."
            )
    (out_dir / "STATISTICAL_SIGNIFICANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel-results", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--primary", default="radiant")
    p.add_argument("--metrics", nargs="+", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse()
    df = run(args.panel_results, args.out_dir, primary=args.primary, metrics=args.metrics)
    print(f"wrote {args.out_dir / 'statistical_significance.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
