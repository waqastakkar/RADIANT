"""R-group SAR consistency analysis.

This analysis asks whether a model preserves potency ordering across
substituent changes on the same Bemis-Murcko scaffold. It is intentionally
prediction-file driven, so it can be run on the full benchmark panel without
loading checkpoints.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    discover_predictions,
    load_predictions,
    publication_style,
    save_figure,
    save_table,
    spearman_pearson,
    write_summary_md,
)
from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles
from radiant_qsar.pretrain.corpus import _murcko_scaffold_smiles


def _annotate_sar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scaffold_smiles"] = out["smiles"].map(_murcko_scaffold_smiles)
    out["rgroup_smiles"] = out["smiles"].map(murcko_rgroup_smiles)
    out = out[(out["scaffold_smiles"] != "") & (out["rgroup_smiles"] != "")]
    return out.reset_index(drop=True)


def _sample_pairs(group: pd.DataFrame, *, max_pairs: int, seed: int) -> list[tuple[int, int]]:
    pairs = list(itertools.combinations(group.index.to_list(), 2))
    if len(pairs) <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[int(i)] for i in take]


def _pair_rows(
    df: pd.DataFrame,
    *,
    max_pairs_per_scaffold: int,
    min_abs_true_delta: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for scaffold, group in df.groupby("scaffold_smiles", sort=False):
        if group["rgroup_smiles"].nunique() < 2 or len(group) < 2:
            continue
        for i, j in _sample_pairs(group, max_pairs=max_pairs_per_scaffold, seed=seed):
            a = df.loc[i]
            b = df.loc[j]
            true_delta = float(a["true_pchembl"] - b["true_pchembl"])
            pred_delta = float(a["pred_pchembl"] - b["pred_pchembl"])
            if abs(true_delta) < min_abs_true_delta:
                continue
            rows.append({
                "target": a.get("target_chembl_id", ""),
                "split": a.get("split_kind", ""),
                "scaffold_smiles": scaffold,
                "smiles_a": a["smiles"],
                "smiles_b": b["smiles"],
                "rgroup_a": a["rgroup_smiles"],
                "rgroup_b": b["rgroup_smiles"],
                "true_delta": true_delta,
                "pred_delta": pred_delta,
                "delta_abs_error": abs(pred_delta - true_delta),
                "sign_correct": float(np.sign(true_delta) == np.sign(pred_delta)),
            })
    return pd.DataFrame(rows)


def run(
    *,
    panel_root: Path | str,
    out_dir: Path | str,
    model: str = "radiant",
    split: str | None = "scaffold",
    min_abs_true_delta: float = 0.3,
    max_pairs_per_scaffold: int = 250,
    seed: int = 0,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_rgroup_sar")
    manifest = discover_predictions(panel_root)
    if manifest.empty:
        raise FileNotFoundError(f"no predictions.csv found under {panel_root}")
    manifest = manifest[manifest["model"] == model]
    if split:
        manifest = manifest[manifest["split"] == split]
    if manifest.empty:
        raise FileNotFoundError(f"no predictions for model={model!r}, split={split!r}")

    all_pairs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for _, item in manifest.iterrows():
        preds = _annotate_sar(load_predictions(item["path"]))
        pairs = _pair_rows(
            preds,
            max_pairs_per_scaffold=max_pairs_per_scaffold,
            min_abs_true_delta=min_abs_true_delta,
            seed=seed,
        )
        if pairs.empty:
            summary_rows.append({
                "model": item["model"],
                "target": item["target"],
                "split": item["split"],
                "n_scaffolds": int(preds["scaffold_smiles"].nunique()),
                "n_pairs": 0,
                "delta_mae": np.nan,
                "delta_pearson": np.nan,
                "delta_spearman": np.nan,
                "sign_accuracy": np.nan,
            })
            continue
        pairs.insert(0, "model", item["model"])
        pairs.insert(1, "panel_target", item["target"])
        pairs.insert(2, "panel_split", item["split"])
        stats = spearman_pearson(pairs["true_delta"], pairs["pred_delta"], n_bootstrap=200, seed=seed)
        summary_rows.append({
            "model": item["model"],
            "target": item["target"],
            "split": item["split"],
            "n_scaffolds": int(preds["scaffold_smiles"].nunique()),
            "n_pairs": int(len(pairs)),
            "delta_mae": float(pairs["delta_abs_error"].mean()),
            "delta_pearson": stats["pearson"],
            "delta_spearman": stats["spearman"],
            "sign_accuracy": float(pairs["sign_correct"].mean()),
        })
        all_pairs.append(pairs)

    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    save_table(pairs_df, paths, "rgroup_sar_pairs")
    save_table(summary, paths, "rgroup_sar_summary")

    if not pairs_df.empty:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.46, 3.0))
        ax.scatter(pairs_df["true_delta"], pairs_df["pred_delta"], s=12, alpha=0.6)
        lo = float(np.nanmin([pairs_df["true_delta"].min(), pairs_df["pred_delta"].min()]))
        hi = float(np.nanmax([pairs_df["true_delta"].max(), pairs_df["pred_delta"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.axhline(0, color="#666666", linewidth=0.6)
        ax.axvline(0, color="#666666", linewidth=0.6)
        ax.set_xlabel("Observed R-group delta pChEMBL")
        ax.set_ylabel("Predicted R-group delta pChEMBL")
        ax.set_title("R-group SAR ordering")
        save_figure(fig, paths, "rgroup_sar_delta")
        plt.close(fig)

    headline = "No eligible same-scaffold R-group pairs were found."
    if not summary.empty and summary["n_pairs"].sum() > 0:
        pooled_sign = float(np.average(summary["sign_accuracy"].dropna(), weights=summary.loc[summary["sign_accuracy"].notna(), "n_pairs"]))
        pooled_mae = float(np.average(summary["delta_mae"].dropna(), weights=summary.loc[summary["delta_mae"].notna(), "n_pairs"]))
        headline = f"Across {int(summary['n_pairs'].sum())} R-group pairs, sign accuracy={pooled_sign:.3f}, delta MAE={pooled_mae:.3f}."
    write_summary_md(
        paths,
        title="R-group SAR Consistency",
        claim="The model should preserve potency changes caused by R-group substitutions on a fixed scaffold.",
        headline=headline,
        details={
            "Model": model,
            "Split": split or "all",
            "Minimum observed delta": f"{min_abs_true_delta:.2f} pChEMBL",
        },
        tables_referenced=["rgroup_sar_pairs.csv", "rgroup_sar_summary.csv"],
        figures_referenced=["rgroup_sar_delta.png"] if not pairs_df.empty else [],
    )
    return {"paths": paths, "pairs": pairs_df, "summary": summary}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R-group SAR consistency analysis")
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="radiant")
    p.add_argument("--split", default="scaffold")
    p.add_argument("--min-abs-true-delta", type=float, default=0.3)
    p.add_argument("--max-pairs-per-scaffold", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
