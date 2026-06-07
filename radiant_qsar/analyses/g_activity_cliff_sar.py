"""Activity-cliff error analysis for benchmark predictions."""

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


def _fingerprints(smiles: list[str]):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [gen.GetFingerprint(m) if m is not None else None for m in mols]
    return fps


def _cliff_pairs(
    df: pd.DataFrame,
    *,
    tanimoto_threshold: float,
    activity_delta_threshold: float,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    from rdkit import DataStructs

    fps = _fingerprints(df["smiles"].astype(str).tolist())
    rows: list[dict] = []
    for i, j in itertools.combinations(range(len(df)), 2):
        if fps[i] is None or fps[j] is None:
            continue
        true_delta = float(df.iloc[i]["true_pchembl"] - df.iloc[j]["true_pchembl"])
        if abs(true_delta) < activity_delta_threshold:
            continue
        sim = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        if sim < tanimoto_threshold:
            continue
        pred_delta = float(df.iloc[i]["pred_pchembl"] - df.iloc[j]["pred_pchembl"])
        rows.append({
            "target": df.iloc[i].get("target_chembl_id", ""),
            "split": df.iloc[i].get("split_kind", ""),
            "smiles_a": df.iloc[i]["smiles"],
            "smiles_b": df.iloc[j]["smiles"],
            "tanimoto": sim,
            "true_delta": true_delta,
            "pred_delta": pred_delta,
            "delta_abs_error": abs(pred_delta - true_delta),
            "sign_correct": float(np.sign(true_delta) == np.sign(pred_delta)),
        })
    if len(rows) <= max_pairs:
        return pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(rows), size=max_pairs, replace=False)
    return pd.DataFrame([rows[int(i)] for i in keep])


def run(
    *,
    panel_root: Path | str,
    out_dir: Path | str,
    model: str | None = "radiant",
    split: str | None = "activity_cliff",
    tanimoto_threshold: float = 0.55,
    activity_delta_threshold: float = 1.0,
    max_pairs_per_cell: int = 2000,
    seed: int = 0,
) -> dict:
    """Activity-cliff SAR error analysis.

    Pass ``model=None`` to run the analysis across all models in the panel
    and emit a per-model comparison table + bar chart of Δ-MAE, sign
    accuracy, and pairwise-rank Spearman. This is the manuscript-grade
    figure for cliff handling -- it answers "which model captures SAR
    cliffs the best?"
    """
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_activity_cliff_sar")
    manifest = discover_predictions(panel_root)
    if manifest.empty:
        raise FileNotFoundError(f"no predictions.csv found under {panel_root}")
    if model is not None:
        manifest = manifest[manifest["model"] == model]
    if split:
        manifest = manifest[manifest["split"] == split]
    if manifest.empty:
        raise FileNotFoundError(f"no predictions for model={model!r}, split={split!r}")

    all_pairs: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for _, item in manifest.iterrows():
        preds = load_predictions(item["path"])
        pairs = _cliff_pairs(
            preds,
            tanimoto_threshold=tanimoto_threshold,
            activity_delta_threshold=activity_delta_threshold,
            max_pairs=max_pairs_per_cell,
            seed=seed,
        )
        if pairs.empty:
            summary_rows.append({
                "model": item["model"], "target": item["target"], "split": item["split"],
                "n_pairs": 0, "delta_mae": np.nan, "delta_pearson": np.nan,
                "delta_spearman": np.nan, "sign_accuracy": np.nan,
            })
            continue
        pairs.insert(0, "model", item["model"])
        pairs.insert(1, "panel_target", item["target"])
        pairs.insert(2, "panel_split", item["split"])
        stats = spearman_pearson(pairs["true_delta"], pairs["pred_delta"], n_bootstrap=200, seed=seed)
        summary_rows.append({
            "model": item["model"], "target": item["target"], "split": item["split"],
            "n_pairs": int(len(pairs)),
            "delta_mae": float(pairs["delta_abs_error"].mean()),
            "delta_pearson": stats["pearson"],
            "delta_spearman": stats["spearman"],
            "sign_accuracy": float(pairs["sign_correct"].mean()),
        })
        all_pairs.append(pairs)

    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    save_table(pairs_df, paths, "activity_cliff_pairs")
    save_table(summary, paths, "activity_cliff_summary")

    if not pairs_df.empty:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.46, 3.0))
        sc = ax.scatter(
            pairs_df["true_delta"],
            pairs_df["pred_delta"],
            c=pairs_df["tanimoto"],
            s=12,
            alpha=0.7,
            cmap="viridis",
        )
        lo = float(np.nanmin([pairs_df["true_delta"].min(), pairs_df["pred_delta"].min()]))
        hi = float(np.nanmax([pairs_df["true_delta"].max(), pairs_df["pred_delta"].max()]))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
        ax.set_xlabel("Observed cliff delta pChEMBL")
        ax.set_ylabel("Predicted cliff delta pChEMBL")
        ax.set_title("Activity-cliff SAR")
        fig.colorbar(sc, ax=ax, label="Tanimoto")
        save_figure(fig, paths, "activity_cliff_delta")
        plt.close(fig)

    headline = "No eligible activity cliffs were found."
    if not summary.empty and summary["n_pairs"].sum() > 0:
        sign = float(np.average(summary["sign_accuracy"].dropna(), weights=summary.loc[summary["sign_accuracy"].notna(), "n_pairs"]))
        mae = float(np.average(summary["delta_mae"].dropna(), weights=summary.loc[summary["delta_mae"].notna(), "n_pairs"]))
        headline = f"Across {int(summary['n_pairs'].sum())} cliffs, sign accuracy={sign:.3f}, delta MAE={mae:.3f}."

    # All-models comparison (only when model=None): aggregate per model
    if model is None and not summary.empty:
        per_model = (summary.dropna(subset=["delta_mae"])
                     .groupby("model").apply(lambda g: pd.Series({
                         "n_pairs": int(g["n_pairs"].sum()),
                         "delta_mae_w": float(np.average(g["delta_mae"], weights=g["n_pairs"])),
                         "delta_pearson_w": float(np.average(g["delta_pearson"].dropna(),
                                                              weights=g.loc[g["delta_pearson"].notna(), "n_pairs"]) if g["delta_pearson"].notna().any() else np.nan),
                         "delta_spearman_w": float(np.average(g["delta_spearman"].dropna(),
                                                               weights=g.loc[g["delta_spearman"].notna(), "n_pairs"]) if g["delta_spearman"].notna().any() else np.nan),
                         "sign_accuracy_w": float(np.average(g["sign_accuracy"].dropna(),
                                                              weights=g.loc[g["sign_accuracy"].notna(), "n_pairs"]) if g["sign_accuracy"].notna().any() else np.nan),
                     }))
                     .reset_index()
                     .sort_values("delta_mae_w"))
        save_table(per_model, paths, "activity_cliff_per_model")
        try:
            import matplotlib.pyplot as plt
            from radiant_qsar.analyses.common import NC_DOUBLE_COL, nature_palette
            fig, axes = plt.subplots(1, 3, figsize=(NC_DOUBLE_COL, 2.8))
            colors = nature_palette(len(per_model))
            for ax, col, ylabel, lower in (
                (axes[0], "delta_mae_w", "Cliff Delta-MAE (lower=better)", True),
                (axes[1], "sign_accuracy_w", "Cliff sign accuracy (higher=better)", False),
                (axes[2], "delta_spearman_w", "Cliff rank Spearman (higher=better)", False),
            ):
                vals = per_model[col].astype(float).values
                ax.bar(per_model["model"], vals, color=colors, edgecolor="none")
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                                fontsize=7, fontweight="bold")
                ax.set_ylabel(ylabel, fontweight="bold", fontsize=8)
                plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=7)
                ax.set_title(("↓" if lower else "↑") + " " + col.replace("_w", "").replace("_", " "),
                             fontweight="bold", fontsize=9)
            fig.suptitle("Activity-cliff handling -- per model",
                         fontweight="bold", y=1.02, fontsize=10)
            fig.tight_layout()
            save_figure(fig, paths, "activity_cliff_per_model_bars")
            plt.close(fig)
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("per-model bar plot failed: %s", exc)

    write_summary_md(
        paths,
        title="Activity-Cliff SAR Error",
        claim="A strong QSAR model should retain potency ordering for high-similarity activity cliffs.",
        headline=headline,
        details={
            "Model": model,
            "Split": split or "all",
            "Tanimoto threshold": f"{tanimoto_threshold:.2f}",
            "Activity delta threshold": f"{activity_delta_threshold:.2f} pChEMBL",
        },
        tables_referenced=["activity_cliff_pairs.csv", "activity_cliff_summary.csv"],
        figures_referenced=["activity_cliff_delta.png"] if not pairs_df.empty else [],
    )
    return {"paths": paths, "pairs": pairs_df, "summary": summary}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Activity-cliff SAR error analysis")
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="radiant")
    p.add_argument("--split", default="activity_cliff")
    p.add_argument("--tanimoto-threshold", type=float, default=0.55)
    p.add_argument("--activity-delta-threshold", type=float, default=1.0)
    p.add_argument("--max-pairs-per-cell", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
