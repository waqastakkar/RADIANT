"""Phase G -- Applicability Domain (AD) analysis.

For every (target, split) cell:
  1. Identify train SMILES = (all activities for target) - (test inchikeys
     in this cell's predictions.csv). Val is folded into "train" for AD
     purposes because the question is "how close is a test molecule to the
     chemistry the model has SEEN."
  2. Compute Morgan-FP Tanimoto from each test molecule to its nearest
     neighbour in the train set.
  3. Bin: high (>=0.7), medium (0.5-0.7), low (0.3-0.5), novel (<0.3).
  4. Per bin and per model, report MAE.
  5. Aggregate across the panel: a "MAE vs distance" curve per model,
     plus a per-bin bar chart.

This is one of the most NMI-relevant figures: it directly answers
"does the model degrade gracefully when the test chemistry drifts away
from training?" Reviewers care because virtual screening lives in the
low-similarity regime by construction.

Outputs
-------
tables/
    ad_per_molecule.csv            -- test molecule + max Tanimoto + abs_err per model
    ad_per_bin_mae.csv             -- bin x model MAE
    ad_per_cell_summary.csv        -- per (target, split): bin counts and per-model MAE
figures/
    g_ad_mae_vs_distance.{png,svg} -- MAE vs Tanimoto curve per model
    g_ad_per_bin.{png,svg}         -- grouped-bar per (bin x model)
    g_ad_distribution.{png,svg}    -- histogram of max-Tanimoto per split
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


# Bin edges (right-closed) and labels in display order
BIN_EDGES = (0.0, 0.3, 0.5, 0.7, 1.01)  # 1.01 to include max=1.0
BIN_LABELS = ("novel", "low", "medium", "high")
SPLIT_ORDER = ("random", "scaffold", "time", "cluster", "activity_cliff")


def _morgan_fps(smiles_list: list[str]):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        fps.append(gen.GetFingerprint(m) if m is not None else None)
    return fps


def _max_tanimoto_to_set(query_fps, ref_fps_valid) -> np.ndarray:
    from rdkit import DataStructs
    out = np.full(len(query_fps), np.nan, dtype=float)
    if not ref_fps_valid:
        return out
    for i, q in enumerate(query_fps):
        if q is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps_valid)
        out[i] = float(max(sims)) if sims else np.nan
    return out


def _load_activities(activities_path: Path) -> pd.DataFrame:
    """Returns activities frame with at least target_chembl_id, inchikey14, smiles.

    Accepts either ``smiles`` or ``standard_smiles`` (the column the curated
    activities.parquet writes) and normalises to ``smiles``.
    """
    if activities_path.suffix == ".parquet":
        df = pd.read_parquet(activities_path)
    else:
        df = pd.read_csv(activities_path)
    if "smiles" not in df.columns and "standard_smiles" in df.columns:
        df = df.rename(columns={"standard_smiles": "smiles"})
    needed = {"target_chembl_id", "inchikey14", "smiles"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"activities file missing columns {miss}; got {list(df.columns)}")
    return df


def _per_cell_ad(
    *,
    panel_root: Path,
    activities: pd.DataFrame,
    cell: pd.Series,
) -> pd.DataFrame:
    """Compute max-Tanimoto-to-train for every test molecule in one cell."""
    pred = load_predictions(cell["path"])
    target = cell["target"]
    test_inchikeys = set(pred["inchikey14"].astype(str).tolist())

    tgt_data = activities[activities["target_chembl_id"] == target]
    train_data = tgt_data[~tgt_data["inchikey14"].astype(str).isin(test_inchikeys)]
    train_smiles = train_data["smiles"].astype(str).tolist()

    train_fps = _morgan_fps(train_smiles)
    train_fps_valid = [f for f in train_fps if f is not None]
    test_smiles = pred["smiles"].astype(str).tolist()
    test_fps = _morgan_fps(test_smiles)
    max_sim = _max_tanimoto_to_set(test_fps, train_fps_valid)

    out = pred[["inchikey14", "smiles", "true_pchembl", "pred_pchembl"]].copy()
    out["max_tanimoto_to_train"] = max_sim
    out["abs_err"] = (out["pred_pchembl"] - out["true_pchembl"]).abs()
    out.insert(0, "model", cell["model"])
    out.insert(1, "target", target)
    out.insert(2, "split", cell["split"])
    return out


def _assign_bin(s: np.ndarray) -> pd.Categorical:
    return pd.cut(s, bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True,
                  right=False, ordered=True)


def _plot_mae_vs_distance(per_mol: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    if per_mol.empty:
        return
    df = per_mol.dropna(subset=["max_tanimoto_to_train", "abs_err"]).copy()
    df["sim_bin"] = pd.cut(df["max_tanimoto_to_train"],
                           bins=np.linspace(0, 1, 11), include_lowest=True)
    g = df.groupby(["model", "sim_bin"], observed=True)["abs_err"].agg(
        mean="mean", median="median", q25=lambda v: np.nanpercentile(v, 25),
        q75=lambda v: np.nanpercentile(v, 75), n="count").reset_index()
    g["sim_mid"] = g["sim_bin"].apply(lambda iv: iv.mid)
    models = sorted(g["model"].unique())
    colors = nature_palette(len(models))

    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.6, 3.0))
    for c, m in zip(colors, models):
        sub = g[g["model"] == m].sort_values("sim_mid")
        ax.plot(sub["sim_mid"], sub["mean"], "-o", color=c, lw=1.4, ms=4, label=m)
        ax.fill_between(sub["sim_mid"], sub["q25"], sub["q75"],
                        color=c, alpha=0.12, lw=0)
    ax.set_xlabel("Max Tanimoto to training set (Morgan r=2, 2048b)", fontweight="bold")
    ax.set_ylabel("Mean test MAE (with IQR band)", fontweight="bold")
    ax.set_title("Applicability domain: MAE vs distance from training chemistry",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, paths, "g_ad_mae_vs_distance")
    plt.close(fig)


def _plot_per_bin(per_mol: pd.DataFrame, paths: AnalysisPaths) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    if per_mol.empty:
        return pd.DataFrame()
    df = per_mol.dropna(subset=["max_tanimoto_to_train"]).copy()
    df["bin"] = _assign_bin(df["max_tanimoto_to_train"].to_numpy())
    by_bin = df.groupby(["bin", "model"], observed=True)["abs_err"].agg(
        mean="mean", n="count").reset_index()
    pivot = by_bin.pivot(index="bin", columns="model", values="mean").reindex(BIN_LABELS)
    pivot_n = by_bin.pivot(index="bin", columns="model", values="n").reindex(BIN_LABELS)
    save_table(pivot.reset_index(), paths, "ad_per_bin_mae")
    save_table(pivot_n.reset_index(), paths, "ad_per_bin_n")

    models = list(pivot.columns)
    colors = nature_palette(len(models))
    x = np.arange(len(BIN_LABELS))
    w = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.8, 3.0))
    for i, m in enumerate(models):
        vals = pivot[m].values.astype(float)
        ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
               color=colors[i], edgecolor="none", label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl}\n(<{e:.2f})" if i == 0 else f"{lbl}\n({BIN_EDGES[i]:.2f}-{e:.2f})"
                        for i, (lbl, e) in enumerate(zip(BIN_LABELS, BIN_EDGES[1:]))],
                       fontsize=7)
    ax.set_xlabel("Tanimoto similarity to training set", fontweight="bold")
    ax.set_ylabel("Mean test MAE", fontweight="bold")
    ax.set_title("MAE by applicability-domain bin", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, paths, "g_ad_per_bin")
    plt.close(fig)
    return pivot


def _plot_distribution(per_mol: pd.DataFrame, paths: AnalysisPaths) -> None:
    """Histogram of max-Tanimoto per split (one panel per split)."""
    import matplotlib.pyplot as plt
    if per_mol.empty:
        return
    # use radiant only to avoid duplicate molecules across models
    df = per_mol[per_mol["model"] == per_mol["model"].iloc[0]]
    splits = [s for s in SPLIT_ORDER if s in df["split"].unique()]
    if not splits:
        return
    colors = nature_palette(len(splits))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.6, 2.6))
    for c, s in zip(colors, splits):
        vals = df.loc[df["split"] == s, "max_tanimoto_to_train"].dropna().to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=np.linspace(0, 1, 41), histtype="step", lw=1.3,
                color=c, label=f"{s} (n={len(vals)})")
    for e in BIN_EDGES[1:-1]:
        ax.axvline(e, color="black", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Max Tanimoto to training set", fontweight="bold")
    ax.set_ylabel("Test molecules", fontweight="bold")
    ax.set_title("Train-set similarity per split type", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, paths, "g_ad_distribution")
    plt.close(fig)


def run(
    *,
    panel_root: Path | str,
    out_dir: Path | str,
    activities_path: Path | str = "data/processed/v1/activities.parquet",
    splits: tuple[str, ...] | None = None,
) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    paths = AnalysisPaths(Path(out_dir), "g_applicability_domain")
    activities_path = Path(activities_path)
    if not activities_path.exists():
        raise FileNotFoundError(
            f"activities file not found at {activities_path}; "
            "needed to determine train SMILES per target.")

    activities = _load_activities(activities_path)
    manifest = discover_predictions(panel_root)
    if manifest.empty:
        raise FileNotFoundError(f"no predictions.csv under {panel_root}")
    if splits:
        manifest = manifest[manifest["split"].isin(splits)]

    rows: list[pd.DataFrame] = []
    n_cells = 0
    for _, cell in manifest.iterrows():
        try:
            row = _per_cell_ad(panel_root=panel_root, activities=activities, cell=cell)
            rows.append(row)
            n_cells += 1
            if n_cells % 25 == 0:
                logger.info("AD: processed %d cells", n_cells)
        except Exception as exc:
            logger.warning("AD failed for %s/%s/%s: %s",
                           cell["model"], cell["target"], cell["split"], exc)
    if not rows:
        raise RuntimeError("AD analysis produced no rows.")

    per_mol = pd.concat(rows, ignore_index=True)
    save_table(per_mol, paths, "ad_per_molecule")

    per_cell = per_mol.groupby(["model", "target", "split"]).agg(
        n=("max_tanimoto_to_train", "count"),
        mean_sim=("max_tanimoto_to_train", "mean"),
        median_sim=("max_tanimoto_to_train", "median"),
        mae=("abs_err", "mean"),
    ).reset_index()
    save_table(per_cell, paths, "ad_per_cell_summary")

    pivot = _plot_per_bin(per_mol, paths)
    _plot_mae_vs_distance(per_mol, paths)
    _plot_distribution(per_mol, paths)

    # Headline: who is best in the "novel" bin (low Tanimoto)?
    headline = "Applicability-domain analysis produced no bins."
    if not pivot.empty:
        try:
            best_novel = pivot.loc["novel"].idxmin()
            best_novel_mae = float(pivot.loc["novel"].min())
            best_high_mae = float(pivot.loc["high"].min())
            headline = (f"In the novel-scaffold bin (Tanimoto<0.30), best mean MAE is "
                        f"{best_novel} = {best_novel_mae:.3f}. In the high-similarity "
                        f"bin (>=0.70), best MAE = {best_high_mae:.3f} (any model). "
                        f"Gap = {best_novel_mae - best_high_mae:+.3f} pChEMBL units "
                        f"-- the lower the gap, the more robust the model is "
                        f"out-of-distribution.")
        except Exception:
            pass

    write_summary_md(
        paths,
        title="Applicability domain (Tanimoto-to-train)",
        claim=("Per-test-molecule MAE stratified by max Tanimoto similarity to "
               "the training set. Tests whether each model degrades gracefully "
               "when test chemistry drifts away from training."),
        headline=headline,
        details={
            "Bins": ", ".join(f"{lbl} (<{e:.2f})" if i == 0
                              else f"{lbl} ({BIN_EDGES[i]:.2f}-{e:.2f})"
                              for i, (lbl, e) in enumerate(zip(BIN_LABELS, BIN_EDGES[1:]))),
            "Cells processed": str(n_cells),
            "Activities source": str(activities_path),
        },
        tables_referenced=[
            "ad_per_molecule.csv", "ad_per_cell_summary.csv",
            "ad_per_bin_mae.csv", "ad_per_bin_n.csv",
        ],
        figures_referenced=[
            "g_ad_mae_vs_distance.png",
            "g_ad_per_bin.png",
            "g_ad_distribution.png",
        ],
    )
    return {"paths": paths, "per_mol": per_mol, "per_bin": pivot}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--activities", type=Path,
                   default=Path("data/processed/v1/activities.parquet"))
    p.add_argument("--splits", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        activities_path=args.activities,
        splits=tuple(args.splits) if args.splits else None)


if __name__ == "__main__":
    main()
