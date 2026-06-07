"""Phase G -- Scaffold-novelty MAE bins (scaffold split only).

For the scaffold-split test set of every (model, target), bin test
molecules by max Tanimoto similarity to **training scaffolds** (not full
molecules) and report MAE per bin per model.

Tanimoto-to-scaffold is a more conservative novelty measure than
Tanimoto-to-molecule because it isolates the *scaffold* novelty -- a
test molecule with a fresh scaffold but a familiar substituent will land
in a low-similarity bin here whereas it might be classified as
"familiar" by full-molecule Tanimoto.

Inputs
------
* runs/phase_g/g_applicability_domain/tables/ad_per_molecule.csv
  (provides per-test molecule + model + abs_err)
* data/processed/v1/activities.parquet (for train SMILES per target)

Outputs
-------
tables/
    scaffold_novelty_per_molecule.csv  -- + scaffold_sim and bin label
    scaffold_novelty_bin_mae.csv       -- bin x model MAE
figures/
    g_scaffold_novelty_per_bin.{png,svg}
    g_scaffold_novelty_curve.{png,svg}
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    NC_SINGLE_COL,
    nature_palette,
    publication_style,
    save_figure,
    save_table,
    write_summary_md,
)
from radiant_qsar.analyses.g_applicability_domain import (
    _morgan_fps, _max_tanimoto_to_set, _load_activities,
    BIN_EDGES, BIN_LABELS,
)

logger = logging.getLogger(__name__)


def _murcko_scaffold(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if m is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def run(*, panel_root: Path | str, out_dir: Path | str,
        activities_path: Path | str = "data/processed/v1/activities.parquet",
        ad_per_molecule_csv: Path | str | None = None) -> dict:
    publication_style()
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_scaffold_novelty")
    activities_path = Path(activities_path)

    if ad_per_molecule_csv is None:
        ad_per_molecule_csv = (out_dir / "g_applicability_domain"
                                / "tables" / "ad_per_molecule.csv")
    ad_per_molecule_csv = Path(ad_per_molecule_csv)
    if not ad_per_molecule_csv.exists():
        raise FileNotFoundError(
            f"need g_applicability_domain output at {ad_per_molecule_csv}; "
            "run g_applicability_domain first.")
    if not activities_path.exists():
        raise FileNotFoundError(f"activities file not found: {activities_path}")

    per_mol = pd.read_csv(ad_per_molecule_csv)
    per_mol = per_mol[per_mol["split"] == "scaffold"].copy()
    if per_mol.empty:
        raise RuntimeError("no scaffold-split rows in ad_per_molecule.csv")

    activities = _load_activities(activities_path)

    # Compute scaffold for every test molecule and store
    per_mol["scaffold"] = per_mol["smiles"].astype(str).map(_murcko_scaffold)

    # For each target: scaffolds of TRAIN molecules (= activities minus test inchikeys),
    # then per test molecule: max Tanimoto of its scaffold to any train scaffold.
    rows: list[pd.DataFrame] = []
    n_targets = 0
    for target, sub_test in per_mol.groupby("target"):
        # any model row suffices to get test inchikey set (predictions are shared)
        test_inchikeys = set(sub_test["inchikey14"].astype(str).tolist())
        tgt_act = activities[activities["target_chembl_id"] == target]
        train_act = tgt_act[~tgt_act["inchikey14"].astype(str).isin(test_inchikeys)]
        train_scaffolds = (train_act["smiles"].astype(str)
                           .map(_murcko_scaffold))
        train_scaffolds = train_scaffolds[train_scaffolds != ""].unique().tolist()
        if not train_scaffolds:
            continue
        train_scaf_fps = _morgan_fps(train_scaffolds)
        train_scaf_fps_valid = [f for f in train_scaf_fps if f is not None]

        # Per-molecule scaffold and similarity to nearest train scaffold
        uniq_scaf = sub_test["scaffold"].dropna().unique().tolist()
        uniq_fps = _morgan_fps(uniq_scaf)
        scaf_sim_map: dict[str, float] = {}
        sims = _max_tanimoto_to_set(uniq_fps, train_scaf_fps_valid)
        for s, v in zip(uniq_scaf, sims):
            scaf_sim_map[s] = float(v) if v == v else np.nan

        out = sub_test.copy()
        out["scaffold_sim"] = out["scaffold"].map(scaf_sim_map).astype(float)
        rows.append(out)
        n_targets += 1

    if not rows:
        raise RuntimeError("no targets produced scaffold-similarity rows")
    out_df = pd.concat(rows, ignore_index=True)
    out_df["scaffold_bin"] = pd.cut(
        out_df["scaffold_sim"], bins=BIN_EDGES, labels=BIN_LABELS,
        include_lowest=True, right=False, ordered=True)
    save_table(out_df, paths, "scaffold_novelty_per_molecule")

    # bin x model MAE
    by_bin = out_df.groupby(["scaffold_bin", "model"],
                            observed=True)["abs_err"].agg(
        mean="mean", n="count").reset_index()
    pivot = (by_bin.pivot(index="scaffold_bin", columns="model", values="mean")
             .reindex(BIN_LABELS))
    pivot_n = (by_bin.pivot(index="scaffold_bin", columns="model", values="n")
               .reindex(BIN_LABELS))
    save_table(pivot.reset_index(), paths, "scaffold_novelty_bin_mae")
    save_table(pivot_n.reset_index(), paths, "scaffold_novelty_bin_n")

    # Bar chart per bin x model
    import matplotlib.pyplot as plt
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
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlabel("Scaffold similarity bin to train scaffolds",
                  fontweight="bold")
    ax.set_ylabel("Mean test MAE", fontweight="bold")
    ax.set_title(f"Scaffold-novelty MAE (scaffold split, {n_targets} targets)",
                 fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, paths, "g_scaffold_novelty_per_bin")
    plt.close(fig)

    # Smooth curve: MAE vs continuous scaffold sim, decile bins
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.6, 3.0))
    for c, m in zip(colors, models):
        sub = out_df[out_df["model"] == m].dropna(subset=["scaffold_sim", "abs_err"])
        if sub.empty:
            continue
        sub = sub.copy()
        sub["b"] = pd.cut(sub["scaffold_sim"], bins=np.linspace(0, 1, 11),
                          include_lowest=True)
        g = sub.groupby("b", observed=True)["abs_err"].agg(
            mean="mean", q25=lambda v: np.nanpercentile(v, 25),
            q75=lambda v: np.nanpercentile(v, 75)).reset_index()
        g["mid"] = g["b"].apply(lambda iv: iv.mid)
        ax.plot(g["mid"], g["mean"], "-o", color=c, lw=1.4, ms=4, label=m)
        ax.fill_between(g["mid"], g["q25"], g["q75"], color=c, alpha=0.12, lw=0)
    ax.set_xlabel("Max Tanimoto of test scaffold to any train scaffold",
                  fontweight="bold")
    ax.set_ylabel("MAE (mean, IQR band)", fontweight="bold")
    ax.set_title("Scaffold-novelty curve", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, paths, "g_scaffold_novelty_curve")
    plt.close(fig)

    headline = "scaffold-novelty: no bins produced."
    if not pivot.empty:
        try:
            best_novel = pivot.loc["novel"].idxmin()
            best_novel_mae = float(pivot.loc["novel"].min())
            high_mae = float(pivot.loc["high"].min())
            headline = (f"Best model in novel-scaffold bin "
                        f"(scaffold-Tanimoto<0.30): {best_novel} "
                        f"= {best_novel_mae:.3f} MAE. Gap to high-similarity "
                        f"bin: {best_novel_mae - high_mae:+.3f}.")
        except Exception:
            pass

    write_summary_md(
        paths,
        title="Scaffold-novelty MAE bins (scaffold split)",
        claim=("On the scaffold split, bin test molecules by max Tanimoto "
               "of their Bemis-Murcko scaffold to any TRAINING scaffold. "
               "Tests whether the model generalises to novel scaffolds."),
        headline=headline,
        details={
            "Bins": ", ".join(BIN_LABELS),
            "Targets with scaffold-sim rows": str(n_targets),
        },
        tables_referenced=[
            "scaffold_novelty_per_molecule.csv",
            "scaffold_novelty_bin_mae.csv",
            "scaffold_novelty_bin_n.csv",
        ],
        figures_referenced=[
            "g_scaffold_novelty_per_bin.png",
            "g_scaffold_novelty_curve.png",
        ],
    )
    return {"paths": paths, "per_mol": out_df, "per_bin": pivot}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--activities", type=Path,
                   default=Path("data/processed/v1/activities.parquet"))
    p.add_argument("--ad-per-molecule-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        activities_path=args.activities,
        ad_per_molecule_csv=args.ad_per_molecule_csv)


if __name__ == "__main__":
    main()
