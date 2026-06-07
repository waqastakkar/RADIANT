"""Failure-mode analysis for RADIANT-QSAR prediction panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    discover_predictions,
    load_predictions,
    publication_style,
    regression_metrics,
    save_figure,
    save_table,
    write_summary_md,
)
from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles
from radiant_qsar.pretrain.corpus import _murcko_scaffold_smiles


def _render_worst_structures(worst: pd.DataFrame, paths: AnalysisPaths,
                             top_k_render: int = 24) -> None:
    """Render RDKit thumbnails for the top-K worst predictions.

    Emits per-molecule PNG + SVG and a single grid PNG+SVG with each
    tile labelled by target, split, true vs predicted pChEMBL, and the
    absolute error. Failures silently if RDKit / cairo are missing.
    """
    if worst.empty:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return
    sub = worst.head(top_k_render).reset_index(drop=True)
    structures_dir = paths.figures / "worst_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    tile_w, tile_h = 460, 460
    # Per-molecule renders
    for i, row in sub.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        legend = (f"{row['panel_target']} / {row['panel_split']}  "
                  f"|err|={row['abs_error']:.2f}\n"
                  f"true={row['true_pchembl']:.2f}  pred={row['pred_pchembl']:.2f}")
        # SVG
        d_svg = rdMolDraw2D.MolDraw2DSVG(tile_w, tile_h)
        d_svg.drawOptions().legendFontSize = 16
        d_svg.DrawMolecule(mol, legend=legend)
        d_svg.FinishDrawing()
        (structures_dir / f"worst_{i:02d}.svg").write_text(
            d_svg.GetDrawingText(), encoding="utf-8")
        # PNG via Cairo
        try:
            d_png = rdMolDraw2D.MolDraw2DCairo(tile_w, tile_h)
            d_png.drawOptions().legendFontSize = 16
            d_png.DrawMolecule(mol, legend=legend)
            d_png.FinishDrawing()
            d_png.WriteDrawingText(str(structures_dir / f"worst_{i:02d}.png"))
        except Exception:
            pass

    # Combined grid PNG
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        n = len(sub)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.8, nrows * 3.1),
                                 squeeze=False)
        for i in range(nrows * ncols):
            ax = axes[i // ncols][i % ncols]
            png = structures_dir / f"worst_{i:02d}.png"
            if i < n and png.exists():
                ax.imshow(mpimg.imread(str(png)))
            ax.set_axis_off()
        fig.suptitle(f"Worst {n} predictions  (|pred - true| pChEMBL)",
                     fontweight="bold", fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        save_figure(fig, paths, "failure_worst_structures_grid")
        plt.close(fig)
    except Exception:
        pass


def _load_panel(panel_root: Path | str, model: str | None) -> pd.DataFrame:
    manifest = discover_predictions(panel_root)
    if manifest.empty:
        raise FileNotFoundError(f"no predictions.csv found under {panel_root}")
    if model:
        manifest = manifest[manifest["model"] == model]
    if manifest.empty:
        raise FileNotFoundError(f"no predictions for model={model!r}")
    frames = []
    for _, item in manifest.iterrows():
        df = load_predictions(item["path"])
        df.insert(0, "model", item["model"])
        df.insert(1, "panel_target", item["target"])
        df.insert(2, "panel_split", item["split"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run(
    *,
    panel_root: Path | str,
    out_dir: Path | str,
    model: str = "radiant",
    top_n: int = 100,
    min_scaffold_n: int = 3,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), "g_failure_modes")
    df = _load_panel(panel_root, model)
    df = df.copy()
    df["error"] = df["pred_pchembl"] - df["true_pchembl"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    df["scaffold_smiles"] = df["smiles"].map(_murcko_scaffold_smiles)
    df["rgroup_smiles"] = df["smiles"].map(murcko_rgroup_smiles)

    worst = df.sort_values("abs_error", ascending=False).head(top_n)
    metrics_rows = []
    for (m, target, split), group in df.groupby(["model", "panel_target", "panel_split"], sort=False):
        row = {"model": m, "target": target, "split": split}
        row.update(regression_metrics(group))
        metrics_rows.append(row)
    cell_metrics = pd.DataFrame(metrics_rows)

    scaffold = (
        df[df["scaffold_smiles"] != ""]
        .groupby(["model", "panel_target", "panel_split", "scaffold_smiles"], as_index=False)
        .agg(
            n=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            mean_true=("true_pchembl", "mean"),
            mean_pred=("pred_pchembl", "mean"),
            distinct_rgroups=("rgroup_smiles", "nunique"),
        )
    )
    scaffold = scaffold[scaffold["n"] >= min_scaffold_n].sort_values("mae", ascending=False)

    save_table(worst, paths, "worst_predictions")
    save_table(cell_metrics, paths, "failure_metrics_by_cell")
    save_table(scaffold, paths, "failure_metrics_by_scaffold")

    if not cell_metrics.empty:
        import matplotlib.pyplot as plt

        plot = cell_metrics.sort_values("mae", ascending=False).head(20)
        labels = (plot["target"].astype(str) + "/" + plot["split"].astype(str)).to_list()
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.bar(np.arange(len(plot)), plot["mae"].to_numpy())
        ax.set_xticks(np.arange(len(plot)))
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title("Highest-error benchmark cells")
        save_figure(fig, paths, "failure_cells_mae")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Worst-K structure thumbnails (PNG + SVG) -- one image per failure
    # plus a combined grid. Reviewers want to *see* the molecules the
    # model gets most wrong.
    _render_worst_structures(worst, paths, top_k_render=24)

    headline = f"Worst-cell MAE={cell_metrics['mae'].max():.3f}; median-cell MAE={cell_metrics['mae'].median():.3f}."
    write_summary_md(
        paths,
        title="Failure Mode Analysis",
        claim="Large errors should be traceable to targets, splits, scaffolds, and R-group contexts.",
        headline=headline,
        details={
            "Model": model,
            "Worst rows reported": str(top_n),
            "Scaffold minimum n": str(min_scaffold_n),
        },
        tables_referenced=[
            "worst_predictions.csv",
            "failure_metrics_by_cell.csv",
            "failure_metrics_by_scaffold.csv",
        ],
        figures_referenced=["failure_cells_mae.png"],
    )
    return {"paths": paths, "worst": worst, "cell_metrics": cell_metrics, "scaffold_metrics": scaffold}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure mode analysis")
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", default="radiant")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--min-scaffold-n", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
