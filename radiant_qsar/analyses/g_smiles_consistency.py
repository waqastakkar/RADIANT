"""Phase G -- Test-time SMILES augmentation consistency.

For each scaffold-split test molecule, predict it under K different random
SMILES enumerations of the SAME molecule and measure prediction variance.
This serves as:

1. A test-time consistency check ("does the model give the same answer
   for the same molecule written differently?")
2. A confidence proxy independent of the (collapsed) halting head: low
   prediction-stdev = high confidence.
3. A diagnostic for SMILES-augmentation training robustness (item #4 on
   the user's ablation wishlist).

For each cell we report:

* per-molecule mean prediction, stdev, abs_err
* Spearman correlation between stdev and abs_err (does uncertainty track
  actual error?)
* Top-k retention MAE curve using stdev as confidence

This module loads the same checkpoints G.4 loads, so it inherits the
chem_config + ZINC20 vocab fixes. Resumable: each cell caches its
``smiles_consistency_K{K}.csv`` and we skip if present.

Outputs
-------
tables/
    smiles_consistency_per_cell.csv    -- per-cell summary stats
    smiles_consistency_per_molecule.csv -- one row per (cell, molecule)
figures/
    g_smiles_consistency_stdev_vs_err.{png,svg}
    g_smiles_consistency_retention.{png,svg}
    g_smiles_consistency_per_cell_bars.{png,svg}
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
    spearman_pearson,
    write_summary_md,
)

logger = logging.getLogger(__name__)


@dataclass
class _Cfg:
    panel_root: Path
    config_path: Path
    vocab_path: Path
    lf_model_dir: str = "radiant"
    split: str = "scaffold"
    n_augmentations: int = 5
    device: str = "cuda"
    batch_size: int = 64
    task_name: str = "pchembl"
    seed: int = 0


def _randomize_smiles(smiles_list: Sequence[str], k: int, seed: int) -> list[list[str]]:
    """Return K random-SMILES variants per input. Falls back to original
    when RDKit randomization fails."""
    from rdkit import Chem
    rng = np.random.default_rng(seed)
    out: list[list[str]] = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            out.append([s] * k)
            continue
        variants = []
        for _ in range(k):
            try:
                v = Chem.MolToSmiles(m, doRandom=True,
                                      canonical=False,
                                      isomericSmiles=True)
                variants.append(v)
            except Exception:
                variants.append(s)
        # Always include canonical SMILES as variant 0 for stability
        try:
            canon = Chem.MolToSmiles(m, canonical=True)
            variants[0] = canon
        except Exception:
            pass
        out.append(variants)
    _ = rng  # rng kept for future reproducibility hooks
    return out


def _run_cell(cell_dir: Path, cfg: _Cfg) -> pd.DataFrame:
    """Predict every test molecule of a cell K times under random SMILES."""
    import torch
    from radiant import RadiantConfig
    from radiant_chem.config import RadiantChemConfig
    from radiant_chem.model_chem import RadiantChemModel
    from radiant_chem.tasks import TaskRegistry, TaskSpec
    from radiant_chem.tokenizer import SmilesTokenizer

    # Skip if cached
    cache = cell_dir / f"smiles_consistency_K{cfg.n_augmentations}.csv"
    if cache.exists():
        return pd.read_csv(cache)

    # Build model with the cell's chem_config so the head dimensions match
    tok = SmilesTokenizer.load(cfg.vocab_path)
    base_cfg = RadiantConfig.from_json(cfg.config_path).replace(
        vocab_size=tok.vocab_size, pad_token_id=tok.pad_id,
    )
    cell_chem_cfg_path = cell_dir / "chem_config.json"
    if cell_chem_cfg_path.exists():
        import json
        cc = json.loads(cell_chem_cfg_path.read_text(encoding="utf-8"))
        chem_kwargs = {k: v for k, v in cc.items() if k != "base"}
        chem_cfg = RadiantChemConfig(base=base_cfg, **chem_kwargs)
    else:
        chem_cfg = RadiantChemConfig(base=base_cfg)
    tasks = TaskRegistry([TaskSpec(cfg.task_name, "regression",
                                   cfg.task_name, num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(cfg.device)
    ckpt = torch.load(cell_dir / "best.pt", map_location=cfg.device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    df = pd.read_csv(cell_dir / "predictions.csv")
    test_smiles = df["smiles"].astype(str).tolist()
    variants = _randomize_smiles(test_smiles, k=cfg.n_augmentations, seed=cfg.seed)

    max_seq = base_cfg.max_seq_len
    preds = np.full((len(df), cfg.n_augmentations), np.nan, dtype=float)
    with torch.no_grad():
        for var_idx in range(cfg.n_augmentations):
            batch_smiles = [v[var_idx] for v in variants]
            for start in range(0, len(batch_smiles), cfg.batch_size):
                chunk = batch_smiles[start:start + cfg.batch_size]
                input_ids, attn = tok.encode_batch(chunk)
                if input_ids.shape[1] > max_seq:
                    input_ids = input_ids[:, :max_seq]
                    attn = attn[:, :max_seq]
                input_ids = input_ids.to(cfg.device)
                attn = attn.to(cfg.device)
                out = model(input_ids, attention_mask=attn,
                            return_loop_metrics=True)
                pred = out.task_outputs[cfg.task_name].squeeze(-1).cpu().numpy()
                preds[start:start + len(chunk), var_idx] = pred

    mean_pred = np.nanmean(preds, axis=1)
    std_pred = np.nanstd(preds, axis=1, ddof=0)
    out_df = pd.DataFrame({
        "inchikey14": df["inchikey14"],
        "smiles": df["smiles"],
        "true_pchembl": df["true_pchembl"],
        "canonical_pred": preds[:, 0],
        "mean_pred": mean_pred,
        "std_pred": std_pred,
        "abs_err": np.abs(mean_pred - df["true_pchembl"]),
    })
    out_df.to_csv(cache, index=False)
    return out_df


def run(*, panel_root: Path | str, out_dir: Path | str,
        config_path: Path | str = "configs/radiant_75m.json",
        vocab_path: Path | str = "data/zinc20/smiles_vocab.json",
        lf_model_dir: str = "radiant",
        split: str = "scaffold",
        n_augmentations: int = 5,
        device: str = "cuda",
        batch_size: int = 64,
        max_cells: int | None = None) -> dict:
    publication_style()
    panel_root = Path(panel_root)
    paths = AnalysisPaths(Path(out_dir), "g_smiles_consistency")
    cfg = _Cfg(panel_root=panel_root, config_path=Path(config_path),
               vocab_path=Path(vocab_path), lf_model_dir=lf_model_dir,
               split=split, n_augmentations=n_augmentations,
               device=device, batch_size=batch_size)

    cell_dirs = sorted(
        d for d in (panel_root / lf_model_dir).glob(f"*/{split}")
        if (d / "best.pt").exists() and (d / "predictions.csv").exists()
    )
    if max_cells:
        cell_dirs = cell_dirs[:max_cells]
    if not cell_dirs:
        raise FileNotFoundError(
            f"no eligible cells at {panel_root / lf_model_dir}/*/{split}/")
    logger.info("processing %d cells with K=%d augmentations",
                len(cell_dirs), n_augmentations)

    rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for cell_dir in cell_dirs:
        target = cell_dir.parent.name
        try:
            df = _run_cell(cell_dir, cfg)
        except Exception as exc:
            logger.warning("cell %s failed: %s", target, exc)
            continue
        df["target"] = target
        df["split"] = split
        rows.append(df)

        # Per-cell stats
        valid = df.dropna(subset=["std_pred", "abs_err"])
        if len(valid) < 3:
            continue
        sp = spearman_pearson(valid["std_pred"].to_numpy(),
                              valid["abs_err"].to_numpy(),
                              n_bootstrap=100, seed=cfg.seed)
        summary_rows.append({
            "target": target, "split": split,
            "n": int(len(df)),
            "mean_std": float(valid["std_pred"].mean()),
            "median_std": float(valid["std_pred"].median()),
            "spearman_std_err": float(sp["spearman"]),
            "pearson_std_err": float(sp["pearson"]),
            "canonical_mae": float(np.abs(df["canonical_pred"] - df["true_pchembl"]).mean()),
            "ensemble_mae": float(np.abs(df["mean_pred"] - df["true_pchembl"]).mean()),
        })

    if not rows:
        raise RuntimeError("no cells produced consistency predictions")

    per_mol = pd.concat(rows, ignore_index=True)
    save_table(per_mol, paths, "smiles_consistency_per_molecule")
    summary = pd.DataFrame(summary_rows)
    save_table(summary, paths, "smiles_consistency_per_cell")

    _plot_stdev_vs_err(per_mol, paths)
    _plot_retention(per_mol, paths)
    _plot_per_cell_bars(summary, paths)

    headline = "no consistency data produced"
    if not summary.empty:
        mean_std = float(summary["mean_std"].mean())
        mean_corr = float(summary["spearman_std_err"].mean())
        ens_gain = float((summary["canonical_mae"] - summary["ensemble_mae"]).mean())
        headline = (
            f"K={n_augmentations} random SMILES per molecule across "
            f"{len(summary)} scaffold-split cells: "
            f"mean per-molecule prediction std = {mean_std:.3f} pChEMBL; "
            f"mean Spearman(std, abs_err) = {mean_corr:+.3f} "
            f"(positive = stdev tracks error); "
            f"ensemble-vs-canonical MAE delta = {ens_gain:+.3f}.")

    write_summary_md(
        paths,
        title=f"SMILES augmentation consistency (K={n_augmentations})",
        claim=("Predict every scaffold-split test molecule under K random "
               "SMILES enumerations of the SAME molecule; the standard "
               "deviation of the K predictions is an inference-time "
               "confidence proxy and a SMILES-aug robustness check."),
        headline=headline,
        details={
            "K augmentations": str(n_augmentations),
            "Split": split,
            "Cells processed": str(len(summary)),
        },
        tables_referenced=[
            "smiles_consistency_per_molecule.csv",
            "smiles_consistency_per_cell.csv",
        ],
        figures_referenced=[
            "g_smiles_consistency_stdev_vs_err.png",
            "g_smiles_consistency_retention.png",
            "g_smiles_consistency_per_cell_bars.png",
        ],
    )
    return {"paths": paths, "per_mol": per_mol, "summary": summary}


def _plot_stdev_vs_err(per_mol: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    df = per_mol.dropna(subset=["std_pred", "abs_err"])
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 3.0))
    # Decile bin
    df = df.copy()
    df["bin"] = pd.qcut(df["std_pred"], q=10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(
        mean_std=("std_pred", "mean"),
        mean_err=("abs_err", "mean"),
        q25=("abs_err", lambda v: np.nanpercentile(v, 25)),
        q75=("abs_err", lambda v: np.nanpercentile(v, 75)),
        n=("abs_err", "count")).reset_index()
    ax.plot(g["mean_std"], g["mean_err"], "-o",
            color=nature_palette(1)[0], lw=1.6, ms=4)
    ax.fill_between(g["mean_std"], g["q25"], g["q75"],
                    color=nature_palette(1)[0], alpha=0.15, lw=0)
    ax.set_xlabel("Prediction stdev across K SMILES enumerations",
                  fontweight="bold")
    ax.set_ylabel("Mean absolute error (with IQR band)", fontweight="bold")
    ax.set_title("Does test-time-aug stdev track actual error?",
                 fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_smiles_consistency_stdev_vs_err")
    plt.close(fig)


def _plot_retention(per_mol: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    df = per_mol.dropna(subset=["std_pred", "abs_err"])
    if df.empty:
        return
    sorted_df = df.sort_values("std_pred", ascending=True)
    n = len(sorted_df)
    ks = np.arange(0.05, 1.001, 0.05)
    xs = []
    ys = []
    for k in ks:
        keep = max(int(n * k), 1)
        xs.append(float(k) * 100)
        ys.append(float(sorted_df.head(keep)["abs_err"].mean()))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL * 1.4, 3.0))
    ax.plot(xs, ys, "-o", color=nature_palette(1)[0], lw=1.6, ms=4)
    ax.set_xlabel("Retention (%) ranked by stdev (low first = confident)",
                  fontweight="bold")
    ax.set_ylabel("MAE on retained subset", fontweight="bold")
    ax.set_title("Top-k confidence-filtered MAE\n(stdev-based confidence)",
                 fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.3, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_smiles_consistency_retention")
    plt.close(fig)


def _plot_per_cell_bars(summary: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt
    if summary.empty:
        return
    sub = summary.sort_values("mean_std", ascending=False)
    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL * 0.7, 3.0))
    ax.bar(sub["target"], sub["mean_std"],
           color=nature_palette(1)[0], edgecolor="none")
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean per-molecule prediction stdev",
                  fontweight="bold")
    ax.set_title("Test-time SMILES augmentation: per-cell mean stdev",
                 fontweight="bold", fontsize=9)
    fig.tight_layout()
    save_figure(fig, paths, "g_smiles_consistency_per_cell_bars")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--config", type=Path,
                   default=Path("configs/radiant_75m.json"))
    p.add_argument("--vocab", type=Path,
                   default=Path("data/zinc20/smiles_vocab.json"))
    p.add_argument("--lf-model-dir", default="radiant")
    p.add_argument("--split", default="scaffold")
    p.add_argument("--n-augmentations", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-cells", type=int, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(panel_root=args.panel_root, out_dir=args.out_dir,
        config_path=args.config, vocab_path=args.vocab,
        lf_model_dir=args.lf_model_dir, split=args.split,
        n_augmentations=args.n_augmentations,
        device=args.device, batch_size=args.batch_size,
        max_cells=args.max_cells)


if __name__ == "__main__":
    main()
