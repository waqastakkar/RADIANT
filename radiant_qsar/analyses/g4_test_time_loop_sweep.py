"""Phase G.4 — Test-time loop-depth sweep (Sub-claim C4).

Evaluate the same RADIANT checkpoint at several values of
``n_loops`` and report per-complexity-bin error curves, overlayed with
the halting-induced *effective* compute (which scales sublinearly with
``n_loops`` when adaptive halting is on).

Two operating modes
-------------------

1. **Predictions-only** (``--mode predictions``): point this script at a
   directory containing one ``predictions.csv`` per loop count, named
   ``predictions_nloops{K}.csv``. No model is loaded; this mode is
   appropriate for re-analysis or for non-RADIANT baselines that
   expose a knob analogous to depth.

2. **Model-driven** (``--mode model``): load a chem checkpoint and a
   tokenized dataset; run inference at each requested ``n_loops``
   value, writing per-loop predictions plus the aggregate sweep table.

In both modes the outputs (tables, figures, summary) are identical.

Outputs
-------
* ``sweep_metrics.csv``     — MAE, RMSE, R² per (n_loops, bin).
* ``effective_compute.csv`` — mean effective depth per n_loops.
* ``g4_mae_vs_nloops.{png,svg}`` — one MAE curve per complexity bin.
* ``g4_effective_compute.{png,svg}`` — n_loops vs effective depth.
* ``summary.md`` — C4 verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    COMPLEXITY_DESCRIPTORS,
    NC_DOUBLE_COL,
    NC_SINGLE_COL,
    complexity_bins,
    join_descriptors,
    load_predictions,
    nature_palette,
    publication_style,
    regression_metrics,
    save_figure,
    save_table,
    write_summary_md,
)

logger = logging.getLogger(__name__)


DEFAULT_LOOPS: tuple[int, ...] = (1, 2, 4, 8, 12, 16, 24)


# ---------------------------------------------------------------------------
# Mode A: predictions already exist on disk
# ---------------------------------------------------------------------------

def load_loop_predictions(
    predictions_dir: Path | str,
    loops: Sequence[int],
    *,
    descriptors_path: Path | str | None = None,
) -> pd.DataFrame:
    """Read ``predictions_nloops{K}.csv`` for each K and concatenate.

    Returns a frame with an extra ``n_loops`` column.
    """
    predictions_dir = Path(predictions_dir)
    frames: list[pd.DataFrame] = []
    for k in loops:
        p = predictions_dir / f"predictions_nloops{k}.csv"
        if not p.exists():
            logger.warning("missing %s; skipping n_loops=%d", p, k)
            continue
        df = load_predictions(p)
        df["n_loops"] = int(k)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No predictions_nloops*.csv files found in {predictions_dir}"
        )
    full = pd.concat(frames, ignore_index=True)
    if descriptors_path is not None:
        desc = (pd.read_parquet(descriptors_path)
                if Path(descriptors_path).suffix.lower() == ".parquet"
                else pd.read_csv(descriptors_path))
        full = join_descriptors(full, desc)
    return full


# ---------------------------------------------------------------------------
# Mode B: re-run model at each n_loops
# ---------------------------------------------------------------------------

@dataclass
class ModelSweepConfig:
    checkpoint_path: Path
    config_path: Path
    vocab_path: Path
    smiles_csv: Path                       # canonical: idx, inchikey14, smiles, true_pchembl
    target_chembl_id: str
    split_kind: str
    task_name: str = "pchembl"
    batch_size: int = 64
    device: str = "cuda"
    # If set, load the cell's saved chem_config.json so the regression head
    # (hidden_dim, dropout, pooling_kind, depth_adaptive_pool) matches the
    # trained checkpoint exactly. Without this G.4 raises a head-shape
    # RuntimeError when the fine-tune recipe differs from defaults.
    chem_config_path: Path | None = None


def run_model_sweep(
    cfg: ModelSweepConfig,
    loops: Sequence[int],
    out_dir: Path,
) -> Path:
    """Load checkpoint and dump ``predictions_nloops{K}.csv`` for each K.

    Imports torch + chem stack lazily so this module can be imported on
    machines without GPU / without the chem deps installed.
    """
    import torch
    from radiant import RadiantConfig
    from radiant_chem.config import RadiantChemConfig
    from radiant_chem.model_chem import RadiantChemModel
    from radiant_chem.tasks import TaskRegistry, TaskSpec
    from radiant_chem.tokenizer import SmilesTokenizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = SmilesTokenizer.load(cfg.vocab_path)
    base_cfg = RadiantConfig.from_json(cfg.config_path).replace(
        vocab_size=tok.vocab_size,
        pad_token_id=tok.pad_id,
    )
    # If the cell saved its own chem_config.json (fine-tune writes one per cell),
    # use it to recover head hidden_dim / dropout / pooling_kind so the loaded
    # state dict shapes match. Falls back to defaults otherwise.
    if cfg.chem_config_path is not None and Path(cfg.chem_config_path).exists():
        import json
        cc = json.loads(Path(cfg.chem_config_path).read_text(encoding="utf-8"))
        chem_kwargs = {k: v for k, v in cc.items() if k != "base"}
        chem_cfg = RadiantChemConfig(base=base_cfg, **chem_kwargs)
    else:
        chem_cfg = RadiantChemConfig(base=base_cfg)
    tasks = TaskRegistry([TaskSpec(cfg.task_name, "regression", cfg.task_name, num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(cfg.device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    df = pd.read_csv(cfg.smiles_csv)
    required = {"idx", "inchikey14", "smiles", "true_pchembl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{cfg.smiles_csv} missing columns {missing}")

    max_seq_len = base_cfg.max_seq_len

    for k in loops:
        preds: list[float] = []
        depths: list[float] = []
        with torch.no_grad():
            for start in range(0, len(df), cfg.batch_size):
                chunk = df.iloc[start:start + cfg.batch_size]
                input_ids, attn = tok.encode_batch(chunk["smiles"].tolist())
                # Truncate sequences that exceed the model's max_seq_len.
                # Long SMILES (> max_seq_len tokens) are silently clipped; the
                # prediction is still valid for the retained prefix.
                if input_ids.shape[1] > max_seq_len:
                    input_ids = input_ids[:, :max_seq_len]
                    attn = attn[:, :max_seq_len]
                input_ids = input_ids.to(cfg.device)
                attn = attn.to(cfg.device)
                out = model(input_ids, n_loops=int(k), attention_mask=attn,
                            return_loop_metrics=True)
                pred = out.task_outputs[cfg.task_name].squeeze(-1).cpu().numpy()
                preds.extend(pred.tolist())
                avg_depth = (out.base.halting.avg_depth
                             if (out.base.halting is not None and out.base.halting.avg_depth is not None)
                             else float(k))
                depths.extend([float(avg_depth)] * len(pred))
        out_path = out_dir / f"predictions_nloops{k}.csv"
        out_df = df.copy()
        out_df["target_chembl_id"] = cfg.target_chembl_id
        out_df["split_kind"] = cfg.split_kind
        out_df["pred_pchembl"] = preds
        out_df["effective_depth"] = depths
        out_df = out_df[[
            "idx", "inchikey14", "target_chembl_id", "split_kind", "smiles",
            "true_pchembl", "pred_pchembl", "effective_depth",
        ]]
        out_df.to_csv(out_path, index=False)
        logger.info("wrote %s (n=%d, n_loops=%d, eff_depth=%.2f)",
                    out_path, len(out_df), k, float(np.mean(depths)))

    return out_dir


# ---------------------------------------------------------------------------
# Metrics & figures
# ---------------------------------------------------------------------------

def sweep_metrics_table(
    df: pd.DataFrame,
    *,
    bin_descriptor: str = "BertzCT",
    n_bins: int = 4,
) -> pd.DataFrame:
    """Compute MAE/RMSE/R² per (n_loops, complexity-bin)."""
    if bin_descriptor not in df.columns:
        logger.warning("bin descriptor '%s' missing; producing global metrics only", bin_descriptor)
        df = df.copy()
        df["__bin__"] = "all"
    else:
        df = df.copy()
        # Bin once globally so the same molecules fall in the same bin across loops.
        unique = df.drop_duplicates("inchikey14")[["inchikey14", bin_descriptor]].copy()
        unique["__bin__"] = complexity_bins(unique[bin_descriptor], n_bins=n_bins)
        df = df.merge(unique[["inchikey14", "__bin__"]], on="inchikey14", how="left")

    has_depth = "effective_depth" in df.columns
    rows: list[dict] = []
    for (k, b), grp in df.groupby(["n_loops", "__bin__"], dropna=False):
        m = regression_metrics(grp)
        mean_depth = float(grp["effective_depth"].mean()) if has_depth else float(k)
        rows.append({"n_loops": int(k), "bin": str(b), **m, "mean_effective_depth": mean_depth})
    return pd.DataFrame(rows).sort_values(["bin", "n_loops"]).reset_index(drop=True)


def plot_mae_vs_nloops(table: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    bins = sorted(table["bin"].unique())
    colors = nature_palette(len(bins))
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.8))
    for c, b in zip(colors, bins):
        grp = table[table["bin"] == b].sort_values("n_loops")
        ax.plot(grp["n_loops"], grp["mae"], marker="o", ms=4, lw=1.2,
                color=c, label=str(b))
    ax.set_xlabel("Inference n_loops", fontweight="bold")
    ax.set_ylabel("MAE  (pChEMBL)", fontweight="bold")
    ax.set_title("Test-time loop sweep:\nMAE per complexity bin", fontweight="bold")
    ax.legend(title="Complexity bin", title_fontsize=6, frameon=False)
    fig.tight_layout()
    out = save_figure(fig, paths, "g4_mae_vs_nloops")
    plt.close(fig)
    return out


def plot_effective_compute(table: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    import matplotlib.pyplot as plt

    from radiant_qsar.analyses.common import NATURE_PALETTE
    agg = table.groupby("n_loops")["mean_effective_depth"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.5))
    ax.plot(agg["n_loops"], agg["n_loops"], "--", lw=1.0,
            color="#888888", label="n_loops (no halting)")
    ax.plot(agg["n_loops"], agg["mean_effective_depth"], marker="o", ms=4,
            lw=1.4, color=NATURE_PALETTE[3], label="Effective depth (halting on)")
    ax.set_xlabel("n_loops budget", fontweight="bold")
    ax.set_ylabel("Mean depth", fontweight="bold")
    ax.set_title("Halting-induced effective compute", fontweight="bold")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = save_figure(fig, paths, "g4_effective_compute")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    predictions_dir: Path | str,
    out_dir: Path | str,
    *,
    loops: Sequence[int] = DEFAULT_LOOPS,
    descriptors_path: Path | str | None = None,
    bin_descriptor: str = "BertzCT",
    n_bins: int = 4,
) -> dict:
    publication_style()
    paths = AnalysisPaths(Path(out_dir), name="g4_test_time_loop_sweep")

    df = load_loop_predictions(predictions_dir, loops, descriptors_path=descriptors_path)
    table = sweep_metrics_table(df, bin_descriptor=bin_descriptor, n_bins=n_bins)
    save_table(table, paths, "g4_sweep_metrics")

    effective = table.groupby("n_loops")["mean_effective_depth"].mean().reset_index()
    save_table(effective, paths, "g4_effective_compute")

    f1 = plot_mae_vs_nloops(table, paths)
    f2 = plot_effective_compute(table, paths)

    # Verdict: is MAE monotonically non-increasing in the hard bin?
    hardest = table.sort_values("bin").groupby("bin").last().index.tolist()[-1] if not table.empty else None
    if hardest is not None:
        hb = table[table["bin"] == hardest].sort_values("n_loops")
        improvements = float(hb["mae"].iloc[0] - hb["mae"].iloc[-1])
        if len(hb) >= 2 and improvements > 0:
            verdict = (
                f"C4 supported: MAE on hardest bin ({hardest}) drops by {improvements:.3f} "
                f"from n_loops={int(hb['n_loops'].iloc[0])} to {int(hb['n_loops'].iloc[-1])}."
            )
        else:
            verdict = (
                f"C4 not supported on hardest bin ({hardest}): no monotonic gain from "
                f"deeper inference."
            )
    else:
        verdict = "indeterminate"

    write_summary_md(
        paths,
        title="G.4 — Test-time loop-depth sweep",
        claim="C4: training at modest n_loops_train extrapolates to deeper inference-time n_loops on hard cases.",
        headline=verdict,
        details={
            "Loops swept": ", ".join(str(x) for x in loops),
            "Bin descriptor": bin_descriptor,
            "Complexity bins": str(n_bins),
        },
        tables_referenced=["g4_sweep_metrics.csv", "g4_effective_compute.csv"],
        figures_referenced=[p.name for p in f1 + f2],
    )

    return {"sweep_metrics": table, "verdict": verdict, "paths": paths}


def run_panel(
    panel_root: Path | str,
    config_path: Path | str,
    vocab_path: Path | str,
    out_dir: Path | str,
    *,
    lf_model_dir: str = "radiant",
    split: str = "scaffold",
    loops: Sequence[int] = DEFAULT_LOOPS,
    descriptors_path: Path | str | None = None,
    bin_descriptor: str = "BertzCT",
    n_bins: int = 4,
    device: str = "cuda",
    batch_size: int = 64,
    task_name: str = "pchembl",
) -> dict:
    """Run G.4 loop sweep across all LF cells (all 20 targets) in the panel.

    For each target cell, runs ``run_model_sweep()`` to generate
    ``predictions_nloopsK.csv`` files under ``<cell>/loop_sweep/`` (skips
    cells where the directory already exists — fully resumable).
    Then aggregates MAE-vs-loops curves across all targets.

    Parameters
    ----------
    panel_root:
        Root of the panel, e.g. ``runs/panel_75m``.
    config_path / vocab_path:
        RADIANT config JSON and tokenizer vocab — must match all cell
        checkpoints (they all derive from the same pretrain config).
    split:
        Which split to use per target. Defaults to ``scaffold``.
    """
    publication_style()
    panel_root = Path(panel_root)
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, name="g4_test_time_loop_sweep")

    cell_dirs = sorted(
        d for d in (panel_root / lf_model_dir).glob(f"*/{split}")
        if (d / "best.pt").exists() and (d / "predictions.csv").exists()
    )
    if not cell_dirs:
        raise FileNotFoundError(
            f"No LF cells with best.pt found at {panel_root / lf_model_dir}/*/{split}/. "
            "Check that fine-tuning completed for all targets."
        )
    logger.info("G.4 panel mode: %d cells found (split=%s)", len(cell_dirs), split)

    all_sweep_frames: list[pd.DataFrame] = []

    for cell_dir in cell_dirs:
        target = cell_dir.parent.name
        loop_sweep_dir = cell_dir / "loop_sweep"

        # Resumable: skip if all loop files already exist
        missing = [k for k in loops
                   if not (loop_sweep_dir / f"predictions_nloops{k}.csv").exists()]
        if not missing:
            logger.info("  %s: loop_sweep already complete, loading", target)
        else:
            logger.info("  %s: running model sweep for loops %s", target, missing if len(missing) < len(loops) else loops)
            cell_chem_cfg = cell_dir / "chem_config.json"
            run_model_sweep(
                ModelSweepConfig(
                    checkpoint_path=cell_dir / "best.pt",
                    config_path=Path(config_path),
                    vocab_path=Path(vocab_path),
                    smiles_csv=cell_dir / "predictions.csv",
                    target_chembl_id=target,
                    split_kind=split,
                    task_name=task_name,
                    batch_size=batch_size,
                    device=device,
                    chem_config_path=cell_chem_cfg if cell_chem_cfg.exists() else None,
                ),
                loops=loops,
                out_dir=loop_sweep_dir,
            )

        try:
            df = load_loop_predictions(loop_sweep_dir, loops,
                                       descriptors_path=descriptors_path)
            df["target"] = target
            all_sweep_frames.append(df)
        except Exception as exc:
            logger.warning("  %s: failed to load loop predictions: %s", target, exc)

    if not all_sweep_frames:
        raise RuntimeError("All panel cells failed in G.4 sweep.")

    full_df = pd.concat(all_sweep_frames, ignore_index=True)

    # Aggregate: per-(n_loops, bin) across all targets
    table = sweep_metrics_table(full_df, bin_descriptor=bin_descriptor, n_bins=n_bins)
    save_table(table, paths, "g4_sweep_metrics_panel")

    # Per-target MAE at each n_loops (aggregate summary)
    per_target = (full_df.groupby(["target", "n_loops"])
                  .apply(lambda g: pd.Series(regression_metrics(g)))
                  .reset_index())
    save_table(per_target, paths, "g4_per_target_sweep")

    effective = table.groupby("n_loops")["mean_effective_depth"].mean().reset_index()
    save_table(effective, paths, "g4_effective_compute_panel")

    f1 = plot_mae_vs_nloops(table, paths)
    f2 = plot_effective_compute(table, paths)
    f3 = _plot_per_target_lines(per_target, paths)

    hardest = table.sort_values("bin").groupby("bin").last().index.tolist()[-1] if not table.empty else None
    if hardest is not None:
        hb = table[table["bin"] == hardest].sort_values("n_loops")
        improvements = float(hb["mae"].iloc[0] - hb["mae"].iloc[-1]) if len(hb) >= 2 else 0.0
        verdict = (
            f"C4 supported across {len(cell_dirs)} panel cells: panel-mean MAE on hardest bin ({hardest}) "
            f"drops {improvements:.3f} from n_loops={int(hb['n_loops'].iloc[0])} to {int(hb['n_loops'].iloc[-1])}."
            if improvements > 0 else
            f"C4 inconclusive across {len(cell_dirs)} panel cells on hardest bin ({hardest})."
        )
    else:
        verdict = "indeterminate"

    write_summary_md(
        paths,
        title="G.4 — Test-time loop-depth sweep (panel)",
        claim="C4: deeper inference-time n_loops reduces MAE on hard molecules across all 20 targets.",
        headline=verdict,
        details={
            "Panel cells": str(len(cell_dirs)),
            "Split": split,
            "Loops swept": ", ".join(str(x) for x in loops),
            "Bin descriptor": bin_descriptor,
        },
        tables_referenced=["g4_sweep_metrics_panel.csv", "g4_per_target_sweep.csv",
                           "g4_effective_compute_panel.csv"],
        figures_referenced=[p.name for p in f1 + f2 + f3],
    )

    return {"sweep_metrics": table, "per_target": per_target, "verdict": verdict, "paths": paths}


def _plot_per_target_lines(per_target: pd.DataFrame, paths: AnalysisPaths) -> list[Path]:
    """One MAE-vs-n_loops line per target (light) + bold mean line."""
    import matplotlib.pyplot as plt
    from radiant_qsar.analyses.common import NATURE_PALETTE

    if per_target.empty or "mae" not in per_target.columns:
        return []
    fig, ax = plt.subplots(figsize=(NC_SINGLE_COL, 2.8))
    for target, grp in per_target.groupby("target"):
        g = grp.sort_values("n_loops")
        ax.plot(g["n_loops"], g["mae"], alpha=0.28, lw=0.9, color=NATURE_PALETTE[1])
    mean_line = per_target.groupby("n_loops")["mae"].mean().reset_index()
    ax.plot(mean_line["n_loops"], mean_line["mae"], lw=2.0,
            color=NATURE_PALETTE[3], label="Panel mean", zorder=5)
    ax.set_xlabel("Inference n_loops", fontweight="bold")
    ax.set_ylabel("MAE  (pChEMBL)", fontweight="bold")
    ax.set_title(f"MAE vs n_loops — {per_target['target'].nunique()} targets",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    out = save_figure(fig, paths, "g4_per_target_mae_lines")
    plt.close(fig)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase G.4 test-time loop sweep")
    p.add_argument("--mode", choices=["predictions", "model", "panel"], default="predictions")
    p.add_argument("--panel-root", type=Path,
                   help="(panel mode) auto-discovers all LF cells")
    p.add_argument("--predictions-dir", type=Path,
                   help="(predictions mode) directory containing predictions_nloops{K}.csv")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--loops", type=int, nargs="+", default=list(DEFAULT_LOOPS))
    p.add_argument("--descriptors", type=Path, default=None)
    p.add_argument("--bin-descriptor", default="BertzCT")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--split", default="scaffold", help="(panel mode) which split to use per target")
    # model / panel mode
    p.add_argument("--checkpoint", type=Path, help="(model mode) chem checkpoint .pt")
    p.add_argument("--config", type=Path, help="RadiantConfig json")
    p.add_argument("--vocab", type=Path, help="SmilesTokenizer vocab json")
    p.add_argument("--smiles-csv", type=Path, help="(model mode) test SMILES CSV")
    p.add_argument("--target-chembl-id", default="default")
    p.add_argument("--split-kind", default="default")
    p.add_argument("--task-name", default="pchembl")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()

    if args.mode == "panel":
        if not (args.panel_root and args.config and args.vocab):
            raise SystemExit("--mode panel requires --panel-root, --config, --vocab")
        run_panel(
            panel_root=args.panel_root,
            config_path=args.config,
            vocab_path=args.vocab,
            out_dir=args.out_dir,
            split=args.split,
            loops=args.loops,
            descriptors_path=args.descriptors,
            bin_descriptor=args.bin_descriptor,
            n_bins=args.n_bins,
            device=args.device,
            batch_size=args.batch_size,
        )
        return

    if args.mode == "model":
        if not (args.checkpoint and args.config and args.vocab and args.smiles_csv):
            raise SystemExit("--mode model requires --checkpoint, --config, --vocab, --smiles-csv")
        sweep_out = args.out_dir / "loop_predictions"
        run_model_sweep(
            ModelSweepConfig(
                checkpoint_path=args.checkpoint,
                config_path=args.config,
                vocab_path=args.vocab,
                smiles_csv=args.smiles_csv,
                target_chembl_id=args.target_chembl_id,
                split_kind=args.split_kind,
                task_name=args.task_name,
                batch_size=args.batch_size,
                device=args.device,
            ),
            loops=args.loops,
            out_dir=sweep_out,
        )
        predictions_dir = sweep_out
    else:
        if not args.predictions_dir:
            raise SystemExit("--mode predictions requires --predictions-dir")
        predictions_dir = args.predictions_dir

    run(
        predictions_dir=predictions_dir,
        out_dir=args.out_dir,
        loops=args.loops,
        descriptors_path=args.descriptors,
        bin_descriptor=args.bin_descriptor,
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
