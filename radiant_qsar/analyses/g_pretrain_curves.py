"""Phase G -- Stage-1 (ZINC20) + Stage-2 (ChEMBL activity-pretrain) curves.

Manuscript Figure 2 material: the three-stage training story.

Stage 1 (ZINC20 self-supervised pretrain). Reads
``checkpoints/pretrain/train_log.jsonl`` (~10k logged steps over the full
~500k training steps) and plots:

* loss curves: MLM, contrastive (3 variants), total -- log-y, EMA-smoothed
* learning rate schedule
* gradient norm
* per-step n_loops (the loop-sampling distribution)
* training throughput (steps/sec)

Stage 2 (ChEMBL activity-pretrain). Reads
``checkpoints/activity_pretrain/result.json["history"]`` and plots:

* val MAE per epoch
* val Pearson per epoch
* train loss per epoch

Headline metrics are written to summary.md so reviewers can audit the
pretraining without having to re-run anything.

Inputs are optional -- this module skips cleanly if either log is absent.
"""
from __future__ import annotations

import argparse
import json
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


def _ema(x: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """Simple exponential-moving-average smoother."""
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _read_zinc20_log(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _plot_zinc20_losses(df: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt

    loss_cols = [c for c in
                 ["loss_total", "loss_mlm", "loss_contrastive",
                  "loss_scaffold_contrastive", "loss_rgroup_contrastive",
                  "loss_rgroup_mlm"] if c in df.columns]
    if not loss_cols:
        return
    colors = nature_palette(len(loss_cols))
    fig, ax = plt.subplots(figsize=(NC_DOUBLE_COL * 0.7, 3.4))
    for c, col in zip(colors, loss_cols):
        vals = df[col].astype(float).to_numpy()
        mask = np.isfinite(vals) & (vals > 0)
        if not mask.any():
            continue
        steps = df.loc[mask, "step"].to_numpy()
        smoothed = _ema(vals[mask])
        ax.plot(steps, vals[mask], color=c, alpha=0.15, lw=0.5)
        ax.plot(steps, smoothed, color=c, lw=1.4, label=col.replace("loss_", ""))
    ax.set_yscale("log")
    ax.set_xlabel("Training step", fontweight="bold")
    ax.set_ylabel("Loss (log scale)", fontweight="bold")
    ax.set_title("Stage 1: ZINC20 self-supervised pretrain -- loss curves",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    fig.tight_layout()
    save_figure(fig, paths, "g_pretrain_zinc20_loss")
    plt.close(fig)


def _plot_zinc20_lr_grad(df: pd.DataFrame, paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(NC_DOUBLE_COL, 2.6))
    colors = nature_palette(3)

    if "lr" in df.columns:
        ax = axes[0]
        ax.plot(df["step"], df["lr"], color=colors[0], lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Step", fontweight="bold")
        ax.set_ylabel("Learning rate", fontweight="bold")
        ax.set_title("LR schedule", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    if "grad_norm" in df.columns:
        ax = axes[1]
        vals = df["grad_norm"].astype(float).to_numpy()
        mask = np.isfinite(vals) & (vals > 0)
        ax.plot(df.loc[mask, "step"], vals[mask], color=colors[1],
                alpha=0.25, lw=0.5)
        ax.plot(df.loc[mask, "step"], _ema(vals[mask]),
                color=colors[1], lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("Step", fontweight="bold")
        ax.set_ylabel("Grad norm", fontweight="bold")
        ax.set_title("Gradient norm", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    if "n_loops" in df.columns:
        ax = axes[2]
        # Histogram of sampled n_loops across training
        counts = df["n_loops"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=colors[2], edgecolor="none")
        ax.set_xlabel("Loop count", fontweight="bold")
        ax.set_ylabel("Step count (log)", fontweight="bold")
        ax.set_yscale("log")
        ax.set_title("Loop sampling during pretrain",
                     fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle("Stage 1: ZINC20 pretrain diagnostics",
                 fontweight="bold", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, paths, "g_pretrain_zinc20_diagnostics")
    plt.close(fig)


def _plot_activity_pretrain(history: list[dict], paths: AnalysisPaths) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return
    df = pd.DataFrame(history)
    if df.empty:
        return
    colors = nature_palette(3)
    fig, axes = plt.subplots(1, 3, figsize=(NC_DOUBLE_COL, 2.8))

    if "train_loss" in df.columns:
        ax = axes[0]
        ax.plot(df["epoch"], df["train_loss"], "-o",
                color=colors[0], lw=1.6, ms=4)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Train loss", fontweight="bold")
        ax.set_title("Stage 2 train loss", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    if "val_mae" in df.columns:
        ax = axes[1]
        ax.plot(df["epoch"], df["val_mae"], "-o",
                color=colors[1], lw=1.6, ms=4)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Val MAE", fontweight="bold")
        ax.set_title("Stage 2 val MAE", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    if "val_pearson" in df.columns:
        ax = axes[2]
        ax.plot(df["epoch"], df["val_pearson"], "-o",
                color=colors[2], lw=1.6, ms=4)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Val Pearson r", fontweight="bold")
        ax.set_title("Stage 2 val Pearson", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle("Stage 2: ChEMBL activity-pretrain",
                 fontweight="bold", y=1.02, fontsize=10)
    fig.tight_layout()
    save_figure(fig, paths, "g_pretrain_activity")
    plt.close(fig)


def _plot_three_stage_overview(zinc_df: pd.DataFrame, act_history: list[dict],
                               ft_best_csv: Path | None,
                               paths: AnalysisPaths) -> None:
    """Single 3-panel figure summarising the three training stages."""
    import matplotlib.pyplot as plt
    colors = nature_palette(3)

    fig, axes = plt.subplots(1, 3, figsize=(NC_DOUBLE_COL, 3.0))

    # Stage 1: ZINC20 total loss
    if not zinc_df.empty and "loss_total" in zinc_df.columns:
        ax = axes[0]
        vals = zinc_df["loss_total"].astype(float).to_numpy()
        mask = np.isfinite(vals) & (vals > 0)
        ax.plot(zinc_df.loc[mask, "step"], vals[mask],
                color=colors[0], alpha=0.2, lw=0.4)
        ax.plot(zinc_df.loc[mask, "step"], _ema(vals[mask]),
                color=colors[0], lw=1.5)
        ax.set_xlabel("ZINC20 step", fontweight="bold")
        ax.set_ylabel("Total loss", fontweight="bold")
        ax.set_yscale("log")
        ax.set_title("Stage 1: ZINC20 pretrain", fontweight="bold", fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.4)

    # Stage 2: activity-pretrain val MAE
    if act_history:
        df = pd.DataFrame(act_history)
        if "val_mae" in df.columns:
            ax = axes[1]
            ax.plot(df["epoch"], df["val_mae"], "-o",
                    color=colors[1], lw=1.6, ms=4)
            ax.set_xlabel("ChEMBL epoch", fontweight="bold")
            ax.set_ylabel("Val MAE (pChEMBL)", fontweight="bold")
            ax.set_title("Stage 2: activity-pretrain",
                         fontweight="bold", fontsize=9)
            ax.grid(True, alpha=0.3, lw=0.4)

    # Stage 3: fine-tune best-epoch val MAE across 20 targets
    if ft_best_csv is not None and ft_best_csv.exists():
        bdf = pd.read_csv(ft_best_csv)
        if {"val_mae", "target", "split"}.issubset(bdf.columns):
            ax = axes[2]
            scaff = bdf[bdf["split"] == "scaffold"].sort_values("val_mae")
            if not scaff.empty:
                ax.bar(range(len(scaff)), scaff["val_mae"].values,
                       color=colors[2], edgecolor="none")
                ax.set_xticks(range(len(scaff)))
                ax.set_xticklabels(scaff["target"].values, rotation=80,
                                   fontsize=6)
                ax.set_xlabel("Target (scaffold split)", fontweight="bold")
                ax.set_ylabel("Best val MAE (pChEMBL)", fontweight="bold")
                ax.set_title("Stage 3: per-target fine-tune",
                             fontweight="bold", fontsize=9)
                ax.grid(True, axis="y", alpha=0.3, lw=0.4)

    fig.suptitle("Three-stage training pipeline",
                 fontweight="bold", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, paths, "g_pretrain_three_stage_overview")
    plt.close(fig)


def run(
    *,
    out_dir: Path | str,
    zinc20_log: Path | str = "checkpoints/pretrain/train_log.jsonl",
    activity_pretrain_result: Path | str = "checkpoints/activity_pretrain/result.json",
    finetune_best_csv: Path | str | None = None,
) -> dict:
    publication_style()
    out_dir = Path(out_dir)
    paths = AnalysisPaths(out_dir, "g_pretrain_curves")

    zinc20_log = Path(zinc20_log)
    activity_pretrain_result = Path(activity_pretrain_result)

    headlines = []

    # Stage 1: ZINC20 ----------------------------------------------------
    zinc_df = pd.DataFrame()
    if zinc20_log.exists():
        zinc_df = _read_zinc20_log(zinc20_log)
        save_table(zinc_df, paths, "zinc20_train_log")
        if not zinc_df.empty:
            _plot_zinc20_losses(zinc_df, paths)
            _plot_zinc20_lr_grad(zinc_df, paths)
            final_total = float(zinc_df["loss_total"].iloc[-1]) if "loss_total" in zinc_df.columns else float("nan")
            final_mlm = float(zinc_df["loss_mlm"].iloc[-1]) if "loss_mlm" in zinc_df.columns else float("nan")
            total_steps = int(zinc_df["step"].max()) if "step" in zinc_df.columns else len(zinc_df)
            total_hours = float(zinc_df["elapsed_s"].max()) / 3600.0 if "elapsed_s" in zinc_df.columns else float("nan")
            headlines.append(
                f"Stage 1 (ZINC20): {total_steps:,} training steps "
                f"({total_hours:.1f} h wall), final total loss = {final_total:.3f}, "
                f"final MLM loss = {final_mlm:.3f}.")
    else:
        logger.warning("ZINC20 log not found at %s", zinc20_log)

    # Stage 2: activity-pretrain -----------------------------------------
    act_history: list[dict] = []
    if activity_pretrain_result.exists():
        rec = json.loads(activity_pretrain_result.read_text(encoding="utf-8"))
        act_history = rec.get("history") or rec.get("train_history") or []
        if act_history:
            adf = pd.DataFrame(act_history)
            save_table(adf, paths, "activity_pretrain_history")
            _plot_activity_pretrain(act_history, paths)
            best_mae = float(adf["val_mae"].min()) if "val_mae" in adf else float("nan")
            best_pe = float(adf["val_pearson"].max()) if "val_pearson" in adf else float("nan")
            n_targets = rec.get("n_targets", "?")
            headlines.append(
                f"Stage 2 (ChEMBL activity-pretrain): {len(act_history)} epochs "
                f"on {n_targets} targets, best val MAE = {best_mae:.3f}, "
                f"best val Pearson = {best_pe:.3f}.")
    else:
        logger.warning("activity-pretrain result not found at %s", activity_pretrain_result)

    # Stage 3 overlay: fine-tune best epoch from g_training_curves -------
    ft_best = None
    if finetune_best_csv is not None:
        ft_best = Path(finetune_best_csv)
    else:
        cand = out_dir / "g_training_curves" / "tables" / "training_curves_best_epoch.csv"
        if cand.exists():
            ft_best = cand

    _plot_three_stage_overview(zinc_df, act_history, ft_best, paths)

    if not headlines:
        headlines.append(
            "No pretrain artifacts found; nothing to plot. Expected "
            "checkpoints/pretrain/train_log.jsonl and "
            "checkpoints/activity_pretrain/result.json.")

    write_summary_md(
        paths,
        title="Pretraining curves (ZINC20 + ChEMBL activity)",
        claim=("The three-stage pretrain story: ZINC20 self-supervised "
               "pretrain -> ChEMBL activity-pretrain -> per-target fine-tune. "
               "Each stage's loss / metric trajectory is plotted from its "
               "own logged history -- no model is re-run."),
        headline="  ".join(headlines),
        details={
            "ZINC20 log": str(zinc20_log) if zinc20_log.exists() else "not found",
            "Activity-pretrain": str(activity_pretrain_result) if activity_pretrain_result.exists() else "not found",
            "Stage-3 best-epoch table": str(ft_best) if ft_best else "not found",
        },
        tables_referenced=[
            "zinc20_train_log.csv",
            "activity_pretrain_history.csv",
        ],
        figures_referenced=[
            "g_pretrain_zinc20_loss.png",
            "g_pretrain_zinc20_diagnostics.png",
            "g_pretrain_activity.png",
            "g_pretrain_three_stage_overview.png",
        ],
    )
    return {"paths": paths, "zinc_df": zinc_df, "act_history": act_history}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--zinc20-log", type=Path,
                   default=Path("checkpoints/pretrain/train_log.jsonl"))
    p.add_argument("--activity-pretrain-result", type=Path,
                   default=Path("checkpoints/activity_pretrain/result.json"))
    p.add_argument("--finetune-best-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    args = _parse_args()
    run(out_dir=args.out_dir, zinc20_log=args.zinc20_log,
        activity_pretrain_result=args.activity_pretrain_result,
        finetune_best_csv=args.finetune_best_csv)


if __name__ == "__main__":
    main()
