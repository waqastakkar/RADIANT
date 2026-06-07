"""Sweep driver: run a panel x splits x models grid.

Reads the panel manifest produced by
:mod:`radiant_qsar.finetune.select_panel` and dispatches one
:mod:`radiant_qsar.finetune.single_task` (and one
:mod:`radiant_qsar.baselines.morgan_rf`) run per ``(target, split)``
cell. Results are collected from each cell's ``result.json`` into a
single ``panel_results.csv`` for downstream analysis.

Resumability
------------
Each cell is its own directory under ``--out``; if ``result.json``
already exists we skip the cell. So you can ctrl-C the sweep and pick up
where you left off without losing progress.

Outputs
-------
    out/
        radiant/<TARGET>/<SPLIT>/result.json
        radiant/<TARGET>/<SPLIT>/best.pt
        morgan_rf/<TARGET>/<SPLIT>/result.json
        panel_results.csv      <-- aggregated, one row per (target, split, model)
        panel_summary.json     <-- counts + averages

CLI::

    python -m radiant_qsar.finetune.sweep \\
        --panel        data/processed/v1/panel.json \\
        --activities   data/processed/v1/activities.parquet \\
        --vocab        data/processed/v1/smiles_vocab.json \\
        --config       configs/radiant_75m.json \\
        --pretrain-ckpt checkpoints/pretrain_75m/latest.pt \\
        --out          runs/panel \\
        --splits       random scaffold time cluster activity_cliff \\
        --models       radiant morgan_rf \\
        --device       cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


VALID_SPLITS = ("random", "scaffold", "time", "cluster", "activity_cliff")
VALID_MODELS = (
    "radiant",
    "radiant_no_halting",
    "radiant_no_anchor",
    "radiant_no_adapter",
    "radiant_no_depth_pool",
    "radiant_fixed_loops",
    "radiant_no_smiles_aug",
    "radiant_linear_head",
    "morgan_rf",
    "chemberta",
    "molformer",
    "gin",
)


@dataclass
class SweepConfig:
    panel_path: Path
    activities_path: Path
    vocab_path: Path
    config_path: Path
    pretrain_ckpt: Path | None
    out_dir: Path
    splits: list[str]
    models: list[str]
    epochs: int = 100
    batch_size: int = 16
    lr: float = 2e-5
    n_loops_train: int = 8
    warmup_ratio: float = 0.06
    lr_layer_decay: float = 0.75
    grad_accum_steps: int = 2
    head_warmup_epochs: int = 5
    disable_halting_loss: bool = False
    loss_kind: str = "huber"
    huber_beta: float = 0.5
    regression_head_hidden_dim: int = 512
    regression_head_dropout: float = 0.10
    smiles_augment_prob: float = 0.50
    patience: int = 15
    device: str = "cuda"
    rf_n_estimators: int = 500
    rf_n_jobs: int = -1
    # HF transformer baselines
    chemberta_model_id: str = "DeepChem/ChemBERTa-77M-MLM"
    molformer_model_id: str = "ibm/MoLFormer-XL-both-10pct"
    hf_epochs: int = 30
    hf_batch_size: int = 16
    hf_lr: float = 5e-5
    hf_max_seq_len: int = 256
    # GIN
    gin_epochs: int = 60
    gin_hidden_dim: int = 256
    gin_n_layers: int = 5
    gin_lr: float = 1e-3
    gin_batch_size: int = 64
    seed: int = 1337
    skip_existing: bool = True


def _read_panel(panel_path: Path) -> list[dict]:
    data = json.loads(panel_path.read_text(encoding="utf-8"))
    if "entries" not in data:
        raise ValueError(f"panel file {panel_path} missing 'entries' key")
    return data["entries"]


def _cell_dir(out_dir: Path, model: str, target: str, split: str) -> Path:
    return out_dir / model / target / split


def _result_path(cell: Path) -> Path:
    return cell / "result.json"


def _run(cmd: list[str], cwd: Path | None = None, log_file: Path | None = None) -> int:
    """Run a subprocess; tee stdout/stderr to ``log_file`` if given.

    Capturing per-cell logs is essential for sweep debugging: a silent
    crash that kills 100 cells in a row should produce 100 traceback
    files, not one missing aggregated CSV.
    """
    logger.info("$ %s", " ".join(cmd))
    if log_file is None:
        return subprocess.call(cmd, cwd=cwd)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Buffer in a single open handle so stdout and stderr are interleaved
    # in chronological order.
    with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        fh.write(f"\n[exit code {rc}]\n")
    return rc


def _run_radiant_variant(cell: Path, target: str, split: str, cfg: SweepConfig, variant: str = "radiant") -> int:
    cell.mkdir(parents=True, exist_ok=True)
    n_loops_train = 1 if variant == "radiant_fixed_loops" else cfg.n_loops_train
    smiles_augment_prob = 0.0 if variant == "radiant_no_smiles_aug" else cfg.smiles_augment_prob
    hidden_dim = 0 if variant == "radiant_linear_head" else cfg.regression_head_hidden_dim
    cmd = [
        sys.executable, "-m", "radiant_qsar.finetune.single_task",
        "--activities", str(cfg.activities_path),
        "--target", target,
        "--vocab", str(cfg.vocab_path),
        "--config", str(cfg.config_path),
        "--out", str(cell),
        "--split", split,
        "--epochs", str(cfg.epochs),
        "--batch-size", str(cfg.batch_size),
        "--lr", str(cfg.lr),
        "--n-loops-train", str(n_loops_train),
        "--warmup-ratio", str(cfg.warmup_ratio),
        "--lr-layer-decay", str(cfg.lr_layer_decay),
        "--grad-accum-steps", str(cfg.grad_accum_steps),
        "--head-warmup-epochs", str(cfg.head_warmup_epochs),
        "--loss-kind", cfg.loss_kind,
        "--huber-beta", str(cfg.huber_beta),
        "--regression-head-hidden-dim", str(hidden_dim),
        "--regression-head-dropout", str(cfg.regression_head_dropout),
        "--smiles-augment-prob", str(smiles_augment_prob),
        "--pooling-kind", "attention",
        "--patience", str(cfg.patience),
        "--device", cfg.device,
        "--seed", str(cfg.seed),
    ]
    if variant == "radiant_no_depth_pool":
        cmd.append("--no-depth-pool")
    else:
        cmd.append("--use-depth-pool")
    if cfg.pretrain_ckpt is not None:
        cmd.extend(["--pretrain-ckpt", str(cfg.pretrain_ckpt)])
    if cfg.disable_halting_loss or variant == "radiant_fixed_loops":
        cmd.append("--disable-halting-loss")
    if variant == "radiant_no_halting":
        cmd.append("--disable-halting")
    if variant == "radiant_no_anchor":
        cmd.append("--disable-anchor")
    if variant == "radiant_no_adapter":
        cmd.append("--disable-iteration-adapter")
    return _run(cmd, log_file=cell / "cell.log")


def _run_radiant(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
    return _run_radiant_variant(cell, target, split, cfg, "radiant")


def _make_radiant_variant_runner(variant: str):
    def _runner(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
        return _run_radiant_variant(cell, target, split, cfg, variant)
    return _runner


def _run_morgan_rf(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
    cell.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "radiant_qsar.baselines.morgan_rf",
        "--activities", str(cfg.activities_path),
        "--target", target,
        "--out", str(cell),
        "--split", split,
        "--n-estimators", str(cfg.rf_n_estimators),
        "--n-jobs", str(cfg.rf_n_jobs),
        "--seed", str(cfg.seed),
    ]
    return _run(cmd, log_file=cell / "cell.log")


def _run_chemberta(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
    cell.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "radiant_qsar.baselines.chemberta",
        "--activities", str(cfg.activities_path),
        "--target", target,
        "--out", str(cell),
        "--split", split,
        "--model-id", cfg.chemberta_model_id,
        "--epochs", str(cfg.hf_epochs),
        "--batch-size", str(cfg.hf_batch_size),
        "--lr", str(cfg.hf_lr),
        "--max-seq-len", str(cfg.hf_max_seq_len),
        "--device", cfg.device,
        "--seed", str(cfg.seed),
    ]
    return _run(cmd, log_file=cell / "cell.log")


def _run_molformer(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
    cell.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "radiant_qsar.baselines.molformer",
        "--activities", str(cfg.activities_path),
        "--target", target,
        "--out", str(cell),
        "--split", split,
        "--model-id", cfg.molformer_model_id,
        "--epochs", str(cfg.hf_epochs),
        "--batch-size", str(cfg.hf_batch_size),
        "--lr", str(cfg.hf_lr),
        "--max-seq-len", str(cfg.hf_max_seq_len),
        "--device", cfg.device,
        "--seed", str(cfg.seed),
    ]
    return _run(cmd, log_file=cell / "cell.log")


def _run_gin(cell: Path, target: str, split: str, cfg: SweepConfig) -> int:
    cell.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "radiant_qsar.baselines.gin",
        "--activities", str(cfg.activities_path),
        "--target", target,
        "--out", str(cell),
        "--split", split,
        "--n-layers", str(cfg.gin_n_layers),
        "--hidden-dim", str(cfg.gin_hidden_dim),
        "--epochs", str(cfg.gin_epochs),
        "--batch-size", str(cfg.gin_batch_size),
        "--lr", str(cfg.gin_lr),
        "--device", cfg.device,
        "--seed", str(cfg.seed),
    ]
    return _run(cmd, log_file=cell / "cell.log")


_DISPATCH = {
    "radiant": _run_radiant,
    "radiant_no_halting": _make_radiant_variant_runner("radiant_no_halting"),
    "radiant_no_anchor": _make_radiant_variant_runner("radiant_no_anchor"),
    "radiant_no_adapter": _make_radiant_variant_runner("radiant_no_adapter"),
    "radiant_no_depth_pool": _make_radiant_variant_runner("radiant_no_depth_pool"),
    "radiant_fixed_loops": _make_radiant_variant_runner("radiant_fixed_loops"),
    "radiant_no_smiles_aug": _make_radiant_variant_runner("radiant_no_smiles_aug"),
    "radiant_linear_head": _make_radiant_variant_runner("radiant_linear_head"),
    "morgan_rf": _run_morgan_rf,
    "chemberta": _run_chemberta,
    "molformer": _run_molformer,
    "gin":       _run_gin,
}


def run_sweep(cfg: SweepConfig) -> Path:
    """Iterate every (model, target, split) cell. Returns the path to the aggregated CSV."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    panel = _read_panel(cfg.panel_path)
    n_cells = len(cfg.models) * len(panel) * len(cfg.splits)
    logger.info("sweep plan: %d cells (%d models x %d targets x %d splits)",
                n_cells, len(cfg.models), len(panel), len(cfg.splits))

    t0 = time.time()
    completed, skipped, failed = 0, 0, 0
    for model in cfg.models:
        if model not in _DISPATCH:
            raise ValueError(f"unknown model {model!r}; pick from {VALID_MODELS}")
        for entry in panel:
            target = entry["target_chembl_id"]
            for split in cfg.splits:
                if split not in VALID_SPLITS:
                    raise ValueError(f"unknown split {split!r}; pick from {VALID_SPLITS}")
                cell = _cell_dir(cfg.out_dir, model, target, split)
                rpath = _result_path(cell)
                if cfg.skip_existing and rpath.exists():
                    skipped += 1
                    logger.info("[skip] %s/%s/%s -- %s exists", model, target, split, rpath.name)
                    continue
                logger.info("[run ] %s/%s/%s  (%s, n=%d)", model, target, split,
                            entry["target_class"], entry["n_compounds"])
                rc = _DISPATCH[model](cell, target, split, cfg)
                if rc == 0 and rpath.exists():
                    completed += 1
                else:
                    failed += 1
                    logger.warning("[fail] %s/%s/%s rc=%d  result_exists=%s",
                                   model, target, split, rc, rpath.exists())
    logger.info("sweep done: completed=%d skipped=%d failed=%d in %.1f min",
                completed, skipped, failed, (time.time() - t0) / 60.0)

    return aggregate_results(cfg)


def aggregate_results(cfg: SweepConfig) -> Path:
    """Walk every cell directory, collect ``result.json``, write panel_results.csv + summary."""
    panel = _read_panel(cfg.panel_path)
    rows: list[dict] = []

    for model in cfg.models:
        for entry in panel:
            target = entry["target_chembl_id"]
            for split in cfg.splits:
                cell = _cell_dir(cfg.out_dir, model, target, split)
                rpath = _result_path(cell)
                if not rpath.exists():
                    continue
                try:
                    rec = json.loads(rpath.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning("could not parse %s: %s", rpath, exc)
                    continue

                row = {
                    "model": model,
                    "target_chembl_id": target,
                    "target_class": entry["target_class"],
                    "target_name": entry.get("target_name", ""),
                    "uniprot": entry.get("uniprot", ""),
                    "n_compounds": entry.get("n_compounds", ""),
                    "split": split,
                }
                # Canonical schema is ``{train,val,test}: {...metrics}`` at the top
                # level (or nested under "metrics"). We also accept the legacy
                # ``{val_metrics, test_metrics}`` form so older runs aggregate.
                metrics = rec.get("metrics", rec)
                for partition in ("train", "val", "test"):
                    block = metrics.get(partition)
                    if not isinstance(block, dict):
                        block = metrics.get(f"{partition}_metrics")  # legacy
                    if not isinstance(block, dict):
                        continue
                    for k, v in block.items():
                        row[f"{partition}_{k}"] = v
                # Top-level metrics that sit outside the train/val/test blocks.
                for k in ("best_epoch", "n_train", "n_val", "n_test", "n_params",
                          "wallclock_s"):
                    if k in metrics:
                        row[k] = metrics[k]
                rows.append(row)

    # Stable column order: keys in first-seen order.
    cols: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)

    csv_path = cfg.out_dir / "panel_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # Summary: counts + per-(model, split) test-MAE means.
    summary: dict = {
        "n_panel_targets": len(panel),
        "splits": cfg.splits,
        "models": cfg.models,
        "n_rows": len(rows),
        "by_model_split": {},
    }
    for r in rows:
        key = f"{r['model']}/{r['split']}"
        bucket = summary["by_model_split"].setdefault(key, {"count": 0, "test_mae_sum": 0.0})
        bucket["count"] += 1
        if isinstance(r.get("test_mae"), (int, float)):
            bucket["test_mae_sum"] += float(r["test_mae"])
    for key, b in summary["by_model_split"].items():
        b["test_mae_mean"] = b["test_mae_sum"] / b["count"] if b["count"] else None
        b.pop("test_mae_sum", None)

    (cfg.out_dir / "panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows) and panel_summary.json", csv_path, len(rows))
    return csv_path


# ---------------------------------------------------------------------------
def _parse() -> tuple[SweepConfig, bool]:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--vocab", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--pretrain-ckpt", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=list(VALID_SPLITS),
                   choices=VALID_SPLITS)
    p.add_argument("--models", nargs="+", default=list(VALID_MODELS),
                   choices=VALID_MODELS)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--n-loops-train", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--lr-layer-decay", type=float, default=0.75)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--head-warmup-epochs", type=int, default=5)
    p.add_argument("--disable-halting-loss", action="store_true")
    p.add_argument("--loss-kind", default="huber",
                   choices=("mse", "huber", "smooth_l1"))
    p.add_argument("--huber-beta", type=float, default=0.5)
    p.add_argument("--regression-head-hidden-dim", type=int, default=512)
    p.add_argument("--regression-head-dropout", type=float, default=0.10)
    p.add_argument("--smiles-augment-prob", type=float, default=0.50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rf-n-estimators", type=int, default=500)
    p.add_argument("--rf-n-jobs", type=int, default=-1)
    # HF baselines
    p.add_argument("--chemberta-model-id", default="DeepChem/ChemBERTa-77M-MLM")
    p.add_argument("--molformer-model-id", default="ibm/MoLFormer-XL-both-10pct")
    p.add_argument("--hf-epochs", type=int, default=30)
    p.add_argument("--hf-batch-size", type=int, default=16)
    p.add_argument("--hf-lr", type=float, default=5e-5)
    p.add_argument("--hf-max-seq-len", type=int, default=256)
    # GIN
    p.add_argument("--gin-epochs", type=int, default=60)
    p.add_argument("--gin-hidden-dim", type=int, default=256)
    p.add_argument("--gin-n-layers", type=int, default=5)
    p.add_argument("--gin-lr", type=float, default=1e-3)
    p.add_argument("--gin-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-skip-existing", action="store_true",
                   help="rerun cells that already have result.json")
    p.add_argument("--aggregate-only", action="store_true",
                   help="don't run anything; just collect existing result.json files")
    args = p.parse_args()
    return SweepConfig(
        panel_path=args.panel,
        activities_path=args.activities,
        vocab_path=args.vocab,
        config_path=args.config,
        pretrain_ckpt=args.pretrain_ckpt,
        out_dir=args.out,
        splits=args.splits,
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_loops_train=args.n_loops_train,
        warmup_ratio=args.warmup_ratio,
        lr_layer_decay=args.lr_layer_decay,
        grad_accum_steps=args.grad_accum_steps,
        head_warmup_epochs=args.head_warmup_epochs,
        disable_halting_loss=args.disable_halting_loss,
        loss_kind=args.loss_kind,
        huber_beta=args.huber_beta,
        regression_head_hidden_dim=args.regression_head_hidden_dim,
        regression_head_dropout=args.regression_head_dropout,
        smiles_augment_prob=args.smiles_augment_prob,
        patience=args.patience,
        device=args.device,
        rf_n_estimators=args.rf_n_estimators,
        rf_n_jobs=args.rf_n_jobs,
        chemberta_model_id=args.chemberta_model_id,
        molformer_model_id=args.molformer_model_id,
        hf_epochs=args.hf_epochs,
        hf_batch_size=args.hf_batch_size,
        hf_lr=args.hf_lr,
        hf_max_seq_len=args.hf_max_seq_len,
        gin_epochs=args.gin_epochs,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_n_layers=args.gin_n_layers,
        gin_lr=args.gin_lr,
        gin_batch_size=args.gin_batch_size,
        seed=args.seed,
        skip_existing=not args.no_skip_existing,
    ), args.aggregate_only


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, agg_only = _parse()
    if agg_only:
        aggregate_results(cfg)
    else:
        run_sweep(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
