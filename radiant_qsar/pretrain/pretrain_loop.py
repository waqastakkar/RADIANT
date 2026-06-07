"""End-to-end pretraining driver.

Wires :class:`CompoundCorpusDataset` + :class:`MLMContrastiveCollator` +
:func:`combined_pretrain_loss` into the existing :class:`training.Trainer`,
plus checkpointing and resumption.

CLI::

    python -m radiant_qsar.pretrain.pretrain_loop \\
        --compounds data/processed/v1/compounds.parquet \\
        --vocab     data/processed/v1/smiles_vocab.json \\
        --config    configs/radiant_75m.json \\
        --out       checkpoints/pretrain \\
        --steps     200000 --batch-size 64 --lr 3e-4

The driver is deliberately framework-light: AdamW, cosine schedule,
gradient clipping at 1.0, bf16 on CUDA, checkpoint every N steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _jsonable_args(args: argparse.Namespace) -> dict:
    """Return ``vars(args)`` with Path values stringified so the dict is safe
    to embed in a torch checkpoint (PyTorch >= 2.6 ``weights_only=True`` rejects
    pickled ``pathlib.PosixPath`` / ``WindowsPath`` objects)."""
    out = {}
    for k, v in vars(args).items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--compounds", type=Path, default=None,
                   help="ChEMBL compounds.parquet (original Stage 1 source)")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Plain-text SMILES corpus (one per line, e.g. from zinc_corpus.py build). "
                        "Mutually exclusive with --compounds for large-scale pretraining.")
    p.add_argument("--vocab", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path,
                   help="JSON RadiantConfig produced by tiny/small/base presets")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--mlm-mask-prob", type=float, default=0.15)
    p.add_argument("--mlm-weight", type=float, default=1.0)
    p.add_argument("--contrastive-weight", type=float, default=0.1)
    p.add_argument("--scaffold-contrastive-weight", type=float, default=0.05)
    p.add_argument("--rgroup-contrastive-weight", type=float, default=0.05)
    p.add_argument("--rgroup-mlm-weight", type=float, default=0.25)
    p.add_argument("--disable-rgroup-chemistry", action="store_true",
                   help="Disable Stage-1 Murcko scaffold/R-group auxiliary chemistry views.")
    p.add_argument("--contrastive-temperature", type=float, default=0.07)
    p.add_argument("--n-loops-train", type=int, default=4,
                   help="loops used during training (default 4)")
    p.add_argument("--max-loops", type=int, default=None,
                   help="upper bound for inference-time n_loops; overrides the value in --config. "
                        "Set to >= the deepest n_loops you plan to evaluate at (e.g. 24 if you "
                        "want to test sub-claim C4 at n_loops=16). Default: keep config value.")
    p.add_argument("--min-loops", type=int, default=None,
                   help="lower bound; overrides --config. Default: keep config value.")
    p.add_argument("--iteration-signal-kind", default=None,
                   choices=("none", "sinusoidal", "learned", "both"),
                   help="overrides --config. Use 'sinusoidal' to allow inference at n_loops "
                        "beyond max_loops (extrapolation-safe). Default: keep config value.")
    p.add_argument("--curriculum-loops", action="store_true",
                   help="anneal the UPPER loop bound 2 -> n_loops_train over the first third of "
                        "training (the lower bound stays at --min-loops-train)")
    p.add_argument("--loop-sampling", default="range", choices=("range", "fixed"),
                   help="'range' (default, recommended): sample n_loops uniformly in "
                        "[min_loops_train, current_upper] every step so the model learns a "
                        "depth-robust recurrence and never gets a cold-depth gradient shock when "
                        "the curriculum advances. 'fixed': legacy behavior — exactly one loop "
                        "count per step (brittle; causes contrastive collapse on depth increase).")
    p.add_argument("--min-loops-train", type=int, default=2,
                   help="lower bound for 'range' loop sampling (default 2).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true",
                   help="autocast in bf16 on CUDA (recommended for >=Ampere)")
    p.add_argument("--resume-from", type=Path, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Cosine schedule with linear warmup
# ---------------------------------------------------------------------------
def _lr_at(step: int, *, lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # --- Tokenizer + dataset --------------------------------------------
    from radiant_chem import RadiantChemConfig, RadiantChemModel, SmilesTokenizer
    from radiant import RadiantConfig
    from radiant_qsar.pretrain.collator import MLMContrastiveCollator
    from radiant_qsar.pretrain.corpus import CompoundCorpusDataset
    from radiant_qsar.pretrain.objective import combined_pretrain_loss

    tokenizer = SmilesTokenizer.load(args.vocab)
    logger.info("vocab size: %d", tokenizer.vocab_size)

    overrides = dict(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=args.max_len,
        n_loops_train=args.n_loops_train,
    )
    if args.max_loops is not None:
        overrides["max_loops"] = args.max_loops
    if args.min_loops is not None:
        overrides["min_loops"] = args.min_loops
    if args.iteration_signal_kind is not None:
        overrides["iteration_signal_kind"] = args.iteration_signal_kind
    base_cfg = RadiantConfig.from_json(args.config).replace(**overrides)
    logger.info(
        "loop config: n_loops_train=%d  min_loops=%d  max_loops=%d  signal=%s",
        base_cfg.n_loops_train, base_cfg.min_loops, base_cfg.max_loops,
        base_cfg.iteration_signal_kind,
    )
    chem_cfg = RadiantChemConfig(base=base_cfg, mlm_mask_prob=args.mlm_mask_prob)
    model = RadiantChemModel(chem_cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model params: %s", f"{n_params:,}")

    # Persist the configs so the checkpoint is self-describing.
    base_cfg.to_json(args.out / "base_config.json")
    chem_cfg.to_json(args.out / "chem_config.json")

    # --- Dataset: parquet (ChEMBL) or plain-text corpus (ZINC+ChEMBL) ---
    is_streaming = False
    if args.corpus is not None:
        from radiant_qsar.pretrain.zinc_corpus import ZincInMemoryDataset, ZincStreamingDataset
        import os
        corpus_size = os.path.getsize(args.corpus)
        # Use streaming for large corpora (>2GB ≈ >50M molecules)
        if corpus_size > 2_000_000_000:
            dataset = ZincStreamingDataset(
                args.corpus, return_augmented_pair=True,
                return_chemistry_views=not args.disable_rgroup_chemistry,
                shuffle_buffer=500_000,
            )
            is_streaming = True
            logger.info("using STREAMING corpus: %s (%.1f GB) — too large for RAM",
                        args.corpus, corpus_size / 1e9)
        else:
            dataset = ZincInMemoryDataset(
                args.corpus, return_augmented_pair=True,
                return_chemistry_views=not args.disable_rgroup_chemistry,
            )
            logger.info("using in-memory corpus: %s (%d molecules)", args.corpus, len(dataset))
    elif args.compounds is not None:
        dataset = CompoundCorpusDataset(
            parquet_path=args.compounds,
            return_augmented_pair=True,
            return_chemistry_views=not args.disable_rgroup_chemistry,
        )
    else:
        raise ValueError("must provide either --compounds or --corpus")
    collator = MLMContrastiveCollator(
        tokenizer=tokenizer,
        max_len=args.max_len,
        mlm_mask_prob=args.mlm_mask_prob,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(not is_streaming),   # IterableDataset handles its own shuffling
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(args.device != "cpu"),
        drop_last=True,
    )

    # --- Optimizer ------------------------------------------------------
    from radiant.utils import split_param_groups

    param_groups = split_param_groups(model, weight_decay=args.weight_decay)
    optim = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # --- Resume ---------------------------------------------------------
    start_step = 0
    if args.resume_from is not None and args.resume_from.exists():
        # We trust our own checkpoints; weights_only=False is needed because
        # the ckpt also carries the optimizer state and an `args` dict.
        ckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt["step"])
        logger.info("resumed from %s @ step %d", args.resume_from, start_step)

        # The LR cosine schedule and the loop schedule are both functions of
        # `step` *relative to* the original (--steps, --warmup-steps,
        # --n-loops-train, --curriculum-loops, --loop-sampling, --min-loops-train).
        # If those differ on resume, the schedule discontinuously jumps -- which
        # is exactly what corrupts a resumed run. Restore them from the
        # checkpoint's saved args unless the user passed an explicit override,
        # and warn loudly on any mismatch.
        saved = ckpt.get("args", {})
        _passed = set(sys.argv[1:])
        def _flag_passed(*names: str) -> bool:
            return any(n in _passed for n in names)
        _schedule_args = {
            "steps": ("--steps",),
            "warmup_steps": ("--warmup-steps",),
            "n_loops_train": ("--n-loops-train",),
            "curriculum_loops": ("--curriculum-loops",),
            "loop_sampling": ("--loop-sampling",),
            "min_loops_train": ("--min-loops-train",),
        }
        for attr, flags in _schedule_args.items():
            if attr not in saved:
                continue
            old, new = saved[attr], getattr(args, attr, None)
            if _flag_passed(*flags):
                if old != new:
                    logger.warning(
                        "resume: %s overridden on CLI (%r -> %r); schedule will shift -- "
                        "this is the classic 'restart makes it worse' footgun.",
                        attr, old, new,
                    )
            elif old != new:
                logger.info("resume: restoring %s from checkpoint (%r, was %r on CLI)",
                            attr, old, new)
                setattr(args, attr, old)

    # --- Loop counts schedule --------------------------------------------
    # Sampling a *range* of depths every step (default) is the key stability
    # fix: the model learns a depth-robust recurrence instead of overfitting a
    # single depth, and advancing the curriculum can never inject a cold,
    # never-trained loop index at full LR (the gradient shock that collapses the
    # contrastive heads). The curriculum only anneals the UPPER bound.
    loop_rng = random.Random(args.seed + 1)

    def _upper_bound(step: int) -> int:
        if not args.curriculum_loops:
            return args.n_loops_train
        cutoff = max(1, args.steps // 3)
        if step >= cutoff:
            return args.n_loops_train
        frac = step / cutoff
        lo = max(1, args.min_loops_train)
        return max(lo, int(round(lo + frac * (args.n_loops_train - lo))))

    def n_loops_for(step: int) -> int:
        hi = _upper_bound(step)
        if args.loop_sampling == "fixed":
            return hi
        lo = min(max(1, args.min_loops_train), hi)
        return loop_rng.randint(lo, hi)

    # --- Train loop -----------------------------------------------------
    model.train()
    t0 = time.time()
    iter_loader = iter(loader)
    log_path = args.out / "train_log.jsonl"

    use_amp = args.bf16 and args.device.startswith("cuda")
    amp_dtype = torch.bfloat16

    with open(log_path, "a", encoding="utf-8") as log_f:
        for step in range(start_step, args.steps):
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                batch = next(iter_loader)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(args.device, non_blocking=True)

            _set_lr(optim, _lr_at(step, lr=args.lr, warmup=args.warmup_steps, total=args.steps))
            optim.zero_grad(set_to_none=True)

            n_loops = n_loops_for(step)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, metrics = combined_pretrain_loss(
                        model, batch,
                        n_loops=n_loops,
                        mlm_weight=args.mlm_weight,
                        contrastive_weight=args.contrastive_weight,
                        scaffold_contrastive_weight=args.scaffold_contrastive_weight,
                        rgroup_contrastive_weight=args.rgroup_contrastive_weight,
                        rgroup_mlm_weight=args.rgroup_mlm_weight,
                        contrastive_temperature=args.contrastive_temperature,
                    )
            else:
                loss, metrics = combined_pretrain_loss(
                    model, batch,
                    n_loops=n_loops,
                    mlm_weight=args.mlm_weight,
                    contrastive_weight=args.contrastive_weight,
                    scaffold_contrastive_weight=args.scaffold_contrastive_weight,
                    rgroup_contrastive_weight=args.rgroup_contrastive_weight,
                    rgroup_mlm_weight=args.rgroup_mlm_weight,
                    contrastive_temperature=args.contrastive_temperature,
                )

            # Safety net: never let a single non-finite step poison the weights
            # or the AdamW moments. A NaN/Inf loss (or grad) is dropped -- we
            # zero the grads and move on rather than calling optim.step().
            if not torch.isfinite(loss):
                logger.warning("step=%d: non-finite loss (%.4f) -- skipping update",
                               step, float(loss.detach()))
                optim.zero_grad(set_to_none=True)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                logger.warning("step=%d: non-finite grad norm -- skipping update", step)
                optim.zero_grad(set_to_none=True)
                continue
            optim.step()

            if step % args.log_every == 0:
                secs = time.time() - t0
                line = {
                    "step": step,
                    "n_loops": n_loops,
                    "lr": optim.param_groups[0]["lr"],
                    "grad_norm": round(float(grad_norm), 4),
                    "elapsed_s": round(secs, 1),
                    **metrics,
                }
                log_f.write(json.dumps(line) + "\n")
                log_f.flush()
                logger.info(
                    "step=%d  loops=%d  lr=%.2e  loss=%.4f  mlm=%.4f  ctr=%.4f  t=%.0fs",
                    step, n_loops, line["lr"], metrics.get("loss_total", 0.0),
                    metrics.get("loss_mlm", 0.0), metrics.get("loss_contrastive", 0.0), secs,
                )

            if step > 0 and step % args.checkpoint_every == 0:
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "args": _jsonable_args(args),
                }
                ckpt_path = args.out / f"ckpt_step_{step:08d}.pt"
                torch.save(ckpt, ckpt_path)
                # also keep a "latest" symlink-equivalent
                torch.save(ckpt, args.out / "latest.pt")
                logger.info("checkpoint saved: %s", ckpt_path)

    final_path = args.out / "final.pt"
    torch.save({"step": args.steps, "model": model.state_dict()}, final_path)
    logger.info("done; final checkpoint: %s", final_path)


if __name__ == "__main__":
    main()
