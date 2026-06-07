"""Single-target fine-tuning for QSAR.

Given a curated ``activities.parquet`` and a target ChEMBL ID, fine-tune
a :class:`RadiantChemModel` on the regression of ``pchembl`` against
canonical SMILES, evaluate on val/test, and save metrics + checkpoint.

The script is the per-target side of the headline study; the multi-task
hub uses a different driver (``finetune.multi_task``, future work).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class SingleTaskTrainArgs:
    activities: Path
    target_chembl_id: str
    vocab: Path
    config: Path
    out: Path
    pretrain_ckpt: Path | None = None
    split_kind: str = "scaffold"          # random | scaffold | time | cluster | activity_cliff
    n_loops_train: int = 8
    epochs: int = 100
    batch_size: int = 16
    lr: float = 2e-5
    weight_decay: float = 0.01
    early_stopping_patience: int = 15
    warmup_ratio: float = 0.06           # fraction of total steps for linear warmup
    lr_layer_decay: float = 0.75          # multiplicative per-layer LR decay (1.0 = off)
    grad_accum_steps: int = 2             # effective batch = batch_size × grad_accum_steps
    head_warmup_epochs: int = 5           # freeze backbone, train only task head for N epochs
    disable_halting_loss: bool = False    # override config to turn off PonderNet during fine-tune
    disable_halting: bool = False         # ablation: remove halting head entirely
    loss_kind: str = "huber"
    huber_beta: float = 0.5
    regression_head_hidden_dim: int = 512
    regression_head_dropout: float = 0.10
    smiles_augment_prob: float = 0.50
    disable_anchor: bool = False
    disable_iteration_adapter: bool = False
    # --- Architecture: pure deep learning, no fingerprints ---
    pooling_kind: str = "attention"       # learnable attention pooling
    fingerprint_dim: int = 0              # 0 = disabled (ablation only)
    fingerprint_radius: int = 2           # Morgan FP radius (ablation only)
    use_depth_adaptive_pool: bool = True  # depth-weighted halting pooling
    seed: int = 1337
    device: str = "cpu"
    log_every: int = 25
    splits_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    activity_cliff_sim: float = 0.9
    activity_cliff_delta: float = 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class _ActivityDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        smiles,
        pchembl,
        tokenizer,
        max_len: int,
        *,
        augment_prob: float = 0.0,
        fingerprints: np.ndarray | None = None,
    ):
        self.smiles = list(smiles)
        self.pchembl = list(pchembl)
        self.tokenizer = tokenizer
        self.max_len = max_len
        if not (0.0 <= augment_prob <= 1.0):
            raise ValueError("augment_prob must be in [0, 1]")
        self.augment_prob = float(augment_prob)
        self.fingerprints = fingerprints  # (N, fp_dim) or None

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment_prob > 0.0 and np.random.random() < self.augment_prob:
            from radiant_chem.augment import randomize_smiles

            smi = randomize_smiles(smi)
        ids = self.tokenizer.encode(smi, max_len=self.max_len)
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "target": torch.tensor(self.pchembl[idx], dtype=torch.float32),
        }
        if self.fingerprints is not None:
            item["fingerprint"] = torch.from_numpy(self.fingerprints[idx])
        return item


def _collate(batch, pad_id: int):
    L = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    targets = torch.stack([b["target"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = 1
    out = {"input_ids": ids, "attention_mask": attn, "targets": targets}
    if "fingerprint" in batch[0]:
        out["fingerprints"] = torch.stack([b["fingerprint"] for b in batch])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    err = pred - true
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - np.var(err) / max(np.var(true), 1e-12)),
        "pearson": float(pearsonr(pred, true).statistic) if len(pred) > 2 else float("nan"),
        "spearman": float(spearmanr(pred, true).statistic) if len(pred) > 2 else float("nan"),
        "n": int(len(pred)),
    }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _select_target(activities_path: Path, target_chembl_id: str):
    import pandas as pd

    df = pd.read_parquet(activities_path)
    sub = df[df["target_chembl_id"] == target_chembl_id].copy()
    if len(sub) == 0:
        raise SystemExit(f"no rows for target {target_chembl_id} in {activities_path}")
    return sub


def _preflight_state_dict_match(
    model,
    state: dict,
    *,
    ckpt_path: Path,
    config_path: Path,
    min_match_fraction: float = 0.90,
) -> None:
    """Fail fast if the pretrain checkpoint architecture doesn't match the fine-tune config.

    The check validates the *backbone* -- the tensors that are supposed to be
    loaded from the checkpoint -- not the whole model. Fine-tuning legitimately
    adds new, randomly-initialized modules on top of a pretrained backbone:
    attention/depth-adaptive pooling (``pool.*`` / ``depth_pool.*``), the
    fingerprint projection (``fp_proj.*``), and the task head(s)
    (``task_heads.*``). Those are *expected* to be absent from a backbone-only
    checkpoint (e.g. ``backbone_for_finetune.pt``, which Stage 2 saves with mean
    pooling and no task heads), so counting them as "missing" against the total
    falsely trips the threshold.

    A genuine pretrain/fine-tune architecture mismatch instead shows up as
    *shape mismatches* (a key present in both but with different shape) or as
    *missing backbone tensors* (``core.*`` absent from the checkpoint). We abort
    only on those. New-head tensors that the checkpoint doesn't carry are
    reported as fresh-init, not errors. (``strict=False`` still does the load;
    this check only decides when to refuse a clearly-wrong checkpoint.)
    """
    import logging

    # Tensors that are allowed to be absent from a backbone-only checkpoint:
    # they are added by the fine-tune config and trained from scratch.
    NEW_HEAD_PREFIXES = ("pool.", "depth_pool.", "fp_proj.", "task_heads.")

    expected = dict(model.state_dict())

    def _is_new_head(key: str) -> bool:
        return any(key.startswith(p) for p in NEW_HEAD_PREFIXES)

    # "Backbone-expected" = every model tensor we require the checkpoint to
    # provide: all non-head tensors, plus any head tensor the checkpoint
    # happens to carry (so a checkpoint that *does* ship pooling/head weights is
    # still validated for shape).
    backbone_keys = [k for k in expected if (not _is_new_head(k)) or (k in state)]

    n_backbone = len(backbone_keys)
    n_shape_ok = sum(1 for k in backbone_keys if k in state and state[k].shape == expected[k].shape)
    n_shape_mismatch = sum(1 for k in backbone_keys if k in state and state[k].shape != expected[k].shape)
    n_backbone_missing = sum(1 for k in backbone_keys if k not in state)
    n_fresh_heads = sum(1 for k in expected if _is_new_head(k) and k not in state)

    match_rate = n_shape_ok / max(n_backbone, 1)
    summary = (
        f"checkpoint {ckpt_path}\n"
        f"  config                 : {config_path}\n"
        f"  backbone tensors        : {n_backbone}\n"
        f"  shape-matched           : {n_shape_ok}  ({match_rate:.1%} of backbone)\n"
        f"  in ckpt but wrong shape : {n_shape_mismatch}\n"
        f"  backbone tensors missing: {n_backbone_missing}\n"
        f"  new heads (fresh init)  : {n_fresh_heads}  (pooling/task heads added by fine-tune config -- expected)\n"
    )
    if match_rate < min_match_fraction:
        offenders: list[str] = []
        for k in backbone_keys:
            if k in state and state[k].shape != expected[k].shape:
                offenders.append(f"    {k}: model={tuple(expected[k].shape)}  ckpt={tuple(state[k].shape)}")
            elif k not in state:
                offenders.append(f"    {k}: MISSING from checkpoint (model={tuple(expected[k].shape)})")
            if len(offenders) >= 8:
                break
        offender_lines = "\n".join(offenders) if offenders else "    (none)"
        raise SystemExit(
            f"PRETRAIN/FINE-TUNE ARCHITECTURE MISMATCH "
            f"({match_rate:.1%} of backbone matched, < {min_match_fraction:.0%} threshold):\n"
            f"{summary}"
            f"  offending backbone tensors:\n{offender_lines}\n\n"
            f"  Most likely cause: the --config you passed differs from the one used at pretrain.\n"
            f"  The architecture is baked into the saved tensors -- pretrain config = fine-tune config = inference config."
        )
    logging.getLogger(__name__).info(
        "preflight ckpt match: %.1f%% of backbone (%d/%d); %d new head tensors fresh-initialized",
        match_rate * 100, n_shape_ok, n_backbone, n_fresh_heads,
    )


def _split(sub, kind: str, ratios, seed, *, sim, delta):
    """Cache-aware wrapper around the underlying split functions.

    The first call for a (target, split, seed) triple computes the indices
    and persists them to ``data/splits/v1/<target>/<split>__seed<N>.json``;
    subsequent calls (e.g. across the five baselines in the panel sweep)
    read from disk so activity-cliff and cluster splits aren't recomputed.
    """
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    target = sub["target_chembl_id"].iloc[0]
    cfg = SplitCacheConfig(seed=seed, ratios=tuple(ratios), sim=sim, delta=delta)
    return load_or_compute_split(target, kind, sub, cfg)


def run_single_task(args: SingleTaskTrainArgs) -> dict:
    from radiant import RadiantConfig
    from radiant_chem import (
        RadiantChemConfig,
        RadiantChemModel,
        SmilesTokenizer,
    )
    from radiant_chem.objectives import RegressionLoss
    from radiant_chem.tasks import TaskRegistry, TaskSpec

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    sub = _select_target(args.activities, args.target_chembl_id)
    logger.info("target %s: %d rows", args.target_chembl_id, len(sub))
    train_idx, val_idx, test_idx = _split(
        sub, args.split_kind, args.splits_ratios, args.seed,
        sim=args.activity_cliff_sim, delta=args.activity_cliff_delta,
    )
    logger.info("split=%s sizes: train=%d val=%d test=%d",
                args.split_kind, len(train_idx), len(val_idx), len(test_idx))

    tokenizer = SmilesTokenizer.load(args.vocab)
    source_cfg = RadiantConfig.from_json(args.config)
    base_cfg = source_cfg.replace(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        n_loops_train=args.n_loops_train,
        use_state_anchor=not args.disable_anchor,
        use_iteration_adapter=not args.disable_iteration_adapter,
        use_confidence_halting=False if args.disable_halting else source_cfg.use_confidence_halting,
        halting_loss_weight=0.0 if args.disable_halting_loss else source_cfg.halting_loss_weight,
    )
    chem_cfg = RadiantChemConfig(
        base=base_cfg,
        pooling_kind=args.pooling_kind,
        regression_head_hidden_dim=args.regression_head_hidden_dim,
        regression_head_dropout=args.regression_head_dropout,
        fingerprint_dim=args.fingerprint_dim,
        fingerprint_radius=args.fingerprint_radius,
        use_depth_adaptive_pool=args.use_depth_adaptive_pool,
    )
    chem_cfg.to_json(args.out / "chem_config.json")
    base_cfg.to_json(args.out / "base_config.json")
    tasks = TaskRegistry([TaskSpec("pchembl", "regression", "pchembl", num_outputs=1)])
    model = RadiantChemModel(chem_cfg, tasks).to(args.device)

    if args.pretrain_ckpt is not None and args.pretrain_ckpt.exists():
        # We trust our own pretrain checkpoints (they carry an `args` dict and
        # potentially an optimizer state, which the new PyTorch 2.6 default
        # `weights_only=True` would refuse to unpickle).
        ckpt = torch.load(args.pretrain_ckpt, map_location=args.device, weights_only=False)
        state = ckpt.get("model", ckpt)
        _preflight_state_dict_match(model, state, ckpt_path=args.pretrain_ckpt, config_path=args.config)
        # Strict=False so newly-added task heads (random init) don't break loading.
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Count how many param-tensors actually got loaded vs total.
        n_loaded = sum(1 for k in model.state_dict() if k in state and state[k].shape == model.state_dict()[k].shape)
        n_total = len(list(model.state_dict()))
        logger.info("loaded pretrain ckpt: %d/%d tensors matched (%d missing, %d unexpected)",
                    n_loaded, n_total, len(missing), len(unexpected))

    smi = sub["standard_smiles"].tolist()
    pch = sub["pchembl"].astype(float).tolist()

    # ------------------------------------------------------------------
    # Initialise the task head's bias to the training-set mean pChEMBL.
    #
    # Without this, the head starts with bias≈0 but pChEMBL values are
    # ~6.0 on average, so epoch-0 predictions are off by ~6 units.  The
    # resulting huge gradients flow into the pretrained backbone and
    # destroy learned representations.  Setting the bias puts epoch-0
    # predictions near the data mean → small initial loss → gentle
    # gradients → pretrained features are preserved.
    # ------------------------------------------------------------------
    pch_train = [pch[i] for i in train_idx]
    train_mean = float(np.mean(pch_train))
    head = model.task_heads["pchembl"]
    with torch.no_grad():
        # Find the final linear layer's bias (works for both standard
        # RegressionHead and FingerprintAugmentedHead)
        from radiant_chem.model_chem import FingerprintAugmentedHead
        if isinstance(head, FingerprintAugmentedHead):
            head.mlp[-1].bias.fill_(train_mean)
        else:
            head.proj.bias.fill_(train_mean)
    logger.info("task head bias initialised to training mean pChEMBL = %.3f", train_mean)

    # ------------------------------------------------------------------
    # Precompute Morgan fingerprints if fingerprint_dim > 0 (ablation only)
    # ------------------------------------------------------------------
    fp_array = None
    if chem_cfg.fingerprint_dim > 0:
        from radiant_chem.fingerprint import smiles_to_morgan
        logger.info("computing Morgan fingerprints (dim=%d, radius=%d) for %d molecules...",
                     chem_cfg.fingerprint_dim, chem_cfg.fingerprint_radius, len(smi))
        fp_array = np.stack([
            smiles_to_morgan(s, radius=chem_cfg.fingerprint_radius,
                             n_bits=chem_cfg.fingerprint_dim)
            for s in smi
        ])
        logger.info("fingerprints computed: shape %s, density %.3f",
                     fp_array.shape, fp_array.mean())

    train_ds = _ActivityDataset(
        smi, pch, tokenizer, max_len=base_cfg.max_seq_len,
        augment_prob=args.smiles_augment_prob,
        fingerprints=fp_array,
    )
    eval_ds = _ActivityDataset(smi, pch, tokenizer, max_len=base_cfg.max_seq_len,
                               fingerprints=fp_array)
    logger.info(
        "single-target upgrades: regression_head_hidden_dim=%d dropout=%.2f "
        "loss=%s beta=%.3f train_smiles_augment_prob=%.2f",
        args.regression_head_hidden_dim, args.regression_head_dropout,
        args.loss_kind, args.huber_beta, args.smiles_augment_prob,
    )
    pad = tokenizer.pad_id
    loaders = {
        name: DataLoader(
            Subset(train_ds if name == "train" else eval_ds, idxs),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            collate_fn=lambda b: _collate(b, pad),
        )
        for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    }

    # ------------------------------------------------------------------
    # Optimizer: layer-wise LR decay (discriminative fine-tuning).
    #
    # Layers closer to input (stem, early refinement) get lower LR to
    # preserve pretrained representations; task heads + exit decoder get
    # the full LR. This is the standard approach for fine-tuning pretrained
    # transformers (Howard & Ruder 2018, Sun et al. 2019).
    #
    # Layer groups (deepest → shallowest, i.e. lowest LR → highest):
    #   0: core.stem       (embedding + stem blocks)
    #   1: core.refinement (weight-shared loop blocks + halting)
    #   2: core.exit       (exit decoder)
    #   3: task_heads + pool (task-specific, random init)
    # ------------------------------------------------------------------
    def _build_param_groups(model, base_lr, wd, layer_decay):
        """Create parameter groups with layer-wise LR decay."""
        groups = {
            "stem": {"params": [], "lr": base_lr * layer_decay ** 3, "weight_decay": wd},
            "refinement": {"params": [], "lr": base_lr * layer_decay ** 2, "weight_decay": wd},
            "exit": {"params": [], "lr": base_lr * layer_decay ** 1, "weight_decay": wd},
            "heads": {"params": [], "lr": base_lr, "weight_decay": wd},
        }
        # No weight decay on bias and norm params
        groups_no_wd = {
            "stem_no_wd": {"params": [], "lr": base_lr * layer_decay ** 3, "weight_decay": 0.0},
            "refinement_no_wd": {"params": [], "lr": base_lr * layer_decay ** 2, "weight_decay": 0.0},
            "exit_no_wd": {"params": [], "lr": base_lr * layer_decay ** 1, "weight_decay": 0.0},
            "heads_no_wd": {"params": [], "lr": base_lr, "weight_decay": 0.0},
        }

        no_wd_keywords = ("bias", "norm", "layernorm", "layer_norm")

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Classify layer
            if "core.stem" in name or "core.token_embed" in name or "core.embed" in name:
                group_key = "stem"
            elif "core.refinement" in name:
                group_key = "refinement"
            elif "core.exit" in name:
                group_key = "exit"
            else:  # task_heads, pool, anything else
                group_key = "heads"

            is_no_wd = any(kw in name.lower() for kw in no_wd_keywords)
            if is_no_wd:
                groups_no_wd[f"{group_key}_no_wd"]["params"].append(param)
            else:
                groups[group_key]["params"].append(param)

        # Filter out empty groups
        all_groups = [g for g in list(groups.values()) + list(groups_no_wd.values()) if g["params"]]
        return all_groups

    lr_mult = float(base_cfg.halting_head_lr_mult)
    if args.lr_layer_decay < 1.0:
        param_groups = _build_param_groups(model, args.lr, args.weight_decay, args.lr_layer_decay)
        logger.info(
            "layer-wise LR decay=%.2f: %d groups, LR range [%.2e, %.2e]",
            args.lr_layer_decay, len(param_groups),
            min(g["lr"] for g in param_groups),
            max(g["lr"] for g in param_groups),
        )
    else:
        # Flat LR for all params (with halting head multiplier if configured)
        halting_param_names = {
            n for n, _ in model.named_parameters() if "refinement.halting" in n
        }
        halting_params, other_params = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                (halting_params if n in halting_param_names else other_params).append(p)
        if halting_params and lr_mult != 1.0:
            param_groups = [
                {"params": other_params,   "lr": args.lr,              "weight_decay": args.weight_decay},
                {"params": halting_params, "lr": args.lr * lr_mult,    "weight_decay": args.weight_decay},
            ]
        else:
            param_groups = [{"params": list(model.parameters()), "lr": args.lr, "weight_decay": args.weight_decay}]

    import math as _math

    # ------------------------------------------------------------------
    # Head warmup: freeze backbone for the first N epochs so only the
    # random task head trains.  This prevents the large initial loss
    # (from a randomly-initialised head) from sending destructive
    # gradients back into the pretrained backbone.  After warmup the
    # backbone is unfrozen for full end-to-end fine-tuning.
    # ------------------------------------------------------------------
    _backbone_frozen = False

    def _freeze_backbone():
        nonlocal _backbone_frozen
        for n, p in model.named_parameters():
            if "task_heads" not in n and "pool" not in n:
                p.requires_grad = False
        _backbone_frozen = True
        logger.info("backbone FROZEN (head warmup)")

    def _unfreeze_backbone():
        nonlocal _backbone_frozen
        for p in model.parameters():
            p.requires_grad = True
        _backbone_frozen = False
        logger.info("backbone UNFROZEN (head warmup done)")

    # ------------------------------------------------------------------
    # If head warmup is active, start with a head-only optimizer at a
    # higher LR (1e-3) so the random head calibrates fast in ~5 epochs.
    # The full optimizer is built later when the backbone unfreezes.
    # ------------------------------------------------------------------
    _full_param_groups = param_groups  # save for later

    if args.head_warmup_epochs > 0 and args.pretrain_ckpt is not None:
        # Freeze backbone FIRST so the optimizer only allocates state for
        # head + pool params (saves ~2x model size in optimizer memory).
        _freeze_backbone()
        head_lr = 1e-3  # much higher LR for random head warmup
        head_params = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(
            [{"params": head_params, "lr": head_lr, "weight_decay": args.weight_decay}],
            betas=(0.9, 0.999), eps=1e-8,
        )
        warmup_total = max(1, len(loaders["train"]) // args.grad_accum_steps) * args.head_warmup_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s, _t=warmup_total: max(0.0, 0.5 * (1.0 + _math.cos(_math.pi * s / _t)))
        )
        logger.info(
            "head warmup optimizer: lr=%.1e, %d steps (%d epochs), head-only",
            head_lr, warmup_total, args.head_warmup_epochs,
        )
    else:
        optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        # Cosine scheduler with linear warmup
        steps_per_epoch = max(1, len(loaders["train"]) // args.grad_accum_steps)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))

        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + _math.cos(_math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
        logger.info(
            "scheduler: cosine w/ linear warmup, %d warmup / %d total steps, "
            "grad_accum=%d, effective_batch=%d",
            warmup_steps, total_steps, args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps,
        )

    loss_fn = RegressionLoss(kind=args.loss_kind, huber_beta=args.huber_beta)

    best = {"val_pearson": -2.0, "epoch": -1, "state": None}
    bad = 0
    history = []
    # PonderNet supervision is enabled only when the halting head is on
    # and the user has opted in via halting_loss_weight > 0. Computing
    # per-step task predictions adds ~25-50% to training-step compute, so
    # we gate it: when it's off the forward path is identical to before.
    #
    # --disable-halting-loss overrides the config: useful during
    # fine-tuning from an activity-pretrained backbone where the halting
    # regulariser adds noise before the new head has calibrated.
    use_pondernet = (
        base_cfg.use_confidence_halting
        and base_cfg.halting_loss_weight > 0
        and not args.disable_halting_loss
    )
    if args.disable_halting_loss:
        logger.info("PonderNet supervision DISABLED (--disable-halting-loss)")
    warmup_epochs = int(base_cfg.halting_loss_warmup_epochs)
    if use_pondernet:
        from radiant.confidence_halting import pondernet_task_loss
        logger.info(
            "PonderNet supervision ON  (halting_loss_weight=%.4f, prior_lambda=%.3f, "
            "warmup_epochs=%d, head_lr_mult=%.1f)",
            base_cfg.halting_loss_weight, base_cfg.halting_prior_lambda,
            warmup_epochs, lr_mult,
        )

    global_step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        # Unfreeze backbone after head warmup phase
        if _backbone_frozen and epoch >= args.head_warmup_epochs:
            _unfreeze_backbone()
            # Rebuild optimizer with all params now trainable
            if args.lr_layer_decay < 1.0:
                param_groups = _build_param_groups(model, args.lr, args.weight_decay, args.lr_layer_decay)
            else:
                param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                                 "lr": args.lr, "weight_decay": args.weight_decay}]
            optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
            # Re-create scheduler for remaining epochs
            remaining_epochs = args.epochs - epoch
            steps_per_epoch = max(1, len(loaders["train"]) // args.grad_accum_steps)
            remaining_total = steps_per_epoch * remaining_epochs
            remaining_warmup = max(1, int(remaining_total * args.warmup_ratio))

            def _lr_lambda_remaining(current_step: int, _wup=remaining_warmup, _tot=remaining_total) -> float:
                if current_step < _wup:
                    return float(current_step) / float(_wup)
                progress = float(current_step - _wup) / float(max(1, _tot - _wup))
                return max(0.0, 0.5 * (1.0 + _math.cos(_math.pi * progress)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda_remaining)
            global_step = 0  # reset step counter for new scheduler
            logger.info("rebuilt optimizer+scheduler for %d remaining epochs", remaining_epochs)

        # Warmup ramps the halting-loss contribution from 0 to its
        # configured weight over the first `warmup_epochs` epochs. The
        # rest of the model trains pure regression in epochs 0..warmup-1
        # so the backbone is stable before halting supervision kicks in.
        if warmup_epochs > 0 and epoch < warmup_epochs:
            halt_warmup = (epoch + 1) / float(warmup_epochs)
        else:
            halt_warmup = 1.0

        model.train()
        epoch_losses = []
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(loaders["train"]):
            ids = batch["input_ids"].to(args.device)
            attn = batch["attention_mask"].to(args.device)
            tgt = batch["targets"].to(args.device)
            fp = batch.get("fingerprints")
            if fp is not None:
                fp = fp.to(args.device)
            out = model(
                ids, attention_mask=attn,
                n_loops=args.n_loops_train,
                is_causal=False,
                return_per_step_task=use_pondernet,
                fingerprints=fp,
            )
            pred = out.task_outputs["pchembl"].squeeze(-1)
            loss = loss_fn(pred, tgt)
            if out.base.aux_loss is not None and not args.disable_halting_loss:
                # KL halting loss + any MoE aux losses. Scale by warmup so
                # early epochs aren't dominated by the halting regulariser.
                loss = loss + halt_warmup * out.base.aux_loss
            if (
                use_pondernet
                and out.per_step_task_outputs is not None
                and out.base.halting is not None
                and out.base.halting.confidences
                and out.base.halting.confidences[0].requires_grad
            ):
                pn_loss = pondernet_task_loss(
                    out.per_step_task_outputs["pchembl"],
                    target=tgt,
                    confidences=out.base.halting.confidences,
                    attention_mask=attn,
                    task_kind="regression",
                )
                loss = loss + halt_warmup * base_cfg.halting_loss_weight * pn_loss
            # Scale loss for gradient accumulation
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            epoch_losses.append(float(loss.detach().item()))

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        # Val.
        model.eval()
        pv, tv = [], []
        with torch.no_grad():
            for batch in loaders["val"]:
                ids = batch["input_ids"].to(args.device)
                attn = batch["attention_mask"].to(args.device)
                tgt = batch["targets"]
                fp = batch.get("fingerprints")
                if fp is not None:
                    fp = fp.to(args.device)
                out = model(ids, attention_mask=attn, n_loops=args.n_loops_train,
                            is_causal=False, fingerprints=fp)
                pv.extend(out.task_outputs["pchembl"].squeeze(-1).cpu().tolist())
                tv.extend(tgt.tolist())
        val_metrics = regression_metrics(np.array(pv), np.array(tv))
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        current_lr = scheduler.get_last_lr()[0]
        frozen_tag = " [head-only]" if _backbone_frozen else ""
        logger.info(
            "epoch %2d  train_loss=%.4f  val MAE=%.3f Pearson=%.3f  halt_w=%.3f  lr=%.2e%s",
            epoch, train_loss, val_metrics["mae"], val_metrics["pearson"],
            halt_warmup * base_cfg.halting_loss_weight if use_pondernet else 0.0,
            current_lr, frozen_tag,
        )

        if val_metrics["pearson"] > best["val_pearson"]:
            best = {"val_pearson": val_metrics["pearson"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad > args.early_stopping_patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    # Test using the best checkpoint.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    # Phase G needs per-row halting signals (halt_step, effective_depth,
    # confidence_var, tokens). The accumulator collects them across the
    # test loader without changing the metric path.
    from radiant_qsar.eval.halting_extras import HaltingExtrasAccumulator

    halting_acc = HaltingExtrasAccumulator(include_per_atom=False)
    pad_id = tokenizer.pad_id
    id_to_token = tokenizer.id_to_token

    pt, tt = [], []
    with torch.no_grad():
        for batch in loaders["test"]:
            ids = batch["input_ids"].to(args.device)
            attn = batch["attention_mask"].to(args.device)
            tgt = batch["targets"]
            fp = batch.get("fingerprints")
            if fp is not None:
                fp = fp.to(args.device)
            out = model(ids, attention_mask=attn, n_loops=args.n_loops_train,
                        is_causal=False, fingerprints=fp)
            pt.extend(out.task_outputs["pchembl"].squeeze(-1).cpu().tolist())
            tt.extend(tgt.tolist())
            halting_acc.add(
                halting=out.base.halting,
                input_ids=ids,
                attention_mask=attn,
                pad_id=pad_id,
                id_to_token=id_to_token,
            )
    test_metrics = regression_metrics(np.array(pt), np.array(tt))

    # Per-test-molecule predictions for downstream calibration / OOD / C2 / C3 / C5
    # analyses. ``test_idx`` is a stable list of ints into ``sub`` (the per-target
    # DataFrame); the test DataLoader was built without shuffle so prediction
    # order corresponds to test_idx order.
    from radiant_qsar.eval.predictions import write_predictions

    test_inchikeys = sub["inchikey14"].iloc[test_idx].tolist()
    test_smiles = [smi[i] for i in test_idx]
    extras = halting_acc.finalize()
    write_predictions(
        args.out,
        indices=test_idx,
        inchikeys=test_inchikeys,
        smiles=test_smiles,
        true_pchembl=tt,
        pred_pchembl=pt,
        target_chembl_id=args.target_chembl_id,
        split_kind=args.split_kind,
        extra_columns=extras if len(extras["halt_step"]) == len(test_idx) else None,
    )

    elapsed = time.time() - t0
    # Best-of-training val metrics, recovered from history.
    best_val_block = next(
        ({k.replace("val_", ""): v for k, v in row.items() if k.startswith("val_")}
         for row in history if row["epoch"] == best["epoch"]),
        {},
    )
    result = {
        "model": "radiant",
        "target_chembl_id": args.target_chembl_id,
        "split_kind": args.split_kind,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_val_epoch": best["epoch"],
        "best_val_pearson": best["val_pearson"],
        "fine_tune_upgrades": {
            "loss_kind": args.loss_kind,
            "huber_beta": args.huber_beta,
            "regression_head_hidden_dim": args.regression_head_hidden_dim,
            "regression_head_dropout": args.regression_head_dropout,
            "smiles_augment_prob": args.smiles_augment_prob,
            "disable_anchor": args.disable_anchor,
            "disable_iteration_adapter": args.disable_iteration_adapter,
            "use_depth_adaptive_pool": args.use_depth_adaptive_pool,
            "disable_halting_loss": args.disable_halting_loss,
            "disable_halting": args.disable_halting,
        },
        # Canonical metric blocks consumed by sweep.aggregate_results.
        "val": best_val_block,
        "test": test_metrics,
        # Legacy keys kept briefly for backwards-compat with pre-existing analyses.
        "test_metrics": test_metrics,
        "history": history,
        "model_path": "best.pt",
        "predictions_path": "predictions.csv",
        "elapsed_s": round(elapsed, 1),
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if best["state"] is not None:
        torch.save({"model": best["state"]}, args.out / "best.pt")
    logger.info("done: test MAE=%.3f Pearson=%.3f Spearman=%.3f (n=%d) in %.1fs",
                test_metrics["mae"], test_metrics["pearson"], test_metrics["spearman"],
                test_metrics["n"], elapsed)
    return result


def _main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--vocab", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pretrain-ckpt", type=Path, default=None)
    p.add_argument("--split", default="scaffold",
                   choices=("random", "scaffold", "time", "cluster", "activity_cliff"))
    p.add_argument("--n-loops-train", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--lr-layer-decay", type=float, default=0.75)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--head-warmup-epochs", type=int, default=5,
                   help="Freeze backbone for N epochs so only the task head trains "
                        "(prevents destructive gradients from random head init)")
    p.add_argument("--disable-halting-loss", action="store_true",
                   help="Turn off PonderNet halting supervision during fine-tuning")
    p.add_argument("--disable-halting", action="store_true",
                   help="Ablation: remove confidence halting machinery entirely.")
    p.add_argument("--loss-kind", default="huber",
                   choices=("mse", "huber", "smooth_l1"),
                   help="Regression loss for fine-tuning. Huber is more robust to noisy pChEMBL labels.")
    p.add_argument("--huber-beta", type=float, default=0.5,
                   help="SmoothL1/Huber beta in pChEMBL units.")
    p.add_argument("--regression-head-hidden-dim", type=int, default=512,
                   help="Hidden width for the RADIANT-only MLP regression head; 0 keeps the old linear head.")
    p.add_argument("--regression-head-dropout", type=float, default=0.10)
    p.add_argument("--smiles-augment-prob", type=float, default=0.50,
                   help="Probability of randomized-SMILES enumeration for each training example.")
    p.add_argument("--disable-anchor", action="store_true",
                   help="Ablation: remove the recurrent anchor residual path.")
    p.add_argument("--disable-iteration-adapter", action="store_true",
                   help="Ablation: remove loop-conditioned iteration adapters.")
    # Architecture enhancements
    p.add_argument("--pooling-kind", default="attention",
                   choices=("mean", "first", "attention"),
                   help="Pooling strategy: attention (default), mean, first (CLS)")
    p.add_argument("--fingerprint-dim", type=int, default=0,
                   help="Morgan FP dimension (0=disabled, 2048=ablation only)")
    p.add_argument("--fingerprint-radius", type=int, default=2,
                   help="Morgan FP radius (2=ECFP4, ablation only)")
    p.add_argument("--use-depth-pool", action="store_true", default=True,
                   help="Depth-adaptive pooling via halting-weighted intermediates (default: on)")
    p.add_argument("--no-depth-pool", action="store_true",
                   help="Disable depth-adaptive pooling (for ablation)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_single_task(
        SingleTaskTrainArgs(
            activities=args.activities,
            target_chembl_id=args.target,
            vocab=args.vocab,
            config=args.config,
            out=args.out,
            pretrain_ckpt=args.pretrain_ckpt,
            split_kind=args.split,
            n_loops_train=args.n_loops_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            lr_layer_decay=args.lr_layer_decay,
            grad_accum_steps=args.grad_accum_steps,
            head_warmup_epochs=args.head_warmup_epochs,
            disable_halting_loss=args.disable_halting_loss,
            disable_halting=args.disable_halting,
            loss_kind=args.loss_kind,
            huber_beta=args.huber_beta,
            regression_head_hidden_dim=args.regression_head_hidden_dim,
            regression_head_dropout=args.regression_head_dropout,
            smiles_augment_prob=args.smiles_augment_prob,
            disable_anchor=args.disable_anchor,
            disable_iteration_adapter=args.disable_iteration_adapter,
            pooling_kind=args.pooling_kind,
            fingerprint_dim=args.fingerprint_dim,
            fingerprint_radius=args.fingerprint_radius,
            use_depth_adaptive_pool=(args.use_depth_pool and not args.no_depth_pool),
            early_stopping_patience=args.patience,
            device=args.device,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    _main()
