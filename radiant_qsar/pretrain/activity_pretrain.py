"""Stage 2 — Multi-target activity pretraining.

Trains a :class:`RadiantChemModel` on the full ChEMBL activity table
(~1.57M rows, ~947K compounds, ~7.9K targets). The model learns:

    SMILES + target_chembl_id → pChEMBL

via a **target-conditioned regression** objective. A learned target
embedding is added to the pooled molecule representation before the
regression head, so the model conditions its predictions on which protein
it's predicting for.

This intermediate checkpoint bridges the gap between:
  - Stage 1 (SMILES MLM): knows chemistry, not bioactivity.
  - Stage 3 (single-target fine-tune): tiny dataset, needs prior
    bioactivity knowledge to generalise.

Architecture
------------
``RADIANTActivityModel`` wraps ``RadiantChemModel`` and adds:
  * ``target_embed``: ``nn.Embedding(n_targets, d_model)``
  * ``activity_head``: ``d_model → 1`` regression head (2-layer MLP)

Forward:
  1. Encode SMILES → pooled ``(B, D)`` via RADIANT
  2. Look up target embedding → ``(B, D)``
  3. ``h = smiles_pool + target_embed``  (additive fusion)
  4. ``pchembl_pred = activity_head(h)``

The additive fusion is deliberate: it keeps the SMILES encoder output and
the target conditioning in the same representation space, so downstream
single-target fine-tuning can ignore the target embedding and the
backbone transfers cleanly.

Usage
-----
::

    python -m radiant_qsar.pretrain.activity_pretrain \\
        --activities   data/processed/v1/activities.parquet \\
        --vocab        data/processed/v1/smiles_vocab.json \\
        --config       configs/radiant_75m.json \\
        --stage1-ckpt  checkpoints/pretrain_75m/latest.pt \\
        --out          checkpoints/activity_pretrain_75m \\
        --epochs       10 \\
        --batch-size   128 \\
        --lr           5e-5 \\
        --device       cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target vocabulary
# ---------------------------------------------------------------------------

class TargetVocab:
    """Maps target_chembl_id strings to dense integer indices."""

    def __init__(self, targets: list[str]) -> None:
        self._t2i = {t: i for i, t in enumerate(sorted(set(targets)))}
        self._i2t = {i: t for t, i in self._t2i.items()}

    def __len__(self) -> int:
        return len(self._t2i)

    def encode(self, target: str) -> int:
        return self._t2i[target]

    def decode(self, idx: int) -> str:
        return self._i2t[idx]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"targets": list(self._t2i.keys()), "version": 1},
            indent=2,
        ), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TargetVocab":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["targets"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActivityDataset(Dataset):
    """Each sample: (smiles_token_ids, target_idx, pchembl).

    When ``augment_prob > 0``, training samples get randomized SMILES
    enumeration (different atom traversal order) with the given
    probability. This teaches the backbone SMILES equivalence and
    significantly improves generalisation.
    """

    def __init__(
        self,
        smiles: list[str],
        target_ids: list[int],
        pchembl: list[float],
        tokenizer,
        max_len: int,
        augment_prob: float = 0.0,
        rgroup_smiles: list[str] | None = None,
    ):
        self.smiles = smiles
        self.target_ids = target_ids
        self.pchembl = pchembl
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment_prob = augment_prob
        self.rgroup_smiles = rgroup_smiles

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment_prob > 0.0 and np.random.random() < self.augment_prob:
            from radiant_chem.augment import randomize_smiles
            smi = randomize_smiles(smi)
        ids = self.tokenizer.encode(smi, max_len=self.max_len)
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "target_idx": torch.tensor(self.target_ids[idx], dtype=torch.long),
            "pchembl": torch.tensor(self.pchembl[idx], dtype=torch.float32),
        }
        if self.rgroup_smiles is not None:
            rg = self.rgroup_smiles[idx] or self.smiles[idx]
            if self.augment_prob > 0.0 and np.random.random() < self.augment_prob:
                from radiant_chem.augment import randomize_smiles
                rg = randomize_smiles(rg)
            item["rgroup_input_ids"] = torch.tensor(
                self.tokenizer.encode(rg, max_len=self.max_len),
                dtype=torch.long,
            )
        return item


def _collate(batch, pad_id: int):
    L = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = 1
    target_idx = torch.stack([b["target_idx"] for b in batch])
    pchembl = torch.stack([b["pchembl"] for b in batch])
    out = {"input_ids": ids, "attention_mask": attn,
           "target_idx": target_idx, "pchembl": pchembl}
    if "rgroup_input_ids" in batch[0]:
        LR = max(b["rgroup_input_ids"].size(0) for b in batch)
        rg_ids = torch.full((B, LR), pad_id, dtype=torch.long)
        rg_attn = torch.zeros((B, LR), dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["rgroup_input_ids"].size(0)
            rg_ids[i, :n] = b["rgroup_input_ids"]
            rg_attn[i, :n] = 1
        out["rgroup_input_ids"] = rg_ids
        out["rgroup_attention_mask"] = rg_attn
    return out


def murcko_rgroup_smiles(smiles: str) -> str:
    """Return side-chain/R-group SMILES outside the Murcko scaffold.

    This keeps the R-group view in the same learned SMILES/tokenizer path
    rather than injecting fingerprints or external descriptors.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    sidechains = Chem.DeleteSubstructs(mol, scaffold, onlyFrags=False)
    frags = Chem.GetMolFrags(sidechains, asMols=True, sanitizeFrags=True)
    parts = sorted(
        (Chem.MolToSmiles(f, canonical=True) for f in frags if f.GetNumAtoms() > 0),
        key=lambda s: (-len(s), s),
    )
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RADIANTActivityModel(nn.Module):
    """RADIANT + target embedding for multi-target activity prediction.

    Wraps a RadiantChemModel and adds target conditioning.
    The activity_head is a 2-layer MLP with GELU for slightly more
    expressive multi-target fitting.
    """

    def __init__(self, chem_model, n_targets: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.chem = chem_model
        self.target_embed = nn.Embedding(n_targets, d_model)
        nn.init.normal_(self.target_embed.weight, std=0.02)
        self.activity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.rgroup_activity_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Init last layer to small output
        nn.init.normal_(self.activity_head[-1].weight, std=0.01)
        nn.init.zeros_(self.activity_head[-1].bias)
        nn.init.normal_(self.rgroup_activity_head[-1].weight, std=0.01)
        nn.init.zeros_(self.rgroup_activity_head[-1].bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_idx: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_loops: int | None = None,
        rgroup_input_ids: torch.Tensor | None = None,
        rgroup_attention_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Returns full-molecule prediction and optional R-group auxiliary output."""
        out = self.chem(
            input_ids,
            n_loops=n_loops,
            attention_mask=attention_mask,
            is_causal=False,
            run_tasks=False,
            return_pooled=True,
        )
        mol_pool = out.pooled  # (B, D)
        tgt_emb = self.target_embed(target_idx)  # (B, D)
        h = mol_pool + tgt_emb
        pred = self.activity_head(h).squeeze(-1)  # (B,)
        if not return_aux:
            return pred
        aux = {"pred": pred}
        if rgroup_input_ids is not None:
            rg_out = self.chem(
                rgroup_input_ids,
                n_loops=n_loops,
                attention_mask=rgroup_attention_mask,
                is_causal=False,
                run_tasks=False,
                return_pooled=True,
            )
            rg_h = rg_out.pooled + tgt_emb
            aux["rgroup_pred"] = self.rgroup_activity_head(rg_h).squeeze(-1)
        return aux


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    activities_path: Path,
    vocab_path: Path,
    config_path: Path,
    stage1_ckpt: Path | None,
    out_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    n_loops_train: int = 8,
    val_fraction: float = 0.02,
    max_targets: int | None = None,
    min_compounds_per_target: int = 50,
    device: str = "cuda",
    seed: int = 1337,
    log_every: int = 100,
    save_every_epoch: bool = True,
    rgroup_aux_weight: float = 0.25,
) -> dict:
    """Run Stage 2 multi-target activity pretraining."""
    import pandas as pd
    from radiant import RadiantConfig
    from radiant_chem import RadiantChemConfig, RadiantChemModel, SmilesTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # --- Load activities ---
    logger.info("Loading activities from %s", activities_path)
    df = pd.read_parquet(activities_path)
    logger.info("Raw activities: %d rows, %d targets, %d compounds",
                len(df), df["target_chembl_id"].nunique(),
                df["standard_smiles"].nunique())

    # Filter targets with enough data
    target_counts = df["target_chembl_id"].value_counts()
    valid_targets = target_counts[target_counts >= min_compounds_per_target].index.tolist()
    if max_targets is not None:
        valid_targets = valid_targets[:max_targets]
    df = df[df["target_chembl_id"].isin(valid_targets)].reset_index(drop=True)
    logger.info("After filtering (min %d cpds/target): %d rows, %d targets",
                min_compounds_per_target, len(df), df["target_chembl_id"].nunique())

    # --- Build target vocab ---
    target_vocab = TargetVocab(df["target_chembl_id"].unique().tolist())
    target_vocab.save(out_dir / "target_vocab.json")
    logger.info("Target vocab: %d targets", len(target_vocab))

    # --- Tokenizer & model config ---
    tokenizer = SmilesTokenizer.load(vocab_path)
    base_cfg = RadiantConfig.from_json(config_path).replace(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_id,
        n_loops_train=n_loops_train,
    )
    chem_cfg = RadiantChemConfig(base=base_cfg)
    chem_model = RadiantChemModel(chem_cfg)

    # Load Stage 1 checkpoint
    if stage1_ckpt is not None and stage1_ckpt.exists():
        ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        missing, unexpected = chem_model.load_state_dict(state, strict=False)

        # Backbone-aware accounting (consistent with single_task preflight).
        # Pooling / depth-pool / fingerprint / task-head tensors are added by the
        # *next* stage's config and are expected to be absent from a backbone
        # checkpoint, so they're reported as fresh-init rather than "missing".
        _NEW_HEAD_PREFIXES = ("pool.", "depth_pool.", "fp_proj.", "task_heads.")
        model_sd = chem_model.state_dict()

        def _is_new_head(key: str) -> bool:
            return any(key.startswith(p) for p in _NEW_HEAD_PREFIXES)

        backbone_keys = [k for k in model_sd if (not _is_new_head(k)) or (k in state)]
        n_backbone = len(backbone_keys)
        n_loaded = sum(
            1 for k in backbone_keys
            if k in state and state[k].shape == model_sd[k].shape
        )
        n_backbone_missing = n_backbone - n_loaded
        n_fresh_heads = sum(1 for k in model_sd if _is_new_head(k) and k not in state)
        match_rate = n_loaded / max(n_backbone, 1)
        logger.info(
            "Loaded Stage 1 ckpt: %d/%d backbone tensors matched (%.1f%%); "
            "%d backbone missing, %d new heads fresh-init, %d unexpected-in-ckpt",
            n_loaded, n_backbone, match_rate * 100,
            n_backbone_missing, n_fresh_heads, len(unexpected),
        )
        if match_rate < 0.90:
            logger.warning(
                "Stage 1 backbone match is only %.1f%% -- check that --config matches the "
                "config used for Stage 1 pretraining (architecture is baked into the tensors).",
                match_rate * 100,
            )

    model = RADIANTActivityModel(
        chem_model, n_targets=len(target_vocab),
        d_model=base_cfg.d_model, dropout=base_cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d (%.1fM)", n_params, n_params / 1e6)

    # --- Dataset ---
    smiles_list = df["standard_smiles"].tolist()
    target_id_list = [target_vocab.encode(t) for t in df["target_chembl_id"]]
    pchembl_list = df["pchembl"].astype(float).tolist()
    rgroup_list = None
    if rgroup_aux_weight > 0:
        logger.info("Building Murcko side-chain/R-group SMILES views...")
        rgroup_list = [murcko_rgroup_smiles(s) for s in smiles_list]
        n_nonempty = sum(1 for s in rgroup_list if s)
        logger.info("R-group views: %d/%d non-empty (%.1f%%)",
                    n_nonempty, len(rgroup_list),
                    100 * n_nonempty / max(len(rgroup_list), 1))
    full_ds = ActivityDataset(smiles_list, target_id_list, pchembl_list,
                              tokenizer, max_len=base_cfg.max_seq_len,
                              augment_prob=0.5, rgroup_smiles=rgroup_list)
    n_val = max(1000, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(seed))
    pad_id = tokenizer.pad_id
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: _collate(b, pad_id),
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            collate_fn=lambda b: _collate(b, pad_id),
                            num_workers=2, pin_memory=True)
    logger.info("Dataset: %d train, %d val", n_train, n_val)

    # --- Optimizer + scheduler ---
    # Use layer-wise LR decay: backbone gets lower LR, new heads get full LR
    backbone_params, new_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "target_embed" in name or "activity_head" in name:
            new_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.5, "weight_decay": weight_decay},
        {"params": new_params, "lr": lr, "weight_decay": weight_decay},
    ]
    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    logger.info("Scheduler: cosine, %d warmup / %d total steps", warmup_steps, total_steps)

    loss_fn = nn.MSELoss()

    # --- Mixed precision (bf16 on Ampere+; no GradScaler needed for bf16) ---
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info("AMP: %s (dtype=%s)", use_amp, amp_dtype if use_amp else "fp32")

    # --- Training loop ---
    best_val_mae = float("inf")
    history = []

    t0 = time.time()
    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tgt_idx = batch["target_idx"].to(device)
            pch = batch["pchembl"].to(device)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    ids, tgt_idx, attention_mask=attn, n_loops=n_loops_train,
                    rgroup_input_ids=(
                        batch["rgroup_input_ids"].to(device)
                        if "rgroup_input_ids" in batch else None
                    ),
                    rgroup_attention_mask=(
                        batch["rgroup_attention_mask"].to(device)
                        if "rgroup_attention_mask" in batch else None
                    ),
                    return_aux=rgroup_aux_weight > 0,
                )
                if isinstance(out, dict):
                    pred = out["pred"]
                    loss = loss_fn(pred, pch)
                    if "rgroup_pred" in out:
                        loss = loss + rgroup_aux_weight * loss_fn(out["rgroup_pred"], pch)
                else:
                    pred = out
                    loss = loss_fn(pred, pch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.item()))

            if (step + 1) % log_every == 0:
                avg = np.mean(epoch_losses[-log_every:])
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    "  epoch %d step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, step + 1, len(train_loader), avg, lr_now,
                )

        train_loss = float(np.mean(epoch_losses))

        # --- Validation ---
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                tgt_idx = batch["target_idx"].to(device)
                with torch.autocast(device, dtype=amp_dtype, enabled=use_amp):
                    pred = model(ids, tgt_idx, attention_mask=attn, n_loops=n_loops_train)
                val_preds.extend(pred.cpu().tolist())
                val_trues.extend(batch["pchembl"].tolist())

        val_preds = np.array(val_preds)
        val_trues = np.array(val_trues)
        val_mae = float(np.mean(np.abs(val_preds - val_trues)))
        val_rmse = float(np.sqrt(np.mean((val_preds - val_trues) ** 2)))
        from scipy.stats import pearsonr
        val_pearson = float(pearsonr(val_preds, val_trues).statistic) if len(val_preds) > 2 else 0.0

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_mae": val_mae, "val_rmse": val_rmse, "val_pearson": val_pearson,
        })
        logger.info(
            "epoch %d  train_loss=%.4f  val_mae=%.3f  val_rmse=%.3f  val_pearson=%.3f",
            epoch, train_loss, val_mae, val_rmse, val_pearson,
        )

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "val_mae": val_mae,
            "n_targets": len(target_vocab),
        }
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(ckpt_data, out_dir / "best.pt")
            logger.info("  -> New best val_mae=%.3f, saved best.pt", val_mae)

        if save_every_epoch:
            torch.save(ckpt_data, out_dir / f"epoch_{epoch:03d}.pt")

    # Save final
    torch.save(ckpt_data, out_dir / "latest.pt")
    elapsed = time.time() - t0

    # --- Extract the RADIANT backbone for Stage 3 ---
    # Save just the chem_model (RadiantChemModel) state_dict so Stage 3
    # fine-tuning loads it with the standard `--pretrain-ckpt` flag.
    backbone_state = {}
    for k, v in model.state_dict().items():
        if k.startswith("chem."):
            backbone_state[k[len("chem."):]] = v.cpu()
    torch.save({"model": backbone_state}, out_dir / "backbone_for_finetune.pt")
    logger.info("Saved backbone_for_finetune.pt (%d tensors) for Stage 3",
                len(backbone_state))

    result = {
        "best_val_mae": best_val_mae,
        "epochs": epochs,
        "n_targets": len(target_vocab),
        "n_train": n_train,
        "n_val": n_val,
        "elapsed_s": round(elapsed, 1),
        "rgroup_aux_weight": rgroup_aux_weight,
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Done in %.1f min. Best val_mae=%.3f", elapsed / 60, best_val_mae)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description="Stage 2: Multi-target activity pretraining")
    p.add_argument("--activities", required=True, type=Path,
                   help="activities.parquet from data curation")
    p.add_argument("--vocab", required=True, type=Path,
                   help="SMILES tokenizer vocab JSON")
    p.add_argument("--config", required=True, type=Path,
                   help="RADIANT model config JSON (e.g., radiant_75m.json)")
    p.add_argument("--stage1-ckpt", type=Path, default=None,
                   help="Stage 1 SMILES-pretrained checkpoint (latest.pt)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--n-loops-train", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-cpds", type=int, default=50,
                   help="Min compounds per target to include")
    p.add_argument("--max-targets", type=int, default=None,
                   help="Cap on number of targets (None = all)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--rgroup-aux-weight", type=float, default=0.25,
                   help="Auxiliary target-conditioned Murcko R-group activity loss weight. Set 0 to disable.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    train(
        activities_path=args.activities,
        vocab_path=args.vocab,
        config_path=args.config,
        stage1_ckpt=args.stage1_ckpt,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_loops_train=args.n_loops_train,
        min_compounds_per_target=args.min_cpds,
        max_targets=args.max_targets,
        device=args.device,
        seed=args.seed,
        rgroup_aux_weight=args.rgroup_aux_weight,
    )


if __name__ == "__main__":
    _main()
