"""Shared fine-tuning helper for HuggingFace transformer baselines.

Both ChemBERTa-2 and MolFormer are SMILES-tokenizing transformers exposed
through the standard ``transformers`` API. The differences between them
that matter for our purposes are:

* the HF model ID (passed in as ``model_id``)
* whether the tokenizer needs ``trust_remote_code=True`` (MolFormer's
  custom tokenizer lives in the model repo)
* the input column the tokenizer expects (always raw SMILES)

Everything else -- regression head replacement, AdamW with warmup,
canonical metric schema, predictions CSV, model bundle -- is shared.
This module exposes a single :func:`train_hf_baseline` that the per-model
CLIs (chemberta.py / molformer.py) wrap with their own defaults.

Outputs (per ``--out`` directory)
---------------------------------
* ``result.json``   -- canonical schema (``train``/``val``/``test`` blocks).
* ``model.pt``      -- ``{"state_dict": ..., "config": {model_id, max_len, regression_target}}``.
* ``predictions.csv`` -- per-test-molecule rows.

The trained checkpoint is loadable by :func:`predict_smiles_from_ckpt`,
which the screening pipeline uses through the
``HFTransformerPotency`` filter (see ``screening/filters/ml_scoring.py``
when wired up).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


MODEL_FILENAME = "model.pt"


@dataclass
class HFBaselineConfig:
    activities: Path
    target_chembl_id: str
    out: Path
    model_id: str                            # HF model identifier
    baseline_name: str                       # "chemberta" / "molformer" -- writes through to result.json
    split_kind: str = "scaffold"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = 256
    seed: int = 1337
    device: str = "cuda"
    grad_clip: float = 1.0
    early_stopping_patience: int = 5
    splits_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    activity_cliff_sim: float = 0.9
    activity_cliff_delta: float = 1.0
    trust_remote_code: bool = False          # MolFormer needs True
    pooling: str = "cls"                     # "cls" | "mean"
    extra_meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Splits (delegated to the shared splits package; same as morgan_rf).
# ---------------------------------------------------------------------------
def _split(sub, kind, ratios, seed, *, sim, delta):
    """Cache-aware split. See :mod:`radiant_qsar.splits.cache`."""
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    target = sub["target_chembl_id"].iloc[0]
    cfg = SplitCacheConfig(seed=seed, ratios=tuple(ratios), sim=sim, delta=delta)
    return load_or_compute_split(target, kind, sub, cfg)


# ---------------------------------------------------------------------------
# Model: backbone + regression head
# ---------------------------------------------------------------------------
def _make_model(model_id: str, *, trust_remote_code: bool, dropout: float = 0.1):
    """Load the HF backbone and attach a fresh 1-output regression head.

    We deliberately do NOT use ``AutoModelForSequenceClassification`` --
    its head can have unexpected layer names that don't checkpoint
    cleanly. Instead we attach our own simple ``[Linear -> GELU ->
    Dropout -> Linear(1)]`` head over the pooled hidden state.
    """
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    backbone = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or getattr(cfg, "n_embd", 768)

    class _Wrapper(nn.Module):
        def __init__(self, backbone, hidden_size, head_dropout: float = 0.1):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, input_ids, attention_mask, *, pooling: str = "cls"):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state                  # (B, S, H)
            if pooling == "cls":
                pooled = h[:, 0]                       # CLS / first token
            else:                                       # mean over non-pad
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            return self.head(pooled).squeeze(-1)        # (B,)

    return _Wrapper(backbone, hidden, head_dropout=dropout), hidden


# ---------------------------------------------------------------------------
# Save / load / predict
# ---------------------------------------------------------------------------
def _library_versions() -> dict:
    out = {}
    for mod in ("torch", "transformers", "rdkit", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not-installed"
    return out


def save_bundle(model, tokenizer, cfg: HFBaselineConfig, dest: Path) -> Path:
    """Persist enough to reload + score new SMILES later. Tokenizer is saved
    via ``save_pretrained`` so even for HF models with custom code the
    reload path is local-only after this point."""
    import torch

    tokenizer_dir = dest.parent / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    payload = {
        "state_dict": model.state_dict(),
        "model_id": cfg.model_id,
        "baseline_name": cfg.baseline_name,
        "max_seq_len": cfg.max_seq_len,
        "trust_remote_code": cfg.trust_remote_code,
        "pooling": cfg.pooling,
        "tokenizer_dir": tokenizer_dir.name,
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "seed": cfg.seed,
        "library_versions": _library_versions(),
        "build_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "schema_version": 1,
    }
    torch.save(payload, dest)
    return dest


def load_bundle(path: Path | str, device: str = "cpu"):
    """Load (model, tokenizer, payload) from a path written by :func:`save_bundle`."""
    import torch
    from transformers import AutoTokenizer

    payload = torch.load(path, map_location=device, weights_only=False)
    tok_dir = Path(path).parent / payload.get("tokenizer_dir", "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tok_dir),
        trust_remote_code=payload.get("trust_remote_code", False),
    )
    model, _ = _make_model(payload["model_id"], trust_remote_code=payload.get("trust_remote_code", False))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, tokenizer, payload


def predict_smiles_from_ckpt(
    ckpt_path: Path | str,
    smiles: Sequence[str],
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Score a list of SMILES with a previously-trained HF baseline.

    Returns ``(N,)`` predicted pchembl values. Does no error-suppression on
    the tokenizer side -- the HF tokenizers gracefully handle anything the
    model was trained on; truly garbage strings return finite numbers
    (the model just predicts based on whatever embedding it produces).
    """
    import torch

    model, tokenizer, payload = load_bundle(ckpt_path, device=device)
    max_len = payload["max_seq_len"]
    pooling = payload.get("pooling", "cls")

    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            chunk = list(smiles[i : i + batch_size])
            enc = tokenizer(chunk, padding=True, truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            y = model(ids, attn, pooling=pooling).cpu().numpy().tolist()
            preds.extend(y)
    return np.asarray(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = int(pred.size)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # Pearson + Spearman; lazy import scipy to keep this module light if a
    # caller only wants train/predict.
    try:
        from scipy.stats import pearsonr, spearmanr
        p = float(pearsonr(pred, true).statistic)
        s = float(spearmanr(pred, true).statistic)
    except Exception:
        p = float(np.corrcoef(pred, true)[0, 1]) if n > 1 else float("nan")
        s = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": p, "spearman": s, "n": n}


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_hf_baseline(cfg: HFBaselineConfig) -> dict:
    """Fine-tune an HF transformer regression head on a single target.

    Returns the same canonical-schema result dict that :mod:`single_task`
    and :mod:`morgan_rf` write, so :mod:`sweep.aggregate_results` consumes
    all three uniformly.
    """
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = pd.read_parquet(cfg.activities)
    sub = df[df["target_chembl_id"] == cfg.target_chembl_id].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"no rows for target {cfg.target_chembl_id}")
    train_idx, val_idx, test_idx = _split(
        sub, cfg.split_kind, cfg.splits_ratios, cfg.seed,
        sim=cfg.activity_cliff_sim, delta=cfg.activity_cliff_delta,
    )
    smi = sub["standard_smiles"].tolist()
    pch = sub["pchembl"].astype(float).values
    logger.info("%s/%s [%s] sizes: train=%d val=%d test=%d",
                cfg.baseline_name, cfg.target_chembl_id, cfg.split_kind,
                len(train_idx), len(val_idx), len(test_idx))

    logger.info("loading %s", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    model, _hidden = _make_model(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    model.to(cfg.device)

    class _DS(Dataset):
        def __init__(self, idxs):
            self.idxs = list(idxs)
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, j):
            i = self.idxs[j]
            return smi[i], float(pch[i]), int(i)

    def _collate(batch):
        smis, ys, idxs = zip(*batch)
        enc = tokenizer(list(smis), padding=True, truncation=True,
                        max_length=cfg.max_seq_len, return_tensors="pt")
        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor(ys, dtype=torch.float32),
                torch.tensor(idxs, dtype=torch.long))

    loaders = {
        "train": DataLoader(_DS(train_idx), batch_size=cfg.batch_size,
                            shuffle=True, collate_fn=_collate),
        "val":   DataLoader(_DS(val_idx),   batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=_collate),
        "test":  DataLoader(_DS(test_idx),  batch_size=cfg.batch_size,
                            shuffle=False, collate_fn=_collate),
    }
    n_steps = max(1, len(loaders["train"]) * cfg.epochs)
    n_warmup = int(n_steps * cfg.warmup_ratio)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=n_warmup, num_training_steps=n_steps)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best = {"val_pearson": -1.0, "epoch": -1, "state": None}
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for ids, attn, y, _ in loaders["train"]:
            ids = ids.to(cfg.device); attn = attn.to(cfg.device); y = y.to(cfg.device)
            pred = model(ids, attn, pooling=cfg.pooling)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
            sched.step()
            epoch_loss += loss.item() * y.size(0); n_seen += y.size(0)
        train_loss = epoch_loss / max(n_seen, 1)

        # val
        model.eval()
        with torch.no_grad():
            pv, tv = [], []
            for ids, attn, y, _ in loaders["val"]:
                pv.extend(model(ids.to(cfg.device), attn.to(cfg.device), pooling=cfg.pooling).cpu().tolist())
                tv.extend(y.tolist())
        val_m = _metrics(np.array(pv), np.array(tv))
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        if epoch % 2 == 0 or epoch == cfg.epochs - 1:
            logger.info("ep %2d  train_loss=%.4f  val_mae=%.3f  val_rho=%.3f",
                        epoch, train_loss, val_m["mae"], val_m["pearson"])
        if val_m["pearson"] > best["val_pearson"]:
            best.update(val_pearson=val_m["pearson"], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > cfg.early_stopping_patience:
                logger.info("early stop at epoch %d", epoch)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        # Train metrics (on best checkpoint, for consistency with morgan_rf)
        ptr, ttr = [], []
        for ids, attn, y, _ in loaders["train"]:
            ptr.extend(model(ids.to(cfg.device), attn.to(cfg.device), pooling=cfg.pooling).cpu().tolist())
            ttr.extend(y.tolist())
        train_m = _metrics(np.array(ptr), np.array(ttr))
        # Test metrics
        pt, tt, idxs = [], [], []
        for ids, attn, y, idx in loaders["test"]:
            pt.extend(model(ids.to(cfg.device), attn.to(cfg.device), pooling=cfg.pooling).cpu().tolist())
            tt.extend(y.tolist()); idxs.extend(idx.tolist())
    test_m = _metrics(np.array(pt), np.array(tt))
    elapsed = time.time() - t0

    # 1. save bundle
    save_bundle(model, tokenizer, cfg, cfg.out / MODEL_FILENAME)

    # 2. predictions.csv (canonical schema -- joinable to descriptors.parquet via inchikey14)
    from radiant_qsar.eval.predictions import write_predictions

    test_smi = [smi[int(i)] for i in idxs]
    test_inchikeys = sub["inchikey14"].iloc[list(idxs)].tolist()
    write_predictions(
        cfg.out,
        indices=idxs,
        inchikeys=test_inchikeys,
        smiles=test_smi,
        true_pchembl=tt,
        pred_pchembl=pt,
        target_chembl_id=cfg.target_chembl_id,
        split_kind=cfg.split_kind,
    )

    # 3. result.json
    best_val_block = next(
        ({k.replace("val_", ""): v for k, v in row.items() if k.startswith("val_")}
         for row in history if row["epoch"] == best["epoch"]),
        {},
    )
    result = {
        "model": cfg.baseline_name,
        "model_id": cfg.model_id,
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "best_val_epoch": best["epoch"], "best_val_pearson": best["val_pearson"],
        "train": train_m,
        "val": best_val_block,
        "test": test_m,
        "model_path": MODEL_FILENAME,
        "predictions_path": "predictions.csv",
        "elapsed_s": round(elapsed, 1),
        "library_versions": _library_versions(),
        **cfg.extra_meta,
    }
    (cfg.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("%s/%s [%s] done: test MAE=%.3f rho=%.3f (n=%d) in %.1fs",
                cfg.baseline_name, cfg.target_chembl_id, cfg.split_kind,
                test_m["mae"], test_m["pearson"], test_m["n"], elapsed)
    return result
