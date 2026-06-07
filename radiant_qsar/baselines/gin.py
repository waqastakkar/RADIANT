"""Graph Isomorphism Network (GIN) baseline -- pure torch + rdkit.

A standard message-passing GNN: 5 GIN layers, atom-level features from
rdkit, mean+max pooling, MLP regression head. Trained from scratch
(no pretrain) on each (target, split) cell.

We deliberately avoid PyTorch Geometric / DGL -- both are large
dependencies and the GIN layer is short enough to write in pure torch
using ``index_add_`` for scatter-aggregation. Memory-efficient enough
for batches of ~1000 medium-sized molecules per GPU step.

Citation
--------
Xu et al., 'How Powerful are Graph Neural Networks?', ICLR 2019.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


MODEL_FILENAME = "model.pt"


# ---------------------------------------------------------------------------
# Atom + bond featurization
# ---------------------------------------------------------------------------
# Atom features: 11+7+6+6+5+2+5 = 42 dims
_ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14]      # C N O F P S Cl Br I B Si
_DEGREES = list(range(7))                                   # 0..6
_FORMAL_CHARGES = [-2, -1, 0, 1, 2, 3]
_HYBRIDIZATIONS = ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"]
_NUM_HS = list(range(5))
_CHIRAL_TAGS = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER", "_OTHER"]


def _onehot(value, allowed: list) -> list[int]:
    out = [0] * len(allowed)
    if value in allowed:
        out[allowed.index(value)] = 1
    else:
        out[-1] = 1   # last slot = "other"
    return out


def _atom_features(atom) -> list[float]:
    f = []
    f += _onehot(atom.GetAtomicNum(), _ATOM_LIST)
    f += _onehot(atom.GetDegree(), _DEGREES)
    f += _onehot(atom.GetFormalCharge(), _FORMAL_CHARGES)
    f += _onehot(str(atom.GetHybridization()).rsplit(".", 1)[-1], _HYBRIDIZATIONS)
    f += _onehot(atom.GetTotalNumHs(), _NUM_HS)
    f += [int(atom.GetIsAromatic()), int(atom.IsInRing())]
    f += _onehot(str(atom.GetChiralTag()).rsplit(".", 1)[-1], _CHIRAL_TAGS)
    return [float(x) for x in f]


def _smiles_to_graph(smi: str):
    """Returns (atom_features [N, F], edge_index [2, 2E]) or (None, None) on failure."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    feats = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    src, dst = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [a, b]
        dst += [b, a]
    if not src:
        # Single-atom (or all-isolated) "molecule" -- give it a self-loop
        # so message passing has at least an identity step.
        for i in range(mol.GetNumAtoms()):
            src.append(i); dst.append(i)
    edge_index = np.array([src, dst], dtype=np.int64)
    return feats, edge_index


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class GINConfig:
    activities: Path
    target_chembl_id: str
    out: Path
    split_kind: str = "scaffold"
    n_layers: int = 5
    hidden_dim: int = 256
    dropout: float = 0.1
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 1337
    device: str = "cuda"
    early_stopping_patience: int = 8
    splits_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    activity_cliff_sim: float = 0.9
    activity_cliff_delta: float = 1.0


def _atom_feat_dim() -> int:
    return (len(_ATOM_LIST) + len(_DEGREES) + len(_FORMAL_CHARGES)
            + len(_HYBRIDIZATIONS) + len(_NUM_HS) + 2 + len(_CHIRAL_TAGS))


class GINLayer(nn.Module):
    """Single GIN layer: ``h_i' = MLP((1 + eps) h_i + sum_j h_j)``."""

    def __init__(self, dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Aggregate neighbor messages via index_add_ on the destination dim.
        agg = torch.zeros_like(h)
        agg.index_add_(0, edge_index[1], h[edge_index[0]])
        return self.mlp((1.0 + self.eps) * h + agg)


class GINRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GINLayer(hidden_dim) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),  # mean+max concat -> 2*hidden
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index)
        n_graphs = int(batch_idx.max().item()) + 1
        D = h.size(1)
        # Per-graph mean.
        sum_per_graph = torch.zeros(n_graphs, D, device=h.device).index_add_(0, batch_idx, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=torch.float))
        mean_per_graph = sum_per_graph / counts.unsqueeze(-1).clamp(min=1)
        # Per-graph max via scatter-style max -- pure torch.
        max_per_graph = torch.full((n_graphs, D), -float("inf"), device=h.device)
        max_per_graph.index_reduce_(0, batch_idx, h, reduce="amax", include_self=True)
        # Mask out unfilled rows (graphs with zero atoms shouldn't occur here, but be safe).
        finite = torch.isfinite(max_per_graph)
        max_per_graph = torch.where(finite, max_per_graph, torch.zeros_like(max_per_graph))
        pooled = torch.cat([mean_per_graph, max_per_graph], dim=-1)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def _collate_graphs(batch):
    feats, edges, ys, idxs = zip(*batch)
    n_atoms = [f.shape[0] for f in feats]
    offsets = np.cumsum([0] + n_atoms[:-1])
    x = np.concatenate(feats, axis=0)
    big_edges = np.concatenate([e + off for e, off in zip(edges, offsets)], axis=1) if edges else np.zeros((2, 0), dtype=np.int64)
    batch_idx = np.concatenate([np.full(n, i, dtype=np.int64) for i, n in enumerate(n_atoms)])
    return (
        torch.from_numpy(x),
        torch.from_numpy(big_edges),
        torch.from_numpy(batch_idx),
        torch.tensor(ys, dtype=torch.float32),
        torch.tensor(idxs, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Save / load / predict
# ---------------------------------------------------------------------------
def save_bundle(model, cfg: GINConfig, dest: Path) -> Path:
    payload = {
        "state_dict": model.state_dict(),
        "in_dim": _atom_feat_dim(),
        "hidden_dim": cfg.hidden_dim,
        "n_layers": cfg.n_layers,
        "dropout": cfg.dropout,
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "seed": cfg.seed,
        "build_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "schema_version": 1,
    }
    torch.save(payload, dest)
    return dest


def load_bundle(path: Path | str, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GINRegressor(payload["in_dim"], payload["hidden_dim"], payload["n_layers"], payload.get("dropout", 0.1))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def predict_smiles_from_ckpt(
    ckpt_path: Path | str,
    smiles: Sequence[str],
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    model, _ = load_bundle(ckpt_path, device=device)
    preds: list[float] = []
    with torch.no_grad():
        for i in range(0, len(smiles), batch_size):
            chunk = list(smiles[i : i + batch_size])
            graphs = []
            valid_pos = []
            for j, s in enumerate(chunk):
                f, e = _smiles_to_graph(s)
                if f is not None:
                    graphs.append((f, e, 0.0, 0))
                    valid_pos.append(j)
            chunk_pred: list[float] = [float("nan")] * len(chunk)
            if graphs:
                x, ei, b, _, _ = _collate_graphs(graphs)
                x = x.to(device); ei = ei.to(device); b = b.to(device)
                y = model(x, ei, b).cpu().numpy().tolist()
                for pos, val in zip(valid_pos, y):
                    chunk_pred[pos] = float(val)
            preds.extend(chunk_pred)
    return np.asarray(preds, dtype=np.float32)


# ---------------------------------------------------------------------------
# Splits + metrics (delegated)
# ---------------------------------------------------------------------------
def _split(sub, kind, ratios, seed, *, sim, delta):
    """Cache-aware split. See :mod:`radiant_qsar.splits.cache`."""
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    target = sub["target_chembl_id"].iloc[0]
    cfg = SplitCacheConfig(seed=seed, ratios=tuple(ratios), sim=sim, delta=delta)
    return load_or_compute_split(target, kind, sub, cfg)


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float); true = np.asarray(true, dtype=float)
    n = int(pred.size)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
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
def train_gin(cfg: GINConfig) -> dict:
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset

    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

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
    logger.info("gin/%s [%s] sizes: train=%d val=%d test=%d",
                cfg.target_chembl_id, cfg.split_kind,
                len(train_idx), len(val_idx), len(test_idx))

    # Pre-featurize once (graphs are small; this keeps training tight).
    graphs: list[tuple] = []
    bad = []
    for i, s in enumerate(smi):
        f, e = _smiles_to_graph(s)
        if f is None:
            bad.append(i)
            graphs.append(None)
        else:
            graphs.append((f, e))
    train_idx = [i for i in train_idx if graphs[i] is not None]
    val_idx   = [i for i in val_idx   if graphs[i] is not None]
    test_idx  = [i for i in test_idx  if graphs[i] is not None]
    if bad:
        logger.warning("dropped %d unparseable molecules from this target", len(bad))

    class _DS(Dataset):
        def __init__(self, idxs):
            self.idxs = list(idxs)
        def __len__(self):
            return len(self.idxs)
        def __getitem__(self, j):
            i = self.idxs[j]
            f, e = graphs[i]
            return f, e, float(pch[i]), int(i)

    loaders = {
        "train": DataLoader(_DS(train_idx), batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_graphs),
        "val":   DataLoader(_DS(val_idx),   batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_graphs),
        "test":  DataLoader(_DS(test_idx),  batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_graphs),
    }

    model = GINRegressor(_atom_feat_dim(), cfg.hidden_dim, cfg.n_layers, cfg.dropout).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best = {"val_pearson": -1.0, "epoch": -1, "state": None}
    bad_epochs = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for x, ei, b, y, _ in loaders["train"]:
            x = x.to(cfg.device); ei = ei.to(cfg.device); b = b.to(cfg.device); y = y.to(cfg.device)
            pred = model(x, ei, b)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item() * y.size(0); n_seen += y.size(0)
        train_loss = epoch_loss / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            pv, tv = [], []
            for x, ei, b, y, _ in loaders["val"]:
                x = x.to(cfg.device); ei = ei.to(cfg.device); b = b.to(cfg.device)
                pv.extend(model(x, ei, b).cpu().tolist()); tv.extend(y.tolist())
        val_m = _metrics(np.array(pv), np.array(tv))
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
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
        pt, tt, idxs = [], [], []
        for x, ei, b, y, idx in loaders["test"]:
            x = x.to(cfg.device); ei = ei.to(cfg.device); b = b.to(cfg.device)
            pt.extend(model(x, ei, b).cpu().tolist()); tt.extend(y.tolist()); idxs.extend(idx.tolist())
    test_m = _metrics(np.array(pt), np.array(tt))
    elapsed = time.time() - t0

    save_bundle(model, cfg, cfg.out / MODEL_FILENAME)

    # Canonical predictions.csv (joinable to descriptors.parquet via inchikey14).
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

    best_val_block = next(
        ({k.replace("val_", ""): v for k, v in row.items() if k.startswith("val_")}
         for row in history if row["epoch"] == best["epoch"]),
        {},
    )
    result = {
        "model": "gin",
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "best_val_epoch": best["epoch"], "best_val_pearson": best["val_pearson"],
        "n_layers": cfg.n_layers, "hidden_dim": cfg.hidden_dim,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "val": best_val_block, "test": test_m,
        "model_path": MODEL_FILENAME, "predictions_path": "predictions.csv",
        "elapsed_s": round(elapsed, 1),
    }
    (cfg.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("gin/%s [%s] done: test MAE=%.3f rho=%.3f (n=%d) in %.1fs",
                cfg.target_chembl_id, cfg.split_kind,
                test_m["mae"], test_m["pearson"], test_m["n"], elapsed)
    return result


# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="scaffold",
                   choices=("random", "scaffold", "time", "cluster", "activity_cliff"))
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_gin(GINConfig(
        activities=args.activities, target_chembl_id=args.target, out=args.out,
        split_kind=args.split, n_layers=args.n_layers, hidden_dim=args.hidden_dim,
        dropout=args.dropout, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=args.device, seed=args.seed,
    ))


if __name__ == "__main__":
    _main()
