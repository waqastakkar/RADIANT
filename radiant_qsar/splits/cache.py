"""On-disk cache for the (target, split) -> (train_idx, val_idx, test_idx) map.

A single sweep over the 20-target panel × 5 splits × 5 models invokes the
split logic 500 times. The activity-cliff and Butina-cluster splits do
O(n²) Tanimoto work on each call -- recomputing them per cell wastes
hours.

The cache is content-addressed by both:

* a filename ``<target>__<split>__seed<N>.json`` (so different seeds
  coexist, and warm-cache lookups are O(1) file open), and
* a SHA-256 fingerprint of the sorted ``inchikey14`` list embedded in
  each cache file (so any change to the curated data automatically
  invalidates the cache and forces recomputation).

The cache is read-after-write consistent within one process and safe to
share across processes -- json.loads + json.dumps are atomic-enough for
this use case (the sweep runs cells sequentially in subprocesses, and
the precompute CLI is single-process).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


DEFAULT_CACHE_DIR = Path(
    os.environ.get("RADIANT_SPLITS_DIR", "data/splits/v1")
)


@dataclass
class SplitCacheConfig:
    """Per-call config: cache location + parameters that drive the split."""

    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    seed: int = 1337
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    sim: float = 0.9                  # activity_cliff Tanimoto threshold
    delta: float = 1.0                # activity_cliff pchembl delta threshold

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)


# ---------------------------------------------------------------------------
def split_cache_path(target_chembl_id: str, split_kind: str, cfg: SplitCacheConfig) -> Path:
    return cfg.cache_dir / target_chembl_id / f"{split_kind}__seed{cfg.seed}.json"


def _data_fingerprint(sub) -> str:
    """SHA-256 prefix over the sorted inchikey14 list of the target's rows.

    Same molecules -> same fingerprint, regardless of row order. If the
    curation pipeline reruns and the molecule set changes, the fingerprint
    changes and the cache invalidates."""
    keys = sorted(sub["inchikey14"].astype(str).tolist())
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:16]


def _compute_split(
    target: str,
    split_kind: str,
    sub,
    cfg: SplitCacheConfig,
) -> tuple[list[int], list[int], list[int]]:
    smi = sub["standard_smiles"].tolist()
    pch = sub["pchembl"].astype(float).tolist()
    if split_kind == "random":
        from radiant_qsar.splits import random_split
        return random_split(smi, ratios=cfg.ratios, seed=cfg.seed)
    if split_kind == "scaffold":
        from radiant_qsar.splits import scaffold_split
        return scaffold_split(smi, ratios=cfg.ratios, seed=cfg.seed)
    if split_kind == "cluster":
        from radiant_qsar.splits import cluster_split
        return cluster_split(smi, ratios=cfg.ratios, seed=cfg.seed)
    if split_kind == "time":
        from radiant_qsar.splits import time_split
        years = sub["doc_year_max"].astype("Int64").tolist()
        return time_split(years)
    if split_kind == "activity_cliff":
        from radiant_qsar.splits import activity_cliff_split
        return activity_cliff_split(
            smi, pch, ratios=cfg.ratios,
            sim_threshold=cfg.sim, delta_pchembl=cfg.delta, seed=cfg.seed,
        )
    raise ValueError(f"unknown split kind {split_kind!r}")


# ---------------------------------------------------------------------------
def load_or_compute_split(
    target_chembl_id: str,
    split_kind: str,
    sub,
    cfg: SplitCacheConfig | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Cache-aware split. Returns ``(train_idx, val_idx, test_idx)`` lists.

    On a cold cache, computes the split, persists it as JSON, and returns.
    On a warm cache with matching data fingerprint, reads from disk without
    recomputing. On a warm cache whose fingerprint disagrees with the
    current data, logs a warning and recomputes.
    """
    cfg = cfg or SplitCacheConfig()
    fp = _data_fingerprint(sub)
    path = split_cache_path(target_chembl_id, split_kind, cfg)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("split cache %s unreadable (%s); recomputing", path, exc)
            data = None
        if (
            data is not None
            and data.get("data_hash") == fp
            and data.get("seed") == cfg.seed
            and tuple(data.get("ratios", ())) == cfg.ratios
        ):
            return list(data["train_idx"]), list(data["val_idx"]), list(data["test_idx"])
        logger.warning(
            "split cache %s stale (data_hash mismatch or seed/ratios changed); recomputing",
            path,
        )

    train, val, test = _compute_split(target_chembl_id, split_kind, sub, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_chembl_id": target_chembl_id,
        "split_kind": split_kind,
        "seed": cfg.seed,
        "ratios": list(cfg.ratios),
        "activity_cliff_sim": cfg.sim,
        "activity_cliff_delta": cfg.delta,
        "data_hash": fp,
        "n_compounds": int(len(sub)),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_idx": list(train),
        "val_idx": list(val),
        "test_idx": list(test),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info(
        "split cache wrote %s (train=%d val=%d test=%d)",
        path, len(train), len(val), len(test),
    )
    return list(train), list(val), list(test)
