"""Tests for the on-disk split cache.

Behaviours covered:

* cold cache miss → compute, persist a JSON, return the indices
* warm cache hit  → no recomputation; the cached file is returned verbatim
* cache invalidation when the data fingerprint disagrees (same target,
  different molecule set)
* different seed → different file (no cross-contamination)
* round-trip exact equality of indices (no list/tuple drift)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pandas")


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1",
    "CN(C)CC", "O=C1CCC1", "CCBr", "NCCc1ccc(O)c(O)c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ClCCl", "OCC(O)C(O)C(O)C(O)CO",
    "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "CCN(CC)CC", "CCCO", "CCCCO",
    "CCCCCO", "CC(C)O", "CC(C)(C)O",
]


def _make_sub(n: int = 20, seed: int = 0):
    import pandas as pd

    smi = (SMILES * 2)[:n]
    return pd.DataFrame({
        "target_chembl_id": ["TGT_X"] * n,
        "standard_smiles": smi,
        "inchikey14": [f"K{i:013d}" for i in range(n)],
        "pchembl": [4.0 + (i % 7) * 0.5 for i in range(n)],
        "pchembl_iqr": [0.0] * n,
        "n_replicates": [1] * n,
        "doc_year_min": [2018 + (i % 5) for i in range(n)],
        "doc_year_max": [2018 + (i % 5) for i in range(n)],
    })


def test_cold_miss_writes_cache(tmp_path: Path):
    from radiant_qsar.splits.cache import (
        SplitCacheConfig,
        load_or_compute_split,
        split_cache_path,
    )

    sub = _make_sub()
    cfg = SplitCacheConfig(cache_dir=tmp_path, seed=42)
    cache_path = split_cache_path("TGT_X", "scaffold", cfg)
    assert not cache_path.exists()

    train, val, test = load_or_compute_split("TGT_X", "scaffold", sub, cfg)
    assert cache_path.exists()
    assert sorted(train + val + test) == list(range(len(sub)))
    blob = json.loads(cache_path.read_text())
    assert blob["target_chembl_id"] == "TGT_X"
    assert blob["split_kind"] == "scaffold"
    assert blob["seed"] == 42
    assert blob["data_hash"]   # non-empty


def test_warm_hit_returns_cached(tmp_path: Path, monkeypatch):
    """Warm cache must not call the underlying split function."""
    from radiant_qsar.splits.cache import (
        SplitCacheConfig,
        load_or_compute_split,
    )
    from radiant_qsar.splits import cache as cache_mod

    sub = _make_sub()
    cfg = SplitCacheConfig(cache_dir=tmp_path, seed=42)
    a = load_or_compute_split("TGT_X", "scaffold", sub, cfg)

    # Sabotage the underlying compute fn: a warm hit must not invoke it.
    def _explode(*args, **kwargs):
        raise RuntimeError("warm cache should NOT recompute")
    monkeypatch.setattr(cache_mod, "_compute_split", _explode)

    b = load_or_compute_split("TGT_X", "scaffold", sub, cfg)
    assert a == b


def test_data_fingerprint_invalidates(tmp_path: Path, caplog):
    """If the molecule set changes, the cache must NOT serve stale indices."""
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    cfg = SplitCacheConfig(cache_dir=tmp_path, seed=0)
    sub_a = _make_sub(n=20)
    a = load_or_compute_split("TGT_X", "scaffold", sub_a, cfg)

    sub_b = _make_sub(n=15)  # different molecule set -> different fingerprint
    with caplog.at_level("WARNING"):
        b = load_or_compute_split("TGT_X", "scaffold", sub_b, cfg)
    # Indices must respect the new size.
    assert sorted(b[0] + b[1] + b[2]) == list(range(15))
    assert any("stale" in r.message for r in caplog.records)


def test_different_seed_different_file(tmp_path: Path):
    from radiant_qsar.splits.cache import (
        SplitCacheConfig,
        load_or_compute_split,
        split_cache_path,
    )

    sub = _make_sub()
    cfg_a = SplitCacheConfig(cache_dir=tmp_path, seed=1)
    cfg_b = SplitCacheConfig(cache_dir=tmp_path, seed=2)
    load_or_compute_split("TGT_X", "scaffold", sub, cfg_a)
    load_or_compute_split("TGT_X", "scaffold", sub, cfg_b)
    pa = split_cache_path("TGT_X", "scaffold", cfg_a)
    pb = split_cache_path("TGT_X", "scaffold", cfg_b)
    assert pa != pb
    assert pa.exists() and pb.exists()


def test_all_split_kinds_roundtrip(tmp_path: Path):
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    sub = _make_sub()
    cfg = SplitCacheConfig(cache_dir=tmp_path, seed=0)
    for kind in ("random", "scaffold", "time", "cluster", "activity_cliff"):
        train, val, test = load_or_compute_split("TGT_X", kind, sub, cfg)
        assert sorted(train + val + test) == list(range(len(sub))), f"{kind}: not a partition"


def test_different_targets_different_dirs(tmp_path: Path):
    from radiant_qsar.splits.cache import (
        SplitCacheConfig,
        load_or_compute_split,
        split_cache_path,
    )

    sub_a = _make_sub()
    sub_b = _make_sub()
    sub_a["target_chembl_id"] = "TGT_A"
    sub_b["target_chembl_id"] = "TGT_B"
    cfg = SplitCacheConfig(cache_dir=tmp_path, seed=0)
    load_or_compute_split("TGT_A", "scaffold", sub_a, cfg)
    load_or_compute_split("TGT_B", "scaffold", sub_b, cfg)
    assert split_cache_path("TGT_A", "scaffold", cfg).parent.name == "TGT_A"
    assert split_cache_path("TGT_B", "scaffold", cfg).parent.name == "TGT_B"
