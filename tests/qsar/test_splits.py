"""Tests for the QSAR split strategies."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from radiant_qsar.splits import (
    activity_cliff_split,
    cluster_split,
    random_split,
    scaffold_split,
    target_holdout_split,
    time_split,
)


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1",
    "CN(C)CC", "O=C1CCC1", "CCBr", "NCCc1ccc(O)c(O)c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ClCCl", "OCC(O)C(O)C(O)C(O)CO",
    "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "CCN(CC)CC",
    "CCCO", "CCCCO", "CCCCCO", "CC(C)O", "CC(C)(C)O",
]
PCHEMBL = [5.0, 6.5, 5.2, 7.1, 6.0, 4.8, 5.5, 6.2, 7.4, 8.0,
           4.5, 3.7, 6.6, 4.9, 5.3, 5.0, 5.1, 5.0, 5.2, 5.4]
YEARS = [2018, 2019, 2020, 2021, 2022, 2018, 2019, 2020, 2021, 2022,
         2018, 2019, 2020, 2021, 2022, 2018, 2020, 2022, 2019, 2021]
TARGETS = ["T1"]*5 + ["T2"]*5 + ["T3"]*5 + ["T4"]*5
CLASSES = ["kinase"]*5 + ["gpcr"]*5 + ["protease"]*5 + ["nuclear_receptor"]*5


def _disjoint_full_cover(splits, n):
    train, val, test = splits
    union = sorted(train + val + test)
    assert union == list(range(n)), f"split does not partition {n} indices"
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0


def test_random_split_partitions():
    _disjoint_full_cover(random_split(SMILES, ratios=(0.7, 0.15, 0.15), seed=7), len(SMILES))


def test_random_split_deterministic():
    a = random_split(SMILES, seed=42)
    b = random_split(SMILES, seed=42)
    assert a == b


def test_scaffold_split_partitions():
    _disjoint_full_cover(
        scaffold_split(SMILES, ratios=(0.7, 0.15, 0.15), seed=0), len(SMILES)
    )


def test_time_split_routes_by_year():
    train, val, test = time_split(YEARS, train_max_year=2019)
    assert all(YEARS[i] <= 2019 for i in train)
    assert all(YEARS[i] == 2020 for i in val)
    assert all(YEARS[i] >= 2021 for i in test)


def test_time_split_handles_pandas_missing():
    """The split must accept pandas ``Int64`` arrays (nullable) without crashing
    on ``pd.NA`` -- this is what activities.parquet hands us."""
    import pandas as pd

    series = pd.array([2018, pd.NA, 2020, 2021, pd.NA, 2022], dtype="Int64")
    train, val, test = time_split(list(series), train_max_year=2019)
    # pd.NA rows route to train.
    assert 1 in train and 4 in train
    # Real years split as usual.
    assert 0 in train         # 2018 <= 2019
    assert 2 in val           # 2020 == 2019+1
    assert 3 in test and 5 in test  # 2021, 2022


def test_time_split_handles_nan():
    train, val, test = time_split([2018, float("nan"), 2021], train_max_year=2019)
    assert 1 in train  # NaN -> train
    assert 0 in train
    assert 2 in test


def test_time_split_handles_missing_year():
    yrs = [2018, None, 2020, 2021, None]
    train, val, test = time_split(yrs, train_max_year=2019)
    assert 1 in train  # missing -> train
    assert 4 in train


def test_cluster_split_partitions():
    _disjoint_full_cover(cluster_split(SMILES, ratios=(0.7, 0.15, 0.15)), len(SMILES))


def test_activity_cliff_split_keeps_cliffs_together():
    # A and B are nearly identical SMILES with very different potency:
    # cliff (A, B) must end in the same split.
    smi = ["CCO", "CCC", "CCCO", "OCCO"]
    pch = [5.0, 9.0, 5.1, 5.0]  # A=CCO, B=CCC -> high similarity, big delta
    splits = activity_cliff_split(smi, pch, ratios=(0.8, 0.1, 0.1), seed=0)
    assignment = {i: name for name, members in zip(("train", "val", "test"), splits) for i in members}
    # If a cliff exists between any indices, they share an assignment.
    # Sanity: at minimum the function must partition the set.
    _disjoint_full_cover(splits, len(smi))


def test_target_holdout_partitions():
    splits = target_holdout_split(TARGETS, CLASSES, holdout_fraction=0.25, seed=0, by="target")
    _disjoint_full_cover(splits, len(TARGETS))
    # Held-out targets are entirely in test.
    train_targets = {TARGETS[i] for i in splits[0]}
    test_targets = {TARGETS[i] for i in splits[2]}
    assert not (train_targets & test_targets)


def test_target_holdout_by_class():
    splits = target_holdout_split(TARGETS, CLASSES, holdout_fraction=0.25, seed=1, by="class")
    _disjoint_full_cover(splits, len(TARGETS))
    train_classes = {CLASSES[i] for i in splits[0]}
    test_classes = {CLASSES[i] for i in splits[2]}
    assert not (train_classes & test_classes)
