"""Train/val/test splitters for molecular datasets.

* :func:`random_split` -- deterministic random split by ``seed``.
* :func:`scaffold_split` -- groups molecules by Bemis-Murcko scaffold and
  assigns whole groups to splits, so molecules sharing a scaffold do not
  leak across train/test. Falls back to a deterministic-hash split when
  rdkit is unavailable, with a warning.
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections import defaultdict
from typing import Iterable


def _three_way_lengths(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    a, b, c = ratios
    s = a + b + c
    if s <= 0:
        raise ValueError("ratios must sum to a positive number")
    a, b, c = a / s, b / s, c / s
    n_train = int(round(n * a))
    n_val = int(round(n * b))
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def random_split(
    items: Iterable,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Return three lists of *indices* into ``items`` for train/val/test."""
    items_list = list(items)
    n = len(items_list)
    n_train, n_val, n_test = _three_way_lengths(n, ratios)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    return idxs[:n_train], idxs[n_train : n_train + n_val], idxs[n_train + n_val :]


def _murcko_scaffold(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except ImportError:
        return None
    except Exception:
        return None


def _hash_bucket(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def scaffold_split(
    smiles: list[str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Group molecules by scaffold; assign groups (largest first) to train, then val, then test.

    With rdkit installed this uses Bemis-Murcko scaffolds. Without rdkit it
    falls back to grouping by a deterministic SHA-256 prefix of each SMILES,
    which is *not* chemically meaningful but at least gives a reproducible,
    seed-independent grouping.
    """
    n = len(smiles)
    n_train, n_val, _ = _three_way_lengths(n, ratios)

    # Group indices by scaffold key.
    groups: dict[str, list[int]] = defaultdict(list)
    rdkit_used = False
    for i, s in enumerate(smiles):
        key = _murcko_scaffold(s)
        if key is None:
            key = _hash_bucket(s)
        else:
            rdkit_used = True
        groups[key].append(i)

    if not rdkit_used:
        warnings.warn(
            "scaffold_split fell back to hash-based grouping (rdkit unavailable).",
            stacklevel=2,
        )

    # Sort groups by descending size, then deterministic hash for tie-breaking.
    rng = random.Random(seed)
    group_keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]), rng.random()))

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for key in group_keys:
        idxs = groups[key]
        if len(train_idx) + len(idxs) <= n_train:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    return train_idx, val_idx, test_idx
