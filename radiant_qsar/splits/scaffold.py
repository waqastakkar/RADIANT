"""Bemis-Murcko scaffold split.

Groups molecules by their Murcko scaffold and assigns whole groups to
splits, so molecules sharing a scaffold do not leak across train/test.
Falls back to a deterministic SHA-256-prefix grouping when rdkit is
unavailable -- not chemically meaningful but reproducible.

The convention used here is "largest scaffolds in train, smallest in test"
(MoleculeNet convention), which gives the test set the most novel
scaffolds.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from typing import Sequence

from radiant_qsar.splits.random import _three_lengths


def _murcko_scaffold(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaffold, canonical=True)
        return s if s else "EMPTY"
    except ImportError:
        return None
    except Exception:
        return None


def _hash_bucket(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def scaffold_split(
    smiles: Sequence[str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    largest_to_train: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Split by Murcko scaffold groups.

    Args:
        smiles: list of SMILES (one per row of the activity table).
        ratios: train/val/test fractions; will be normalized.
        seed:   tie-break randomness for equal-sized scaffolds.
        largest_to_train: if True, biggest scaffolds go to train (MoleculeNet
            convention). If False, biggest go to test (gives larger but more
            in-distribution test sets, occasionally desired).

    Returns:
        (train_idx, val_idx, test_idx) sorted lists.
    """
    n = len(smiles)
    n_train, n_val, _ = _three_lengths(n, ratios)

    rdkit_used = False
    groups: dict[str, list[int]] = defaultdict(list)
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

    import random as _r

    rng = _r.Random(seed)
    keys = sorted(groups.keys(), key=lambda k: (-len(groups[k]) if largest_to_train else len(groups[k]), rng.random()))

    train, val, test = [], [], []
    for key in keys:
        idxs = groups[key]
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)
    return sorted(train), sorted(val), sorted(test)
