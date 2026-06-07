"""Activity-cliff-aware split.

A pair (i, j) is an *activity cliff* if Tanimoto(MorganFP_i, MorganFP_j) >= 0.9
and |pchembl_i - pchembl_j| >= 1.0 (one log unit of potency).

This splitter keeps cliff pairs **together** in the same split (whichever
split the larger member of the pair goes to) so that test-set cliffs cannot
be solved by simple memorization of the closely related training molecule.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from radiant_qsar.splits.random import _three_lengths


def _find_cliff_pairs(
    smiles: Sequence[str],
    pchembl: Sequence[float],
    *,
    sim_threshold: float,
    delta_pchembl: float,
) -> list[tuple[int, int]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        warnings.warn("activity_cliff_split requires rdkit; returning no cliffs.", stacklevel=2)
        return []

    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None)

    pairs = []
    for i in range(len(fps)):
        if fps[i] is None:
            continue
        # Compare against j > i only (symmetric).
        cmp_fps = [f for f in fps[i + 1 :]]
        if not cmp_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], cmp_fps)
        for offset, sim in enumerate(sims):
            j = i + 1 + offset
            if fps[j] is None:
                continue
            if sim >= sim_threshold and abs(float(pchembl[i]) - float(pchembl[j])) >= delta_pchembl:
                pairs.append((i, j))
    return pairs


def activity_cliff_split(
    smiles: Sequence[str],
    pchembl: Sequence[float],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    sim_threshold: float = 0.9,
    delta_pchembl: float = 1.0,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Split such that cliff pairs land in the same split.

    Implementation: union-find over cliff pairs to form connected
    components; each component is assigned as a unit to train/val/test.
    """
    if len(smiles) != len(pchembl):
        raise ValueError("smiles and pchembl must be the same length")

    n = len(smiles)
    pairs = _find_cliff_pairs(
        smiles, pchembl, sim_threshold=sim_threshold, delta_pchembl=delta_pchembl
    )

    # Union-find.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs:
        union(a, b)

    # Group by root.
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    n_train, n_val, _ = _three_lengths(n, ratios)
    import random as _r

    rng = _r.Random(seed)
    keys = sorted(components.keys(), key=lambda k: (-len(components[k]), rng.random()))

    train, val, test = [], [], []
    for k in keys:
        idxs = components[k]
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)
    return sorted(train), sorted(val), sorted(test)
