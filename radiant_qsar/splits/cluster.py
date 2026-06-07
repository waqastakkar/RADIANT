"""Butina clustering on 2048-bit Morgan fingerprints.

Tanimoto-distance threshold 0.65 (i.e., similarity >= 0.35 within a cluster).
Whole clusters are partitioned to splits so chemical-space leakage is
prevented at a stronger level than scaffold split.
"""

from __future__ import annotations

import warnings
from typing import Sequence

from radiant_qsar.splits.random import _three_lengths


def _morgan_fps(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits))
    return fps


def cluster_split(
    smiles: Sequence[str],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    distance_threshold: float = 0.65,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Butina-cluster the molecules and assign whole clusters to splits."""
    try:
        from rdkit.Chem import AllChem  # noqa: F401
        from rdkit.ML.Cluster import Butina
        from rdkit import DataStructs
    except ImportError:
        warnings.warn(
            "cluster_split requires rdkit; falling back to scaffold split.",
            stacklevel=2,
        )
        from radiant_qsar.splits.scaffold import scaffold_split

        return scaffold_split(smiles, ratios=ratios, seed=seed)

    n = len(smiles)
    n_train, n_val, _ = _three_lengths(n, ratios)
    fps = _morgan_fps(smiles)

    # Map of valid indices.
    valid_idx = [i for i, f in enumerate(fps) if f is not None]
    if not valid_idx:
        return [], [], []

    # Build pairwise Tanimoto distance vector in the order Butina expects.
    dists = []
    valid_fps = [fps[i] for i in valid_idx]
    for i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(
        dists,
        len(valid_fps),
        distance_threshold,
        isDistData=True,
        reordering=True,
    )
    # Translate cluster element ids back to original indices.
    clusters = [tuple(valid_idx[j] for j in cluster) for cluster in clusters]

    import random as _r

    rng = _r.Random(seed)
    clusters = sorted(clusters, key=lambda c: (-len(c), rng.random()))

    train, val, test = [], [], []
    for cluster in clusters:
        idxs = list(cluster)
        if len(train) + len(idxs) <= n_train:
            train.extend(idxs)
        elif len(val) + len(idxs) <= n_val:
            val.extend(idxs)
        else:
            test.extend(idxs)

    # Anything left out (rdkit-failed parses) goes to train.
    placed = set(train) | set(val) | set(test)
    leftover = [i for i in range(n) if i not in placed]
    train.extend(leftover)
    return sorted(train), sorted(val), sorted(test)
