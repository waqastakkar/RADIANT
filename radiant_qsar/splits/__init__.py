"""Five reproducible split strategies for ChEMBL-shaped activity tables.

All splits return ``(train_idx, val_idx, test_idx)`` -- lists of integer
positions into the input row order. None of them shuffle the rows in
place, so split files are stable across machines for a given seed.

Splits:
    random           -- seeded shuffle baseline
    scaffold         -- Bemis-Murcko scaffold groups (rdkit when available)
    time             -- by document publication year
    cluster          -- Butina clustering on Morgan fingerprints (rdkit)
    activity_cliff   -- preserves cliff pairs within splits (MoleculeACE-style)
    target_holdout   -- leave-target-(family-)out
"""

from radiant_qsar.splits.random import random_split
from radiant_qsar.splits.scaffold import scaffold_split
from radiant_qsar.splits.time_split import time_split
from radiant_qsar.splits.cluster import cluster_split
from radiant_qsar.splits.activity_cliff import activity_cliff_split
from radiant_qsar.splits.target_holdout import target_holdout_split
from radiant_qsar.splits.cache import (
    DEFAULT_CACHE_DIR,
    SplitCacheConfig,
    load_or_compute_split,
    split_cache_path,
)

__all__ = [
    "random_split",
    "scaffold_split",
    "time_split",
    "cluster_split",
    "activity_cliff_split",
    "target_holdout_split",
    # cache
    "DEFAULT_CACHE_DIR",
    "SplitCacheConfig",
    "load_or_compute_split",
    "split_cache_path",
]
