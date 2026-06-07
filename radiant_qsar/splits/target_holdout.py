"""Leave-target-(family-)out splits.

Two flavors:

* By target id. The held-out targets contribute *all* their rows to test;
  no compound for those targets appears in train/val. Tests target-leakage-
  free OOD generalization.

* By target class (a coarse bucket like 'kinase' / 'gpcr' / ...). Holds out
  an entire class. This is the hardest OOD setup we run.

The function selects targets / classes to hold out using a deterministic
seeded sample, stratifying on target_class to keep the test set diverse.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence


def target_holdout_split(
    target_ids: Sequence[str],
    target_classes: Sequence[str],
    *,
    holdout_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0,
    by: str = "target",
) -> tuple[list[int], list[int], list[int]]:
    """Hold out 20% of targets (or classes) as test; carve val from the remainder.

    Args:
        target_ids:    one per row.
        target_classes: one per row, used for stratification when ``by='target'``.
        holdout_fraction: fraction of *targets* (or classes) to put in test.
        val_fraction:   fraction of remaining rows that go to val (random by row).
        seed:          deterministic.
        by:            ``'target'`` -- choose individual targets to hold out.
                       ``'class'``  -- hold out entire target classes.

    Returns ``(train_idx, val_idx, test_idx)``.
    """
    if by not in {"target", "class"}:
        raise ValueError("by must be 'target' or 'class'")
    if len(target_ids) != len(target_classes):
        raise ValueError("target_ids and target_classes must be same length")

    rng = random.Random(seed)
    n = len(target_ids)

    if by == "class":
        unique_classes = sorted({c or "other" for c in target_classes})
        n_holdout = max(1, int(round(len(unique_classes) * holdout_fraction)))
        held = set(rng.sample(unique_classes, k=min(n_holdout, len(unique_classes))))
        test = [i for i in range(n) if (target_classes[i] or "other") in held]
    else:
        # Stratified-by-class sample of targets.
        by_class: dict[str, list[str]] = defaultdict(list)
        for tid, cls in zip(target_ids, target_classes):
            by_class[cls or "other"].append(tid)
        held: set[str] = set()
        for cls, tids in by_class.items():
            uniq = sorted(set(tids))
            n_take = max(1, int(round(len(uniq) * holdout_fraction)))
            held.update(rng.sample(uniq, k=min(n_take, len(uniq))))
        test = [i for i in range(n) if target_ids[i] in held]

    remaining = [i for i in range(n) if i not in set(test)]
    rng.shuffle(remaining)
    n_val = int(round(len(remaining) * val_fraction))
    val = sorted(remaining[:n_val])
    train = sorted(remaining[n_val:])
    return train, val, sorted(test)
