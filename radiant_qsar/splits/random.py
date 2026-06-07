"""Seeded random split.

Returned indices are sorted within each split so the file diff is stable
across runs that produce the same split.
"""

from __future__ import annotations

import random
from typing import Sequence


def _three_lengths(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
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
    items: Sequence,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    n = len(items)
    n_train, n_val, _ = _three_lengths(n, ratios)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    train = sorted(idxs[:n_train])
    val = sorted(idxs[n_train : n_train + n_val])
    test = sorted(idxs[n_train + n_val :])
    return train, val, test
