"""Time split by ``doc_year``.

Train: rows whose document year is <= ``train_max_year``.
Val:   rows with year == ``train_max_year + 1``.
Test:  rows with year >= ``train_max_year + 2``.

Rows with missing year are routed to ``train`` by default (configurable).
This is the most realistic generalization test for QSAR -- it asks "did
the model learn chemistry that generalizes to tomorrow's chemists?"
"""

from __future__ import annotations

import math
from typing import Sequence


def _is_missing(y) -> bool:
    """True when ``y`` is any flavor of "missing": ``None``, ``pd.NA``, NaN, or
    a value that cannot be coerced to int. We use ``int(y)`` as the implicit
    test and treat any failure as "missing" so the split function works on
    plain Python lists, numpy arrays, and pandas nullable ``Int64`` columns
    interchangeably."""
    if y is None:
        return True
    try:
        f = float(y)
    except (TypeError, ValueError):
        return True
    return math.isnan(f)


def time_split(
    years: Sequence[int | None],
    *,
    train_max_year: int = 2020,
    val_year_offset: int = 1,
    missing_year_to: str = "train",
) -> tuple[list[int], list[int], list[int]]:
    if missing_year_to not in {"train", "drop"}:
        raise ValueError("missing_year_to must be 'train' or 'drop'")

    train, val, test = [], [], []
    val_year = train_max_year + val_year_offset
    test_year_min = val_year + 1

    for i, y in enumerate(years):
        if _is_missing(y):
            if missing_year_to == "train":
                train.append(i)
            continue
        y = int(y)
        if y <= train_max_year:
            train.append(i)
        elif y < test_year_min:
            val.append(i)
        else:
            test.append(i)
    return sorted(train), sorted(val), sorted(test)
