"""Canonical per-test-molecule predictions writer.

Every model in the panel sweep writes a ``predictions.csv`` with the
same leading columns. Phase G analyses (calibration, complexity bins,
per-atom attribution) read these files and join them against
``descriptors.parquet`` on ``inchikey14``. Without a uniform schema the
analyses would have to special-case each model.

Schema (canonical leading columns; model-specific extras append to the right)::

    idx, inchikey14, target_chembl_id, split_kind, smiles, true_pchembl, pred_pchembl

``idx`` is the row position inside the per-target ``sub`` DataFrame --
i.e. the same indices the split cache stores, so a downstream consumer
can re-derive train / val / test membership without re-running splits.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


PREDICTIONS_FILENAME = "predictions.csv"

PREDICTIONS_SCHEMA: tuple[str, ...] = (
    "idx",
    "inchikey14",
    "target_chembl_id",
    "split_kind",
    "smiles",
    "true_pchembl",
    "pred_pchembl",
)


def write_predictions(
    out_dir: Path | str,
    *,
    indices: Sequence[int],
    inchikeys: Sequence[str],
    smiles: Sequence[str],
    true_pchembl: Sequence[float],
    pred_pchembl: Sequence[float],
    target_chembl_id: str,
    split_kind: str,
    extra_columns: dict[str, Sequence] | None = None,
    filename: str = PREDICTIONS_FILENAME,
) -> Path:
    """Write canonical predictions.csv to ``out_dir/<filename>``.

    All sequence args must be the same length. ``extra_columns`` is an
    optional dict of per-row data the calling model wants to attach
    (e.g. ``halt_step`` for RADIANT with halting on, ``confidence``
    for ConfidenceHalting outputs). Extra columns are appended after the
    canonical ones in the order keys appear in the dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    n = len(indices)
    for name, vec in (
        ("inchikeys", inchikeys),
        ("smiles", smiles),
        ("true_pchembl", true_pchembl),
        ("pred_pchembl", pred_pchembl),
    ):
        if len(vec) != n:
            raise ValueError(
                f"write_predictions: '{name}' has length {len(vec)}, expected {n}"
            )
    if extra_columns:
        for name, vec in extra_columns.items():
            if len(vec) != n:
                raise ValueError(
                    f"write_predictions: extra column '{name}' has length {len(vec)}, expected {n}"
                )

    extra_keys = list(extra_columns.keys()) if extra_columns else []
    header = list(PREDICTIONS_SCHEMA) + extra_keys

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = [
                int(indices[i]),
                str(inchikeys[i]),
                target_chembl_id,
                split_kind,
                str(smiles[i]),
                f"{float(true_pchembl[i]):.4f}",
                f"{float(pred_pchembl[i]):.4f}",
            ]
            for k in extra_keys:
                v = extra_columns[k][i]
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            w.writerow(row)
    return out_path
