"""Streaming readers for compound libraries.

Yields ``(mol_id, smiles, mol)`` tuples from:

* ``.smi`` -- whitespace- or tab-separated, two columns (auto-detect order)
* ``.csv`` -- with at least an id column and a smiles column
* ``.sdf`` -- molecule blocks; the SDF property used as id is configurable

Streaming I/O so libraries of millions of molecules don't blow up RAM.
Bad rows (unparseable SMILES, missing id, etc.) are reported through a
counter instead of crashing the run.
"""

from __future__ import annotations

import csv
import gzip
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TextIO

logger = logging.getLogger(__name__)


@dataclass
class LoaderStats:
    n_total: int = 0
    n_parsed: int = 0
    n_unparseable: int = 0
    n_missing_id: int = 0
    skipped_lines: list[int] = field(default_factory=list)


def _open_text(path: Path) -> TextIO:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _iter_smi(
    path: Path,
    *,
    id_first: bool | None = None,
    delimiter: str | None = None,
):
    """Iterate ``(mol_id, smiles)`` from a .smi file.

    ``id_first``: if None, auto-detect from the first non-blank line. SMILES
    are heuristically the longer token containing chem characters; ID is
    the other.
    """
    f = _open_text(path)
    try:
        first = None
        # Probe first non-blank, non-comment line.
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            first = line
            break
        if first is None:
            return

        sep = delimiter
        if sep is None:
            sep = "\t" if "\t" in first else None  # None -> any whitespace

        parts = first.split(sep) if sep else first.split()
        if id_first is None and len(parts) >= 2:
            # Heuristic: SMILES contains at least one of: [ ] = # @ / \\ ( )
            chem_chars = set("[]=#@\\/()")
            score_a = sum(c in chem_chars for c in parts[0])
            score_b = sum(c in chem_chars for c in parts[1])
            id_first = score_b > score_a

        # Yield first line.
        yield _smi_split(first, id_first, sep)
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            yield _smi_split(line, id_first, sep)
    finally:
        f.close()


def _smi_split(line: str, id_first: bool, sep: str | None) -> tuple[str | None, str | None]:
    parts = line.split(sep) if sep else line.split()
    if len(parts) < 2:
        # SMI may legitimately omit the ID; auto-assign one downstream.
        return None, parts[0] if parts else None
    if id_first:
        return parts[0], parts[1]
    return parts[1], parts[0]


def _iter_csv(
    path: Path,
    *,
    id_column: str,
    smiles_column: str,
    delimiter: str = ",",
):
    f = _open_text(path)
    try:
        reader = csv.DictReader(f, delimiter=delimiter)
        if id_column not in (reader.fieldnames or []) or smiles_column not in (reader.fieldnames or []):
            raise KeyError(
                f"CSV must have columns {id_column!r} and {smiles_column!r}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            yield row.get(id_column), row.get(smiles_column)
    finally:
        f.close()


def _iter_sdf(path: Path, *, id_property: str = "_Name"):
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    for i, mol in enumerate(suppl):
        if mol is None:
            yield None, None, None
            continue
        if id_property == "_Name":
            mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}"
        else:
            mol_id = mol.GetProp(id_property) if mol.HasProp(id_property) else f"mol_{i}"
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = None
        yield mol_id, smi, mol


# ---------------------------------------------------------------------------
def load_library(
    path: str | Path,
    *,
    id_first: bool | None = None,
    id_column: str = "id",
    smiles_column: str = "smiles",
    delimiter: str | None = None,
    sdf_id_property: str = "_Name",
    parse_mol: bool = True,
) -> Iterator[tuple[str, str, "object | None"]]:
    """Iterate ``(mol_id, smiles, mol_or_None)`` from a library file.

    For .smi / .csv inputs, ``mol`` is parsed lazily via rdkit unless
    ``parse_mol=False``. SDF inputs already carry ``mol`` from the supplier.

    The function logs counts and skipped lines; downstream code only sees
    well-formed rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = path.with_suffix("").suffix.lower()

    stats = LoaderStats()

    if suffix in (".smi", ".smiles"):
        gen = _iter_smi(path, id_first=id_first, delimiter=delimiter)
        for i, (mol_id, smi) in enumerate(gen):
            stats.n_total += 1
            if smi is None:
                stats.n_unparseable += 1
                continue
            if mol_id is None:
                mol_id = f"mol_{i}"
                stats.n_missing_id += 1
            mol = _parse_smiles(smi) if parse_mol else None
            if parse_mol and mol is None:
                stats.n_unparseable += 1
                continue
            stats.n_parsed += 1
            yield mol_id, smi, mol
    elif suffix == ".csv":
        gen = _iter_csv(path, id_column=id_column, smiles_column=smiles_column,
                        delimiter=delimiter or ",")
        for mol_id, smi in gen:
            stats.n_total += 1
            if not smi:
                stats.n_unparseable += 1
                continue
            mol = _parse_smiles(smi) if parse_mol else None
            if parse_mol and mol is None:
                stats.n_unparseable += 1
                continue
            stats.n_parsed += 1
            yield mol_id or f"mol_{stats.n_total}", smi, mol
    elif suffix == ".sdf":
        for mol_id, smi, mol in _iter_sdf(path, id_property=sdf_id_property):
            stats.n_total += 1
            if mol is None or smi is None:
                stats.n_unparseable += 1
                continue
            stats.n_parsed += 1
            yield mol_id or f"mol_{stats.n_total}", smi, mol
    else:
        raise ValueError(
            f"Unsupported library file type {suffix!r}; use .smi, .csv, or .sdf"
        )

    logger.info(
        "loader: total=%d parsed=%d unparseable=%d missing_id=%d",
        stats.n_total, stats.n_parsed, stats.n_unparseable, stats.n_missing_id,
    )


def _parse_smiles(smi: str):
    from rdkit import Chem

    return Chem.MolFromSmiles(smi)
