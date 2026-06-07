"""SMILES augmentation utilities.

When ``rdkit`` is installed, :func:`randomize_smiles` returns a non-canonical
random walk over the same molecule (a different atom-traversal order). This
is the standard recipe for SMILES enumeration used in molecular
pretraining and contrastive learning.

Without ``rdkit`` the function returns the input unchanged and prints a one-
time warning so the caller knows they're getting the identity augmentation.
"""

from __future__ import annotations

import functools
import random
import warnings


@functools.lru_cache(maxsize=1)
def _have_rdkit() -> bool:
    """Thread-safe, cached check for rdkit availability."""
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        warnings.warn(
            "rdkit not available; randomize_smiles returns identity. "
            "Install rdkit for SMILES enumeration.",
            stacklevel=2,
        )
        return False


def randomize_smiles(s: str, *, seed: int | None = None) -> str:
    """Return a randomized but equivalent SMILES; identity if rdkit missing."""
    if not _have_rdkit():
        return s
    from rdkit import Chem

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    rng = random.Random(seed) if seed is not None else random
    n = mol.GetNumAtoms()
    if n == 0:
        return s
    root = rng.randrange(n)
    try:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False, rootedAtAtom=root)
    except Exception:
        return s


def canonicalize_smiles(s: str) -> str:
    """Return the rdkit-canonical SMILES; identity if rdkit missing or invalid."""
    if not _have_rdkit():
        return s
    from rdkit import Chem

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return s
