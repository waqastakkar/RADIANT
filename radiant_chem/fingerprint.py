"""Morgan fingerprint computation for hybrid RADIANT+FP models.

Provides :func:`smiles_to_morgan` for single molecules and
:func:`batch_morgan_fp` for batched computation. Returns dense
numpy/torch arrays suitable for concatenation with RADIANT embeddings.

Requires ``rdkit``; raises ``ImportError`` if not installed.
"""

from __future__ import annotations

import numpy as np
import torch


def smiles_to_morgan(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Compute a Morgan fingerprint bit vector for a single SMILES.

    Returns a (n_bits,) float32 numpy array. Invalid SMILES return a
    zero vector with a one-time warning.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    fp.ToNumpyArray(arr)
    return arr


def batch_morgan_fp(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Compute Morgan fingerprints for a batch of SMILES.

    Returns ``(B, n_bits)`` float32 tensor on ``device``.
    """
    fps = np.stack([smiles_to_morgan(s, radius, n_bits) for s in smiles_list])
    return torch.from_numpy(fps).to(device=device)
