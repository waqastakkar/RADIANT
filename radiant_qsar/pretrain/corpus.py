"""Compound corpus dataset for pretraining.

Wraps a curated ``compounds.parquet`` file (output of Phase A) into an
indexable Dataset that yields one molecule at a time. The collator
(see :mod:`collator`) is responsible for tokenization and masking; the
dataset stays simple so it composes cleanly with PyTorch DataLoader
workers.

Optionally yields a *pair* of randomized SMILES per molecule (one
canonical-ish, one randomized via rdkit if available), which the
collator turns into positive pairs for the contrastive objective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompoundCorpusDataset:
    """A torch Dataset over standard SMILES from a compounds.parquet release.

    Attributes
    ----------
    smiles : list[str]
        the standard (canonical) SMILES strings.
    return_augmented_pair : bool
        when True, ``__getitem__`` returns ``(smiles, smiles_random)``.
        When False, returns just ``smiles``.
    return_chemistry_views : bool
        when True, also returns Murcko scaffold and side-chain/R-group
        SMILES views for chemistry-aware Stage-1 pretraining.
    """

    parquet_path: Path
    return_augmented_pair: bool = True
    return_chemistry_views: bool = True
    smiles_column: str = "standard_smiles"

    def __post_init__(self) -> None:
        import pandas as pd

        self.parquet_path = Path(self.parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(self.parquet_path)
        df = pd.read_parquet(self.parquet_path, columns=[self.smiles_column])
        self.smiles: list[str] = df[self.smiles_column].dropna().astype(str).tolist()
        logger.info(
            "CompoundCorpusDataset: loaded %d compounds from %s",
            len(self.smiles), self.parquet_path,
        )

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        s = self.smiles[idx]
        if not self.return_augmented_pair and not self.return_chemistry_views:
            return s
        views: list[str] = [s]
        # Augmented partner; falls back to identity if rdkit missing or fails.
        # Use seed=None so each epoch sees a different randomized SMILES,
        # maximising the diversity of positive pairs for contrastive learning.
        if self.return_augmented_pair:
            from radiant_chem.augment import randomize_smiles

            views.append(randomize_smiles(s))
        if self.return_chemistry_views:
            from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles

            views.append(_murcko_scaffold_smiles(s))
            views.append(murcko_rgroup_smiles(s) or s)
        return tuple(views)


def _murcko_scaffold_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return smiles
    return Chem.MolToSmiles(scaffold, canonical=True)
