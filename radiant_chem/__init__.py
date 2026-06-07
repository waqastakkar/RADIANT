"""RADIANT-Chem: ChEMBL-oriented variant of RADIANT.

Wraps the domain-agnostic core with a SMILES tokenizer, dataset loaders,
property/regression/classification heads, and pre-training objectives
(masked token, contrastive). The core architecture is unchanged --
only the input/output stack is chemistry-specific.
"""

from radiant_chem.config import RadiantChemConfig
from radiant_chem.tokenizer import SmilesTokenizer
from radiant_chem.dataset import ChemblCsvDataset, MaskedSmilesDataset
from radiant_chem.splits import random_split, scaffold_split
from radiant_chem.augment import randomize_smiles
from radiant_chem.objectives import (
    MaskedLMLoss,
    RegressionLoss,
    ClassificationLoss,
    ContrastiveLoss,
)
from radiant_chem.tasks import TaskSpec, TaskRegistry
from radiant_chem.model_chem import RadiantChemModel, FingerprintAugmentedHead
from radiant_chem.depth_pool import DepthAdaptivePool
from radiant_chem.fingerprint import smiles_to_morgan, batch_morgan_fp

__all__ = [
    "RadiantChemConfig",
    "SmilesTokenizer",
    "ChemblCsvDataset",
    "MaskedSmilesDataset",
    "random_split",
    "scaffold_split",
    "randomize_smiles",
    "MaskedLMLoss",
    "RegressionLoss",
    "ClassificationLoss",
    "ContrastiveLoss",
    "TaskSpec",
    "TaskRegistry",
    "RadiantChemModel",
    "FingerprintAugmentedHead",
    "DepthAdaptivePool",
    "smiles_to_morgan",
    "batch_morgan_fp",
]
