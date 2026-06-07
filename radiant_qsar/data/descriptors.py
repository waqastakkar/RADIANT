"""RDKit molecular descriptors used by sub-claim C1 (halt-vs-complexity).

Computes a fixed, deterministic descriptor vector per compound. The
descriptor names + ordering are stable so analysis code can rely on the
column layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# Stable column order used downstream.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "ExactMolWt",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumRings",
    "NumAromaticRings",
    "NumAliphaticRings",
    "FractionCSP3",
    "TPSA",
    "MolLogP",
    "MolMR",
    "NumHBA",
    "NumHBD",
    "NumValenceElectrons",
    "BertzCT",
    "QED",
    "SAscore_proxy",
)


def _safe(fn, mol, default=float("nan")) -> float:
    try:
        return float(fn(mol))
    except Exception:
        return default


def compute_descriptors(smiles_iterable: Sequence[str]) -> "pandas.DataFrame":
    """Compute the descriptor table for a list of standard SMILES.

    Returns a DataFrame whose columns are exactly :data:`DESCRIPTOR_NAMES`
    plus a leading ``standard_smiles`` column for joining back. Rows that
    fail to parse have NaN throughout.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    rows = []
    n = len(smiles_iterable)
    for i, smi in enumerate(smiles_iterable):
        if i % 10_000 == 0 and i > 0:
            logger.info("  descriptors: %d / %d", i, n)
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            rows.append({"standard_smiles": smi, **{k: float("nan") for k in DESCRIPTOR_NAMES}})
            continue
        rec = {
            "standard_smiles": smi,
            "MolWt": _safe(Descriptors.MolWt, mol),
            "ExactMolWt": _safe(Descriptors.ExactMolWt, mol),
            "HeavyAtomCount": _safe(Descriptors.HeavyAtomCount, mol),
            "NumHeteroatoms": _safe(Descriptors.NumHeteroatoms, mol),
            "NumRotatableBonds": _safe(rdMolDescriptors.CalcNumRotatableBonds, mol),
            "NumRings": _safe(rdMolDescriptors.CalcNumRings, mol),
            "NumAromaticRings": _safe(rdMolDescriptors.CalcNumAromaticRings, mol),
            "NumAliphaticRings": _safe(rdMolDescriptors.CalcNumAliphaticRings, mol),
            "FractionCSP3": _safe(rdMolDescriptors.CalcFractionCSP3, mol),
            "TPSA": _safe(rdMolDescriptors.CalcTPSA, mol),
            "MolLogP": _safe(Crippen.MolLogP, mol),
            "MolMR": _safe(Crippen.MolMR, mol),
            "NumHBA": _safe(Lipinski.NumHAcceptors, mol),
            "NumHBD": _safe(Lipinski.NumHDonors, mol),
            "NumValenceElectrons": _safe(Descriptors.NumValenceElectrons, mol),
            "BertzCT": _safe(Descriptors.BertzCT, mol),
            "QED": _safe(QED.qed, mol),
            # SAscore is hosted in RDKit Contrib; we proxy it by a cheap heuristic
            # combining ring count, fraction sp3, and stereocenter count. The
            # real SAscore can be plugged in by users who install it.
            "SAscore_proxy": _safe(
                lambda m: rdMolDescriptors.CalcNumRings(m)
                + (1 - rdMolDescriptors.CalcFractionCSP3(m))
                + 0.5 * len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
                mol,
            ),
        }
        rows.append(rec)
    return pd.DataFrame(rows, columns=("standard_smiles",) + DESCRIPTOR_NAMES)


def compute_and_save_descriptors(compounds_parquet: Path, out_path: Path) -> Path:
    """Read compounds.parquet, compute descriptors, write descriptors.parquet."""
    import pandas as pd

    out_path = Path(out_path)
    compounds = pd.read_parquet(compounds_parquet, columns=["inchikey14", "standard_smiles"])
    df = compute_descriptors(compounds["standard_smiles"].tolist())
    df.insert(0, "inchikey14", compounds["inchikey14"].values)
    df = df.drop_duplicates(subset=["inchikey14"]).reset_index(drop=True)
    df.to_parquet(out_path, compression="zstd", index=False)
    logger.info("descriptors written: %s (%d rows)", out_path, len(df))
    return out_path


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--compounds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compute_and_save_descriptors(args.compounds, args.out)


if __name__ == "__main__":
    _main()
