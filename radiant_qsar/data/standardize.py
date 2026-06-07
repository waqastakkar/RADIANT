"""Compound standardization.

Step 2 of the data pipeline. Reads the raw extraction output, runs each
SMILES through the standard ChEMBL structure-curation pipeline (or a
RDKit fallback when the official ``chembl_structure_pipeline`` is not
installed), and emits a deduplicated `compounds.parquet` keyed by
InChIKey-14.

Standardization steps (in order):

1. Parse SMILES with RDKit. Drop rows that fail to parse.
2. Strip salts: keep the largest organic fragment (LargestFragmentChooser).
3. Neutralize charges where the ionization is uncontroversial (Uncharger).
4. Tautomer canonicalization (TautomerEnumerator.Canonicalize).
5. Re-canonicalize SMILES.
6. Compute InChI / InChIKey; ``inchikey14`` (first 14 chars) used as
   compound identity for deduplication.

Each step's failures are counted in the manifest so we can report
exactly how many compounds dropped at each gate.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class StandardizeConfig:
    in_path: Path                    # raw_activities.parquet
    out_dir: Path                    # produces compounds.parquet
    n_jobs: int = 1                  # parallelize with joblib if >1
    verbose: bool = True

    def __post_init__(self) -> None:
        self.in_path = Path(self.in_path)
        self.out_dir = Path(self.out_dir)
        if not self.in_path.exists():
            raise FileNotFoundError(self.in_path)


# ---------------------------------------------------------------------------
# Per-SMILES standardizer
# ---------------------------------------------------------------------------
def _have_chembl_pipeline() -> bool:
    try:
        import chembl_structure_pipeline  # noqa: F401
        return True
    except Exception:
        return False


def standardize_one(smiles: str) -> dict:
    """Standardize a single SMILES. Returns {smiles, inchikey14, status}.

    ``status`` is one of: ok, parse_fail, frag_fail, neut_fail, taut_fail,
    inchi_fail. The first non-ok status short-circuits and is returned.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem  # noqa: F401  (import side effects)
    from rdkit.Chem.MolStandardize import rdMolStandardize

    out = {
        "canonical_smiles": None,
        "standard_smiles": None,
        "inchikey": None,
        "inchikey14": None,
        "status": "ok",
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["status"] = "parse_fail"
        return out

    # 1. Largest fragment.
    try:
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)
    except Exception:
        out["status"] = "frag_fail"
        return out

    # 2. Neutralize.
    try:
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
    except Exception:
        out["status"] = "neut_fail"
        return out

    # 3. Tautomer canonicalization.
    try:
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
    except Exception:
        out["status"] = "taut_fail"
        return out

    # 4. Canonical SMILES + InChIKey.
    try:
        canon = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.MolToInchiKey(mol)
        if not ikey:
            out["status"] = "inchi_fail"
            return out
        out["canonical_smiles"] = smiles
        out["standard_smiles"] = canon
        out["inchikey"] = ikey
        out["inchikey14"] = ikey[:14]
    except Exception:
        out["status"] = "inchi_fail"
    return out


def standardize_one_pipeline(smiles: str) -> dict:
    """Variant that uses the official ``chembl_structure_pipeline`` when present."""
    from chembl_structure_pipeline import standardizer
    from rdkit import Chem

    out = {
        "canonical_smiles": None,
        "standard_smiles": None,
        "inchikey": None,
        "inchikey14": None,
        "status": "ok",
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        out["status"] = "parse_fail"
        return out
    try:
        std = standardizer.standardize_mol(mol)
        std = standardizer.get_parent_mol(std)[0]
    except Exception:
        out["status"] = "pipeline_fail"
        return out
    try:
        canon = Chem.MolToSmiles(std, canonical=True)
        ikey = Chem.MolToInchiKey(std)
        if not ikey:
            out["status"] = "inchi_fail"
            return out
        out["canonical_smiles"] = smiles
        out["standard_smiles"] = canon
        out["inchikey"] = ikey
        out["inchikey14"] = ikey[:14]
    except Exception:
        out["status"] = "inchi_fail"
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def standardize_compounds(cfg: StandardizeConfig) -> Path:
    """Read raw activities, standardize unique SMILES, write `compounds.parquet`."""
    import pandas as pd

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "compounds.parquet"

    use_pipeline = _have_chembl_pipeline()
    fn = standardize_one_pipeline if use_pipeline else standardize_one
    logger.info("standardize: using %s", "chembl_structure_pipeline" if use_pipeline else "rdkit fallback")

    logger.info("reading raw activities: %s", cfg.in_path)
    raw = pd.read_parquet(cfg.in_path, columns=["canonical_smiles"])
    unique = raw.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
    logger.info("  unique input SMILES: %d", len(unique))

    t0 = time.time()
    rows = []
    n = len(unique)
    log_every = max(1, n // 50)

    if cfg.n_jobs > 1:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=cfg.n_jobs, batch_size=512)(
                delayed(fn)(s) for s in unique["canonical_smiles"].tolist()
            )
            for raw_smi, r in zip(unique["canonical_smiles"].tolist(), results):
                r["raw_smiles"] = raw_smi
            rows = list(results)
        except ImportError:
            logger.warning("joblib not installed; falling back to single-process")
            cfg = StandardizeConfig(in_path=cfg.in_path, out_dir=cfg.out_dir, n_jobs=1)

    if cfg.n_jobs <= 1:
        for i, smi in enumerate(unique["canonical_smiles"].tolist()):
            r = fn(smi)
            r["raw_smiles"] = smi
            rows.append(r)
            if cfg.verbose and (i + 1) % log_every == 0:
                ok = sum(1 for x in rows if x["status"] == "ok")
                logger.info(
                    "  %d / %d  (ok=%d, %.1f s)", i + 1, n, ok, time.time() - t0
                )

    df = pd.DataFrame(rows)
    status_counts = df["status"].value_counts().to_dict()

    ok = df[df["status"] == "ok"].copy()
    # Keep the canonical (raw) smiles too -- some reviewers want it.
    ok = ok.rename(columns={"raw_smiles": "input_smiles"})
    # Deduplicate on inchikey14 (the chemical-identity key).
    deduped = ok.drop_duplicates(subset=["inchikey14"]).reset_index(drop=True)
    logger.info("  ok rows: %d  unique inchikey14: %d", len(ok), len(deduped))

    deduped.to_parquet(out_path, compression="zstd", index=False)

    meta = {
        "stage": "standardize",
        "in_path": str(cfg.in_path),
        "rows_in": int(len(unique)),
        "rows_out_unique_inchikey14": int(len(deduped)),
        "status_counts": {k: int(v) for k, v in status_counts.items()},
        "elapsed_s": round(time.time() - t0, 1),
        "pipeline": "chembl_structure_pipeline" if use_pipeline else "rdkit_fallback",
    }
    (cfg.out_dir / "compounds.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    standardize_compounds(StandardizeConfig(in_path=args.in_path, out_dir=args.out, n_jobs=args.n_jobs))


if __name__ == "__main__":
    _main()
