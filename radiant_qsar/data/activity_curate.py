"""Activity curation -> pXC50.

Step 3 of the data pipeline. Reads the raw extraction + standardized
compounds and produces a clean ``activities.parquet`` with one row per
``(inchikey14, target, standard_type)`` triple, where the value is a
median pXC50 across replicates and the IQR is recorded for replicate
agreement filtering.

Outputs
-------
``activities.parquet`` columns:

* ``inchikey14``        -- compound identity (joins to compounds.parquet)
* ``standard_smiles``    -- canonical SMILES of the compound
* ``target_chembl_id``   -- ChEMBL target id
* ``uniprot``            -- UniProt accession when available
* ``target_name``        -- human-readable name
* ``organism``           -- target organism
* ``standard_type``      -- one of {IC50, Ki, Kd, EC50, AC50, XC50}
* ``pchembl``            -- median pXC50 across replicates
* ``pchembl_iqr``        -- IQR across replicates (= 0 for singletons)
* ``n_replicates``       -- count
* ``doc_year_min``       -- earliest doc year (for time split)
* ``doc_year_max``       -- latest doc year
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unit conversion to pchembl
# ---------------------------------------------------------------------------
_UNITS_TO_MOLAR = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "nM": 1e-9,
}


def value_to_pchembl(value: float, unit: str) -> float | None:
    """Convert a value+unit to ``pchembl = -log10(M)``. Returns None if invalid."""
    if value is None or value <= 0:
        return None
    factor = _UNITS_TO_MOLAR.get(unit)
    if factor is None:
        return None
    molar = float(value) * factor
    if molar <= 0 or not math.isfinite(molar):
        return None
    return -math.log10(molar)


@dataclass
class CurateConfig:
    raw_path: Path                  # raw_activities.parquet
    compounds_path: Path            # compounds.parquet (output of standardize)
    out_dir: Path                   # writes activities.parquet
    pchembl_min: float = 3.0        # mM upper bound
    pchembl_max: float = 12.0       # fM lower bound
    iqr_max: float = 1.0            # log unit; >1 means severe replicate disagreement
    standard_types: tuple[str, ...] = ("IC50", "Ki", "Kd", "EC50", "AC50", "XC50")

    def __post_init__(self) -> None:
        self.raw_path = Path(self.raw_path)
        self.compounds_path = Path(self.compounds_path)
        self.out_dir = Path(self.out_dir)
        for p in (self.raw_path, self.compounds_path):
            if not p.exists():
                raise FileNotFoundError(p)


def curate_activities(cfg: CurateConfig) -> Path:
    """Run the curation pipeline and write `activities.parquet`."""
    import numpy as np
    import pandas as pd

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "activities.parquet"

    t0 = time.time()
    logger.info("loading raw activities + compounds")
    cols = [
        "canonical_smiles",
        "target_chembl_id",
        "uniprot",
        "target_name",
        "organism",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "doc_year",
    ]
    raw = pd.read_parquet(cfg.raw_path, columns=cols)
    compounds = pd.read_parquet(
        cfg.compounds_path, columns=["input_smiles", "standard_smiles", "inchikey14"]
    )
    raw = raw.merge(
        compounds.rename(columns={"input_smiles": "canonical_smiles"}),
        on="canonical_smiles",
        how="inner",
    )
    n_after_join = len(raw)
    logger.info("  rows after compound join: %d", n_after_join)

    # 1. Filter standard_type.
    raw = raw[raw["standard_type"].isin(cfg.standard_types)].copy()

    # 2. pchembl: prefer the existing pchembl_value, else derive from value+unit.
    raw["pchembl"] = raw["pchembl_value"].astype("Float64")
    mask_use_derived = raw["pchembl"].isna()
    if mask_use_derived.any():
        sub_v = raw.loc[mask_use_derived, "standard_value"].tolist()
        sub_u = raw.loc[mask_use_derived, "standard_units"].tolist()
        derived_vals = [value_to_pchembl(v, u) for v, u in zip(sub_v, sub_u)]
        raw.loc[mask_use_derived, "pchembl"] = pd.array(derived_vals, dtype="Float64")

    # 3. Range filter.
    n_before_range = len(raw)
    raw = raw.dropna(subset=["pchembl"]).copy()
    raw = raw[(raw["pchembl"] >= cfg.pchembl_min) & (raw["pchembl"] <= cfg.pchembl_max)].copy()
    logger.info("  after pchembl range [%g, %g]: %d (was %d)",
                cfg.pchembl_min, cfg.pchembl_max, len(raw), n_before_range)

    # 4. Aggregate replicates -- vectorized via named aggregation. The
    #    apply-style alternative is 10-100x slower at 1-2M-row scale.
    keys = ["inchikey14", "target_chembl_id", "standard_type"]
    raw["pchembl_f"] = raw["pchembl"].astype(float)
    raw["doc_year_n"] = pd.to_numeric(raw["doc_year"], errors="coerce")

    grouped = raw.groupby(keys, dropna=False, sort=False, observed=True)
    activities = grouped.agg(
        standard_smiles=("standard_smiles", "first"),
        uniprot=("uniprot", "first"),
        target_name=("target_name", "first"),
        organism=("organism", "first"),
        pchembl=("pchembl_f", "median"),
        pchembl_q25=("pchembl_f", lambda x: np.quantile(x.values, 0.25) if len(x) else float("nan")),
        pchembl_q75=("pchembl_f", lambda x: np.quantile(x.values, 0.75) if len(x) else float("nan")),
        n_replicates=("pchembl_f", "size"),
        doc_year_min=("doc_year_n", "min"),
        doc_year_max=("doc_year_n", "max"),
    ).reset_index()
    activities["pchembl_iqr"] = activities["pchembl_q75"] - activities["pchembl_q25"]
    activities = activities.drop(columns=["pchembl_q25", "pchembl_q75"])
    for c in ("doc_year_min", "doc_year_max", "n_replicates"):
        activities[c] = activities[c].astype("Int64")

    # 5. Drop high-IQR replicates.
    n_before_iqr = len(activities)
    activities = activities[activities["pchembl_iqr"] <= cfg.iqr_max].copy()
    logger.info("  after IQR <= %.2f: %d (was %d)", cfg.iqr_max, len(activities), n_before_iqr)

    # 6. Schema tidy.
    activities = activities[[
        "inchikey14", "standard_smiles", "target_chembl_id", "uniprot",
        "target_name", "organism", "standard_type",
        "pchembl", "pchembl_iqr", "n_replicates",
        "doc_year_min", "doc_year_max",
    ]]
    activities = activities.sort_values(["target_chembl_id", "inchikey14"]).reset_index(drop=True)

    activities.to_parquet(out_path, compression="zstd", index=False)
    elapsed = time.time() - t0

    n_targets = activities["target_chembl_id"].nunique()
    n_compounds = activities["inchikey14"].nunique()
    logger.info(
        "wrote %s: %d (compound, target, type) rows | %d targets | %d compounds | %.1f s",
        out_path, len(activities), n_targets, n_compounds, elapsed,
    )

    meta = {
        "stage": "activity_curate",
        "rows_in": int(n_after_join),
        "rows_out": int(len(activities)),
        "n_unique_compounds": int(n_compounds),
        "n_unique_targets": int(n_targets),
        "pchembl_range": [cfg.pchembl_min, cfg.pchembl_max],
        "iqr_max": cfg.iqr_max,
        "standard_types": list(cfg.standard_types),
        "elapsed_s": round(elapsed, 1),
    }
    (cfg.out_dir / "activities.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--compounds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pchembl-min", type=float, default=3.0)
    p.add_argument("--pchembl-max", type=float, default=12.0)
    p.add_argument("--iqr-max", type=float, default=1.0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    curate_activities(
        CurateConfig(
            raw_path=args.raw,
            compounds_path=args.compounds,
            out_dir=args.out,
            pchembl_min=args.pchembl_min,
            pchembl_max=args.pchembl_max,
            iqr_max=args.iqr_max,
        )
    )


if __name__ == "__main__":
    _main()
