"""Target consolidation and target-class taxonomy.

Reads `activities.parquet`, attaches a coarse target class derived from
ChEMBL's protein classification table, and writes `targets.parquet`.

Class taxonomy is intentionally small (8 buckets) so that target-class-
stratified evaluation has enough data per bucket on a single GPU budget.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# Coarse buckets used in stratified evaluation.
COARSE_CLASSES = (
    "kinase",
    "gpcr",
    "ion_channel",
    "nuclear_receptor",
    "protease",
    "other_enzyme",
    "transporter",
    "other",
)


# Heuristic mapping from ChEMBL protein-classification level-1 strings to our
# coarse buckets. Names are matched lower-cased and substring-wise.
_CLASS_MAP = (
    ("kinase", "kinase"),
    ("gpcr", "gpcr"),
    ("7tm", "gpcr"),
    ("g protein-coupled receptor", "gpcr"),
    ("ion channel", "ion_channel"),
    ("nuclear hormone receptor", "nuclear_receptor"),
    ("nuclear receptor", "nuclear_receptor"),
    ("protease", "protease"),
    ("peptidase", "protease"),
    ("transporter", "transporter"),
    ("solute carrier", "transporter"),
    ("transferase", "other_enzyme"),
    ("hydrolase", "other_enzyme"),
    ("oxidoreductase", "other_enzyme"),
    ("lyase", "other_enzyme"),
    ("isomerase", "other_enzyme"),
    ("ligase", "other_enzyme"),
    ("enzyme", "other_enzyme"),
)


def _coarse_class(level1: str | None) -> str:
    if not level1:
        return "other"
    s = level1.lower()
    for needle, bucket in _CLASS_MAP:
        if needle in s:
            return bucket
    return "other"


def _load_target_classification(db_path: Path) -> "pandas.DataFrame":
    """Pull a flattened class-name string per target from chembl_36.

    ChEMBL's ``protein_classification`` table stores a hierarchy via
    ``parent_id``; rather than walking it, we group by target and join all
    the class ``pref_name`` strings reachable from any of the target's
    components. The downstream :func:`_coarse_class` runs substring matches
    against this concatenated string, which is robust to schema variants
    across ChEMBL releases.
    """
    import pandas as pd

    sql = """
    SELECT td.chembl_id                                        AS target_chembl_id,
           GROUP_CONCAT(LOWER(pc.pref_name), ' | ')            AS class_names,
           GROUP_CONCAT(LOWER(pc.protein_class_desc), ' | ')   AS class_descs,
           MIN(pc.class_level)                                  AS min_class_level
    FROM   target_dictionary td
    JOIN   target_components tc      ON tc.tid = td.tid
    JOIN   component_class    cc     ON cc.component_id = tc.component_id
    JOIN   protein_classification pc ON pc.protein_class_id = cc.protein_class_id
    WHERE  td.target_type LIKE 'SINGLE PROTEIN'
    GROUP  BY td.chembl_id
    """
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        df = pd.read_sql(sql, con)
    finally:
        con.close()
    return df


@dataclass
class ConsolidateConfig:
    activities_path: Path
    db_path: Path
    out_dir: Path

    def __post_init__(self) -> None:
        for p in (self.activities_path, self.db_path):
            p = Path(p)
            if not p.exists():
                raise FileNotFoundError(p)
        self.activities_path = Path(self.activities_path)
        self.db_path = Path(self.db_path)
        self.out_dir = Path(self.out_dir)


def consolidate_targets(cfg: ConsolidateConfig) -> Path:
    import json
    import pandas as pd

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / "targets.parquet"

    activities = pd.read_parquet(
        cfg.activities_path,
        columns=["target_chembl_id", "uniprot", "target_name", "organism", "inchikey14"],
    )
    target_compound_counts = activities.groupby("target_chembl_id")["inchikey14"].nunique().rename("n_compounds")
    target_meta = activities.drop_duplicates(subset=["target_chembl_id"])[
        ["target_chembl_id", "uniprot", "target_name", "organism"]
    ]

    classification = _load_target_classification(cfg.db_path)
    target_meta = target_meta.merge(
        classification[["target_chembl_id", "class_names", "class_descs"]],
        on="target_chembl_id",
        how="left",
    )
    # ChEMBL's `protein_class_desc` stores the FULL hierarchy path (e.g.
    # "membrane receptor  7tm1  smallmol  monoamine receptor  dopamine receptor")
    # while `pref_name` is just the leaf ("dopamine receptor"). Keywords like
    # `7tm1`, `gpcr`, `ion channel` only live in the descs string, so we must
    # match against the union of both.
    combined = (
        target_meta["class_names"].fillna("")
        + " | "
        + target_meta["class_descs"].fillna("")
    )
    target_meta["target_class"] = combined.map(_coarse_class)
    target_meta = target_meta.merge(target_compound_counts, on="target_chembl_id", how="left")
    target_meta = target_meta.sort_values("n_compounds", ascending=False).reset_index(drop=True)

    target_meta.to_parquet(out_path, compression="zstd", index=False)

    counts = target_meta["target_class"].value_counts().to_dict()
    meta = {
        "stage": "target_consolidate",
        "n_targets": int(len(target_meta)),
        "class_counts": {k: int(v) for k, v in counts.items()},
    }
    (cfg.out_dir / "targets.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("targets.parquet: %d targets, classes=%s", len(target_meta), counts)
    return out_path


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    consolidate_targets(
        ConsolidateConfig(
            activities_path=args.activities,
            db_path=args.db,
            out_dir=args.out,
        )
    )


if __name__ == "__main__":
    _main()
