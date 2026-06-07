"""SQL extraction from ``chembl_36.db`` into a raw-activities Parquet.

Step 1 of the data pipeline. Pulls a single denormalized table covering
activities, compounds, assays, targets, and document years. Filters at the
SQL level for: binding/functional assays, single-protein targets,
confidence_score >= 7, the six standard potency types we care about, and
sane units / relations / data-validity comments.

The output is intentionally *raw*: we do compound / activity standardization
in dedicated downstream steps so this query can be cached and reused.

Typical run on a 29 GB ChEMBL 36 SQLite is 10-30 minutes (read-bound on
HDDs; SSDs are dramatically faster).

CLI::

    python -m radiant_qsar.data.chembl_extract \\
        --db D:/My-Work/RADIANT/chembl_36.db \\
        --out D:/My-Work/RADIANT/data/raw

Programmatic::

    from radiant_qsar.data import extract_activities, ExtractConfig
    extract_activities(ExtractConfig(db_path=..., out_dir=...))
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------
EXTRACTION_SQL = """
WITH active_assays AS (
    SELECT a.assay_id,
           a.tid,
           a.confidence_score,
           a.assay_type,
           td.target_type,
           td.pref_name        AS target_name,
           td.organism,
           td.chembl_id        AS target_chembl_id,
           tc.component_id,
           comp.accession      AS uniprot
    FROM   assays a
    JOIN   target_dictionary td  ON td.tid          = a.tid
    LEFT JOIN target_components tc ON tc.tid        = a.tid
    LEFT JOIN component_sequences comp ON comp.component_id = tc.component_id
    WHERE  a.assay_type IN ('B','F')
      AND  a.confidence_score >= 7
      AND  td.target_type LIKE 'SINGLE PROTEIN'
)
SELECT  act.activity_id,
        md.chembl_id              AS molecule_chembl_id,
        md.molregno,
        cs.canonical_smiles,
        aa.target_chembl_id, aa.uniprot, aa.target_name, aa.organism,
        act.standard_type, act.standard_relation, act.standard_value,
        act.standard_units, act.pchembl_value,
        act.activity_comment, act.data_validity_comment,
        COALESCE(act.potential_duplicate, 0) AS potential_duplicate,
        d.year                    AS doc_year, d.journal, d.doi,
        aa.assay_id, aa.confidence_score
FROM    activities act
JOIN    active_assays         aa ON aa.assay_id  = act.assay_id
JOIN    molecule_dictionary   md ON md.molregno  = act.molregno
JOIN    compound_structures   cs ON cs.molregno  = act.molregno
LEFT JOIN docs                d  ON d.doc_id     = act.doc_id
WHERE   act.standard_type IN ('IC50','Ki','Kd','EC50','AC50','XC50')
  AND   act.standard_relation IN ('=','~')
  AND   act.standard_units    IN ('nM','uM','M','mM')
  AND   act.standard_value IS NOT NULL
  AND   cs.canonical_smiles IS NOT NULL
  AND   COALESCE(act.potential_duplicate, 0) = 0
  AND   ( act.data_validity_comment IS NULL
          OR act.data_validity_comment NOT IN
             ('Outside typical range',
              'Non standard unit for type',
              'Potential transcription error',
              'Author confirmed error',
              'Outside typical range and Potential transcription error') )
"""


# Selected columns -- the order is what the Parquet schema will use.
COLUMNS: tuple[str, ...] = (
    "activity_id",
    "molecule_chembl_id",
    "molregno",
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
    "activity_comment",
    "data_validity_comment",
    "potential_duplicate",
    "doc_year",
    "journal",
    "doi",
    "assay_id",
    "confidence_score",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class ExtractConfig:
    """Inputs to :func:`extract_activities`."""

    db_path: Path
    out_dir: Path
    chunk_size: int = 200_000           # rows per Parquet chunk to bound RAM
    max_rows: int | None = None         # for dev iteration; None = all rows
    parquet_compression: str = "zstd"   # zstd compresses ChEMBL well
    extra_meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.out_dir = Path(self.out_dir)
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)
        if self.chunk_size < 1_000:
            raise ValueError("chunk_size must be >= 1000")


def extract_activities(cfg: ExtractConfig) -> Path:
    """Run the extraction SQL and write `raw_activities.parquet`.

    Returns the path to the written Parquet directory.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = cfg.out_dir / "raw_activities.parquet"
    if out_path.exists():
        logger.warning("Output exists and will be overwritten: %s", out_path)
        # We keep the file open via ParquetWriter; explicit unlink first.
        out_path.unlink()

    sql = EXTRACTION_SQL
    if cfg.max_rows is not None:
        sql = sql + f"\nLIMIT {int(cfg.max_rows)}"

    t0 = time.time()
    logger.info("Opening SQLite (read-only): %s", cfg.db_path)
    # ``mode=ro`` requires a URI connection.
    uri = f"file:{cfg.db_path.as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        con.execute("PRAGMA temp_store = MEMORY")
        con.execute("PRAGMA cache_size = -200000")  # ~200 MB page cache
        cur = con.execute(sql)
        # Build a pyarrow schema once from the first chunk's pandas dtypes.
        writer: pq.ParquetWriter | None = None
        total = 0
        chunk_idx = 0
        while True:
            rows = cur.fetchmany(cfg.chunk_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=list(COLUMNS))
            # Make integer columns nullable to handle NULLs cleanly.
            for c in ("molregno", "doc_year", "assay_id", "confidence_score"):
                if c in df.columns:
                    df[c] = df[c].astype("Int64")
            for c in ("standard_value", "pchembl_value"):
                if c in df.columns:
                    df[c] = df[c].astype("Float64")
            for c in (
                "molecule_chembl_id",
                "canonical_smiles",
                "target_chembl_id",
                "uniprot",
                "target_name",
                "organism",
                "standard_type",
                "standard_relation",
                "standard_units",
                "activity_comment",
                "data_validity_comment",
                "journal",
                "doi",
            ):
                if c in df.columns:
                    df[c] = df[c].astype("string")
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression=cfg.parquet_compression,
                )
            writer.write_table(table)
            total += len(df)
            chunk_idx += 1
            if chunk_idx % 5 == 0 or chunk_idx == 1:
                logger.info(
                    "  ... %d rows (%.1f s, %.0f rows/s)",
                    total,
                    time.time() - t0,
                    total / max(time.time() - t0, 1e-3),
                )
        if writer is not None:
            writer.close()
    finally:
        con.close()

    elapsed = time.time() - t0
    logger.info("Extraction complete: %d rows in %.1f s -> %s", total, elapsed, out_path)

    meta = {
        "stage": "chembl_extract",
        "db_path": str(cfg.db_path),
        "row_count": total,
        "elapsed_s": round(elapsed, 1),
        "sql_filters": {
            "assay_type_in": ["B", "F"],
            "confidence_score_min": 7,
            "target_type": "SINGLE PROTEIN",
            "standard_types": ["IC50", "Ki", "Kd", "EC50", "AC50", "XC50"],
            "standard_relations": ["=", "~"],
            "standard_units": ["nM", "uM", "M", "mM"],
        },
        "chembl_version": "36",
        "extra": cfg.extra_meta,
    }
    (cfg.out_dir / "raw_activities.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> ExtractConfig:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True, type=Path, help="path to chembl_36.db")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--compression", default="zstd")
    args = p.parse_args()
    return ExtractConfig(
        db_path=args.db,
        out_dir=args.out,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        parquet_compression=args.compression,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _parse_args()
    extract_activities(cfg)
