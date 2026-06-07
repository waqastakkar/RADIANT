"""One-shot Phase-A driver.

Runs every step of the data-curation pipeline in order against the
provided ChEMBL SQLite, producing the full
``data/processed/v1/{compounds,activities,descriptors,targets}.parquet``
release plus a ``manifest.json`` summarizing it.

Usage::

    python -m radiant_qsar.data.run_phase_a \\
        --db D:/My-Work/RADIANT/chembl_36.db \\
        --raw-dir D:/My-Work/RADIANT/data/raw \\
        --processed-dir D:/My-Work/RADIANT/data/processed/v1 \\
        [--skip-extract]            # if raw/raw_activities.parquet already exists
        [--n-jobs 4]                 # parallelize standardization
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--processed-dir", required=True, type=Path)
    p.add_argument("--skip-extract", action="store_true",
                   help="re-use raw_activities.parquet if it exists")
    p.add_argument("--skip-standardize", action="store_true",
                   help="re-use compounds.parquet if it exists")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=500_000)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from radiant_qsar.data.chembl_extract import ExtractConfig, extract_activities
    from radiant_qsar.data.standardize import StandardizeConfig, standardize_compounds
    from radiant_qsar.data.activity_curate import CurateConfig, curate_activities
    from radiant_qsar.data.descriptors import compute_and_save_descriptors
    from radiant_qsar.data.target_consolidate import ConsolidateConfig, consolidate_targets
    from radiant_qsar.data.manifest import ManifestConfig, build_manifest

    raw_path = args.raw_dir / "raw_activities.parquet"

    if args.skip_extract and raw_path.exists():
        logger.info("[1/6] extract: skipped (using existing %s)", raw_path)
    else:
        logger.info("[1/6] extract")
        extract_activities(
            ExtractConfig(
                db_path=args.db,
                out_dir=args.raw_dir,
                chunk_size=args.chunk_size,
                max_rows=args.max_rows,
            )
        )

    compounds_path = args.processed_dir / "compounds.parquet"
    if args.skip_standardize and compounds_path.exists():
        logger.info("[2/6] standardize: skipped (using existing %s)", compounds_path)
    else:
        logger.info("[2/6] standardize")
        standardize_compounds(
            StandardizeConfig(
                in_path=raw_path,
                out_dir=args.processed_dir,
                n_jobs=args.n_jobs,
            )
        )

    logger.info("[3/6] curate activities")
    curate_activities(
        CurateConfig(
            raw_path=raw_path,
            compounds_path=args.processed_dir / "compounds.parquet",
            out_dir=args.processed_dir,
        )
    )

    logger.info("[4/6] descriptors")
    compute_and_save_descriptors(
        args.processed_dir / "compounds.parquet",
        args.processed_dir / "descriptors.parquet",
    )

    logger.info("[5/6] consolidate targets")
    consolidate_targets(
        ConsolidateConfig(
            activities_path=args.processed_dir / "activities.parquet",
            db_path=args.db,
            out_dir=args.processed_dir,
        )
    )

    logger.info("[6/6] manifest")
    build_manifest(ManifestConfig(processed_dir=args.processed_dir))

    logger.info("done. Inspect %s/manifest.json for the release manifest.", args.processed_dir)


if __name__ == "__main__":
    main()
