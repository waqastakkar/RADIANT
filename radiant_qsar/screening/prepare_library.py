"""CLI: prepare a compound library for virtual screening.

Two modes for selecting filters:

* ``--profile <name>``                 -- one of the predefined profiles.
* ``--filters lipinski,veber,pains``   -- explicit comma-separated list,
                                          all defaults.

Use ``--list-filters`` or ``--list-profiles`` to discover what's available.

Examples
--------
A CNS run on a vendor SMI file::

    python -m radiant_qsar.screening.prepare_library \\
        --input    library.smi \\
        --output   filtered.smi \\
        --profile  cns_brain_penetrant \\
        --rejects  rejected.csv \\
        --summary  summary.json

Custom subset for a kinase library::

    python -m radiant_qsar.screening.prepare_library \\
        --input    kinase_lib.smi \\
        --output   kinase_clean.smi \\
        --filters  lipinski,veber,kinase_hinge,pains,reactive_groups,dedup_inchikey
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from radiant_qsar.screening import (
    Pipeline,
    PROFILES,
    available_filters,
    get_filter,
)

logger = logging.getLogger(__name__)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, help="library file (.smi/.csv/.sdf, optionally .gz)")
    p.add_argument("--output", type=Path, help="passed-molecule .smi (smiles\\tid per line)")
    p.add_argument("--rejects", type=Path, default=None, help="optional rejected-molecule audit CSV")
    p.add_argument("--audit", type=Path, default=None,
                   help="optional full-audit CSV (every molecule with passed/failed_at/reason)")
    p.add_argument("--summary", type=Path, default=None, help="optional summary JSON")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--profile", type=str, help="named filter profile, e.g. cns_brain_penetrant")
    g.add_argument("--filters", type=str,
                   help="explicit comma-separated filter names (all defaults)")

    p.add_argument("--id-first", action="store_true",
                   help="for .smi: ID is the first whitespace token (default: auto-detect)")
    p.add_argument("--id-column", default="id", help="for .csv: id column name")
    p.add_argument("--smiles-column", default="smiles", help="for .csv: smiles column name")
    p.add_argument("--sdf-id-property", default="_Name",
                   help="for .sdf: SDF property to use as molecule id "
                        "(default: _Name; HY-Selleck NP libraries: Catalog_NO)")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--list-filters", action="store_true", help="list all registered filters and exit")
    p.add_argument("--list-profiles", action="store_true", help="list profiles and exit")
    return p.parse_args()


def _print_filters() -> None:
    print(f"{len(available_filters())} registered filters:")
    for name in available_filters():
        try:
            f = get_filter(name)
            desc = f.describe()
        except Exception as exc:
            desc = f"(error instantiating: {exc})"
        print(f"  {name:30s}  {desc}")


def _print_profiles() -> None:
    print(f"{len(PROFILES)} profiles:")
    for name, prof in PROFILES.items():
        print(f"  {name:25s}  {prof.description}")
        print(f"      filters: {', '.join(n for n, _ in prof.filters)}")


def main() -> int:
    args = _parse()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.list_filters:
        _print_filters()
        return 0
    if args.list_profiles:
        _print_profiles()
        return 0

    if args.input is None or args.output is None:
        print("ERROR: --input and --output are required (unless using --list-*).", file=sys.stderr)
        return 2

    if args.profile is None and args.filters is None:
        print("ERROR: pick --profile <name> or --filters a,b,c.", file=sys.stderr)
        return 2

    if args.profile:
        pipe = Pipeline.from_profile(args.profile)
    else:
        names = [n.strip() for n in args.filters.split(",") if n.strip()]
        pipe = Pipeline.from_specs([(n, {}) for n in names], profile_name="custom")

    logger.info("filters: %s", " | ".join(pipe.names()))

    summary = pipe.run(
        args.input,
        args.output,
        rejects_path=args.rejects,
        audit_path=args.audit,
        summary_path=args.summary,
        id_first=args.id_first,
        id_column=args.id_column,
        smiles_column=args.smiles_column,
        sdf_id_property=args.sdf_id_property,
        log_every=args.log_every,
    )

    print()
    print(f"=== summary ({pipe.profile_name or 'custom'}) ===")
    print(f"  input   : {summary.n_input}")
    print(f"  passed  : {summary.n_passed}  ({100*summary.n_passed/max(summary.n_input,1):.2f}%)")
    print(f"  failed  : {summary.n_failed}")
    print(f"  rejects by filter:")
    for name, n in sorted(summary.rejects_by_filter.items(), key=lambda kv: -kv[1]):
        print(f"    {name:30s}  {n}")
    print(f"  elapsed : {summary.elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
