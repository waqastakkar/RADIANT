"""Pick a reproducible target panel for the headline experiment.

Reads ``activities.parquet`` + ``targets.parquet`` from the curated v1
release, then picks the top-N most-data-rich targets *per class* subject
to minimum-compound and maximum-compound bounds. The result is a JSON
manifest (panel.json) consumed by :mod:`radiant_qsar.finetune.sweep`.

Why per-class
-------------
NMI-grade rigor requires generalization across target classes. Picking
"top 20 by compound count" alone ends up dominated by kinases (the most
data-rich class). Stratified-by-class selection forces the panel to
exercise every class with enough samples to support per-class statistics.

Default panel breakdown (20 targets / 7 classes)
-------------------------------------------------
    kinase            6
    gpcr              4
    protease          3
    nuclear_receptor  3
    other_enzyme      2
    transporter       1
    ion_channel       1
    -----            ---
    total             20

Override by passing ``--per-class 'kinase=8,gpcr=5,...'``.

CLI::

    python -m radiant_qsar.finetune.select_panel \\
        --activities data/processed/v1/activities.parquet \\
        --targets    data/processed/v1/targets.parquet \\
        --out        data/processed/v1/panel.json \\
        --min-compounds 1000 \\
        --max-compounds 50000
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Default per-class quotas summing to 20.
DEFAULT_QUOTAS: dict[str, int] = {
    "kinase": 6,
    "gpcr": 4,
    "protease": 3,
    "nuclear_receptor": 3,
    "other_enzyme": 2,
    "transporter": 1,
    "ion_channel": 1,
}


@dataclass
class PanelEntry:
    target_chembl_id: str
    target_class: str
    target_name: str | None
    organism: str | None
    uniprot: str | None
    n_compounds: int


@dataclass
class Panel:
    entries: list[PanelEntry] = field(default_factory=list)
    quotas: dict[str, int] = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)

    def by_class(self) -> dict[str, list[PanelEntry]]:
        out: dict[str, list[PanelEntry]] = {}
        for e in self.entries:
            out.setdefault(e.target_class, []).append(e)
        return out

    def to_dict(self) -> dict:
        return {
            "quotas": self.quotas,
            "constraints": self.constraints,
            "n_targets": len(self.entries),
            "by_class": {
                cls: [e.target_chembl_id for e in lst]
                for cls, lst in self.by_class().items()
            },
            "entries": [e.__dict__ for e in self.entries],
        }


def select_panel(
    activities_path: Path,
    targets_path: Path,
    *,
    per_class: dict[str, int] | None = None,
    min_compounds: int = 1000,
    max_compounds: int | None = 50000,
    organism_filter: str | None = "Homo sapiens",
) -> Panel:
    """Run selection. Returns a :class:`Panel` ready for serialization."""
    import pandas as pd

    quotas = dict(per_class or DEFAULT_QUOTAS)
    activities = pd.read_parquet(activities_path, columns=["target_chembl_id", "inchikey14"])
    targets = pd.read_parquet(targets_path)

    # Recount compounds-per-target from the curated activities table -- this is
    # the authoritative number after IQR / range / type filters.
    counts = (
        activities.groupby("target_chembl_id")["inchikey14"].nunique().rename("n_compounds")
    )
    targets = targets.merge(counts, left_on="target_chembl_id", right_index=True, how="left", suffixes=("_old", ""))
    if "n_compounds_old" in targets.columns:
        targets = targets.drop(columns=["n_compounds_old"])
    targets["n_compounds"] = targets["n_compounds"].fillna(0).astype(int)

    # Optional organism filter (Homo sapiens by default; reviewers expect human).
    if organism_filter:
        before = len(targets)
        targets = targets[targets["organism"].fillna("").str.contains(organism_filter, case=False)]
        logger.info("organism filter %r: %d -> %d", organism_filter, before, len(targets))

    # Compound-count bounds.
    targets = targets[targets["n_compounds"] >= min_compounds]
    if max_compounds is not None:
        # Cap to avoid one giant target dominating the per-class slice.
        targets = targets[targets["n_compounds"] <= max_compounds]

    panel = Panel(quotas=dict(quotas), constraints={
        "min_compounds": min_compounds,
        "max_compounds": max_compounds,
        "organism": organism_filter,
    })

    for cls, n in quotas.items():
        slice_ = targets[targets["target_class"] == cls].copy()
        slice_ = slice_.sort_values(["n_compounds"], ascending=False).head(n)
        if len(slice_) == 0:
            logger.warning("class %r: NO eligible targets after filters; skipping", cls)
            continue
        if len(slice_) < n:
            logger.warning("class %r: requested %d but only %d available", cls, n, len(slice_))
        for _, row in slice_.iterrows():
            panel.entries.append(PanelEntry(
                target_chembl_id=row["target_chembl_id"],
                target_class=row["target_class"],
                target_name=(row.get("target_name") if isinstance(row.get("target_name"), str) else None),
                organism=(row.get("organism") if isinstance(row.get("organism"), str) else None),
                uniprot=(row.get("uniprot") if isinstance(row.get("uniprot"), str) else None),
                n_compounds=int(row["n_compounds"]),
            ))

    return panel


def _parse_per_class(s: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(f"--per-class expects key=value pairs, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--targets", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path,
                   help="output JSON manifest")
    p.add_argument("--per-class", type=_parse_per_class, default=None,
                   help="override quotas: 'kinase=8,gpcr=5,...'")
    p.add_argument("--min-compounds", type=int, default=1000)
    p.add_argument("--max-compounds", type=int, default=50000)
    p.add_argument("--organism", type=str, default="Homo sapiens",
                   help="substring match on the organism column; pass '' to disable")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    panel = select_panel(
        activities_path=args.activities,
        targets_path=args.targets,
        per_class=args.per_class,
        min_compounds=args.min_compounds,
        max_compounds=args.max_compounds,
        organism_filter=args.organism or None,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(panel.to_dict(), indent=2), encoding="utf-8")

    print(f"=== panel ({len(panel.entries)} targets) ===")
    for cls, lst in panel.by_class().items():
        print(f"  {cls:18s} ({len(lst):2d}):")
        for e in lst:
            tag = e.target_chembl_id
            name = (e.target_name or "")[:40]
            print(f"    {tag:14s}  n_compounds={e.n_compounds:>6}  uniprot={e.uniprot or '?':10s}  {name}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
