"""The Pipeline: chains filters and runs over a library.

Behavior:
  * Streams molecules from the loader so memory stays flat.
  * Short-circuits on first failed filter (fastest case for HTS triage).
  * Records every per-molecule result for an audit CSV.
  * Optionally writes a kept-molecule .smi and a rejected-molecule CSV.
  * Reports per-filter pass / reject counts at the end.

Pipelines compose with :class:`radiant_qsar.screening.profiles.Profile`
or by passing an explicit ``filters=[(name, kwargs), ...]`` list.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from radiant_qsar.screening.base import (
    Filter,
    FilterContext,
    MolReport,
    get_filter,
)
from radiant_qsar.screening.library_loader import load_library
from radiant_qsar.screening.profiles import Profile, get_profile

logger = logging.getLogger(__name__)


@dataclass
class PipelineSummary:
    n_input: int = 0
    n_passed: int = 0
    n_failed: int = 0
    elapsed_s: float = 0.0
    rejects_by_filter: dict[str, int] = field(default_factory=dict)
    profile_name: str = ""

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "pass_rate": (self.n_passed / max(self.n_input, 1)),
            "elapsed_s": round(self.elapsed_s, 2),
            "rejects_by_filter": dict(self.rejects_by_filter),
            "profile": self.profile_name,
        }


class Pipeline:
    """An ordered sequence of filters applied to each molecule."""

    def __init__(self, filters: list[Filter], *, profile_name: str = "") -> None:
        if not filters:
            raise ValueError("Pipeline requires at least one filter.")
        self.filters = filters
        self.profile_name = profile_name

    # ------------------------------------------------------------------
    @classmethod
    def from_profile(cls, profile_name: str) -> "Pipeline":
        profile = get_profile(profile_name)
        return cls.from_specs(profile.filters, profile_name=profile.name)

    @classmethod
    def from_specs(cls, specs: list[tuple[str, dict]], *, profile_name: str = "") -> "Pipeline":
        filters = [get_filter(name, **kwargs) for name, kwargs in specs]
        return cls(filters, profile_name=profile_name)

    # ------------------------------------------------------------------
    def apply_to_mol(self, mol_id: str, smiles: str, mol) -> MolReport:
        """Run all filters on one molecule, short-circuit on first failure."""
        ctx = FilterContext(smiles=smiles, mol_id=mol_id)
        report = MolReport(mol_id=mol_id, smiles=smiles, passed=True)

        for f in self.filters:
            try:
                result = f(mol, ctx)
            except Exception as exc:  # pragma: no cover -- filters are tested
                report.passed = False
                report.failed_at = f.name or f.__class__.__name__
                report.failed_reason = f"filter error: {exc}"
                return report
            report.results.append(result)
            if result.score is not None:
                report.scores[result.name] = float(result.score)
            if not result.passed:
                report.passed = False
                report.failed_at = result.name
                report.failed_reason = result.reason
                return report
        return report

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        rejects_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        summary_path: str | Path | None = None,
        id_first: bool | None = None,
        id_column: str = "id",
        smiles_column: str = "smiles",
        sdf_id_property: str = "_Name",
        log_every: int = 10_000,
    ) -> PipelineSummary:
        """Stream a library through the pipeline and write outputs."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rejects_w = audit_w = None
        rejects_f = audit_f = None
        if rejects_path is not None:
            rejects_path = Path(rejects_path)
            rejects_path.parent.mkdir(parents=True, exist_ok=True)
            rejects_f = open(rejects_path, "w", encoding="utf-8", newline="")
            rejects_w = csv.writer(rejects_f)
            rejects_w.writerow(["mol_id", "smiles", "failed_at", "reason"])
        if audit_path is not None:
            audit_path = Path(audit_path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_f = open(audit_path, "w", encoding="utf-8", newline="")
            audit_w = csv.writer(audit_f)
            audit_w.writerow(["mol_id", "smiles", "passed", "failed_at", "reason"])

        out_f = open(output_path, "w", encoding="utf-8")
        try:
            summary = PipelineSummary(profile_name=self.profile_name)
            t0 = time.time()
            for mol_id, smi, mol in load_library(
                input_path,
                id_first=id_first,
                id_column=id_column,
                smiles_column=smiles_column,
                sdf_id_property=sdf_id_property,
            ):
                summary.n_input += 1
                report = self.apply_to_mol(mol_id, smi, mol)
                if report.passed:
                    summary.n_passed += 1
                    out_f.write(f"{smi}\t{mol_id}\n")
                else:
                    summary.n_failed += 1
                    bucket = report.failed_at or "unknown"
                    summary.rejects_by_filter[bucket] = summary.rejects_by_filter.get(bucket, 0) + 1
                    if rejects_w is not None:
                        rejects_w.writerow([mol_id, smi, report.failed_at, report.failed_reason])
                if audit_w is not None:
                    audit_w.writerow([mol_id, smi, int(report.passed),
                                      report.failed_at or "", report.failed_reason])
                if summary.n_input % log_every == 0:
                    logger.info(
                        "  %d processed (%d passed, %d failed) in %.1fs",
                        summary.n_input, summary.n_passed, summary.n_failed,
                        time.time() - t0,
                    )
            summary.elapsed_s = time.time() - t0
        finally:
            out_f.close()
            if rejects_f is not None:
                rejects_f.close()
            if audit_f is not None:
                audit_f.close()

        if summary_path is not None:
            Path(summary_path).write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

        logger.info(
            "pipeline done: input=%d passed=%d (%.2f%%) failed=%d  elapsed=%.1fs",
            summary.n_input, summary.n_passed,
            100.0 * summary.n_passed / max(summary.n_input, 1),
            summary.n_failed, summary.elapsed_s,
        )
        return summary

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return [f.name or f.__class__.__name__ for f in self.filters]
