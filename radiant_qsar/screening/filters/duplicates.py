"""Duplicate removal by InChIKey-14.

A *stateful* filter -- it tracks the InChIKey-14 of every molecule it has
already passed. The first occurrence passes; subsequent occurrences fail
with reason "duplicate of <id>".
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


@register_filter("dedup_inchikey")
class DedupByInchiKey(Filter):
    """Reject any molecule whose InChIKey-14 has already passed the filter."""

    name = "dedup_inchikey"

    def __init__(self) -> None:
        # Maps inchikey14 -> the mol_id we kept it under.
        self._seen: dict[str, str] = {}

    def reset(self) -> None:
        """Clear the dedup memory (e.g. between independent libraries)."""
        self._seen.clear()

    def apply(self, mol, ctx) -> FilterResult:
        from rdkit import Chem

        try:
            ikey = Chem.MolToInchiKey(mol)
        except Exception as exc:
            return FilterResult(self.name, False, f"InChIKey failed: {exc}")
        if not ikey:
            return FilterResult(self.name, False, "empty InChIKey")
        ikey14 = ikey[:14]
        ctx.extras.setdefault("inchikey14", ikey14)
        prior = self._seen.get(ikey14)
        if prior is not None:
            return FilterResult(self.name, False, f"duplicate of {prior}")
        self._seen[ikey14] = ctx.mol_id
        return FilterResult(self.name, True)
