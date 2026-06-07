"""Reactive-group SMARTS filters.

Beyond the published catalogs (PAINS / Brenk), some functional groups are
near-universally undesirable in HTS (acid chlorides, anhydrides, perchlorates).
This module bundles a deterministic, hand-curated list with a clear
"why each entry exists" comment.
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


# (description, SMARTS) pairs. Hand-curated; conservative.
REACTIVE_GROUPS: list[tuple[str, str]] = [
    ("acid_chloride",        "[CX3](=O)[Cl,Br,I]"),
    ("acid_anhydride",       "[CX3](=O)[OX2][CX3](=O)"),
    ("alkyl_halide_primary", "[CX4H2][Cl,Br,I]"),
    ("isocyanate",           "N=C=O"),
    ("isothiocyanate",       "N=C=S"),
    ("phosphoryl_halide",    "P(=O)([Cl,Br,F])"),
    ("sulfonyl_halide",      "S(=O)(=O)[Cl,Br,F]"),
    ("perchlorate",          "Cl(=O)(=O)(=O)O"),
    ("disulfide",            "[#16]-[#16]"),
    ("hydrazine",            "[NX3;H1,H2][NX3;H1,H2]"),
    ("nitroso",              "[NX2]=[OX1]"),
    ("nitrate_ester",        "[OX2][NX3](=O)=O"),
    ("azide",                "[NX2]=[NX2]=[NX1]"),
    ("acyl_cyanide",         "[CX3](=O)[CX2]#N"),
    ("imino_chloride",       "[Cl][CX3]=N"),
    ("phosphine",            "[PX3]"),
]


@register_filter("reactive_groups")
class ReactiveGroups(Filter):
    """Reject molecules containing any of the curated reactive-group SMARTS."""

    name = "reactive_groups"

    def __init__(self) -> None:
        self._compiled: list[tuple[str, object]] | None = None

    def _ensure(self):
        if self._compiled is None:
            from rdkit import Chem

            self._compiled = []
            for label, smarts in REACTIVE_GROUPS:
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._compiled.append((label, pat))

    def apply(self, mol, ctx) -> FilterResult:
        self._ensure()
        for label, pat in self._compiled:
            if mol.HasSubstructMatch(pat):
                return FilterResult(self.name, False, f"reactive group: {label}")
        return FilterResult(self.name, True)
