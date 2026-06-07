"""Target-class-specific structural requirements.

Each filter encodes a coarse pharmacophore that medicinal chemists use as
a rough "is this molecule even in the right ballpark for target X?" gate.
None of them is sufficient for triage on its own; they're meant to be
composed with the other filter families.
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


# ---------------------------------------------------------------------------
# Kinase hinge-binder pharmacophore.
# The kinase hinge "donor-acceptor" pair (most kinases) is satisfied by:
#   - aminopyrimidine, aminopyridine, indazole, indole-3, pyrrolopyrimidine,
#     7-azaindole, ...
# We approximate via a few SMARTS that cover the most common scaffolds.
# ---------------------------------------------------------------------------
_KINASE_HINGE_SMARTS = (
    "n1cncc1[NH2]",                     # aminopyrimidine / aminoimidazole
    "c1cc[nH]c1",                       # pyrrole-ish hinge
    "c1ccc2[nH]ncc2c1",                 # indazole
    "c1ccc2[nH]ccc2c1",                 # indole
    "c1cc2nc[nH]c2nc1",                 # purine-like
    "c1ncc2[nH]ccc2n1",                 # 7-azaindole / pyrrolopyrimidine
    "c1cc(N)nc(N)n1",                   # diaminopyrimidine
)


@register_filter("kinase_hinge")
class KinaseHingeBinder(Filter):
    """Require at least one canonical kinase-hinge SMARTS match."""

    name = "kinase_hinge"

    def __init__(self) -> None:
        self._patterns = None

    def _ensure_patterns(self):
        if self._patterns is None:
            from rdkit import Chem

            self._patterns = [Chem.MolFromSmarts(s) for s in _KINASE_HINGE_SMARTS]

    def apply(self, mol, ctx) -> FilterResult:
        self._ensure_patterns()
        for pat in self._patterns:
            if pat is not None and mol.HasSubstructMatch(pat):
                return FilterResult(self.name, True)
        return FilterResult(self.name, False, "no kinase-hinge SMARTS match")


# ---------------------------------------------------------------------------
# GPCR aminergic basic amine.
# Most aminergic GPCR ligands (5HT, dopaminergic, adrenergic, histaminergic)
# carry a basic amine that ionizes at pH 7 and salt-bridges with Asp3.32.
# A "basic amine" includes primary, secondary, *and* tertiary aliphatic N,
# plus pyridyl-type aromatic N. Excludes amides, sulfonamides, imines,
# nitroso, etc., where the lone pair is delocalized.
# ---------------------------------------------------------------------------
BASIC_AMINE_SMARTS_LIST: tuple[str, ...] = (
    # Aliphatic sp3 N (any H count: 0/1/2 -- tertiary still counts)
    "[NX3;!$(N-C(=O));!$(N-C(=N));!$(N-S(=O)(=O));!$(N-N=O);!$(N=*);!$(N#*);!$(N-O)]",
    # Pyridyl-type aromatic N (basic). Excludes N-H pyrrole and pyridinium.
    "[nX2;!$([nH]);!$([n+])]",
)


def _has_basic_amine(mol) -> bool:
    from rdkit import Chem

    for smarts in BASIC_AMINE_SMARTS_LIST:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            return True
    return False


@register_filter("basic_amine_required")
class BasicAmineRequired(Filter):
    """Require at least one basic-amine nitrogen (aminergic-GPCR-style)."""

    name = "basic_amine_required"

    def apply(self, mol, ctx) -> FilterResult:
        if _has_basic_amine(mol):
            return FilterResult(self.name, True)
        return FilterResult(self.name, False, "no basic amine present")


# ---------------------------------------------------------------------------
# Covalent warheads.
# Useful as either a *positive* requirement (covalent-inhibitor screening)
# or a *negative* filter (general HTS). The default mode is "require".
# ---------------------------------------------------------------------------
_COVALENT_WARHEAD_SMARTS = (
    # Acrylamide / Michael acceptor
    "C=CC(=O)N",
    # Vinyl sulfone
    "C=CS(=O)(=O)",
    # Chloroacetamide
    "ClCC(=O)N",
    # Fluoromethyl ketone
    "FCC(=O)",
    # Boronic acid / ester (proteasome, serine hydrolases)
    "B(O)O",
    # Aldehyde (calpain etc.)
    "[CX3H1](=O)",
    # Epoxide
    "C1OC1",
    # Aziridine
    "C1NC1",
    # Cyanamide / nitrile (e.g. saxagliptin, cathepsin K)
    "[CX2]#[NX1]",
)


@register_filter("covalent_warhead_required")
class CovalentWarheadRequired(Filter):
    """Require at least one canonical covalent-warhead SMARTS hit."""

    name = "covalent_warhead_required"

    def __init__(self) -> None:
        self._patterns = None

    def _ensure_patterns(self):
        if self._patterns is None:
            from rdkit import Chem

            self._patterns = [Chem.MolFromSmarts(s) for s in _COVALENT_WARHEAD_SMARTS]

    def apply(self, mol, ctx) -> FilterResult:
        self._ensure_patterns()
        for pat in self._patterns:
            if pat is not None and mol.HasSubstructMatch(pat):
                return FilterResult(self.name, True)
        return FilterResult(self.name, False, "no covalent warhead present")


@register_filter("covalent_warhead_forbidden")
class CovalentWarheadForbidden(Filter):
    """Reject any molecule containing a covalent-warhead SMARTS hit."""

    name = "covalent_warhead_forbidden"

    def __init__(self) -> None:
        self._patterns = None

    def _ensure_patterns(self):
        if self._patterns is None:
            from rdkit import Chem

            self._patterns = [Chem.MolFromSmarts(s) for s in _COVALENT_WARHEAD_SMARTS]

    def apply(self, mol, ctx) -> FilterResult:
        self._ensure_patterns()
        for pat in self._patterns:
            if pat is not None and mol.HasSubstructMatch(pat):
                return FilterResult(self.name, False, "covalent warhead present")
        return FilterResult(self.name, True)
