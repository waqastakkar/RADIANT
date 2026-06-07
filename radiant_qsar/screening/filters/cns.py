"""CNS / brain-penetrance filters.

* :func:`cns_mpo` -- the Wager *et al.* 6-property MPO score (range 0-6,
  higher is better; threshold 4 by default).
* :func:`bbb_egan` -- a permissive BBB filter based on TPSA + LogP.
* :func:`bbb_strict` -- a stricter BBB-likely envelope.
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter
from radiant_qsar.screening.filters.physchem import _descriptors


def _hump(value: float, lo: float, hi: float, lo_zero: float, hi_zero: float) -> float:
    """Trapezoidal MPO sub-score: 1.0 in [lo,hi], 0 outside [lo_zero,hi_zero], linear in between."""
    if value < lo_zero or value > hi_zero:
        return 0.0
    if lo <= value <= hi:
        return 1.0
    if value < lo:
        return (value - lo_zero) / max(lo - lo_zero, 1e-9)
    return (hi_zero - value) / max(hi_zero - hi, 1e-9)


def _cns_mpo(d: dict) -> dict:
    """Wager 2010 6-component MPO. Returns sub-scores + total (0-6)."""
    parts = {
        "mw":   _hump(d["MolWt"], 0, 360, 0, 500),
        "logp": _hump(d["LogP"], 0, 3, 0, 5),
        "logd": _hump(d["LogP"], 0, 2, 0, 4),  # logD ~ logP if no ionizable, OK proxy
        "tpsa": _hump(d["TPSA"], 40, 90, 20, 120),
        "hbd":  _hump(d["HBD"], 0, 0.5, 0, 3.5),
        # pKa most-basic; we have no rdkit tool that handles this cheaply,
        # so use a structural proxy: presence of basic N -> higher score.
        # Returns 0 (no basic N) or 1 (one or more basic N atoms).
    }
    parts["pka"] = _basic_amine_score(d.get("_mol"))
    parts["total"] = sum(parts[k] for k in ("mw", "logp", "logd", "tpsa", "hbd", "pka"))
    return parts


def _basic_amine_score(mol) -> float:
    """1.0 if the molecule has a basic nitrogen (pyridyl or non-amide aliphatic N), else 0.0."""
    if mol is None:
        return 0.0
    # Defer import to avoid load-order coupling; both filter modules are imported
    # by ``radiant_qsar.screening.__init__``.
    from radiant_qsar.screening.filters.target_specific import _has_basic_amine

    return 1.0 if _has_basic_amine(mol) else 0.0


@register_filter("cns_mpo")
class CNSMpoFilter(Filter):
    """CNS MPO score (Wager 2010); reject below threshold (default >= 4 of 6)."""

    name = "cns_mpo"

    def __init__(self, threshold: float = 4.0) -> None:
        if threshold < 0 or threshold > 6:
            raise ValueError("CNS MPO threshold must be in [0, 6]")
        self.threshold = threshold

    def apply(self, mol, ctx) -> FilterResult:
        d = dict(_descriptors(mol, ctx))  # copy; don't pollute cache with _mol
        d["_mol"] = mol
        scores = _cns_mpo(d)
        total = scores["total"]
        ctx.extras.setdefault("cns_mpo", total)
        ok = total >= self.threshold
        return FilterResult(self.name, ok, f"CNS MPO={total:.2f} < {self.threshold}" if not ok else "", total)


@register_filter("bbb_egan")
class BbbEgan(Filter):
    """Permissive BBB filter (Egan's envelope: TPSA <= 90, LogP in [1, 4])."""

    name = "bbb_egan"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if d["TPSA"] > 90:
            return FilterResult(self.name, False, f"TPSA={d['TPSA']:.1f} > 90", d["TPSA"])
        if not (1.0 <= d["LogP"] <= 4.0):
            return FilterResult(self.name, False, f"LogP={d['LogP']:.2f} outside [1,4]", d["LogP"])
        return FilterResult(self.name, True)


@register_filter("bbb_strict")
class BbbStrict(Filter):
    """Strict BBB envelope (TPSA <= 70, MW <= 450, HBD <= 3, LogP in [1, 3.5])."""

    name = "bbb_strict"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if d["TPSA"] > 70:
            return FilterResult(self.name, False, f"TPSA={d['TPSA']:.1f} > 70")
        if d["MolWt"] > 450:
            return FilterResult(self.name, False, f"MolWt={d['MolWt']:.0f} > 450")
        if d["HBD"] > 3:
            return FilterResult(self.name, False, f"HBD={d['HBD']} > 3")
        if not (1.0 <= d["LogP"] <= 3.5):
            return FilterResult(self.name, False, f"LogP={d['LogP']:.2f} outside [1,3.5]")
        return FilterResult(self.name, True)
