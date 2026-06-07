"""Physicochemical / drug-likeness rule-based filters.

Each filter computes a small set of rdkit descriptors and applies a
classical published cutoff. Descriptors are cached on the
:class:`FilterContext` so subsequent filters re-use them.
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


def _descriptors(mol, ctx: FilterContext) -> dict:
    """Compute (or fetch from cache) the standard descriptor pack."""
    cache = ctx.extras
    if "physchem" in cache:
        return cache["physchem"]
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    d = {
        "MolWt": float(Descriptors.MolWt(mol)),
        "LogP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "RotBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "Rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "AromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "HeavyAtomCount": int(Descriptors.HeavyAtomCount(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
        "FormalCharge": int(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
    }
    cache["physchem"] = d
    return d


# ---------------------------------------------------------------------------
@register_filter("lipinski")
class LipinskiRo5(Filter):
    """Lipinski's Rule of 5 (oral bioavailability)."""

    name = "lipinski"

    def __init__(self, max_violations: int = 1) -> None:
        self.max_violations = max_violations

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        viol = (
            (d["MolWt"] > 500)
            + (d["LogP"] > 5)
            + (d["HBA"] > 10)
            + (d["HBD"] > 5)
        )
        passed = viol <= self.max_violations
        return FilterResult(
            name=self.name, passed=passed,
            reason=f"{viol} Lipinski violations" if not passed else "",
            score=float(viol),
        )


@register_filter("veber")
class Veber(Filter):
    """Veber rules: oral bioavailability via TPSA and rotatable-bond cutoffs."""

    name = "veber"

    def __init__(self, tpsa_max: float = 140.0, rotbonds_max: int = 10) -> None:
        self.tpsa_max = tpsa_max
        self.rotbonds_max = rotbonds_max

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if d["TPSA"] > self.tpsa_max:
            return FilterResult(self.name, False, f"TPSA={d['TPSA']:.1f} > {self.tpsa_max}", d["TPSA"])
        if d["RotBonds"] > self.rotbonds_max:
            return FilterResult(self.name, False, f"RotBonds={d['RotBonds']} > {self.rotbonds_max}", d["RotBonds"])
        return FilterResult(self.name, True)


@register_filter("ghose")
class Ghose(Filter):
    """Ghose drug-likeness window (small/lead-like)."""

    name = "ghose"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        ok = (
            160 <= d["MolWt"] <= 480
            and -0.4 <= d["LogP"] <= 5.6
            and 20 <= d["HeavyAtomCount"] <= 70
            and 40 <= d["MolMR"] <= 130
        )
        return FilterResult(self.name, ok, "outside Ghose window" if not ok else "")


@register_filter("egan")
class Egan(Filter):
    """Egan rules: oral bioavailability via TPSA + LogP envelope."""

    name = "egan"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if d["LogP"] > 5.88:
            return FilterResult(self.name, False, f"LogP={d['LogP']:.2f} > 5.88", d["LogP"])
        if d["TPSA"] > 131.6:
            return FilterResult(self.name, False, f"TPSA={d['TPSA']:.1f} > 131.6", d["TPSA"])
        return FilterResult(self.name, True)


@register_filter("muegge")
class Muegge(Filter):
    """Muegge rules: more stringent drug-likeness."""

    name = "muegge"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if not (200 <= d["MolWt"] <= 600):
            return FilterResult(self.name, False, f"MolWt {d['MolWt']:.0f} outside [200,600]")
        if not (-2 <= d["LogP"] <= 5):
            return FilterResult(self.name, False, f"LogP {d['LogP']:.2f} outside [-2,5]")
        if d["TPSA"] > 150:
            return FilterResult(self.name, False, f"TPSA {d['TPSA']:.1f} > 150")
        if d["Rings"] > 7:
            return FilterResult(self.name, False, f"Rings {d['Rings']} > 7")
        if d["RotBonds"] > 15:
            return FilterResult(self.name, False, f"RotBonds {d['RotBonds']} > 15")
        if d["HBA"] > 10 or d["HBD"] > 5:
            return FilterResult(self.name, False, "HBA>10 or HBD>5")
        return FilterResult(self.name, True)


@register_filter("rule_of_three")
class RuleOfThree(Filter):
    """Astex Rule of 3 (fragment-likeness)."""

    name = "rule_of_three"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        ok = (
            d["MolWt"] <= 300
            and d["LogP"] <= 3
            and d["HBA"] <= 3
            and d["HBD"] <= 3
            and d["RotBonds"] <= 3
            and d["TPSA"] <= 60
        )
        return FilterResult(self.name, ok, "outside Rule of 3" if not ok else "")


@register_filter("formal_charge")
class FormalChargeFilter(Filter):
    """Reject molecules whose net formal charge is outside ``[charge_min, charge_max]``."""

    name = "formal_charge"

    def __init__(self, charge_min: int = -2, charge_max: int = 2) -> None:
        self.charge_min = charge_min
        self.charge_max = charge_max

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        c = d["FormalCharge"]
        ok = self.charge_min <= c <= self.charge_max
        return FilterResult(self.name, ok, f"charge {c} outside [{self.charge_min},{self.charge_max}]" if not ok else "", float(c))


@register_filter("heavy_atom_range")
class HeavyAtomRange(Filter):
    """Reject molecules outside the heavy-atom range."""

    name = "heavy_atom_range"

    def __init__(self, min_heavy: int = 8, max_heavy: int = 70) -> None:
        self.min_heavy = min_heavy
        self.max_heavy = max_heavy

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        n = d["HeavyAtomCount"]
        ok = self.min_heavy <= n <= self.max_heavy
        return FilterResult(self.name, ok,
                            f"HeavyAtoms={n} outside [{self.min_heavy},{self.max_heavy}]" if not ok else "",
                            float(n))


@register_filter("molwt_range")
class MolWtRange(Filter):
    """Reject molecules outside a configurable MW window."""

    name = "molwt_range"

    def __init__(self, min_mw: float = 100.0, max_mw: float = 800.0) -> None:
        self.min_mw = min_mw
        self.max_mw = max_mw

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        mw = d["MolWt"]
        ok = self.min_mw <= mw <= self.max_mw
        return FilterResult(self.name, ok,
                            f"MolWt={mw:.1f} outside [{self.min_mw},{self.max_mw}]" if not ok else "",
                            mw)
