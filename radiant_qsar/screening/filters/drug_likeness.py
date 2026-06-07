"""Drug-likeness scores: QED and SAscore."""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


@register_filter("qed_min")
class QEDMin(Filter):
    """Reject molecules with quantitative drug-likeness QED below a threshold."""

    name = "qed_min"

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("QED threshold must be in [0, 1]")
        self.threshold = threshold

    def apply(self, mol, ctx) -> FilterResult:
        from rdkit.Chem import QED

        if "qed" not in ctx.extras:
            ctx.extras["qed"] = float(QED.qed(mol))
        q = ctx.extras["qed"]
        ok = q >= self.threshold
        return FilterResult(self.name, ok, f"QED={q:.3f} < {self.threshold}" if not ok else "", q)


@register_filter("sa_max")
class SyntheticAccessibilityMax(Filter):
    """Reject molecules with predicted synthetic difficulty above the threshold.

    Uses RDKit Contrib's SAscore when available (1 = trivial, 10 = very
    hard); falls back to a fast structural proxy (rings + non-sp3 fraction
    + chiral-center count) if not. The proxy is documented and
    deterministic but should not be confused with the published SAscore.
    """

    name = "sa_max"

    def __init__(self, threshold: float = 6.0, prefer_rdkit: bool = True) -> None:
        self.threshold = threshold
        self.prefer_rdkit = prefer_rdkit

    @staticmethod
    def _fast_proxy(mol) -> float:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        rings = rdMolDescriptors.CalcNumRings(mol)
        sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        # Keep the scale comparable to SAscore's 1-10 range.
        return float(min(10.0, 1.0 + rings + (1 - sp3) + 0.5 * chiral))

    def apply(self, mol, ctx) -> FilterResult:
        score: float | None = None
        if self.prefer_rdkit:
            try:
                # RDKit Contrib ships sascorer.py at Contrib/SA_Score/. It's
                # not part of the default rdkit namespace. We try-import it
                # and fall back gracefully when the contrib isn't on PYTHONPATH.
                from rdkit.Chem import RDConfig
                import os, sys
                contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
                if contrib not in sys.path:
                    sys.path.append(contrib)
                import sascorer  # type: ignore
                score = float(sascorer.calculateScore(mol))
            except Exception:
                score = None
        if score is None:
            score = self._fast_proxy(mol)

        ctx.extras.setdefault("sascore", score)
        ok = score <= self.threshold
        return FilterResult(self.name, ok, f"SA={score:.2f} > {self.threshold}" if not ok else "", score)
