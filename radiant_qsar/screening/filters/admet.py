"""ADMET-related filters.

The truly accurate ADMET endpoints (CYP inhibition, hERG blockage, DILI)
require trained ML models -- the standard external sources are TDC's
DeepPurpose or ADMETlab2. This module provides:

* :class:`EsolSolubility` -- Delaney's published ESOL formula. Deterministic,
  no extra deps.
* :class:`HergProxy` -- a structural alert for the canonical hERG-blocker
  motif (basic N + lipophilic core + aromatic). It is *not* a substitute
  for a trained model; use it as a coarse pre-filter only.
* :class:`PgpProxy` -- coarse P-gp efflux liability proxy.
* :class:`MlAdmetHook` -- generic plugin point for a user-supplied
  ``Callable[[Mol], float]`` (e.g. a TDC model's predict). Returns
  ``passed=score <= threshold`` or its inverse.

All proxies are documented as proxies in their docstrings so reviewers
don't mistake them for endpoint predictions.
"""

from __future__ import annotations

import math
from typing import Callable

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter
from radiant_qsar.screening.filters.physchem import _descriptors


# ---------------------------------------------------------------------------
# ESOL: Delaney's empirical aqueous-solubility model.
#   logS = 0.16 - 0.63 * logP - 0.0062 * MW + 0.066 * RB - 0.74 * AP
# where AP = (#aromatic atoms) / (#heavy atoms)
# ---------------------------------------------------------------------------
def _esol_logs(mol, d: dict) -> float:
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    n = max(d["HeavyAtomCount"], 1)
    ap = aromatic_atoms / n
    return 0.16 - 0.63 * d["LogP"] - 0.0062 * d["MolWt"] + 0.066 * d["RotBonds"] - 0.74 * ap


@register_filter("esol_min")
class EsolSolubility(Filter):
    """Aqueous solubility via ESOL (Delaney 2004); reject below ``threshold`` log10(M)."""

    name = "esol_min"

    def __init__(self, threshold: float = -5.0) -> None:
        # -5 ~= 10 uM aqueous solubility, a common minimum for HTS-friendly libraries.
        self.threshold = threshold

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        logs = _esol_logs(mol, d)
        ctx.extras.setdefault("esol_logs", logs)
        ok = logs >= self.threshold
        return FilterResult(self.name, ok,
                            f"ESOL logS={logs:.2f} < {self.threshold}" if not ok else "",
                            logs)


# ---------------------------------------------------------------------------
# hERG proxy. Canonical hERG-blocker pharmacophore:
#   - basic nitrogen (typically protonated at pH 7)
#   - lipophilic aromatic core (pi-stack with hERG aromatic cage)
#   - LogP > 3
# This is a coarse rule-out used by medicinal chemists; it is *not* an
# endpoint prediction.
# ---------------------------------------------------------------------------
@register_filter("herg_proxy")
class HergProxy(Filter):
    """COARSE hERG-blocker proxy (basic N + LogP>3 + aromatic). Not a model."""

    name = "herg_proxy"

    def __init__(self, logp_max: float = 3.5) -> None:
        self.logp_max = logp_max

    def apply(self, mol, ctx) -> FilterResult:
        from rdkit import Chem

        d = _descriptors(mol, ctx)
        basic_n = mol.HasSubstructMatch(Chem.MolFromSmarts(
            "[#7;!H0;!$(N=*);!$(N#*);!$(N-S(=O)(=O));!$(N-C=O)]"
        ))
        aromatic = d["AromaticRings"] >= 2
        risky = basic_n and aromatic and d["LogP"] > self.logp_max
        if risky:
            return FilterResult(self.name, False,
                                f"hERG-like motif: basic N + {d['AromaticRings']} arom rings + LogP {d['LogP']:.2f}",
                                d["LogP"])
        return FilterResult(self.name, True)


# ---------------------------------------------------------------------------
# P-gp efflux proxy: high MW + multiple HBA -> increased efflux risk.
# ---------------------------------------------------------------------------
@register_filter("pgp_proxy")
class PgpProxy(Filter):
    """COARSE P-gp efflux proxy (MW >= 400 AND HBA >= 8). Not a model."""

    name = "pgp_proxy"

    def apply(self, mol, ctx) -> FilterResult:
        d = _descriptors(mol, ctx)
        if d["MolWt"] >= 400 and d["HBA"] >= 8:
            return FilterResult(self.name, False,
                                f"P-gp risk: MW={d['MolWt']:.0f}, HBA={d['HBA']}")
        return FilterResult(self.name, True)


# ---------------------------------------------------------------------------
# Generic plug-in for trained ADMET models (TDC, ADMETlab, custom).
# ---------------------------------------------------------------------------
class MlAdmetHook(Filter):
    """Wraps an external ``Callable[[Mol], float]`` as a filter.

    Use this to integrate a trained CYP / hERG / DILI / PPB model. The
    callable should return a single float; the filter passes the molecule
    if ``score <= threshold`` (or ``>= threshold`` when ``higher_is_safer``).

    Not registered by default -- callers instantiate directly.
    """

    def __init__(
        self,
        name: str,
        scorer: Callable[[object], float],
        threshold: float,
        *,
        higher_is_safer: bool = True,
        description: str = "",
    ) -> None:
        self.name = name
        self._scorer = scorer
        self.threshold = threshold
        self.higher_is_safer = higher_is_safer
        self._description = description

    def describe(self) -> str:  # pragma: no cover
        return self._description or f"ML ADMET hook ({self.name})"

    def apply(self, mol, ctx) -> FilterResult:
        try:
            score = float(self._scorer(mol))
        except Exception as exc:
            return FilterResult(self.name, False, f"scorer error: {exc}")
        if math.isnan(score):
            return FilterResult(self.name, False, "scorer returned NaN", score)
        ok = (score >= self.threshold) if self.higher_is_safer else (score <= self.threshold)
        sign = ">=" if self.higher_is_safer else "<="
        return FilterResult(self.name, ok,
                            f"{self.name} score {score:.3f} not {sign} {self.threshold}" if not ok else "",
                            score)
