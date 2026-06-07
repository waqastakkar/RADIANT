"""Structural-alert filters using RDKit's FilterCatalog.

Catalogs available:
    PAINS  -- pan-assay interference (Baell & Holloway 2010), aggregated
    PAINS_A / PAINS_B / PAINS_C -- the three sub-catalogs separately
    BRENK  -- Brenk et al. 2008 medicinal-chem alerts
    NIH    -- NIH unwanted compounds
    ZINC   -- ZINC's "drug-like" filter

Each catalog is wrapped as a filter returning passed=False for any hit.
Hit details are reported in the ``reason`` field for downstream auditing.
"""

from __future__ import annotations

from radiant_qsar.screening.base import Filter, FilterContext, FilterResult, register_filter


_CATALOG_KEYS = {
    "pains": "PAINS",
    "pains_a": "PAINS_A",
    "pains_b": "PAINS_B",
    "pains_c": "PAINS_C",
    "brenk": "BRENK",
    "nih": "NIH",
    "zinc": "ZINC",
    "all_alerts": "ALL",
}


def _make_catalog(key: str):
    from rdkit.Chem import FilterCatalog

    params = FilterCatalog.FilterCatalogParams()
    cat_enum = getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, _CATALOG_KEYS[key])
    params.AddCatalog(cat_enum)
    return FilterCatalog.FilterCatalog(params)


class _CatalogFilter(Filter):
    """Generic wrapper around an RDKit FilterCatalog."""

    catalog_key: str = ""

    def __init__(self) -> None:
        self._catalog = None

    def _ensure_catalog(self):
        if self._catalog is None:
            self._catalog = _make_catalog(self.catalog_key)

    def apply(self, mol, ctx) -> FilterResult:
        self._ensure_catalog()
        entry = self._catalog.GetFirstMatch(mol)
        if entry is None:
            return FilterResult(self.name, True)
        desc = entry.GetDescription() or "<no description>"
        return FilterResult(self.name, False, f"{self.name} hit: {desc}")


@register_filter("pains")
class PainsFilter(_CatalogFilter):
    """PAINS (pan-assay interference)."""
    name = "pains"
    catalog_key = "pains"


@register_filter("pains_a")
class PainsAFilter(_CatalogFilter):
    """PAINS A sub-catalog."""
    name = "pains_a"
    catalog_key = "pains_a"


@register_filter("pains_b")
class PainsBFilter(_CatalogFilter):
    """PAINS B sub-catalog."""
    name = "pains_b"
    catalog_key = "pains_b"


@register_filter("pains_c")
class PainsCFilter(_CatalogFilter):
    """PAINS C sub-catalog."""
    name = "pains_c"
    catalog_key = "pains_c"


@register_filter("brenk")
class BrenkFilter(_CatalogFilter):
    """Brenk medicinal-chemistry alerts."""
    name = "brenk"
    catalog_key = "brenk"


@register_filter("nih")
class NihFilter(_CatalogFilter):
    """NIH unwanted-compounds catalog."""
    name = "nih"
    catalog_key = "nih"


@register_filter("zinc")
class ZincFilter(_CatalogFilter):
    """ZINC's drug-like filter."""
    name = "zinc"
    catalog_key = "zinc"


@register_filter("all_alerts")
class AllAlertsFilter(_CatalogFilter):
    """Union of all RDKit catalogs (PAINS + BRENK + NIH + ZINC)."""
    name = "all_alerts"
    catalog_key = "all_alerts"
