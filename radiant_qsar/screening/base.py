"""Filter abstractions and the global registry.

A *filter* is a deterministic callable that, given a molecule and a
shared context, decides whether the molecule should pass to the next
stage of the pipeline. Every filter returns a :class:`FilterResult`
recording the decision, an optional reason, and any auxiliary scores.

Filters register themselves via :func:`register_filter` (or the
``@register_filter("name")`` decorator) so pipelines and CLI commands
can compose them by name only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# A "Mol" here is rdkit.Chem.Mol; we type-hint loosely so this module
# imports without rdkit available (filter modules pull it in lazily).
Mol = Any


@dataclass
class FilterContext:
    """Per-molecule shared state, accumulated across filters in one pass.

    Filters can stash precomputed values here (e.g. ``mol_wt``, ``logp``)
    so later filters don't recompute. Keys are filter-defined; collisions
    are the caller's responsibility.
    """

    smiles: str
    mol_id: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Outcome of one filter on one molecule."""

    name: str
    passed: bool
    reason: str = ""
    score: float | None = None


@dataclass
class MolReport:
    """Final outcome for one molecule across an entire pipeline."""

    mol_id: str
    smiles: str
    passed: bool
    failed_at: str | None = None       # name of the first filter that rejected
    failed_reason: str = ""
    results: list[FilterResult] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Filter base class
# ---------------------------------------------------------------------------
class Filter:
    """Subclass and override :py:meth:`apply` to define a filter.

    Attributes
    ----------
    name : str
        Unique short identifier used by profiles and the CLI. Subclasses
        usually set this as a class attribute.
    """

    name: str = ""

    def apply(self, mol: Mol, ctx: FilterContext) -> FilterResult:
        raise NotImplementedError

    def __call__(self, mol: Mol, ctx: FilterContext) -> FilterResult:
        return self.apply(mol, ctx)

    def describe(self) -> str:
        """One-line description; overridden by subclasses for the CLI ``--list``."""
        return self.__class__.__doc__.splitlines()[0].strip() if self.__class__.__doc__ else ""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable[..., Filter]] = {}


def register_filter(name: str):
    """Decorator: register a filter factory under ``name``.

    The decorated callable can be either:
      * a :class:`Filter` subclass (instantiated with no args), or
      * a factory function returning a :class:`Filter` instance.

    Re-registration is allowed (later wins) but logged.
    """
    def deco(factory):
        if isinstance(factory, type) and issubclass(factory, Filter):
            if not getattr(factory, "name", ""):
                factory.name = name

        _REGISTRY[name] = factory
        return factory

    return deco


def get_filter(name: str, **kwargs) -> Filter:
    """Instantiate a registered filter by name. ``kwargs`` are forwarded."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown filter {name!r}. Known filters: {sorted(_REGISTRY)}"
        )
    factory = _REGISTRY[name]
    if isinstance(factory, type):
        return factory(**kwargs) if kwargs else factory()
    return factory(**kwargs)


def available_filters() -> list[str]:
    return sorted(_REGISTRY)


def filter_names_iter() -> Iterable[str]:
    return iter(sorted(_REGISTRY))
