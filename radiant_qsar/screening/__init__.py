"""Virtual-screening library preparation.

Modular filter pipeline for triaging compound libraries before docking or
ML scoring. Each filter is an independent, registered callable; profiles
compose filters for common target classes.

Public API
----------

* :class:`Filter` -- base class.
* :class:`FilterResult` / :class:`MolReport` -- per-molecule outcomes.
* :class:`Pipeline` -- chains filters and runs over a library.
* :func:`load_library` -- iterator over (id, smiles, mol) triples from
  ``.smi``, ``.csv``, or ``.sdf`` files.
* :func:`get_filter` / :func:`available_filters` -- registry access.
* :class:`Profile` / :func:`get_profile` / :data:`PROFILES` -- preset
  filter combinations per target class.

Typical usage::

    from radiant_qsar.screening import Pipeline, get_profile

    pipe = Pipeline.from_profile("cns_brain_penetrant")
    pipe.run("library.smi", "filtered.smi")
"""

from radiant_qsar.screening.base import (
    Filter,
    FilterContext,
    FilterResult,
    MolReport,
    available_filters,
    get_filter,
    register_filter,
)
from radiant_qsar.screening.library_loader import load_library
from radiant_qsar.screening.pipeline import Pipeline
from radiant_qsar.screening.profiles import PROFILES, Profile, get_profile

# Importing the filter modules registers their filters as a side effect.
from radiant_qsar.screening.filters import (  # noqa: F401
    physchem,
    drug_likeness,
    structural_alerts,
    cns,
    admet,
    target_specific,
    reactive,
    duplicates,
    ml_scoring,
)

__all__ = [
    "Filter",
    "FilterContext",
    "FilterResult",
    "MolReport",
    "Pipeline",
    "Profile",
    "PROFILES",
    "available_filters",
    "get_filter",
    "get_profile",
    "load_library",
    "register_filter",
]
