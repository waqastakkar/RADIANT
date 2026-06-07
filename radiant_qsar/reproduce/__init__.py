"""Reproducibility bundle for the RADIANT-QSAR study.

This package is the Phase I deliverable. The four artifacts it carries
are *static* (hand-authored, version-controlled), not pipeline outputs:

* ``manifest.yaml``      — single source of truth for versions,
  artifact hashes, and submission-bundle pointers.
* ``Dockerfile``         — single-image, single-command reproduction
  of the whole pipeline.
* ``run_all.sh``         — orchestration script that chains every
  phase's existing CLI driver.
* ``repro_checklist.md`` — Nature Machine Intelligence's mandatory
  Reporting Summary + Code/Data Availability checklist, kept in lock-
  step with the manifest.

The package also exposes a small Python helper, :func:`load_manifest`,
so other modules and tests can read the manifest without re-parsing the
YAML in multiple places.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["MANIFEST_PATH", "REPRODUCE_ROOT", "load_manifest"]

REPRODUCE_ROOT: Path = Path(__file__).resolve().parent
MANIFEST_PATH: Path = REPRODUCE_ROOT / "manifest.yaml"


def load_manifest() -> dict[str, Any]:
    """Parse ``manifest.yaml`` and return it as a plain dict.

    Raises :class:`RuntimeError` if PyYAML isn't installed, with the
    extras-install hint, so a user trying to reproduce the study isn't
    left guessing at a cryptic ``ModuleNotFoundError``.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read the reproducibility manifest; "
            "install it with `pip install -e .[yaml]` or `conda install pyyaml`."
        ) from exc

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file missing: {MANIFEST_PATH}")
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
