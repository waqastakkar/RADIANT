"""Smoke tests for the Phase I reproducibility bundle.

These tests are deliberately cheap: they verify the static artifacts
under ``radiant_qsar/reproduce/`` are parseable, internally
consistent, and that every *local* path the manifest references
actually exists. They do *not* exercise the pipeline -- ``run_all.sh``
does that.

The goal is that a green test run on a fresh clone is sufficient
evidence that the reproducibility bundle is well-formed; the only way
to land an inconsistent ``manifest.yaml`` is to also fix this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from radiant_qsar.reproduce import MANIFEST_PATH, REPRODUCE_ROOT, load_manifest


REPO_ROOT = REPRODUCE_ROOT.parent.parent  # radiant_qsar/reproduce -> repo root


def test_manifest_yaml_is_well_formed():
    manifest = load_manifest()
    assert isinstance(manifest, dict)
    for required in ("study", "environment", "dependencies", "data_source",
                     "data_release", "tokenizer", "pretrain", "finetune",
                     "baselines", "phase_g", "tests", "bundle"):
        assert required in manifest, f"manifest.yaml missing top-level section: {required}"


def test_required_sibling_files_exist():
    for name in ("Dockerfile", "run_all.sh", "repro_checklist.md", "__init__.py"):
        assert (REPRODUCE_ROOT / name).exists(), f"reproduce/ missing {name}"


def test_phase_g_modules_resolve_to_importable_packages():
    manifest = load_manifest()
    modules = manifest["phase_g"]["modules"]
    for name, info in modules.items():
        mod_path = info["module"]
        # We don't import (saves heavy deps in CI); we check the file exists.
        rel = mod_path.replace(".", "/") + ".py"
        assert (REPO_ROOT / rel).exists(), f"{name}: source file missing for module '{mod_path}' ({rel})"


def test_data_release_files_listed_match_local_manifest():
    """If `data/processed/v1/manifest.json` exists locally, the hashes in
    `manifest.yaml:data_release.files` must agree. If the curated release
    isn't built yet, skip silently — Phase A hasn't been run."""
    import json

    local_manifest = REPO_ROOT / "data" / "processed" / "v1" / "manifest.json"
    if not local_manifest.exists():
        pytest.skip("Phase A output not present; nothing to cross-check.")
    local = json.loads(local_manifest.read_text())
    declared = load_manifest()["data_release"]["files"]
    for fname, decl in declared.items():
        if fname not in local["files"]:
            pytest.fail(f"manifest.yaml references {fname} but local manifest.json doesn't")
        assert local["files"][fname]["sha256"] == decl["sha256"], (
            f"SHA-256 mismatch for {fname}; update manifest.yaml"
        )


def test_local_paths_in_manifest_exist_when_phase_complete():
    """Best-effort: for every path-shaped string in the manifest, if its
    *parent directory* exists we expect the leaf to exist too (i.e. a
    phase has been started but the artifact is missing). Paths that
    point into never-built directories are skipped silently."""
    manifest = load_manifest()

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        elif isinstance(node, str):
            yield node

    # Artifacts produced only at pipeline run time; missing != broken manifest.
    runtime_outputs = {"best.pt", "PHASE_G_REPORT.md"}
    runtime_roots = ("runs/", "checkpoints/")

    seen: set[Path] = set()
    for s in walk(manifest):
        if "/" not in s or s == "TBD" or s.startswith(("http://", "https://")):
            continue
        # Trailing-slash strings name *directories* that the pipeline
        # creates on first run; they aren't required to exist statically.
        if s.endswith("/"):
            continue
        # Only validate things that look like repo-relative paths to source-controlled files.
        if not s.startswith(("data/", "configs/", "paper/", "tests/", "radiant", "docs/")) \
                and not s.startswith(runtime_roots):
            continue
        p = REPO_ROOT / s
        if p in seen:
            continue
        seen.add(p)
        if p.name in runtime_outputs:
            continue
        # For runtime roots, missing leaves are fine — the phase just hasn't run.
        if s.startswith(runtime_roots):
            continue
        if p.parent.exists() and not p.exists():
            pytest.fail(f"manifest.yaml references {s} but it doesn't exist on disk")
