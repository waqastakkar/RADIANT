"""Originality enforcement.

Scans the RADIANT source tree for strings borrowed from any reference work
the user explicitly forbade. Also enforces a basic modularity ceiling: no
single source file should be a god-file.
"""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

# These are the strings we are NOT allowed to use anywhere in our source tree.
# This list itself is the only place they appear in the repo.
FORBIDDEN_NAMES = (
    "Mythos",
    "OpenMythos",
    "open_mythos",
    "MythosConfig",
    "MythosTokenizer",
    "LTIInjection",
    "ACTHalting",
    "MoEFFN",
    "MLAttention",
    "max_loop_iters",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "mythos_1b",
    "mythos_3b",
    "mythos_10b",
)


def _scanned_paths() -> list[Path]:
    paths: list[Path] = []
    for sub in ("radiant", "radiant_chem", "training", "examples", "configs"):
        paths.extend((REPO_ROOT / sub).rglob("*.py"))
        paths.extend((REPO_ROOT / sub).rglob("*.json"))
    paths.extend((REPO_ROOT / "docs").rglob("*.md"))
    paths.extend((REPO_ROOT / "docs").rglob("*.txt"))
    if (REPO_ROOT / "README.md").exists():
        paths.append(REPO_ROOT / "README.md")
    return paths


@pytest.mark.parametrize("forbidden", FORBIDDEN_NAMES)
def test_forbidden_names_absent(forbidden):
    offenders = []
    for path in _scanned_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if forbidden in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"{forbidden!r} found in: {offenders}"


def test_modularity_no_god_files():
    """No core source file should exceed 450 lines -- forces splitting."""
    big = []
    for path in (REPO_ROOT / "radiant").rglob("*.py"):
        n = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        if n > 450:
            big.append((str(path.relative_to(REPO_ROOT)), n))
    assert not big, f"Modules exceeding 450 lines: {big}"


def test_required_modules_present():
    """The architectural concepts the user requested must each have a home file."""
    expected = [
        "radiant/state_anchor.py",
        "radiant/iteration_signal.py",
        "radiant/iteration_adapter.py",
        "radiant/confidence_halting.py",
        "radiant/refinement_core.py",
        "radiant/stem_encoder.py",
        "radiant/exit_decoder.py",
    ]
    missing = [p for p in expected if not (REPO_ROOT / p).exists()]
    assert not missing, f"Missing required modules: {missing}"
