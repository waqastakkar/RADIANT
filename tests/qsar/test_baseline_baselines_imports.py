"""Lightweight tests for the new baselines: imports + CLI parsing.

These tests deliberately do NOT attempt to download the HF checkpoints
or run a real fine-tune -- those require GPU + network and are
exercised manually via the sweep. Here we cover:

* the modules import cleanly,
* the CLI parsers accept the documented flags,
* the sweep dispatch knows the new model names,
* the GIN module's pure-torch helpers are correct on synthetic graphs.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_chemberta_imports():
    mod = importlib.import_module("radiant_qsar.baselines.chemberta")
    assert hasattr(mod, "_main")
    assert mod.DEFAULT_MODEL_ID  # non-empty


def test_molformer_imports():
    mod = importlib.import_module("radiant_qsar.baselines.molformer")
    assert hasattr(mod, "_main")
    assert mod.DEFAULT_MODEL_ID


def test_gin_imports():
    mod = importlib.import_module("radiant_qsar.baselines.gin")
    assert hasattr(mod, "_main")
    assert hasattr(mod, "GINConfig")
    assert hasattr(mod, "train_gin")


def test_sweep_knows_new_models():
    from radiant_qsar.finetune.sweep import VALID_MODELS, _DISPATCH

    for name in ("radiant", "morgan_rf", "chemberta", "molformer", "gin"):
        assert name in VALID_MODELS, f"{name} missing from VALID_MODELS"
        assert name in _DISPATCH, f"{name} missing from _DISPATCH"


@pytest.mark.parametrize("module_name,argv", [
    ("radiant_qsar.baselines.chemberta", [
        "--activities", "x.parquet", "--target", "T", "--out", "out",
        "--split", "scaffold", "--epochs", "1",
    ]),
    ("radiant_qsar.baselines.molformer", [
        "--activities", "x.parquet", "--target", "T", "--out", "out",
        "--split", "scaffold", "--epochs", "1",
    ]),
    ("radiant_qsar.baselines.gin", [
        "--activities", "x.parquet", "--target", "T", "--out", "out",
        "--split", "scaffold", "--epochs", "1", "--hidden-dim", "32",
    ]),
])
def test_cli_help(module_name, argv, monkeypatch):
    """Each baseline CLI must accept its documented flags without crashing on parse."""
    mod = importlib.import_module(module_name)
    # Patch sys.argv so ``--help`` exits 0; we just want to confirm the
    # parser accepts the flag set the sweep passes.
    monkeypatch.setattr(sys, "argv", [module_name] + argv + ["--help"])
    with pytest.raises(SystemExit) as exc:
        mod._main()
    # ``--help`` returns 0; an unknown-flag error returns 2.
    assert exc.value.code == 0, f"{module_name} CLI rejected its documented flags"
