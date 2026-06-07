"""Pytest fixtures shared across the suite.

Every test that needs a model uses ``tiny_config`` so the suite stays under a
minute on CPU. The deterministic seed is applied automatically per test.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Make every test reproducible. Runs before each test."""
    import torch

    seed = 1337
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    yield


@pytest.fixture
def tiny_cfg():
    """Tiny but valid config; ~thousands of params, suitable for shape tests."""
    from radiant import tiny_config

    return tiny_config()


@pytest.fixture
def tiny_model(tiny_cfg):
    """Instantiated tiny model on CPU."""
    from radiant import RadiantModel

    return RadiantModel(tiny_cfg).eval()
