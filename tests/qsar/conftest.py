"""Per-test isolation for the splits cache.

Without this fixture, baseline / sweep tests that exercise ``_split``
would write into the repo's real ``data/splits/v1/`` directory because
``SplitCacheConfig`` defaults to ``DEFAULT_CACHE_DIR`` from the cache
module. We monkey-patch that module attribute so every QSAR test gets
its own throwaway cache under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_splits_cache(tmp_path: Path, monkeypatch):
    """Redirect ``radiant_qsar.splits.cache.DEFAULT_CACHE_DIR`` to a tmp dir.

    ``SplitCacheConfig`` resolves its ``cache_dir`` via a default_factory
    that reads the module attribute at instantiation time, so a runtime
    monkey-patch propagates correctly into every newly created config.
    """
    from radiant_qsar.splits import cache

    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path / "splits_cache")
    yield
