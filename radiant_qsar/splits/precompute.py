"""Pre-compute every (target, split) cell of the panel before the sweep.

Running the panel sweep cold means every cell that needs a split kicks
off the split computation independently. With 5 models per (target,
split), the activity-cliff and cluster computations run up to 5× as
often as needed.

This script walks the panel manifest and forces a cache fill for every
``(target, split)`` pair, in series, with progress logging. After it
finishes, every cell in the sweep finds a warm cache and skips
recomputation entirely.

Usage::

    python -m radiant_qsar.splits.precompute \\
        --panel      data/processed/v1/panel.json \\
        --activities data/processed/v1/activities.parquet \\
        --splits     random scaffold time cluster activity_cliff \\
        --seed       1337
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from radiant_qsar.splits.cache import (
    DEFAULT_CACHE_DIR,
    SplitCacheConfig,
    load_or_compute_split,
    split_cache_path,
)


logger = logging.getLogger(__name__)


VALID_SPLITS = ("random", "scaffold", "time", "cluster", "activity_cliff")


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=list(VALID_SPLITS), choices=VALID_SPLITS)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help=f"override RADIANT_SPLITS_DIR (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--force", action="store_true",
                   help="recompute even if a cache file already exists")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if "entries" not in panel:
        raise SystemExit(f"{args.panel} is not a panel manifest (missing 'entries')")
    targets = [e["target_chembl_id"] for e in panel["entries"]]
    logger.info("precompute plan: %d targets x %d splits = %d cells",
                len(targets), len(args.splits), len(targets) * len(args.splits))

    df = pd.read_parquet(args.activities)
    cfg = SplitCacheConfig(cache_dir=args.cache_dir, seed=args.seed)

    t0 = time.time()
    n_done, n_skipped = 0, 0
    for tgt in targets:
        sub = df[df["target_chembl_id"] == tgt].reset_index(drop=True)
        if len(sub) == 0:
            logger.warning("  %s: no rows; skipping", tgt)
            continue
        for split in args.splits:
            path = split_cache_path(tgt, split, cfg)
            if path.exists() and not args.force:
                n_skipped += 1
                continue
            t_cell = time.time()
            train, val, test = load_or_compute_split(tgt, split, sub, cfg)
            n_done += 1
            logger.info(
                "  %-15s %-15s -> train=%d val=%d test=%d  (%.1fs)",
                tgt, split, len(train), len(val), len(test), time.time() - t_cell,
            )

    elapsed = time.time() - t0
    logger.info(
        "done: %d computed, %d already cached, %.1f minutes total",
        n_done, n_skipped, elapsed / 60.0,
    )


if __name__ == "__main__":
    _main()
