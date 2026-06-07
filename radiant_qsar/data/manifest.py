"""Manifest builder for the curated dataset.

Combines the per-stage `*.meta.json` files into a single
``manifest.json`` describing the whole processed-v1 release. The
manifest is what goes onto Zenodo with the Parquet files for
publication-grade reproducibility.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "rdkit", "pandas", "numpy", "pyarrow", "chembl_structure_pipeline"):
        try:
            module = __import__(mod)
            v = getattr(module, "__version__", None)
            if v is None and mod == "rdkit":
                from rdkit import __version__ as v  # type: ignore
            out[mod] = str(v)
        except Exception:
            out[mod] = "not-installed"
    return out


@dataclass
class ManifestConfig:
    processed_dir: Path
    out_path: Path | None = None      # default: processed_dir/manifest.json

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        if self.out_path is None:
            self.out_path = self.processed_dir / "manifest.json"
        else:
            self.out_path = Path(self.out_path)


def build_manifest(cfg: ManifestConfig) -> Path:
    if not cfg.processed_dir.exists():
        raise FileNotFoundError(cfg.processed_dir)

    files = {}
    for p in sorted(cfg.processed_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(cfg.processed_dir).as_posix()
        files[rel] = {
            "size_bytes": p.stat().st_size,
            "sha256": _hash_file(p),
        }

    stage_meta = {}
    for meta_path in sorted(cfg.processed_dir.rglob("*.meta.json")):
        try:
            stage_meta[meta_path.stem] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            stage_meta[meta_path.stem] = {"error": str(exc)}

    manifest = {
        "build_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "release": "radiant-qsar/data/processed/v1",
        "files": files,
        "stages": stage_meta,
        "tool_versions": _versions(),
    }
    cfg.out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("manifest written: %s (%d files)", cfg.out_path, len(files))
    return cfg.out_path


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--processed", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_manifest(ManifestConfig(processed_dir=args.processed, out_path=args.out))


if __name__ == "__main__":
    _main()
