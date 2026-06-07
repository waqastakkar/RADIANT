"""ZINC20 streaming corpus for large-scale SMILES pretraining.

ZINC20 provides ~1.4B purchasable, drug-like molecules. Downloading
and loading the full set at once is impractical (~50GB uncompressed),
so this module provides:

* :func:`download_zinc20_tranches` -- fetches ZINC20 2D SMILES files
  from the ZINC20 website, selecting tranches by LogP and MW ranges.
* :class:`ZincStreamingDataset` -- memory-mapped streaming dataset
  that reads lines from gzipped or plain SMILES files without loading
  the full corpus into RAM.
* :func:`build_zinc_pretrain_corpus` -- pipeline that downloads,
  deduplicates (optional), and prepares a single SMILES-per-line
  corpus file for Stage 1 pretraining.

Usage
-----
::

    # Step 1: Download ZINC20 tranches (drug-like subset, ~200M mols)
    python -m radiant_qsar.pretrain.zinc_corpus download \\
        --out data/zinc20 \\
        --logp-range -1 5 \\
        --mw-range 200 500 \\
        --max-files 500

    # Step 2: Build merged corpus
    python -m radiant_qsar.pretrain.zinc_corpus build \\
        --zinc-dir data/zinc20 \\
        --chembl data/processed/v1/compounds.parquet \\
        --out data/zinc20/corpus.txt \\
        --deduplicate

The resulting ``smiles.txt`` can be used directly with
:class:`pretrain_loop.py` by swapping ``CompoundCorpusDataset``
for :class:`ZincStreamingDataset``.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, Dataset

logger = logging.getLogger(__name__)


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8", errors="replace")


def _extract_smiles(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if "\t" in stripped:
        smi = stripped.split("\t")[0]
    elif " " in stripped:
        smi = stripped.split()[0]
    else:
        smi = stripped
    if smi.lower() in {"smiles", "smile", "canonical_smiles"}:
        return ""
    return smi


def _discover_zinc_files(zinc_dir: Path) -> list[Path]:
    zinc_files = (
        sorted(zinc_dir.glob("*.gz"))
        + sorted(zinc_dir.glob("*.txt"))
        + sorted(zinc_dir.glob("*.smi"))
        + sorted(zinc_dir.glob("*.smiles"))
        + sorted(zinc_dir.glob("*.csv"))
    )
    excluded = {"catalog.txt", "README.txt", "corpus.txt"}
    return [f for f in zinc_files if f.name not in excluded and f.is_file()]


def _safe_chunk_name(index: int, fpath: Path) -> str:
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in fpath.name)
    return f"{index:05d}_{stem}.clean.smi"


def _process_zinc_file_chunk(task: tuple[str, str, int, bool]) -> dict:
    fpath_s, state_dir_s, index, resume = task
    fpath = Path(fpath_s)
    state_dir = Path(state_dir_s)
    chunk = state_dir / "chunks" / _safe_chunk_name(index, fpath)
    done = state_dir / "done" / f"{_safe_chunk_name(index, fpath)}.json"
    if resume and chunk.exists() and done.exists():
        return json.loads(done.read_text(encoding="utf-8"))

    chunk.parent.mkdir(parents=True, exist_ok=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=chunk.name + ".", suffix=".tmp", dir=chunk.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    count = 0
    try:
        with _open_text(fpath) as src, open(tmp_path, "w", encoding="utf-8") as out_f:
            for line in src:
                smi = _extract_smiles(line)
                if not smi:
                    continue
                out_f.write(smi + "\n")
                count += 1
        tmp_path.replace(chunk)
        record = {
            "file": str(fpath),
            "chunk": str(chunk),
            "count": count,
            "index": index,
        }
        done.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


class ZincStreamingDataset(IterableDataset):
    """Streams SMILES from a text file (one SMILES per line).

    For very large corpora (>100M lines) this is preferable to loading
    everything into RAM.  Each worker gets a disjoint slice of the file.
    Supports both plain ``.txt`` and ``.txt.gz`` files.

    Parameters
    ----------
    path : Path
        SMILES corpus file (one per line, no header).
    shuffle_buffer : int
        Buffer size for reservoir-style shuffling.  Larger = better
        randomisation but more RAM.  Set to 0 to disable.
    """

    def __init__(
        self,
        path: Path,
        shuffle_buffer: int = 100_000,
        return_augmented_pair: bool = True,
        return_chemistry_views: bool = True,
    ) -> None:
        self.path = Path(path)
        self.shuffle_buffer = shuffle_buffer
        self.return_augmented_pair = return_augmented_pair
        self.return_chemistry_views = return_chemistry_views
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        open_fn = gzip.open if str(self.path).endswith(".gz") else open
        buffer: list[str] = []

        with open_fn(self.path, "rt", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f):
                # Distribute lines across workers
                if line_no % num_workers != worker_id:
                    continue
                smi = line.strip()
                if not smi or smi.startswith("#"):
                    continue

                if self.shuffle_buffer > 0:
                    buffer.append(smi)
                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        for s in buffer:
                            yield self._output(s)
                        buffer.clear()
                else:
                    yield self._output(smi)

        # Flush remaining buffer
        if buffer:
            random.shuffle(buffer)
            for s in buffer:
                yield self._output(s)

    def _output(self, smi: str):
        if not self.return_augmented_pair and not self.return_chemistry_views:
            return smi
        views = [smi]
        if self.return_augmented_pair:
            from radiant_chem.augment import randomize_smiles

            views.append(randomize_smiles(smi))
        if self.return_chemistry_views:
            from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles
            from radiant_qsar.pretrain.corpus import _murcko_scaffold_smiles

            views.append(_murcko_scaffold_smiles(smi))
            views.append(murcko_rgroup_smiles(smi) or smi)
        return tuple(views)


class ZincInMemoryDataset(Dataset):
    """Loads all SMILES into RAM. Suitable for <50M molecules.

    Use this when you can afford the memory (~4GB for 50M SMILES)
    and want random access for standard DataLoader shuffling.
    """

    def __init__(
        self,
        path: Path,
        return_augmented_pair: bool = True,
        return_chemistry_views: bool = True,
        max_molecules: int | None = None,
    ) -> None:
        self.return_augmented_pair = return_augmented_pair
        self.return_chemistry_views = return_chemistry_views
        self.smiles: list[str] = []

        logger.info("loading ZINC corpus from %s...", path)
        open_fn = gzip.open if str(path).endswith(".gz") else open
        with open_fn(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                smi = line.strip()
                if not smi or smi.startswith("#"):
                    continue
                self.smiles.append(smi)
                if max_molecules is not None and len(self.smiles) >= max_molecules:
                    break
        logger.info("loaded %d SMILES", len(self.smiles))

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if not self.return_augmented_pair and not self.return_chemistry_views:
            return smi
        views = [smi]
        if self.return_augmented_pair:
            from radiant_chem.augment import randomize_smiles

            views.append(randomize_smiles(smi))
        if self.return_chemistry_views:
            from radiant_qsar.pretrain.activity_pretrain import murcko_rgroup_smiles
            from radiant_qsar.pretrain.corpus import _murcko_scaffold_smiles

            views.append(_murcko_scaffold_smiles(smi))
            views.append(murcko_rgroup_smiles(smi) or smi)
        return tuple(views)


# ---------------------------------------------------------------------------
# ZINC20 download utilities
# ---------------------------------------------------------------------------

def download_zinc20_tranches(
    out_dir: Path,
    *,
    logp_range: tuple[float, float] = (-1.0, 5.0),
    mw_range: tuple[float, float] = (200.0, 500.0),
    max_files: int | None = None,
) -> list[Path]:
    """Download ZINC20 2D SMILES tranches for drug-like molecules.

    Downloads from https://zinc20.docking.org/tranches/download.
    Each tranche is a tab-separated gzipped file with columns:
    smiles, zinc_id, ...

    Parameters
    ----------
    out_dir : Path
        Directory to save downloaded files.
    logp_range : tuple
        LogP filter range for selecting tranches.
    mw_range : tuple
        Molecular weight filter range.
    max_files : int or None
        Cap on number of tranche files to download.

    Returns
    -------
    list[Path]
        Paths to downloaded files.
    """
    import urllib.request

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ZINC20 tranche URL pattern (2D SMILES, standard representation)
    base_url = "https://zinc20.docking.org/tranches/download"

    # First, get the tranche list
    catalog_url = (
        f"{base_url}?"
        f"logp_min={logp_range[0]}&logp_max={logp_range[1]}"
        f"&mwt_min={mw_range[0]}&mwt_max={mw_range[1]}"
        f"&representation=2D&format=txt"
    )

    logger.info("fetching ZINC20 tranche catalog: %s", catalog_url)
    logger.info(
        "filters: LogP=[%.1f, %.1f], MW=[%.0f, %.0f]",
        *logp_range, *mw_range,
    )

    # Save catalog for reproducibility
    catalog_path = out_dir / "catalog.txt"
    if not catalog_path.exists():
        urllib.request.urlretrieve(catalog_url, catalog_path)
        logger.info("saved tranche catalog to %s", catalog_path)

    # Parse tranche URLs from catalog
    urls = []
    with open(catalog_path, "r") as f:
        for line in f:
            url = line.strip()
            if url and url.startswith("http"):
                urls.append(url)

    if max_files is not None:
        urls = urls[:max_files]
    logger.info("will download %d tranche files", len(urls))

    downloaded = []
    for i, url in enumerate(urls):
        fname = url.split("/")[-1]
        fpath = out_dir / fname
        if fpath.exists():
            downloaded.append(fpath)
            continue
        try:
            urllib.request.urlretrieve(url, fpath)
            downloaded.append(fpath)
            if (i + 1) % 50 == 0:
                logger.info("  downloaded %d/%d", i + 1, len(urls))
        except Exception as e:
            logger.warning("  failed to download %s: %s", url, e)

    logger.info("downloaded %d/%d tranche files to %s",
                len(downloaded), len(urls), out_dir)
    return downloaded


def build_pretrain_corpus(
    zinc_dir: Path | None = None,
    chembl_path: Path | None = None,
    out_path: Path = Path("data/zinc20/corpus.txt"),
    *,
    deduplicate: bool = True,
    max_zinc_molecules: int | None = None,
    jobs: int = 1,
    resume: bool = False,
    state_dir: Path | None = None,
) -> dict:
    """Build a merged SMILES corpus from ZINC20 + ChEMBL.

    Writes one canonical SMILES per line to ``out_path``.

    Returns
    -------
    dict with counts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if deduplicate:
        logger.warning(
            "deduplication requires keeping all SMILES in a set in RAM. "
            "For 870M+ molecules this needs ~60GB+ RAM. "
            "Pass --no-deduplicate if memory is limited."
        )
    if deduplicate and jobs > 1:
        logger.warning("parallel build disables itself when --deduplicate is used; using jobs=1")
        jobs = 1
    if max_zinc_molecules is not None and jobs > 1:
        logger.warning("parallel build disables itself when --max-zinc is used; using jobs=1 for exact cap")
        jobs = 1

    all_smiles: set[str] = set() if deduplicate else None
    count = 0
    chembl_count = 0
    zinc_count = 0

    if jobs > 1 and not deduplicate:
        state_dir = Path(state_dir) if state_dir is not None else out_path.parent / f".{out_path.name}.state"
        chunks_dir = state_dir / "chunks"
        state_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        manifest: dict = {
            "out_path": str(out_path),
            "resume": resume,
            "jobs": jobs,
            "deduplicated": False,
            "chembl_path": str(chembl_path) if chembl_path else None,
            "zinc_dir": str(zinc_dir) if zinc_dir else None,
        }

        chembl_chunk = chunks_dir / "00000_chembl.clean.smi"
        if chembl_path is not None:
            done = state_dir / "done" / "00000_chembl.clean.smi.json"
            if resume and chembl_chunk.exists() and done.exists():
                chembl_count = int(json.loads(done.read_text(encoding="utf-8"))["count"])
            else:
                import pandas as pd

                logger.info("loading ChEMBL compounds from %s", chembl_path)
                df = pd.read_parquet(chembl_path, columns=["standard_smiles"])
                chembl_chunk.parent.mkdir(parents=True, exist_ok=True)
                with open(chembl_chunk, "w", encoding="utf-8") as out_f:
                    for smi in df["standard_smiles"].dropna():
                        smi = str(smi).strip()
                        if not smi:
                            continue
                        out_f.write(smi + "\n")
                        chembl_count += 1
                done.parent.mkdir(parents=True, exist_ok=True)
                done.write_text(json.dumps({"file": str(chembl_path), "chunk": str(chembl_chunk), "count": chembl_count, "index": 0}, indent=2), encoding="utf-8")

        records: list[dict] = []
        zinc_files: list[Path] = []
        if zinc_dir is not None:
            zinc_dir = Path(zinc_dir)
            zinc_files = _discover_zinc_files(zinc_dir)
            logger.info("processing %d ZINC files from %s with %d workers", len(zinc_files), zinc_dir, jobs)
            tasks = [(str(f), str(state_dir), i + 1, resume) for i, f in enumerate(zinc_files)]
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                futures = {ex.submit(_process_zinc_file_chunk, task): task for task in tasks}
                for n, fut in enumerate(as_completed(futures), start=1):
                    rec = fut.result()
                    records.append(rec)
                    if n % 10 == 0 or n == len(futures):
                        logger.info("  processed %d/%d ZINC files", n, len(futures))
            records.sort(key=lambda r: int(r["index"]))
            zinc_count = int(sum(int(r["count"]) for r in records))

        tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_out, "w", encoding="utf-8") as out_f:
            if chembl_path is not None and chembl_chunk.exists():
                with open(chembl_chunk, "r", encoding="utf-8", errors="replace") as src:
                    shutil.copyfileobj(src, out_f)
            for rec in records:
                with open(rec["chunk"], "r", encoding="utf-8", errors="replace") as src:
                    shutil.copyfileobj(src, out_f)
        tmp_out.replace(out_path)
        count = chembl_count + zinc_count
        manifest.update({
            "total_molecules": count,
            "chembl_molecules": chembl_count,
            "zinc_molecules": zinc_count,
            "zinc_files": len(zinc_files),
            "chunks": [str(chembl_chunk)] if chembl_path is not None else [],
        })
        manifest["chunks"].extend(str(r["chunk"]) for r in records)
        (state_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        result = {
            "total_molecules": count,
            "chembl_molecules": chembl_count,
            "zinc_molecules": zinc_count,
            "zinc_files": len(zinc_files),
            "deduplicated": False,
            "resumable": True,
            "jobs": jobs,
            "state_dir": str(state_dir),
            "output_path": str(out_path),
        }
        logger.info("corpus built: %s", result)
        return result

    with open(out_path, "w", encoding="utf-8") as out_f:
        # 1. ChEMBL compounds (always included)
        if chembl_path is not None:
            import pandas as pd
            logger.info("loading ChEMBL compounds from %s", chembl_path)
            df = pd.read_parquet(chembl_path, columns=["standard_smiles"])
            for smi in df["standard_smiles"].dropna():
                smi = smi.strip()
                if not smi:
                    continue
                if deduplicate:
                    if smi in all_smiles:
                        continue
                    all_smiles.add(smi)
                out_f.write(smi + "\n")
                count += 1
            logger.info("added %d ChEMBL molecules", count)
            chembl_count = count

        # 2. ZINC20 tranches
        if zinc_dir is not None:
            zinc_dir = Path(zinc_dir)
            zinc_files = _discover_zinc_files(zinc_dir)
            logger.info("processing %d ZINC files from %s", len(zinc_files), zinc_dir)

            for fpath in zinc_files:
                with _open_text(fpath) as f:
                    for line in f:
                        smi = _extract_smiles(line)
                        if not smi:
                            continue
                        if deduplicate:
                            if smi in all_smiles:
                                continue
                            all_smiles.add(smi)
                        out_f.write(smi + "\n")
                        count += 1
                        zinc_count += 1
                        if max_zinc_molecules and zinc_count >= max_zinc_molecules:
                            break
                if max_zinc_molecules and zinc_count >= max_zinc_molecules:
                    break
            logger.info("added %d ZINC molecules (total: %d)", zinc_count, count)

    result = {
        "total_molecules": count,
        "chembl_molecules": chembl_count if chembl_path else 0,
        "zinc_molecules": zinc_count,
        "zinc_files": len(zinc_files) if zinc_dir is not None else 0,
        "deduplicated": deduplicate,
        "resumable": False,
        "jobs": jobs,
        "output_path": str(out_path),
    }
    logger.info("corpus built: %s", result)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="ZINC20 corpus tools")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download ZINC20 tranches")
    dl.add_argument("--out", required=True, type=Path)
    dl.add_argument("--logp-range", nargs=2, type=float, default=[-1.0, 5.0])
    dl.add_argument("--mw-range", nargs=2, type=float, default=[200.0, 500.0])
    dl.add_argument("--max-files", type=int, default=None)

    # build
    bd = sub.add_parser("build", help="Build merged pretrain corpus")
    bd.add_argument("--zinc-dir", type=Path, default=None)
    bd.add_argument("--chembl", type=Path, default=None)
    bd.add_argument("--out", required=True, type=Path)
    bd.add_argument("--deduplicate", action="store_true")
    bd.add_argument("--max-zinc", type=int, default=None)
    bd.add_argument("--jobs", type=int, default=1,
                    help="Parallel ZINC file workers for non-deduplicated builds")
    bd.add_argument("--resume", action="store_true",
                    help="Reuse completed per-file chunks under --state-dir")
    bd.add_argument("--state-dir", type=Path, default=None,
                    help="Directory for resumable per-file chunks and manifest")

    # vocab — rebuild tokenizer vocab from corpus
    vp = sub.add_parser("vocab", help="Rebuild SMILES tokenizer vocab from corpus")
    vp.add_argument("--corpus", required=True, type=Path,
                    help="SMILES corpus (one per line, e.g. output of 'build')")
    vp.add_argument("--out", required=True, type=Path,
                    help="Output vocab JSON path")
    vp.add_argument("--sample", type=int, default=5_000_000,
                    help="Sample N lines for vocab building (default 5M; "
                         "atom-level vocab saturates well under 1M)")
    vp.add_argument("--old-vocab", type=Path, default=None,
                    help="Previous vocab JSON to merge with (ensures all old "
                         "tokens keep their IDs for checkpoint compatibility)")

    args = parser.parse_args()

    if args.command == "download":
        download_zinc20_tranches(
            args.out,
            logp_range=tuple(args.logp_range),
            mw_range=tuple(args.mw_range),
            max_files=args.max_files,
        )
    elif args.command == "build":
        build_pretrain_corpus(
            zinc_dir=args.zinc_dir,
            chembl_path=args.chembl,
            out_path=args.out,
            deduplicate=args.deduplicate,
            max_zinc_molecules=args.max_zinc,
            jobs=args.jobs,
            resume=args.resume,
            state_dir=args.state_dir,
        )
    elif args.command == "vocab":
        _build_vocab(args)


def _build_vocab(args):
    """Build a SMILES tokenizer vocab from a corpus file.

    Reads up to ``--sample`` lines, tokenizes each SMILES, and builds
    a vocab containing every observed atom/bracket/bond token.

    If ``--old-vocab`` is provided, all existing token→ID mappings are
    preserved (for checkpoint compatibility), and new tokens are
    appended with fresh IDs.
    """
    from radiant_chem.tokenizer import SmilesTokenizer, smiles_tokenize
    import json

    corpus_path = Path(args.corpus)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect SMILES sample
    logger.info("reading up to %d lines from %s for vocab...", args.sample, corpus_path)
    sample_smiles = []
    open_fn = gzip.open if str(corpus_path).endswith(".gz") else open
    with open_fn(corpus_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            smi = line.strip()
            if smi and not smi.startswith("#"):
                sample_smiles.append(smi)
            if len(sample_smiles) >= args.sample:
                break
    logger.info("sampled %d SMILES", len(sample_smiles))

    if args.old_vocab is not None:
        # Extend existing vocab — preserve all old IDs
        old_tok = SmilesTokenizer.load(args.old_vocab)
        old_size = old_tok.vocab_size
        logger.info("loaded old vocab: %d tokens from %s", old_size, args.old_vocab)

        # Find new tokens not in old vocab
        from collections import Counter
        new_counts: Counter = Counter()
        for smi in sample_smiles:
            for tok in smiles_tokenize(smi):
                if tok not in old_tok.token_to_id:
                    new_counts[tok] += 1

        # Add new tokens with fresh IDs (after last old ID)
        next_id = max(old_tok.token_to_id.values()) + 1
        merged = dict(old_tok.token_to_id)
        for tok in sorted(new_counts.keys()):
            merged[tok] = next_id
            next_id += 1

        logger.info("found %d new tokens in ZINC (e.g. %s)",
                     len(new_counts),
                     list(new_counts.most_common(10)))
        logger.info("merged vocab: %d → %d tokens", old_size, len(merged))

        new_tok = SmilesTokenizer(vocab=merged)
    else:
        # Build fresh vocab
        new_tok = SmilesTokenizer.from_corpus(sample_smiles)
        logger.info("built fresh vocab: %d tokens", new_tok.vocab_size)

    new_tok.save(out_path)
    logger.info("saved vocab to %s (%d tokens)", out_path, new_tok.vocab_size)

    # Report token coverage on the sample
    unk_count = 0
    total_tokens = 0
    for smi in sample_smiles[:100_000]:
        ids = new_tok.encode(smi, add_bos=False, add_eos=False)
        total_tokens += len(ids)
        unk_count += sum(1 for i in ids if i == new_tok.unk_id)
    unk_rate = unk_count / max(total_tokens, 1)
    logger.info("UNK rate on 100K sample: %.4f%% (%d/%d tokens)",
                unk_rate * 100, unk_count, total_tokens)


if __name__ == "__main__":
    _main()
