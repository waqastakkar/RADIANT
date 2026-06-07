"""Build a tokenizer vocabulary from the curated compound corpus.

Reads ``compounds.parquet`` from the processed-v1 release and produces a
JSON vocab file usable by :class:`radiant_chem.SmilesTokenizer` (or
:class:`radiant_chem.SelfiesTokenizer` when the ``selfies`` extra is
installed). The vocab is **deterministic given the input file** -- token
order is alphabetical within each frequency bucket, special tokens
always at the front.

Typical usage::

    python -m radiant_qsar.data.build_vocab \\
        --compounds D:/My-Work/RADIANT/data/processed/v1/compounds.parquet \\
        --out D:/My-Work/RADIANT/data/processed/v1/smiles_vocab.json \\
        --kind smiles --min-count 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def build_vocab(
    compounds_parquet: Path,
    out_path: Path,
    *,
    kind: str = "smiles",
    min_count: int = 1,
    smiles_column: str = "standard_smiles",
) -> dict[str, int]:
    import pandas as pd

    if kind not in {"smiles", "selfies"}:
        raise ValueError("kind must be 'smiles' or 'selfies'")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(compounds_parquet, columns=[smiles_column])
    smiles = df[smiles_column].dropna().astype(str).tolist()
    logger.info("read %d standard SMILES from %s", len(smiles), compounds_parquet)

    if kind == "smiles":
        from radiant_chem.tokenizer import SmilesTokenizer

        tok = SmilesTokenizer.from_corpus(smiles, min_count=min_count)
    else:
        from radiant_chem.tokenizer import SelfiesTokenizer

        # SelfiesTokenizer.build_vocab consumes SELFIES strings, not SMILES.
        # We convert SMILES -> SELFIES on the fly using the optional `selfies` package.
        try:
            import selfies as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("kind='selfies' requires `pip install selfies`") from exc

        sel = []
        n_failed = 0
        for s in smiles:
            try:
                sel.append(sf.encoder(s))
            except Exception:
                n_failed += 1
        logger.info("converted SMILES->SELFIES: %d ok, %d failed", len(sel), n_failed)
        tok = SelfiesTokenizer()
        tok.build_vocab(sel, min_count=min_count)

    tok.save(out_path)
    meta = {
        "stage": "build_vocab",
        "kind": kind,
        "vocab_size": tok.vocab_size,
        "min_count": min_count,
        "compounds_parquet": str(compounds_parquet),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d tokens)", out_path, tok.vocab_size)
    return tok.token_to_id


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--compounds", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--kind", choices=("smiles", "selfies"), default="smiles")
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--smiles-column", default="standard_smiles")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_vocab(
        args.compounds, args.out, kind=args.kind, min_count=args.min_count,
        smiles_column=args.smiles_column,
    )


if __name__ == "__main__":
    _main()
