"""Tests for the vocabulary-builder."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")


CORPUS = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1", "[NH4+]"]


def _make_compounds_parquet(tmp_path: Path) -> Path:
    import pandas as pd

    df = pd.DataFrame({"standard_smiles": CORPUS})
    p = tmp_path / "compounds.parquet"
    df.to_parquet(p, index=False)
    return p


def test_build_smiles_vocab(tmp_path: Path):
    from radiant_qsar.data.build_vocab import build_vocab

    cp = _make_compounds_parquet(tmp_path)
    out = tmp_path / "vocab.json"
    vocab = build_vocab(cp, out, kind="smiles", min_count=1)
    assert "[PAD]" in vocab and "[BOS]" in vocab and "[MASK]" in vocab
    # Should contain a sensible number of atom-level tokens.
    assert len(vocab) >= 8


def test_build_smiles_vocab_min_count_filters(tmp_path: Path):
    from radiant_qsar.data.build_vocab import build_vocab

    cp = _make_compounds_parquet(tmp_path)
    out_lo = tmp_path / "lo.json"
    out_hi = tmp_path / "hi.json"
    v_lo = build_vocab(cp, out_lo, kind="smiles", min_count=1)
    v_hi = build_vocab(cp, out_hi, kind="smiles", min_count=2)
    # Higher min_count must produce a smaller-or-equal vocab.
    assert len(v_hi) <= len(v_lo)


def test_build_selfies_vocab(tmp_path: Path):
    pytest.importorskip("selfies")
    from radiant_qsar.data.build_vocab import build_vocab

    cp = _make_compounds_parquet(tmp_path)
    out = tmp_path / "vocab.json"
    vocab = build_vocab(cp, out, kind="selfies", min_count=1)
    # SELFIES tokens are bracketed; we should still have all the special tokens.
    assert "[PAD]" in vocab
    assert any(k.startswith("[") and k not in ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]") for k in vocab)
