import random
from pathlib import Path

import torch

from radiant_chem import ChemblCsvDataset, MaskedSmilesDataset, SmilesTokenizer
from radiant_chem.dataset import make_mlm_batch


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = REPO_ROOT / "examples" / "data" / "sample_chembl.csv"


def _tok():
    rows = SAMPLE_CSV.read_text(encoding="utf-8").splitlines()[1:]
    smiles = [r.split(",")[0] for r in rows]
    return SmilesTokenizer.from_corpus(smiles)


def test_csv_dataset_loads_rows_and_targets():
    tok = _tok()
    ds = ChemblCsvDataset(
        SAMPLE_CSV,
        tok,
        target_columns=["logP_demo", "active"],
        max_len=128,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert "input_ids" in sample
    assert sample["targets"].shape == (2,)


def test_csv_collate_pads_to_max_in_batch():
    tok = _tok()
    ds = ChemblCsvDataset(SAMPLE_CSV, tok, target_columns=["logP_demo"], max_len=64)
    batch = ds.collate([ds[0], ds[1], ds[2]])
    B, L = batch["input_ids"].shape
    assert B == 3
    assert batch["attention_mask"].shape == (B, L)
    # Padded positions have id == pad_id and mask == 0.
    pad = batch["input_ids"] == tok.pad_id
    assert (batch["attention_mask"][pad] == 0).all()


def test_csv_dataset_skips_invalid_rows(tmp_path):
    tok = _tok()
    bad = tmp_path / "bad.csv"
    bad.write_text("smiles,target\nCCO,not_a_number\nCC,1.5\n", encoding="utf-8")
    ds = ChemblCsvDataset(bad, tok, target_columns=["target"], max_len=32)
    assert len(ds) == 1
    assert ds.skipped == 1


def test_make_mlm_batch_masks_only_eligible_tokens():
    tok = _tok()
    rng = random.Random(42)
    text = ["CCO", "c1ccccc1"]
    ids, attn = tok.encode_batch(text)
    masked, positions, labels = make_mlm_batch(
        ids, attn, tokenizer=tok, mask_prob=1.0, replace_random_prob=0, keep_orig_prob=0, rng=rng,
    )
    # Where labels != -100, that's a masked-but-with-original-id stored as label.
    for b in range(ids.size(0)):
        for j in range(ids.size(1)):
            if labels[b, j].item() != -100:
                # Should be a non-special, attended-to position.
                assert attn[b, j].item() == 1
                assert int(ids[b, j].item()) not in (tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id)


def test_mlm_dataset_yields_consistent_shapes():
    tok = _tok()
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    ds = MaskedSmilesDataset(smiles, tok, max_len=32, seed=0)
    batch = ds.collate([ds[i] for i in range(3)])
    for k in ("input_ids", "labels", "mask_positions", "attention_mask"):
        assert batch[k].shape[0] == 3
    # labels are -100 wherever input is not masked
    assert (batch["labels"] == -100).any()
