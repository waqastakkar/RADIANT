"""Tests for the pretraining collator + combined objective."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("rdkit")

from radiant import tiny_config
from radiant_chem import RadiantChemConfig, RadiantChemModel, SmilesTokenizer
from radiant_qsar.pretrain.collator import MLMContrastiveCollator
from radiant_qsar.pretrain.activity_pretrain import (
    ActivityDataset,
    RADIANTActivityModel,
    _collate as activity_collate,
    murcko_rgroup_smiles,
)
from radiant_qsar.pretrain.objective import combined_pretrain_loss


CORPUS = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1", "CCBr"]


def _setup():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    base = tiny_config(vocab_size=tok.vocab_size, pad_token_id=tok.pad_id, max_seq_len=32)
    cfg = RadiantChemConfig(base=base)
    model = RadiantChemModel(cfg)
    return tok, model


def test_collator_produces_expected_keys_paired():
    tok, _ = _setup()
    coll = MLMContrastiveCollator(tokenizer=tok, max_len=32, mlm_mask_prob=0.5)
    batch = [(s, s) for s in CORPUS[:3]]
    out = coll(batch)
    for k in ("mlm_input_ids", "mlm_labels", "mlm_attention_mask",
              "view_b_input_ids", "view_b_attention_mask"):
        assert k in out
    B, L = out["mlm_input_ids"].shape
    assert B == 3
    assert out["mlm_labels"].shape == (B, L)
    # Some labels must be != -100 (mask happened).
    assert (out["mlm_labels"] != -100).any()


def test_collator_produces_scaffold_and_rgroup_views():
    tok, _ = _setup()
    coll = MLMContrastiveCollator(tokenizer=tok, max_len=32, mlm_mask_prob=0.5)
    batch = [("CCc1ccccc1", "c1ccccc1CC", "c1ccccc1", "CC") for _ in range(2)]
    out = coll(batch)
    for k in (
        "scaffold_input_ids", "scaffold_attention_mask",
        "rgroup_input_ids", "rgroup_attention_mask",
        "rgroup_mlm_input_ids", "rgroup_mlm_labels",
    ):
        assert k in out


def test_collator_unpaired():
    tok, _ = _setup()
    coll = MLMContrastiveCollator(tokenizer=tok, max_len=32)
    out = coll(CORPUS[:2])
    assert "view_b_input_ids" not in out


def test_combined_pretrain_loss_runs_and_grads_flow():
    tok, model = _setup()
    coll = MLMContrastiveCollator(tokenizer=tok, max_len=32, mlm_mask_prob=0.5)
    batch = coll([(s, s, s, s) for s in CORPUS])
    model.train()
    loss, metrics = combined_pretrain_loss(
        model, batch, n_loops=2, mlm_weight=1.0, contrastive_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert "loss_mlm" in metrics and "loss_contrastive" in metrics
    assert "loss_scaffold_contrastive" in metrics
    assert "loss_rgroup_contrastive" in metrics
    assert "loss_rgroup_mlm" in metrics
    loss.backward()
    g = model.core.stem.token_embed.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_pretrain_loss_without_contrastive():
    tok, model = _setup()
    coll = MLMContrastiveCollator(tokenizer=tok, max_len=32, mlm_mask_prob=0.5)
    out = coll(CORPUS)  # unpaired
    loss, metrics = combined_pretrain_loss(model, out, n_loops=2, contrastive_weight=0.0)
    assert torch.isfinite(loss)
    assert "loss_contrastive" not in metrics


def test_corpus_dataset_pairs(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"standard_smiles": CORPUS})
    p = tmp_path / "compounds.parquet"
    df.to_parquet(p, index=False)
    from radiant_qsar.pretrain.corpus import CompoundCorpusDataset

    ds = CompoundCorpusDataset(parquet_path=p, return_augmented_pair=True)
    a, b, scaffold, rgroup = ds[0]
    assert isinstance(a, str) and isinstance(b, str)
    assert isinstance(scaffold, str) and isinstance(rgroup, str)
    assert len(ds) == len(CORPUS)


def test_zinc_corpus_build_parallel_resume(tmp_path):
    from radiant_qsar.pretrain.zinc_corpus import build_pretrain_corpus

    zinc_dir = tmp_path / "zinc20"
    zinc_dir.mkdir()
    (zinc_dir / "part_a.smi").write_text("smiles\nCCO ZINC1\nCCC ZINC2\n", encoding="utf-8")
    (zinc_dir / "part_b.smi").write_text("c1ccccc1\tZINC3\nCCBr\tZINC4\n", encoding="utf-8")
    out = tmp_path / "corpus.txt"
    state = tmp_path / "state"

    first = build_pretrain_corpus(
        zinc_dir=zinc_dir,
        out_path=out,
        deduplicate=False,
        jobs=2,
        resume=True,
        state_dir=state,
    )
    second = build_pretrain_corpus(
        zinc_dir=zinc_dir,
        out_path=out,
        deduplicate=False,
        jobs=2,
        resume=True,
        state_dir=state,
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["CCO", "CCC", "c1ccccc1", "CCBr"]
    assert first["zinc_files"] == second["zinc_files"] == 2
    assert first["zinc_molecules"] == second["zinc_molecules"] == 4
    assert (state / "manifest.json").exists()


def test_activity_pretrain_rgroup_auxiliary_path_runs():
    tok, chem = _setup()
    rgroups = [murcko_rgroup_smiles("Cc1ccccc1"), murcko_rgroup_smiles("CCOc1ccccc1")]
    assert any(rgroups)
    ds = ActivityDataset(
        ["Cc1ccccc1", "CCOc1ccccc1"],
        [0, 0],
        [6.0, 7.0],
        tok,
        max_len=32,
        rgroup_smiles=rgroups,
    )
    batch = activity_collate([ds[0], ds[1]], tok.pad_id)
    model = RADIANTActivityModel(chem, n_targets=1, d_model=chem.cfg.base.d_model)
    out = model(
        batch["input_ids"],
        batch["target_idx"],
        attention_mask=batch["attention_mask"],
        rgroup_input_ids=batch["rgroup_input_ids"],
        rgroup_attention_mask=batch["rgroup_attention_mask"],
        n_loops=2,
        return_aux=True,
    )
    assert set(out) == {"pred", "rgroup_pred"}
    assert out["pred"].shape == out["rgroup_pred"].shape == (2,)
