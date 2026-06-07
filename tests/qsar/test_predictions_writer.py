"""Tests for the canonical predictions.csv writer + per-baseline contract.

The contract: every baseline (`morgan_rf`, `chemberta`, `molformer`,
`gin`, `radiant` via single_task) writes a `predictions.csv` whose
leading columns are exactly :data:`PREDICTIONS_SCHEMA`. Phase G
analyses depend on this schema being identical across models so the
analyses can read every cell with the same parser.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from radiant_qsar.eval.predictions import (
    PREDICTIONS_SCHEMA,
    PREDICTIONS_FILENAME,
    write_predictions,
)


def test_canonical_schema_is_stable():
    assert PREDICTIONS_SCHEMA[:7] == (
        "idx", "inchikey14", "target_chembl_id",
        "split_kind", "smiles", "true_pchembl", "pred_pchembl",
    )


def test_writer_round_trip(tmp_path: Path):
    out = write_predictions(
        tmp_path,
        indices=[3, 5, 8],
        inchikeys=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"],
        smiles=["CCO", "c1ccccc1", "CC(=O)O"],
        true_pchembl=[5.5, 6.2, 7.0],
        pred_pchembl=[5.3, 6.4, 6.9],
        target_chembl_id="CHEMBL_TEST",
        split_kind="scaffold",
    )
    assert out.name == PREDICTIONS_FILENAME
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 3
    assert rows[0]["target_chembl_id"] == "CHEMBL_TEST"
    assert rows[0]["split_kind"] == "scaffold"
    assert rows[0]["inchikey14"] == "AAAAAAAAAAAAAA"
    assert float(rows[0]["true_pchembl"]) == pytest.approx(5.5)
    assert float(rows[0]["pred_pchembl"]) == pytest.approx(5.3)


def test_writer_extra_columns_appended(tmp_path: Path):
    out = write_predictions(
        tmp_path,
        indices=[0, 1],
        inchikeys=["AAAA", "BBBB"],
        smiles=["CCO", "c1ccccc1"],
        true_pchembl=[5.0, 6.0],
        pred_pchembl=[5.1, 6.1],
        target_chembl_id="T",
        split_kind="random",
        extra_columns={"halt_step": [4, 7], "uncertainty": [0.3, 0.5]},
    )
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert "halt_step" in rows[0] and "uncertainty" in rows[0]
    assert rows[0]["halt_step"] == "4"
    assert float(rows[1]["uncertainty"]) == pytest.approx(0.5)


def test_writer_length_mismatch_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_predictions(
            tmp_path,
            indices=[0, 1, 2],
            inchikeys=["A", "B"],          # length mismatch
            smiles=["CCO", "CCN", "CCC"],
            true_pchembl=[1.0, 2.0, 3.0],
            pred_pchembl=[1.0, 2.0, 3.0],
            target_chembl_id="T", split_kind="random",
        )


# ---------------------------------------------------------------------------
# Per-baseline contract: each baseline writes predictions.csv with the
# canonical schema. We exercise morgan_rf and gin end-to-end (cheap, CPU,
# no network). chemberta / molformer / radiant use the same shared
# helper so their schemas are guaranteed to match by construction.
# ---------------------------------------------------------------------------
def _toy_activities(tmp_path: Path):
    import pandas as pd
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1",
              "CN(C)CC", "O=C1CCC1", "CCBr", "NCCc1ccc(O)c(O)c1",
              "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ClCCl", "OCC(O)C(O)C(O)CO",
              "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "CCN(CC)CC",
              "CCCO", "CCCCO", "CCCCCO", "CC(C)O", "CC(C)(C)O"]
    n = len(smiles)
    df = pd.DataFrame({
        "target_chembl_id": ["TGT_X"] * n,
        "uniprot": ["UX"] * n,
        "target_name": ["t"] * n,
        "organism": ["Homo sapiens"] * n,
        "standard_type": ["IC50"] * n,
        "standard_smiles": smiles,
        "inchikey14": [f"K{i:013d}" for i in range(n)],
        "pchembl": [4.5 + i * 0.2 for i in range(n)],
        "pchembl_iqr": [0.0] * n,
        "n_replicates": [1] * n,
        "doc_year_min": [2018 + i % 5 for i in range(n)],
        "doc_year_max": [2018 + i % 5 for i in range(n)],
    })
    p = tmp_path / "activities.parquet"
    df.to_parquet(p, index=False)
    return p


def test_morgan_rf_emits_canonical_predictions_schema(tmp_path: Path):
    from radiant_qsar.baselines.morgan_rf import MorganRFConfig, train_morgan_rf

    activities = _toy_activities(tmp_path)
    out = tmp_path / "rf"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="TGT_X",
        out=out, split_kind="random", n_estimators=20, n_jobs=1,
    ))
    pcsv = out / PREDICTIONS_FILENAME
    assert pcsv.exists()
    rows = list(csv.DictReader(open(pcsv, encoding="utf-8")))
    assert rows, "predictions.csv must not be empty"
    for col in PREDICTIONS_SCHEMA:
        assert col in rows[0], f"morgan_rf predictions missing column {col!r}"
    # Every row carries the right target / split.
    assert all(r["target_chembl_id"] == "TGT_X" for r in rows)
    assert all(r["split_kind"] == "random" for r in rows)
    # inchikey14 must be present and 14 chars.
    assert all(len(r["inchikey14"]) == 14 for r in rows)


def test_gin_emits_canonical_predictions_schema(tmp_path: Path):
    from radiant_qsar.baselines.gin import GINConfig, train_gin

    activities = _toy_activities(tmp_path)
    out = tmp_path / "gin"
    train_gin(GINConfig(
        activities=activities, target_chembl_id="TGT_X",
        out=out, split_kind="random",
        n_layers=2, hidden_dim=32, epochs=3,
        batch_size=8, lr=1e-2, device="cpu",
    ))
    pcsv = out / PREDICTIONS_FILENAME
    assert pcsv.exists()
    rows = list(csv.DictReader(open(pcsv, encoding="utf-8")))
    assert rows
    for col in PREDICTIONS_SCHEMA:
        assert col in rows[0], f"gin predictions missing column {col!r}"
    assert all(r["target_chembl_id"] == "TGT_X" for r in rows)
