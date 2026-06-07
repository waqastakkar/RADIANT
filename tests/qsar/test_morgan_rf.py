"""End-to-end smoke test for the Morgan/RF baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pd = pytest.importorskip("pandas")


def _toy_activities(tmp_path: Path) -> Path:
    rows = []
    smiles_pool = ["CCO", "CCC", "CCCO", "CCCC", "OCCO", "Cc1ccccc1", "Nc1ccccc1",
                   "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1", "CN(C)CC", "O=C1CCC1",
                   "CCBr", "CC(C)O", "CC(C)Cc1ccccc1", "ClCCl", "OCC(O)C(O)CO",
                   "CC(=O)Nc1ccc(O)cc1", "NCCc1ccc(O)c(O)c1", "CCN(CC)CC"]
    for i, smi in enumerate(smiles_pool):
        rows.append({
            "inchikey14": f"FAKEKEY{i:08d}",
            "standard_smiles": smi,
            "target_chembl_id": "CHEMBL_TEST",
            "uniprot": "P00000",
            "target_name": "Test target",
            "organism": "Homo sapiens",
            "standard_type": "IC50",
            "pchembl": 4.0 + (i % 7) * 0.5,
            "pchembl_iqr": 0.0,
            "n_replicates": 1,
            "doc_year_min": 2018 + (i % 5),
            "doc_year_max": 2018 + (i % 5),
        })
    df = pd.DataFrame(rows)
    p = tmp_path / "activities.parquet"
    df.to_parquet(p, index=False)
    return p


def test_morgan_rf_runs_and_produces_metrics(tmp_path: Path):
    from radiant_qsar.baselines.morgan_rf import (
        MorganRFConfig,
        train_morgan_rf,
    )

    activities = _toy_activities(tmp_path)
    out = tmp_path / "out"
    result = train_morgan_rf(
        MorganRFConfig(
            activities=activities,
            target_chembl_id="CHEMBL_TEST",
            out=out,
            split_kind="random",
            n_estimators=20,
            seed=0,
        )
    )
    # Canonical schema present (the sweep aggregator reads this).
    for partition in ("train", "val", "test"):
        assert partition in result
        assert {"mae", "rmse", "pearson", "spearman", "n"}.issubset(result[partition].keys())
    assert result["test"]["n"] >= 1
    assert (out / "result.json").exists()


def test_morgan_rf_persists_model_and_predictions(tmp_path: Path):
    """The model file (`model.joblib`) and per-test predictions (`predictions.csv`)
    must be written so the screening pipeline can re-use the trained baseline."""
    from radiant_qsar.baselines.morgan_rf import (
        MODEL_FILENAME,
        MorganRFConfig,
        train_morgan_rf,
    )

    activities = _toy_activities(tmp_path)
    out = tmp_path / "out"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="CHEMBL_TEST",
        out=out, split_kind="random", n_estimators=20, n_jobs=1, seed=0,
    ))
    assert (out / MODEL_FILENAME).exists()
    assert (out / MODEL_FILENAME).stat().st_size > 1000
    assert (out / "predictions.csv").exists()
    text = (out / "predictions.csv").read_text(encoding="utf-8")
    # Header is the canonical schema shared by every baseline.
    from radiant_qsar.eval.predictions import PREDICTIONS_SCHEMA
    assert text.startswith(",".join(PREDICTIONS_SCHEMA))
    # At least one data row.
    assert text.count("\n") >= 2


def test_morgan_rf_round_trip_inference(tmp_path: Path):
    """Train, then reload model.joblib and score fresh SMILES.

    This is exactly what the screening pipeline does via MorganRFPotency."""
    import numpy as np
    from radiant_qsar.baselines.morgan_rf import (
        MorganRFConfig,
        load_bundle,
        predict_smiles_from_ckpt,
        train_morgan_rf,
    )

    activities = _toy_activities(tmp_path)
    out = tmp_path / "out"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="CHEMBL_TEST",
        out=out, split_kind="random", n_estimators=20, n_jobs=1, seed=0,
    ))

    bundle = load_bundle(out / "model.joblib")
    assert {"model", "fp_radius", "fp_n_bits", "target_chembl_id"}.issubset(bundle.keys())
    assert bundle["target_chembl_id"] == "CHEMBL_TEST"

    preds = predict_smiles_from_ckpt(out / "model.joblib", ["CCO", "c1ccccc1"])
    assert preds.shape == (2,)
    assert np.isfinite(preds).all()


def test_morgan_rf_predicts_nan_on_unparseable(tmp_path: Path):
    import numpy as np
    from radiant_qsar.baselines.morgan_rf import (
        MorganRFConfig,
        predict_smiles_from_ckpt,
        train_morgan_rf,
    )

    activities = _toy_activities(tmp_path)
    out = tmp_path / "out"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="CHEMBL_TEST",
        out=out, split_kind="random", n_estimators=20, n_jobs=1, seed=0,
    ))
    preds = predict_smiles_from_ckpt(
        out / "model.joblib",
        ["CCO", "definitely_not_a_smiles_!!!"],
    )
    assert np.isfinite(preds[0])
    assert np.isnan(preds[1])


def test_morgan_rf_potency_filter(tmp_path: Path):
    """The screening filter wires up cleanly with a saved RF baseline."""
    import numpy as np
    from rdkit import Chem
    from radiant_qsar.baselines.morgan_rf import MorganRFConfig, train_morgan_rf
    from radiant_qsar.screening.base import FilterContext
    from radiant_qsar.screening.filters.ml_scoring import MorganRFPotency

    activities = _toy_activities(tmp_path)
    out = tmp_path / "out"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="CHEMBL_TEST",
        out=out, split_kind="random", n_estimators=20, n_jobs=1, seed=0,
    ))

    f = MorganRFPotency(out / "model.joblib", min_pchembl=5.0)
    ctx = FilterContext(smiles="CCO", mol_id="x")
    r = f(Chem.MolFromSmiles("CCO"), ctx)
    assert r.score is not None and np.isfinite(r.score)
    assert isinstance(r.passed, bool)


def test_canonical_schema_picked_up_by_aggregator(tmp_path: Path):
    """End-to-end smoke: train an RF cell and run the sweep aggregator on it.

    Regression test: previously the aggregator looked for ``val``/``test`` keys
    while the writer emitted ``val_metrics``/``test_metrics``, silently
    producing empty metric columns in panel_results.csv."""
    import json
    from radiant_qsar.baselines.morgan_rf import MorganRFConfig, train_morgan_rf
    from radiant_qsar.finetune.sweep import SweepConfig, aggregate_results

    activities = _toy_activities(tmp_path)
    runs = tmp_path / "runs"
    cell = runs / "morgan_rf" / "CHEMBL_TEST" / "random"
    train_morgan_rf(MorganRFConfig(
        activities=activities, target_chembl_id="CHEMBL_TEST",
        out=cell, split_kind="random", n_estimators=20, n_jobs=1, seed=0,
    ))

    panel = {"entries": [{
        "target_chembl_id": "CHEMBL_TEST", "target_class": "kinase",
        "n_compounds": 20, "target_name": "test", "uniprot": "P0",
    }]}
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")

    csv_path = aggregate_results(SweepConfig(
        panel_path=panel_path,
        activities_path=activities,
        vocab_path=tmp_path / "v.json",
        config_path=tmp_path / "c.json",
        pretrain_ckpt=None,
        out_dir=runs,
        splits=["random"],
        models=["morgan_rf"],
    ))
    text = csv_path.read_text(encoding="utf-8")
    # CSV has a non-empty test_mae cell.
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = lines[0].split(",")
    data = lines[1].split(",")
    assert "test_mae" in header
    assert data[header.index("test_mae")].strip() != ""
