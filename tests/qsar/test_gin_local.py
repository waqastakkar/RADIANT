"""GIN local correctness tests -- no network, no HF, no GPU.

The HF baselines need a real download to test end-to-end so they're
covered in a separate (optional, skip-if-network-missing) test. GIN is
fully self-contained so we exercise the data path here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("rdkit")
pytest.importorskip("pandas")
pytest.importorskip("scipy")


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1",
    "CN(C)CC", "O=C1CCC1", "CCBr", "NCCc1ccc(O)c(O)c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ClCCl", "OCC(O)C(O)C(O)CO",
    "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "CCN(CC)CC",
    "CCCO", "CCCCO", "CCCCCO", "CC(C)O", "CC(C)(C)O",
    "Nc1n[nH]c2ccccc12", "c1ccc(O)cc1", "Oc1ccc(N)cc1", "Cc1ccccc1",
    "CC(=O)Nc1ccccc1", "CN1CCN(CC1)c1ccccc1", "C1CCCCC1", "C1CCNCC1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)NCCO", "OCC(O)CO", "CC=O",
    "CCN", "CN", "CC#N", "C(=O)O", "CCOCC", "CC(C)CC", "C1CCCO1", "C1CCCCN1",
]
PCHEMBL = list(np.linspace(4.5, 8.5, len(SMILES)))


def _toy_activities(tmp_path: Path) -> Path:
    import pandas as pd

    df = pd.DataFrame({
        "target_chembl_id": ["GIN_TEST"] * len(SMILES),
        "uniprot": ["P0G"] * len(SMILES),
        "target_name": ["t"] * len(SMILES),
        "organism": ["Homo sapiens"] * len(SMILES),
        "standard_type": ["IC50"] * len(SMILES),
        "standard_smiles": SMILES,
        "inchikey14": [f"G{i:013d}" for i in range(len(SMILES))],
        "pchembl": PCHEMBL,
        "pchembl_iqr": [0.0] * len(SMILES),
        "n_replicates": [1] * len(SMILES),
        "doc_year_min": [2018] * len(SMILES),
        "doc_year_max": [2018] * len(SMILES),
    })
    p = tmp_path / "activities.parquet"
    df.to_parquet(p, index=False)
    return p


def test_smiles_to_graph_basic():
    from radiant_qsar.baselines.gin import _atom_feat_dim, _smiles_to_graph

    f, e = _smiles_to_graph("CCO")
    assert f.shape == (3, _atom_feat_dim())
    assert e.shape[0] == 2 and e.shape[1] >= 2  # at least one undirected bond pair


def test_smiles_to_graph_invalid_returns_none():
    from radiant_qsar.baselines.gin import _smiles_to_graph

    f, e = _smiles_to_graph("not_a_real_smiles_!!!")
    assert f is None and e is None


def test_gin_forward_runs_on_synthetic_batch():
    from radiant_qsar.baselines.gin import (
        GINRegressor,
        _atom_feat_dim,
        _collate_graphs,
        _smiles_to_graph,
    )

    graphs = []
    for i, s in enumerate(["CCO", "c1ccccc1", "CC(=O)O"]):
        f, e = _smiles_to_graph(s)
        graphs.append((f, e, float(i), int(i)))
    x, ei, b, y, idx = _collate_graphs(graphs)

    model = GINRegressor(_atom_feat_dim(), hidden_dim=32, n_layers=2, dropout=0.0)
    pred = model(x, ei, b)
    assert pred.shape == (3,)
    assert torch.isfinite(pred).all()


def test_gin_train_smoke(tmp_path: Path):
    """End-to-end CPU smoke: train ~5 epochs on 40 toy rows, persist all artefacts."""
    from radiant_qsar.baselines.gin import GINConfig, MODEL_FILENAME, train_gin

    activities = _toy_activities(tmp_path)
    out = tmp_path / "gin_run"
    res = train_gin(GINConfig(
        activities=activities,
        target_chembl_id="GIN_TEST",
        out=out,
        split_kind="random",
        n_layers=2, hidden_dim=32, epochs=5,
        batch_size=16, lr=1e-2, device="cpu",
    ))
    for partition in ("val", "test"):
        assert partition in res and "mae" in res[partition]
    assert (out / MODEL_FILENAME).exists()
    assert (out / "predictions.csv").exists()
    assert (out / "result.json").exists()


def test_gin_predict_round_trip(tmp_path: Path):
    """Train, then reload and score -- the screening pipeline path."""
    from radiant_qsar.baselines.gin import (
        GINConfig,
        MODEL_FILENAME,
        predict_smiles_from_ckpt,
        train_gin,
    )

    activities = _toy_activities(tmp_path)
    out = tmp_path / "gin_run"
    train_gin(GINConfig(
        activities=activities,
        target_chembl_id="GIN_TEST",
        out=out, split_kind="random",
        n_layers=2, hidden_dim=32, epochs=5,
        batch_size=16, lr=1e-2, device="cpu",
    ))
    preds = predict_smiles_from_ckpt(out / MODEL_FILENAME, ["CCO", "c1ccccc1", "garbage_!!!"])
    assert preds.shape == (3,)
    assert np.isfinite(preds[0]) and np.isfinite(preds[1])
    assert np.isnan(preds[2])
