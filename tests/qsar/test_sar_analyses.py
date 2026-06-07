"""Smoke tests for SAR and failure-mode analyses."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("rdkit")


def _sar_predictions(split: str = "scaffold") -> pd.DataFrame:
    smiles = [
        "Cc1ccccc1",
        "CCc1ccccc1",
        "COc1ccccc1",
        "Clc1ccccc1",
        "CC(=O)Nc1ccccc1",
        "CC(=O)Oc1ccccc1",
        "c1ccccc1",
        "Fc1ccccc1",
    ]
    true = np.array([6.0, 6.7, 7.1, 5.8, 7.5, 6.4, 5.5, 6.2])
    pred = true + np.array([0.05, -0.10, 0.15, -0.05, -0.2, 0.08, 0.1, -0.12])
    return pd.DataFrame({
        "idx": np.arange(len(smiles)),
        "inchikey14": [f"TESTKEY{i:07d}" for i in range(len(smiles))],
        "target_chembl_id": "CHEMBL_TEST",
        "split_kind": split,
        "smiles": smiles,
        "true_pchembl": true,
        "pred_pchembl": pred,
    })


def _write_panel(tmp_path: Path, *, split: str = "scaffold", model: str = "radiant") -> Path:
    panel = tmp_path / "panel"
    out = panel / model / "CHEMBL_TEST" / split
    out.mkdir(parents=True)
    _sar_predictions(split=split).to_csv(out / "predictions.csv", index=False)
    return panel


def test_rgroup_sar_runs(tmp_path: Path):
    from radiant_qsar.analyses import g_rgroup_sar

    panel = _write_panel(tmp_path, split="scaffold")
    res = g_rgroup_sar.run(
        panel_root=panel,
        out_dir=tmp_path / "phase_g",
        model="radiant",
        split="scaffold",
        min_abs_true_delta=0.1,
        max_pairs_per_scaffold=50,
    )
    assert not res["summary"].empty
    assert (res["paths"].tables / "rgroup_sar_summary.csv").exists()
    assert res["paths"].summary_md.exists()


def test_activity_cliff_sar_runs(tmp_path: Path):
    from radiant_qsar.analyses import g_activity_cliff_sar

    panel = _write_panel(tmp_path, split="activity_cliff")
    res = g_activity_cliff_sar.run(
        panel_root=panel,
        out_dir=tmp_path / "phase_g",
        model="radiant",
        split="activity_cliff",
        tanimoto_threshold=0.15,
        activity_delta_threshold=0.2,
        max_pairs_per_cell=100,
    )
    assert not res["summary"].empty
    assert (res["paths"].tables / "activity_cliff_summary.csv").exists()
    assert res["paths"].summary_md.exists()


def test_failure_modes_runs(tmp_path: Path):
    from radiant_qsar.analyses import g_failure_modes

    panel = _write_panel(tmp_path, split="scaffold")
    res = g_failure_modes.run(
        panel_root=panel,
        out_dir=tmp_path / "phase_g",
        model="radiant",
        top_n=4,
        min_scaffold_n=2,
    )
    assert len(res["worst"]) == 4
    assert (res["paths"].tables / "failure_metrics_by_cell.csv").exists()
    assert res["paths"].summary_md.exists()


def test_stage1_representation_probe_runs(tmp_path: Path):
    from radiant_qsar.analyses import g_stage1_representation_probe

    rows = []
    for i in range(12):
        scaffold_id = i // 6
        rgroup_id = (i // 3) % 2
        rows.append({
            "smiles": "Cc1ccccc1" if scaffold_id == 0 else "CC(=O)Oc1ccccc1",
            "scaffold": f"scaffold_{scaffold_id}",
            "rgroup": f"rgroup_{rgroup_id}",
            "emb_0": float(scaffold_id * 3 + 0.05 * i),
            "emb_1": float(rgroup_id * 2 + 0.02 * i),
            "emb_2": float(0.1 * i),
        })
    emb = tmp_path / "stage1_embeddings.csv"
    pd.DataFrame(rows).to_csv(emb, index=False)

    res = g_stage1_representation_probe.run(
        embeddings_csv=emb,
        out_dir=tmp_path / "phase_g",
        k=2,
    )
    assert not res["metrics"].empty
    assert (res["paths"].tables / "stage1_probe_metrics.csv").exists()
    assert res["paths"].summary_md.exists()


def test_rgroup_ablation_runs(tmp_path: Path):
    from radiant_qsar.analyses import g_rgroup_ablation

    rows = []
    for model, mae, pearson in [
        ("radiant", 0.42, 0.78),
        ("radiant_no_stage1_rgroup", 0.55, 0.68),
        ("radiant_no_stage2_rgroup", 0.50, 0.70),
        ("radiant_no_rgroup", 0.60, 0.62),
    ]:
        rows.append({
            "model": model,
            "target": "CHEMBL_TEST",
            "split": "scaffold",
            "mae": mae,
            "pearson": pearson,
        })
    panel_results = tmp_path / "panel_results.csv"
    pd.DataFrame(rows).to_csv(panel_results, index=False)

    res = g_rgroup_ablation.run(
        panel_results=panel_results,
        out_dir=tmp_path / "phase_g",
        primary="radiant",
    )
    assert not res["summary"].empty
    assert (res["paths"].tables / "rgroup_ablation_summary.csv").exists()
    assert res["paths"].summary_md.exists()
