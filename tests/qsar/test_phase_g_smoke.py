"""Smoke tests for Phase G analyses.

Builds tiny synthetic predictions + descriptor tables and runs each
analysis end-to-end on them. These tests don't validate scientific
correctness -- they validate that the scripts import cleanly, accept
the canonical schemas, and emit the promised tables, figures, and
summary.md without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_predictions(
    n: int = 80, *, seed: int = 0, with_depth: bool = True
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "idx": np.arange(n),
        "inchikey14": [f"AAAAAAAAAAAAA{i:02d}" for i in range(n)],
        "target_chembl_id": "CHEMBL_TEST",
        "split_kind": "scaffold",
        "smiles": ["CCO"] * n,                # placeholder; not parsed by these tests
        "true_pchembl": rng.normal(6.0, 1.0, n),
    })
    df["pred_pchembl"] = df["true_pchembl"] + rng.normal(0, 0.5, n)
    if with_depth:
        # depth correlated with one descriptor (MolWt) so G1 has signal
        df["effective_depth"] = 2.0 + 0.02 * rng.normal(0, 1, n)
    return df


def _make_synthetic_descriptors(preds: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(preds)
    base = rng.normal(0, 1, n)
    return pd.DataFrame({
        "inchikey14": preds["inchikey14"],
        "MolWt": 300 + 50 * base + 10 * preds["effective_depth"],
        "NumRotatableBonds": np.clip(np.round(5 + 2 * base), 0, 20),
        "NumRings": np.clip(np.round(3 + base), 0, 8),
        "FractionCSP3": np.clip(0.4 + 0.1 * rng.normal(0, 1, n), 0, 1),
        "BertzCT": 600 + 100 * base + 30 * preds["effective_depth"],
        "SAscore_proxy": 2 + 0.5 * rng.normal(0, 1, n),
    })


@pytest.fixture
def workspace(tmp_path: Path):
    preds = _make_synthetic_predictions()
    desc = _make_synthetic_descriptors(preds)
    pred_path = tmp_path / "predictions.csv"
    preds.to_csv(pred_path, index=False)
    desc_path = tmp_path / "descriptors.csv"
    desc.to_csv(desc_path, index=False)
    return tmp_path, pred_path, desc_path, preds, desc


# ---------------------------------------------------------------------------
# G.1
# ---------------------------------------------------------------------------

def test_g1_runs(workspace):
    from radiant_qsar.analyses import g1_depth_vs_complexity

    tmp_path, pred_path, desc_path, *_ = workspace
    out = tmp_path / "phase_g"
    res = g1_depth_vs_complexity.run(
        predictions_path=pred_path,
        descriptors_path=desc_path,
        out_dir=out,
        n_bootstrap=50,
        n_cv_splits=3,
    )
    assert (res["paths"].tables / "g1_depth_descriptor_correlations.csv").exists()
    assert (res["paths"].figures / "g1_depth_vs_descriptors.png").exists()
    assert res["paths"].summary_md.exists()
    assert "verdict" in res


# ---------------------------------------------------------------------------
# G.4 (predictions mode)
# ---------------------------------------------------------------------------

def test_g4_predictions_mode(workspace):
    from radiant_qsar.analyses import g4_test_time_loop_sweep

    tmp_path, _, desc_path, preds, _ = workspace
    sweep_dir = tmp_path / "loop_sweep"
    sweep_dir.mkdir()
    for k in (1, 2, 4):
        out = preds.copy()
        out["pred_pchembl"] = out["pred_pchembl"] + (0.5 / k)
        out["effective_depth"] = float(k) * 0.8
        out.to_csv(sweep_dir / f"predictions_nloops{k}.csv", index=False)

    res = g4_test_time_loop_sweep.run(
        predictions_dir=sweep_dir,
        out_dir=tmp_path / "phase_g",
        loops=(1, 2, 4),
        descriptors_path=desc_path,
        n_bins=2,
    )
    assert not res["sweep_metrics"].empty
    assert (res["paths"].tables / "g4_sweep_metrics.csv").exists()


# ---------------------------------------------------------------------------
# G.3
# ---------------------------------------------------------------------------

def test_g3_runs(workspace):
    from radiant_qsar.analyses import g3_calibration

    tmp_path, _, _, preds, _ = workspace
    long_rows = []
    for model_name, sigma_scale in [("radiant_mc_loops", 0.5), ("ensemble_5", 0.6)]:
        d = preds.copy()
        d["sigma_pchembl"] = sigma_scale
        d["model"] = model_name
        long_rows.append(d)
    long_df = pd.concat(long_rows, ignore_index=True)
    long_path = tmp_path / "long.csv"
    long_df.to_csv(long_path, index=False)

    res = g3_calibration.run(predictions_csv=long_path, out_dir=tmp_path / "phase_g")
    assert not res["metrics"].empty
    assert (res["paths"].figures / "g3_reliability_diagram.png").exists()


# ---------------------------------------------------------------------------
# G.5 (predictions-only mode; no model required)
# ---------------------------------------------------------------------------

def test_g5_predictions_mode(workspace):
    rdkit = pytest.importorskip("rdkit")  # noqa: F841
    from radiant_qsar.analyses import g5_atom_attribution

    tmp_path, _, _, _, _ = workspace
    # Simple molecules whose per-atom halt arrays we synthesize directly.
    smis = ["CCO", "c1ccncc1", "CC(=O)Oc1ccccc1C(=O)O"]
    rows = []
    for i, smi in enumerate(smis):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        n_atoms = mol.GetNumAtoms()
        rng = np.random.default_rng(i)
        rows.append({
            "smiles": smi,
            "true_pchembl": 6.0 + i * 0.1,
            "pred_pchembl": 6.0 + i * 0.1 + 0.05,
            "per_atom_halt": json.dumps(rng.integers(1, 5, n_atoms).tolist()),
        })
    csv_path = tmp_path / "per_atom_halts.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    res = g5_atom_attribution.run(
        predictions_path=csv_path,
        out_dir=tmp_path / "phase_g",
        n_case_studies=2,
    )
    assert res["paths"].summary_md.exists()


# ---------------------------------------------------------------------------
# G pairwise wins + compute parity (panel layout)
# ---------------------------------------------------------------------------

def _make_panel(tmp_path: Path, preds: pd.DataFrame) -> Path:
    panel = tmp_path / "panel"
    for model in ("radiant_full", "morgan_rf", "chemberta"):
        d = panel / model / "CHEMBL_TEST" / "scaffold"
        d.mkdir(parents=True, exist_ok=True)
        p = preds.copy()
        rng = np.random.default_rng(hash(model) % 2**31)
        noise = rng.normal(0, {"radiant_full": 0.4, "morgan_rf": 0.7,
                               "chemberta": 0.55}[model], len(p))
        p["pred_pchembl"] = p["true_pchembl"] + noise
        p[[
            "idx", "inchikey14", "target_chembl_id", "split_kind", "smiles",
            "true_pchembl", "pred_pchembl",
        ]].to_csv(d / "predictions.csv", index=False)
        (panel / model / "compute.json").write_text(json.dumps({
            "model": model, "params": 1_000_000 * (1 + hash(model) % 30),
            "flops_per_forward": 5e8 * (1 + hash(model) % 5),
        }))
    return panel


def test_pairwise_wins_runs(workspace):
    from radiant_qsar.analyses import g_pairwise_wins

    tmp_path, _, desc_path, preds, _ = workspace
    panel = _make_panel(tmp_path, preds)
    res = g_pairwise_wins.run(
        panel_root=panel,
        out_dir=tmp_path / "phase_g",
        descriptors_path=desc_path,
        n_bootstrap=50,
        n_bins=2,
    )
    assert not res["pairs"].empty
    assert (res["paths"].tables / "pairwise_wins_per_split.csv").exists()


def test_compute_parity_runs(workspace):
    from radiant_qsar.analyses import g_compute_parity

    tmp_path, pred_path, desc_path, preds, _ = workspace
    panel = _make_panel(tmp_path, preds)
    res = g_compute_parity.run(
        panel_root=panel,
        out_dir=tmp_path / "phase_g",
        params_flops_csv=None,
        perm_predictions=pred_path,
        perm_descriptors=desc_path,
        n_bootstrap=50,
        n_perm=20,
    )
    assert not res["compute"].empty
    assert not res["bootstrap"].empty
    assert (res["paths"].tables / "radiant_vs_baselines.csv").exists()
