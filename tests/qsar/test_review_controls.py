import json
from pathlib import Path

import pandas as pd

from radiant_qsar.analyses.build_calibration_input import build_calibration_input
from radiant_qsar.analyses.statistical_significance import run as run_stats


def _prediction_cell(root: Path, model: str, target: str, split: str, preds: list[float]) -> None:
    cell = root / model / target / split
    cell.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "idx": [0, 1],
        "inchikey14": ["A", "B"],
        "target_chembl_id": [target, target],
        "split_kind": [split, split],
        "smiles": ["CC", "CCC"],
        "true_pchembl": [6.0, 7.0],
        "pred_pchembl": preds,
        "confidence_var": [0.04, 0.09],
    }).to_csv(cell / "predictions.csv", index=False)


def test_build_calibration_input_from_halt_variance(tmp_path: Path):
    panel_root = tmp_path / "panel"
    _prediction_cell(panel_root, "radiant", "CHEMBL1", "scaffold", [6.1, 6.8])

    out = tmp_path / "calibration_long.csv"
    df = build_calibration_input(panel_root, out, split="scaffold", include_mc_loops=False)

    assert out.exists()
    assert set(df["model"]) == {"radiant_halt_var"}
    assert df["sigma_pchembl"].round(2).tolist() == [0.2, 0.3]


def test_statistical_significance_from_panel_results(tmp_path: Path):
    panel = tmp_path / "panel_results.csv"
    pd.DataFrame([
        {"model": "radiant", "target_chembl_id": "T1", "split": "scaffold", "test_mae": 0.2, "test_pearson": 0.8},
        {"model": "morgan_rf", "target_chembl_id": "T1", "split": "scaffold", "test_mae": 0.4, "test_pearson": 0.6},
        {"model": "radiant", "target_chembl_id": "T2", "split": "scaffold", "test_mae": 0.3, "test_pearson": 0.7},
        {"model": "morgan_rf", "target_chembl_id": "T2", "split": "scaffold", "test_mae": 0.5, "test_pearson": 0.5},
    ]).to_csv(panel, index=False)

    out = run_stats(panel, tmp_path / "stats", primary="radiant", metrics=["test_mae", "test_pearson"])

    assert set(out["metric"]) == {"test_mae", "test_pearson"}
    assert (tmp_path / "stats" / "STATISTICAL_SIGNIFICANCE.md").exists()
    assert out["wins"].sum() == 4
