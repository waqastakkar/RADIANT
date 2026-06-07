"""Tests for the panel selector and the sweep result aggregator.

The sweep CLI itself shells out to other scripts, so those parts are
covered by integration only -- here we test the data-shaping helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from radiant_qsar.finetune.select_panel import (
    DEFAULT_QUOTAS,
    Panel,
    PanelEntry,
    select_panel,
)
from radiant_qsar.finetune.sweep import SweepConfig, aggregate_results


def _toy_curated(tmp_path: Path) -> tuple[Path, Path]:
    """Build minimal activities.parquet + targets.parquet fixtures."""
    import pandas as pd

    counts = {
        "T_K1": 1100, "T_K2": 2000, "T_G1": 1500, "T_G2": 1100,
        "T_P1": 1200, "T_N1": 1300, "T_O1": 1500, "T_TR1": 1100,
        "T_IC1": 1100, "T_TINY": 50,
    }
    uniprots = {
        "T_K1": "UK1", "T_K2": "UK2", "T_G1": "UG1", "T_G2": "UG2",
        "T_P1": "UP1", "T_N1": "UN1", "T_O1": "UO1", "T_TR1": "UTR1",
        "T_IC1": "UIC1", "T_TINY": "UTI",
    }
    target_names = {
        "T_K1": "JAK_X", "T_K2": "EGFR_X", "T_G1": "D2_X", "T_G2": "MOR_X",
        "T_P1": "FXa_X", "T_N1": "ER_X", "T_O1": "CYP_X", "T_TR1": "SERT_X",
        "T_IC1": "KIR_X", "T_TINY": "TINY",
    }
    target_ids, ups, names = [], [], []
    for tid, n in counts.items():
        target_ids.extend([tid] * n)
        ups.extend([uniprots[tid]] * n)
        names.extend([target_names[tid]] * n)
    n_total = len(target_ids)
    activities = pd.DataFrame({
        "target_chembl_id": target_ids,
        "uniprot": ups,
        "target_name": names,
        "organism": ["Homo sapiens"] * n_total,
        "inchikey14": [f"I{i:013d}" for i in range(n_total)],
    })
    targets = pd.DataFrame({
        "target_chembl_id": ["T_K1", "T_K2", "T_G1", "T_G2", "T_P1", "T_N1",
                              "T_O1", "T_TR1", "T_IC1", "T_TINY"],
        "uniprot":          ["UK1", "UK2", "UG1", "UG2", "UP1", "UN1",
                              "UO1", "UTR1", "UIC1", "UTI"],
        "target_name":      ["JAK_X", "EGFR_X", "D2_X", "MOR_X", "FXa_X", "ER_X",
                              "CYP_X", "SERT_X", "KIR_X", "TINY"],
        "organism":         ["Homo sapiens"] * 10,
        "target_class":     ["kinase", "kinase", "gpcr", "gpcr", "protease",
                              "nuclear_receptor", "other_enzyme", "transporter",
                              "ion_channel", "other_enzyme"],
        "n_compounds":      [1100, 2000, 1500, 1100, 1200, 1300, 1500, 1100, 1100, 50],
    })
    activities_path = tmp_path / "activities.parquet"
    targets_path = tmp_path / "targets.parquet"
    activities.to_parquet(activities_path, index=False)
    targets.to_parquet(targets_path, index=False)
    return activities_path, targets_path


def test_panel_default_quotas_partition_to_20():
    assert sum(DEFAULT_QUOTAS.values()) == 20


def test_panel_respects_quotas_and_min_compounds(tmp_path: Path):
    activities_path, targets_path = _toy_curated(tmp_path)
    panel = select_panel(
        activities_path=activities_path,
        targets_path=targets_path,
        per_class={"kinase": 2, "gpcr": 2, "protease": 1,
                   "nuclear_receptor": 1, "other_enzyme": 1,
                   "transporter": 1, "ion_channel": 1},
        min_compounds=1000,
        organism_filter="Homo sapiens",
    )
    assert len(panel.entries) == 9         # 2+2+1+1+1+1+1
    by_cls = panel.by_class()
    assert len(by_cls["kinase"]) == 2
    assert len(by_cls["gpcr"]) == 2
    # The 50-compound TINY target must not appear (below min).
    assert all(e.target_chembl_id != "T_TINY" for e in panel.entries)


def test_panel_warns_when_class_short(tmp_path: Path):
    activities_path, targets_path = _toy_curated(tmp_path)
    panel = select_panel(
        activities_path=activities_path,
        targets_path=targets_path,
        per_class={"kinase": 5},  # only 2 in the toy data
        min_compounds=1000,
    )
    assert len(panel.entries) == 2  # both kinases included; warning logged


def test_panel_to_dict_has_expected_keys(tmp_path: Path):
    activities_path, targets_path = _toy_curated(tmp_path)
    panel = select_panel(
        activities_path=activities_path,
        targets_path=targets_path,
        per_class={"kinase": 2},
        min_compounds=1000,
    )
    d = panel.to_dict()
    for k in ("quotas", "constraints", "n_targets", "by_class", "entries"):
        assert k in d
    assert d["n_targets"] == len(panel.entries)


def test_aggregate_results_collects_per_cell(tmp_path: Path):
    """Drop synthetic result.json files in the cell layout and check the CSV is built."""
    panel = {
        "entries": [
            {"target_chembl_id": "T1", "target_class": "kinase", "n_compounds": 1500,
             "target_name": "X", "uniprot": "U1"},
            {"target_chembl_id": "T2", "target_class": "gpcr",   "n_compounds": 1200,
             "target_name": "Y", "uniprot": "U2"},
        ],
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")

    out_dir = tmp_path / "runs"
    for model in ("radiant", "morgan_rf"):
        for tgt in ("T1", "T2"):
            for split in ("scaffold", "random"):
                cell = out_dir / model / tgt / split
                cell.mkdir(parents=True)
                (cell / "result.json").write_text(json.dumps({
                    "metrics": {
                        "test": {"mae": 0.5, "rmse": 0.7, "r2": 0.6, "pearson": 0.78, "spearman": 0.74},
                        "val":  {"mae": 0.55},
                        "best_epoch": 12,
                        "n_test": 100,
                    }
                }), encoding="utf-8")

    cfg = SweepConfig(
        panel_path=panel_path,
        activities_path=tmp_path / "activities.parquet",  # not actually read by aggregator
        vocab_path=tmp_path / "vocab.json",
        config_path=tmp_path / "config.json",
        pretrain_ckpt=None,
        out_dir=out_dir,
        splits=["scaffold", "random"],
        models=["radiant", "morgan_rf"],
    )
    csv_path = aggregate_results(cfg)
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    # 2 models x 2 targets x 2 splits = 8 data rows.
    assert text.count("\n") >= 8
    assert "test_mae" in text and "test_pearson" in text

    summary = json.loads((out_dir / "panel_summary.json").read_text(encoding="utf-8"))
    assert summary["n_rows"] == 8
    # Per (model, split): 2 cells averaged.
    for k, v in summary["by_model_split"].items():
        assert v["count"] == 2
        assert v["test_mae_mean"] == pytest.approx(0.5)
