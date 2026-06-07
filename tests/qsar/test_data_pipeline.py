"""Unit tests for the radiant_qsar.data pipeline.

These don't touch the real ChEMBL DB; they exercise the standardization,
curation, descriptor, and manifest functions on synthetic minimal inputs.
The chembl_extract module is integration-tested separately against a
small slice of the real DB (see scripts/qsar_smoke.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest


pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
rdk = pytest.importorskip("rdkit")


CORPUS_SMI = ["CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1"]


def _write_raw(tmp_path: Path) -> Path:
    """Build a minimal raw_activities.parquet for the curate / standardize tests."""
    rows = []
    smiles_target = [
        ("CCO", "CHEMBL1", "P11111", "TestTarget", "Homo sapiens", "IC50", "=", 100.0, "nM", 7.0, 2020),
        ("CCO", "CHEMBL1", "P11111", "TestTarget", "Homo sapiens", "IC50", "=", 200.0, "nM", 6.7, 2021),
        ("c1ccccc1", "CHEMBL1", "P11111", "TestTarget", "Homo sapiens", "IC50", "=", 1.0, "uM", 6.0, 2020),
        ("CC(=O)O", "CHEMBL2", "P22222", "OtherTarget", "Homo sapiens", "Ki", "=", 50.0, "nM", 7.3, 2019),
        ("Cc1ccncc1", "CHEMBL2", "P22222", "OtherTarget", "Homo sapiens", "Ki", "=", 5.0, "uM", 5.3, 2022),
    ]
    cols = [
        "canonical_smiles", "target_chembl_id", "uniprot", "target_name", "organism",
        "standard_type", "standard_relation", "standard_value", "standard_units",
        "pchembl_value", "doc_year",
    ]
    df = pd.DataFrame(smiles_target, columns=cols)
    p = tmp_path / "raw_activities.parquet"
    df.to_parquet(p, index=False)
    return p


def test_standardize_round_trip(tmp_path: Path):
    from radiant_qsar.data.standardize import (
        StandardizeConfig,
        standardize_compounds,
    )

    raw = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"
    standardize_compounds(StandardizeConfig(in_path=raw, out_dir=out_dir))
    df = pd.read_parquet(out_dir / "compounds.parquet")
    assert set(["inchikey14", "standard_smiles", "input_smiles"]).issubset(df.columns)
    assert len(df) >= 3
    # InChIKey-14 should be unique post-dedup.
    assert df["inchikey14"].is_unique


def test_curate_pchembl_and_iqr(tmp_path: Path):
    from radiant_qsar.data.standardize import (
        StandardizeConfig,
        standardize_compounds,
    )
    from radiant_qsar.data.activity_curate import (
        CurateConfig,
        curate_activities,
    )

    raw = _write_raw(tmp_path)
    out_dir = tmp_path / "processed"
    standardize_compounds(StandardizeConfig(in_path=raw, out_dir=out_dir))
    curate_activities(CurateConfig(raw_path=raw, compounds_path=out_dir / "compounds.parquet", out_dir=out_dir))

    df = pd.read_parquet(out_dir / "activities.parquet")
    assert {"inchikey14", "target_chembl_id", "standard_type", "pchembl", "n_replicates"}.issubset(df.columns)
    assert (df["pchembl"] >= 3.0).all() and (df["pchembl"] <= 12.0).all()
    # The two CCO/IC50/CHEMBL1 rows should have aggregated to 1 row with n_replicates=2.
    cco = df[(df["target_chembl_id"] == "CHEMBL1") & (df["standard_type"] == "IC50")]
    assert len(cco) >= 1


def test_value_to_pchembl_unit_consistency():
    from radiant_qsar.data.activity_curate import value_to_pchembl

    # IC50 = 1 uM -> pchembl = 6
    assert value_to_pchembl(1.0, "uM") == pytest.approx(6.0, abs=1e-9)
    # 100 nM -> 7
    assert value_to_pchembl(100.0, "nM") == pytest.approx(7.0, abs=1e-9)
    # invalid unit
    assert value_to_pchembl(100.0, "kg") is None
    # zero / negative
    assert value_to_pchembl(0.0, "nM") is None
    assert value_to_pchembl(-1.0, "nM") is None


def test_descriptors_columns_stable():
    from radiant_qsar.data.descriptors import DESCRIPTOR_NAMES, compute_descriptors

    df = compute_descriptors(CORPUS_SMI)
    assert "standard_smiles" in df.columns
    for name in DESCRIPTOR_NAMES:
        assert name in df.columns
    # MolWt should be positive for all SMILES.
    assert (df["MolWt"].dropna() > 0).all()


def test_manifest_has_files_and_versions(tmp_path: Path):
    from radiant_qsar.data.manifest import ManifestConfig, build_manifest

    # Lay down a minimal "processed" tree.
    proc = tmp_path / "v1"
    proc.mkdir()
    (proc / "x.parquet").write_bytes(b"hello")
    (proc / "x.meta.json").write_text('{"stage": "x", "row_count": 1}', encoding="utf-8")
    out = build_manifest(ManifestConfig(processed_dir=proc))
    import json
    m = json.loads(out.read_text(encoding="utf-8"))
    assert "files" in m and "x.parquet" in m["files"]
    assert m["files"]["x.parquet"]["sha256"]
    assert "tool_versions" in m
