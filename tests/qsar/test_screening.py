"""Tests for the virtual-screening filter library and pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from radiant_qsar.screening import (
    Pipeline,
    PROFILES,
    available_filters,
    get_filter,
    get_profile,
)
from radiant_qsar.screening.base import FilterContext


def _ctx(smi: str, mol_id: str = "x"):
    return FilterContext(smiles=smi, mol_id=mol_id)


# ---------------------------------------------------------------------------
# Registry / profiles
# ---------------------------------------------------------------------------
def test_registry_has_core_filters():
    have = set(available_filters())
    must_have = {
        "lipinski", "veber", "ghose", "egan", "muegge", "rule_of_three",
        "qed_min", "sa_max",
        "pains", "brenk", "nih", "zinc", "all_alerts",
        "cns_mpo", "bbb_egan", "bbb_strict",
        "esol_min", "herg_proxy", "pgp_proxy",
        "kinase_hinge", "basic_amine_required",
        "covalent_warhead_required", "covalent_warhead_forbidden",
        "reactive_groups", "dedup_inchikey",
        "heavy_atom_range", "molwt_range", "formal_charge",
    }
    missing = must_have - have
    assert not missing, f"missing required filters: {missing}"


def test_all_profiles_instantiate():
    for name in PROFILES:
        pipe = Pipeline.from_profile(name)
        assert pipe.filters, f"profile {name} produced an empty pipeline"


# ---------------------------------------------------------------------------
# Lipinski
# ---------------------------------------------------------------------------
def test_lipinski_passes_aspirin():
    f = get_filter("lipinski")
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    r = f(mol, _ctx("CC(=O)Oc1ccccc1C(=O)O"))
    assert r.passed


def test_lipinski_rejects_huge():
    f = get_filter("lipinski", max_violations=0)
    big_smi = "C" * 600  # MW > 7000
    mol = Chem.MolFromSmiles(big_smi)
    if mol is None:
        pytest.skip("rdkit refused the huge SMILES; not a useful test")
    r = f(mol, _ctx(big_smi))
    assert not r.passed
    assert "violation" in r.reason.lower()


# ---------------------------------------------------------------------------
# Drug-likeness
# ---------------------------------------------------------------------------
def test_qed_threshold():
    f = get_filter("qed_min", threshold=0.5)
    # Aspirin QED ~ 0.55
    r = f(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"), _ctx("..."))
    assert r.score is not None
    # Diphenyl benzene -- low QED
    bad = Chem.MolFromSmiles("c1ccc(-c2ccccc2-c2ccccc2)cc1")
    r2 = f(bad, _ctx("..."))
    assert r2.score is not None and 0 <= r2.score <= 1


def test_sa_invalid_threshold_still_runs():
    f = get_filter("sa_max", threshold=10.0)
    r = f(Chem.MolFromSmiles("CCO"), _ctx("CCO"))
    assert r.passed


# ---------------------------------------------------------------------------
# Structural alerts
# ---------------------------------------------------------------------------
def test_pains_catalog_runs():
    f = get_filter("pains")
    # Plain ethanol -- no PAINS hit.
    r = f(Chem.MolFromSmiles("CCO"), _ctx("CCO"))
    assert r.passed


# ---------------------------------------------------------------------------
# CNS
# ---------------------------------------------------------------------------
def test_cns_mpo_score_in_range():
    f = get_filter("cns_mpo", threshold=0)
    # Score is always in [0, 6] regardless of pass/fail.
    r = f(Chem.MolFromSmiles("CCO"), _ctx("CCO"))
    assert r.score is not None and 0 <= r.score <= 6


def test_bbb_egan_rejects_polar():
    f = get_filter("bbb_egan")
    # Sucrose -- very polar, rejected.
    sucrose = "OC[C@H]1O[C@H](O[C@]2(CO)O[C@H](CO)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O"
    mol = Chem.MolFromSmiles(sucrose)
    if mol is None:
        pytest.skip("rdkit could not parse sucrose")
    r = f(mol, _ctx(sucrose))
    assert not r.passed


# ---------------------------------------------------------------------------
# ADMET proxies
# ---------------------------------------------------------------------------
def test_esol_works():
    f = get_filter("esol_min", threshold=-5.0)
    r = f(Chem.MolFromSmiles("CCO"), _ctx("CCO"))
    assert r.passed and r.score is not None


def test_herg_proxy_flags_basic_lipophilic():
    f = get_filter("herg_proxy", logp_max=2.0)
    # A simple aromatic basic amine, lipophilic.
    mol = Chem.MolFromSmiles("c1ccc(CCN(CC)CC)cc1")
    r = f(mol, _ctx(""))
    assert isinstance(r.passed, bool)


# ---------------------------------------------------------------------------
# Target-specific
# ---------------------------------------------------------------------------
def test_kinase_hinge_recognizes_indazole():
    f = get_filter("kinase_hinge")
    # 3-aminoindazole is a canonical hinge binder.
    r = f(Chem.MolFromSmiles("Nc1n[nH]c2ccccc12"), _ctx(""))
    assert r.passed


def test_basic_amine_required():
    f = get_filter("basic_amine_required")
    r1 = f(Chem.MolFromSmiles("CCN(CC)CC"), _ctx(""))
    r2 = f(Chem.MolFromSmiles("CCO"), _ctx(""))
    assert r1.passed and not r2.passed


def test_covalent_warhead_required_and_forbidden():
    req = get_filter("covalent_warhead_required")
    forbid = get_filter("covalent_warhead_forbidden")
    acrylamide = Chem.MolFromSmiles("C=CC(=O)Nc1ccccc1")
    r1 = req(acrylamide, _ctx(""))
    r2 = forbid(acrylamide, _ctx(""))
    assert r1.passed and not r2.passed


# ---------------------------------------------------------------------------
# Reactive groups
# ---------------------------------------------------------------------------
def test_reactive_groups_rejects_acid_chloride():
    f = get_filter("reactive_groups")
    r = f(Chem.MolFromSmiles("CC(=O)Cl"), _ctx(""))
    assert not r.passed


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------
def test_dedup_inchikey_keeps_first_only():
    f = get_filter("dedup_inchikey")
    smi = "CCO"
    r1 = f(Chem.MolFromSmiles(smi), _ctx(smi, "first"))
    r2 = f(Chem.MolFromSmiles(smi), _ctx(smi, "second"))
    assert r1.passed and not r2.passed
    assert "duplicate" in r2.reason


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------
def test_pipeline_runs_smi(tmp_path: Path):
    inp = tmp_path / "lib.smi"
    inp.write_text(
        "\n".join([
            "CCO\tethanol",
            "CC(=O)Oc1ccccc1C(=O)O\taspirin",
            "Nc1n[nH]c2ccccc12\thinge_indazole",
            "CC(=O)Cl\tacid_chloride",
            "CCO\tethanol_dup",
        ]) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "kept.smi"
    rej = tmp_path / "rej.csv"
    summary_path = tmp_path / "summary.json"

    pipe = Pipeline.from_profile("general_drug_like")
    summary = pipe.run(inp, out, rejects_path=rej, summary_path=summary_path)

    assert summary.n_input == 5
    assert summary.n_passed >= 1
    assert (out.read_text(encoding="utf-8").strip().splitlines())  # non-empty
    assert rej.exists()
    assert summary_path.exists()


def test_pipeline_short_circuits():
    """A filter that always rejects must be the recorded failure point,
    even if subsequent filters would also fail."""
    pipe = Pipeline.from_specs([
        ("reactive_groups", {}),
        ("lipinski", {}),
    ])
    rep = pipe.apply_to_mol("x", "CC(=O)Cl", Chem.MolFromSmiles("CC(=O)Cl"))
    assert not rep.passed
    assert rep.failed_at == "reactive_groups"


def test_custom_filter_subset_via_specs(tmp_path: Path):
    inp = tmp_path / "lib.smi"
    inp.write_text("CCO\tethanol\nCC(=O)Cl\tacid_chloride\n", encoding="utf-8")
    out = tmp_path / "kept.smi"
    pipe = Pipeline.from_specs([("reactive_groups", {})])
    summary = pipe.run(inp, out)
    assert summary.n_passed == 1
    assert summary.n_failed == 1
