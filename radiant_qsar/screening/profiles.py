"""Pre-defined filter combinations per target class.

Profiles compose the registered filters into a typical screening recipe
for a given target type. Each profile returns an ordered list of
``(filter_name, kwargs)`` pairs; the pipeline instantiates each filter in
order and runs molecules through them, short-circuiting on the first
failure.

Built-in profiles
-----------------
:py:data:`PROFILES`:

* ``general_drug_like``       -- broad oral-drug HTS triage
* ``cns_brain_penetrant``     -- CNS / brain-penetrant compounds
* ``cns_strict``              -- stringent CNS profile
* ``fragment_screening``      -- fragment-based screening (Rule of 3)
* ``kinase``                  -- kinase-targeted libraries
* ``aminergic_gpcr``          -- aminergic-GPCR-targeted libraries
* ``covalent_inhibitor``      -- covalent-inhibitor screening
* ``oral_safe_liver``         -- liver-safe oral drug profile
* ``htvs_minimal``            -- minimal hygiene for high-throughput VS
* ``ultra_strict``            -- everything turned on; aggressive

Add new profiles by mutating :py:data:`PROFILES` or by passing
``filters=[...]`` directly to :class:`Pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FilterSpec = tuple[str, dict[str, Any]]


@dataclass
class Profile:
    name: str
    description: str
    filters: list[FilterSpec]


# ---------------------------------------------------------------------------
PROFILES: dict[str, Profile] = {
    "general_drug_like": Profile(
        name="general_drug_like",
        description="Broad oral-drug HTS triage. Lipinski/Veber + alerts + dedup.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 8, "max_heavy": 70}),
            ("formal_charge", {"charge_min": -2, "charge_max": 2}),
            ("lipinski", {}),
            ("veber", {}),
            ("qed_min", {"threshold": 0.30}),
            ("pains", {}),
            ("brenk", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "cns_brain_penetrant": Profile(
        name="cns_brain_penetrant",
        description="Brain-penetrant: CNS MPO, BBB Egan envelope, no PAINS / reactive.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 10, "max_heavy": 50}),
            ("lipinski", {}),
            ("veber", {}),
            ("bbb_egan", {}),
            ("cns_mpo", {"threshold": 4.0}),
            ("qed_min", {"threshold": 0.40}),
            ("pains", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "cns_strict": Profile(
        name="cns_strict",
        description="Stringent CNS: BBB strict + CNS MPO >= 5 + Brenk + PAINS.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 10, "max_heavy": 45}),
            ("lipinski", {}),
            ("bbb_strict", {}),
            ("cns_mpo", {"threshold": 5.0}),
            ("qed_min", {"threshold": 0.50}),
            ("pains", {}),
            ("brenk", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "fragment_screening": Profile(
        name="fragment_screening",
        description="Fragment-based screening (Astex Rule of 3) + alerts.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 6, "max_heavy": 22}),
            ("rule_of_three", {}),
            ("qed_min", {"threshold": 0.30}),
            ("pains", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "kinase": Profile(
        name="kinase",
        description="Kinase-targeted: drug-like + kinase-hinge SMARTS gate.",
        filters=[
            ("lipinski", {}),
            ("veber", {}),
            ("qed_min", {"threshold": 0.35}),
            ("kinase_hinge", {}),
            ("pains", {}),
            ("brenk", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "aminergic_gpcr": Profile(
        name="aminergic_gpcr",
        description="Aminergic GPCR-targeted: drug-like + basic-amine requirement.",
        filters=[
            ("lipinski", {}),
            ("veber", {}),
            ("qed_min", {"threshold": 0.40}),
            ("basic_amine_required", {}),
            ("pains", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "covalent_inhibitor": Profile(
        name="covalent_inhibitor",
        description="Covalent-inhibitor screening: warhead REQUIRED + drug-like.",
        filters=[
            ("lipinski", {}),
            ("veber", {}),
            ("qed_min", {"threshold": 0.30}),
            ("covalent_warhead_required", {}),
            ("pains", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "oral_safe_liver": Profile(
        name="oral_safe_liver",
        description="Oral drug avoiding obvious liver-tox proxies (high MW, ESOL, hERG).",
        filters=[
            ("lipinski", {}),
            ("veber", {}),
            ("qed_min", {"threshold": 0.40}),
            ("esol_min", {"threshold": -5.0}),
            ("herg_proxy", {"logp_max": 3.5}),
            ("pgp_proxy", {}),
            ("pains", {}),
            ("brenk", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "htvs_minimal": Profile(
        name="htvs_minimal",
        description="Minimal hygiene only -- for very-large vendor library triage.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 6, "max_heavy": 80}),
            ("formal_charge", {"charge_min": -2, "charge_max": 2}),
            ("reactive_groups", {}),
            ("pains", {}),
            ("dedup_inchikey", {}),
        ],
    ),

    "ultra_strict": Profile(
        name="ultra_strict",
        description="Everything: physchem + drug-likeness + every alert catalog.",
        filters=[
            ("heavy_atom_range", {"min_heavy": 10, "max_heavy": 50}),
            ("formal_charge", {"charge_min": -1, "charge_max": 1}),
            ("lipinski", {"max_violations": 0}),
            ("veber", {}),
            ("ghose", {}),
            ("egan", {}),
            ("muegge", {}),
            ("qed_min", {"threshold": 0.50}),
            ("sa_max", {"threshold": 5.0}),
            ("all_alerts", {}),
            ("reactive_groups", {}),
            ("dedup_inchikey", {}),
        ],
    ),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Known: {sorted(PROFILES)}")
    return PROFILES[name]
