"""Phase G — Compute-aware analyses for the RADIANT-QSAR NMI study.

Each module in this package implements one Phase G sub-claim or
cross-cutting control, as defined in ``docs/qsar_plan.md``:

* ``g0_validation_metrics``    — Model validation metrics (MAE, RMSE,
  R², Pearson, Spearman) across all models × targets × splits. Parity
  plots, comparison bar charts, heatmaps.
* ``g1_depth_vs_complexity``  — Sub-claim C1: effective halt depth vs
  molecular-complexity descriptors (Spearman/Pearson, CV regression,
  feature importance).
* ``g3_calibration``           — Sub-claim C3: reliability diagrams, ECE,
  Brier, NLL, with RADIANT halting / posterior-over-loops uncertainty
  benchmarked against deep ensembles at matched compute.
* ``g4_test_time_loop_sweep`` — Sub-claim C4: evaluate a fixed checkpoint
  at ``n_loops ∈ {1,2,4,8,12,16,24}`` and overlay halting-induced
  effective compute.
* ``g5_atom_attribution``     — Sub-claim C5: per-atom halt-step heat
  maps, rank correlation against gradient*input, MMP-fragment agreement.
* ``g_pairwise_wins``          — Pairwise model benchmarking & win
  analysis across splits, targets, and complexity bins.
* ``g_compute_parity``         — FLOPs / params annotation, paired
  bootstrap CIs, Holm-corrected p-values, label-permutation sanity.
* ``g_rgroup_sar``             — same-scaffold R-group potency-delta checks.
* ``g_activity_cliff_sar``     — high-similarity activity-cliff error checks.
* ``g_failure_modes``          — worst-row, target/split, and scaffold errors.
* ``g_stage1_representation_probe`` — scaffold/R-group embedding probes.
* ``g_rgroup_ablation``        — pure-model R-group chemistry ablation tables.

All analyses are *data-driven*: they consume canonical ``predictions.csv``
files (schema defined in :mod:`radiant_qsar.eval.predictions`) joined
against ``descriptors.parquet``. They do not need to load any model
checkpoint to produce most outputs; only :mod:`g4_test_time_loop_sweep`
and :mod:`g5_atom_attribution` invoke a model directly (and only when a
checkpoint path is supplied).
"""

from __future__ import annotations

from radiant_qsar.analyses.common import (
    AnalysisPaths,
    bootstrap_paired_diff,
    discover_predictions,
    holm_correction,
    load_predictions,
    publication_style,
    save_figure,
    save_table,
    spearman_pearson,
)

__all__ = [
    "AnalysisPaths",
    "bootstrap_paired_diff",
    "discover_predictions",
    "holm_correction",
    "load_predictions",
    "publication_style",
    "save_figure",
    "save_table",
    "spearman_pearson",
]
