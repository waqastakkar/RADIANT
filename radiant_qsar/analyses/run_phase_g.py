"""Phase G orchestrator.

Run every Phase G analysis in sequence and assemble a single
``PHASE_G_REPORT.md`` that the manuscript draft can pull from.

A YAML config drives all paths. Example config::

    panel_root: runs/panel_75m
    out_dir:    runs/phase_g
    descriptors: data/processed/v1/descriptors.parquet

    # G.1 — panel mode: correlate halting depth vs complexity across all 100 cells.
    # Set panel_root to auto-discover radiant/*/predictions.csv files.
    # Alternatively set predictions: <single cell csv> for single-cell mode.
    g1:
      panel_root: runs/panel_75m
      n_bootstrap: 500

    # G.3 — calibration (optional; requires upstream calibration_long.csv)
    g3:
      predictions: runs/phase_g/_inputs/calibration_long.csv

    # G.4 — panel mode: re-runs each target's best.pt at multiple n_loops (resumable)
    g4:
      panel_root: runs/panel_75m
      split: scaffold
      loops: [1, 2, 4, 8, 12, 16, 24]
      config: configs/radiant_75m.json
      vocab: data/processed/v1/smiles_vocab.json
      device: cuda

    # G.5 — panel mode: per-atom attribution from all 20 targets (no checkpoint needed)
    g5:
      panel_root: runs/panel_75m
      split: scaffold
      n_case_studies_per_target: 2

    # Cross-cutting (always run — only need panel_root)
    pairwise:
      n_bootstrap: 10000
    compute_parity:
      params_flops_csv: runs/panel_75m/params_flops.csv
      n_perm: 200
      # label-perm pools all 20 LF scaffold cells automatically — no single cell path needed

Each block is optional; missing blocks skip that analysis.

Usage
-----
    python -m radiant_qsar.analyses.run_phase_g --config configs/phase_g.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from radiant_qsar.analyses import (
    g0_validation_metrics,
    g1_depth_vs_complexity,
    g3_calibration,
    g4_test_time_loop_sweep,
    g5_atom_attribution,
    g_activity_cliff_sar,
    g_applicability_domain,
    g_calibration_extensions,
    g_compute_parity,
    g_confidence_filter,
    g_failure_modes,
    g_halting_toggle,
    g_hard_splits,
    g_pairwise_wins,
    g_per_split_winrate,
    g_pretrain_curves,
    g_ranks,
    g_rgroup_ablation,
    g_rgroup_sar,
    g_scaffold_novelty,
    g_smiles_consistency,
    g_stage1_representation_probe,
    g_stat_tests,
    g_target_family,
    g_training_curves,
)
from radiant_qsar.analyses.common import publication_style

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML required to load the orchestrator config") from exc
    return yaml.safe_load(path.read_text())


def _safe_run(name: str, fn, *args, **kwargs) -> dict | None:
    try:
        logger.info("==> %s", name)
        result = fn(*args, **kwargs)
        logger.info("    OK")
        return result
    except FileNotFoundError as exc:
        logger.warning("%s skipped: %s", name, exc)
        return None
    except Exception:
        logger.error("%s FAILED:\n%s", name, traceback.format_exc())
        return None


def assemble_phase_g_report(out_dir: Path, results: dict[str, dict | None]) -> Path:
    """Concatenate every per-analysis summary.md into a single PHASE_G_REPORT.md."""
    out_path = out_dir / "PHASE_G_REPORT.md"
    parts: list[str] = ["# Phase G — Compute-aware analyses\n",
                        "Assembled from per-analysis summaries written by `run_phase_g.py`.\n"]
    for name, res in results.items():
        if res is None:
            parts.append(f"\n---\n\n## {name}\n\n*(skipped; see logs)*\n")
            continue
        summary_md = res["paths"].summary_md
        if summary_md.exists():
            parts.append("\n---\n")
            parts.append(summary_md.read_text(encoding="utf-8"))
        else:
            parts.append(f"\n---\n\n## {name}\n\n*(no summary.md emitted)*\n")
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def run(config_path: Path | str) -> dict[str, dict | None]:
    publication_style()
    config = _load_yaml(Path(config_path))

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_root = Path(config["panel_root"])
    descriptors = config.get("descriptors")

    results: dict[str, dict | None] = {}

    # G.0 — Model validation metrics (always runs; needs only panel_root)
    g0 = config.get("g0", {})
    results["G.0 validation metrics"] = _safe_run(
        "G.0 validation metrics",
        g0_validation_metrics.run,
        panel_root=panel_root,
        out_dir=out_dir,
        model_filter=g0.get("model"),
    )

    if "g1" in config:
        g1 = config["g1"]
        # Panel mode: discover all LF cells under panel_root (preferred).
        # Single-cell mode: point directly at one predictions.csv.
        g1_panel_root = g1.get("panel_root")
        if g1_panel_root:
            results["G.1 depth-vs-complexity"] = _safe_run(
                "G.1 depth-vs-complexity (panel)",
                g1_depth_vs_complexity.run_panel,
                panel_root=g1_panel_root,
                descriptors_path=g1.get("descriptors", descriptors),
                out_dir=out_dir,
                n_bootstrap=g1.get("n_bootstrap", 500),
                seed=g1.get("seed", 0),
            )
        else:
            results["G.1 depth-vs-complexity"] = _safe_run(
                "G.1 depth-vs-complexity",
                g1_depth_vs_complexity.run,
                predictions_path=g1["predictions"],
                descriptors_path=g1.get("descriptors", descriptors),
                out_dir=out_dir,
                n_bootstrap=g1.get("n_bootstrap", 1000),
                n_cv_splits=g1.get("n_cv_splits", 5),
                seed=g1.get("seed", 0),
                regression_model=g1.get("regression_model", "ridge"),
            )

    if "g3" in config:
        g3 = config["g3"]
        results["G.3 calibration"] = _safe_run(
            "G.3 calibration",
            g3_calibration.run,
            predictions_csv=g3["predictions"],
            out_dir=out_dir,
        )

    if "g4" in config:
        g4 = config["g4"]
        g4_panel_root = g4.get("panel_root")
        if g4_panel_root:
            if not g4.get("config") or not g4.get("vocab"):
                logger.warning("G.4 panel mode requires g4.config and g4.vocab in YAML; skipping")
                results["G.4 test-time loop sweep"] = None
            else:
                results["G.4 test-time loop sweep"] = _safe_run(
                    "G.4 test-time loop sweep (panel)",
                    g4_test_time_loop_sweep.run_panel,
                    panel_root=g4_panel_root,
                    config_path=g4["config"],
                    vocab_path=g4["vocab"],
                    out_dir=out_dir,
                    split=g4.get("split", "scaffold"),
                    loops=g4.get("loops", g4_test_time_loop_sweep.DEFAULT_LOOPS),
                    descriptors_path=g4.get("descriptors", descriptors),
                    bin_descriptor=g4.get("bin_descriptor", "BertzCT"),
                    n_bins=g4.get("n_bins", 4),
                    device=g4.get("device", "cuda"),
                    batch_size=g4.get("batch_size", 64),
                )
        else:
            results["G.4 test-time loop sweep"] = _safe_run(
                "G.4 test-time loop sweep",
                g4_test_time_loop_sweep.run,
                predictions_dir=g4["predictions_dir"],
                out_dir=out_dir,
                loops=g4.get("loops", g4_test_time_loop_sweep.DEFAULT_LOOPS),
                descriptors_path=g4.get("descriptors", descriptors),
                bin_descriptor=g4.get("bin_descriptor", "BertzCT"),
                n_bins=g4.get("n_bins", 4),
            )

    if "g5" in config:
        g5 = config["g5"]
        g5_panel_root = g5.get("panel_root")
        if g5_panel_root:
            results["G.5 per-atom attribution"] = _safe_run(
                "G.5 per-atom attribution (panel)",
                g5_atom_attribution.run_panel,
                panel_root=g5_panel_root,
                out_dir=out_dir,
                split=g5.get("split", "scaffold"),
                n_case_studies_per_target=g5.get("n_case_studies_per_target", 2),
                mmp_delta_pchembl=g5.get("mmp_delta", 1.0),
                mmp_tanimoto=g5.get("mmp_tanimoto", 0.7),
            )
        else:
            results["G.5 per-atom attribution"] = _safe_run(
                "G.5 per-atom attribution",
                g5_atom_attribution.run,
                predictions_path=g5["predictions"],
                out_dir=out_dir,
                descriptors_path=g5.get("descriptors", descriptors),
                n_case_studies=g5.get("n_case_studies", 8),
                mmp_delta_pchembl=g5.get("mmp_delta", 1.0),
                mmp_tanimoto=g5.get("mmp_tanimoto", 0.7),
            )

    if "rgroup_sar" in config:
        rg = config["rgroup_sar"]
        results["R-group SAR"] = _safe_run(
            "R-group SAR",
            g_rgroup_sar.run,
            panel_root=rg.get("panel_root", panel_root),
            out_dir=out_dir,
            model=rg.get("model", "radiant"),
            split=rg.get("split", "scaffold"),
            min_abs_true_delta=rg.get("min_abs_true_delta", 0.3),
            max_pairs_per_scaffold=rg.get("max_pairs_per_scaffold", 250),
            seed=rg.get("seed", 0),
        )

    if "activity_cliff_sar" in config:
        ac = config["activity_cliff_sar"]
        results["Activity-cliff SAR"] = _safe_run(
            "Activity-cliff SAR",
            g_activity_cliff_sar.run,
            panel_root=ac.get("panel_root", panel_root),
            out_dir=out_dir,
            model=ac.get("model", "radiant"),
            split=ac.get("split", "activity_cliff"),
            tanimoto_threshold=ac.get("tanimoto_threshold", 0.55),
            activity_delta_threshold=ac.get("activity_delta_threshold", 1.0),
            max_pairs_per_cell=ac.get("max_pairs_per_cell", 2000),
            seed=ac.get("seed", 0),
        )

    if "failure_modes" in config:
        fm = config["failure_modes"]
        results["Failure modes"] = _safe_run(
            "Failure modes",
            g_failure_modes.run,
            panel_root=fm.get("panel_root", panel_root),
            out_dir=out_dir,
            model=fm.get("model", "radiant"),
            top_n=fm.get("top_n", 100),
            min_scaffold_n=fm.get("min_scaffold_n", 3),
        )

    if "stage1_probe" in config:
        sp = config["stage1_probe"]
        if sp.get("embeddings_csv"):
            results["Stage-1 representation probe"] = _safe_run(
                "Stage-1 representation probe",
                g_stage1_representation_probe.run,
                embeddings_csv=sp["embeddings_csv"],
                out_dir=out_dir,
                k=sp.get("k", 5),
                seed=sp.get("seed", 0),
            )
        else:
            logger.warning("Stage-1 probe requires stage1_probe.embeddings_csv; skipping")
            results["Stage-1 representation probe"] = None

    if "rgroup_ablation" in config:
        ra = config["rgroup_ablation"]
        results["R-group ablation"] = _safe_run(
            "R-group ablation",
            g_rgroup_ablation.run,
            panel_results=ra.get("panel_results", panel_root / "panel_results.csv"),
            out_dir=out_dir,
            primary=ra.get("primary", "radiant"),
            ablations=tuple(ra.get("ablations", [
                "radiant_no_stage1_rgroup",
                "radiant_no_stage2_rgroup",
                "radiant_no_rgroup",
            ])),
        )

    pw = config.get("pairwise", {})
    results["Pairwise wins"] = _safe_run(
        "Pairwise wins",
        g_pairwise_wins.run,
        panel_root=panel_root,
        out_dir=out_dir,
        descriptors_path=pw.get("descriptors", descriptors),
        bin_descriptor=pw.get("bin_descriptor", "BertzCT"),
        n_bins=pw.get("n_bins", 4),
        n_bootstrap=pw.get("n_bootstrap", 10_000),
        seed=pw.get("seed", 0),
    )

    cp = config.get("compute_parity", {})
    results["Compute parity"] = _safe_run(
        "Compute parity",
        g_compute_parity.run,
        panel_root=panel_root,
        out_dir=out_dir,
        params_flops_csv=cp.get("params_flops_csv"),
        perm_predictions=cp.get("perm_predictions"),
        perm_descriptors=cp.get("perm_descriptors", descriptors),
        n_bootstrap=cp.get("n_bootstrap", 10_000),
        n_perm=cp.get("n_perm", 200),
        seed=cp.get("seed", 0),
    )

    # --- Cross-cutting stats / training-curve / hard-split modules ---------
    # All four read g0_validation_metrics outputs (or per-cell result.json
    # for training curves) so they only need panel_root + out_dir; they have
    # safe defaults and are *always* run unless explicitly disabled by the
    # YAML config block ``<name>: {enabled: false}``.
    tc = config.get("training_curves", {})
    if tc.get("enabled", True):
        results["Training curves"] = _safe_run(
            "Training curves",
            g_training_curves.run,
            panel_root=panel_root,
            out_dir=out_dir,
            model=tc.get("model", "radiant"),
        )

    rk = config.get("ranks", {})
    if rk.get("enabled", True):
        results["Average rank + CD diagram"] = _safe_run(
            "Average rank + CD diagram",
            g_ranks.run,
            panel_root=panel_root,
            out_dir=out_dir,
            g0_cell_metrics=rk.get("g0_cell_metrics"),
        )

    st = config.get("stat_tests", {})
    if st.get("enabled", True):
        results["Friedman + Nemenyi"] = _safe_run(
            "Friedman + Nemenyi",
            g_stat_tests.run,
            panel_root=panel_root,
            out_dir=out_dir,
            g0_cell_metrics=st.get("g0_cell_metrics"),
            alpha=st.get("alpha", 0.05),
        )

    hs = config.get("hard_splits", {})
    if hs.get("enabled", True):
        results["Hard-split summary"] = _safe_run(
            "Hard-split summary",
            g_hard_splits.run,
            panel_root=panel_root,
            out_dir=out_dir,
            g0_cell_metrics=hs.get("g0_cell_metrics"),
            reference_model=hs.get("reference_model", "radiant"),
        )

    ad = config.get("applicability_domain", {})
    if ad.get("enabled", True):
        results["Applicability domain"] = _safe_run(
            "Applicability domain",
            g_applicability_domain.run,
            panel_root=panel_root,
            out_dir=out_dir,
            activities_path=ad.get("activities_path",
                                   "data/processed/v1/activities.parquet"),
        )

    cf = config.get("confidence_filter", {})
    if cf.get("enabled", True):
        results["Confidence filter"] = _safe_run(
            "Confidence filter",
            g_confidence_filter.run,
            panel_root=panel_root,
            out_dir=out_dir,
            ad_per_molecule_csv=cf.get("ad_per_molecule_csv"),
        )

    psw = config.get("per_split_winrate", {})
    if psw.get("enabled", True):
        results["Per-split win-rate matrices"] = _safe_run(
            "Per-split win-rate matrices",
            g_per_split_winrate.run,
            panel_root=panel_root,
            out_dir=out_dir,
            g0_cell_metrics=psw.get("g0_cell_metrics"),
            metric_col=psw.get("metric", "mae"),
            lower_better=not psw.get("higher_better", False),
        )

    tf = config.get("target_family", {})
    if tf.get("enabled", True):
        results["Target-family analysis"] = _safe_run(
            "Target-family analysis",
            g_target_family.run,
            panel_root=panel_root,
            out_dir=out_dir,
            g0_cell_metrics=tf.get("g0_cell_metrics"),
        )

    pc = config.get("pretrain_curves", {})
    if pc.get("enabled", True):
        results["Pretrain curves"] = _safe_run(
            "Pretrain curves",
            g_pretrain_curves.run,
            out_dir=out_dir,
            zinc20_log=pc.get("zinc20_log",
                              "checkpoints/pretrain/train_log.jsonl"),
            activity_pretrain_result=pc.get("activity_pretrain_result",
                                            "checkpoints/activity_pretrain/result.json"),
            finetune_best_csv=pc.get("finetune_best_csv"),
        )

    cex = config.get("calibration_extensions", {})
    if cex.get("enabled", True):
        results["Calibration extensions"] = _safe_run(
            "Calibration extensions",
            g_calibration_extensions.run,
            panel_root=panel_root,
            out_dir=out_dir,
            splits=tuple(cex["splits"]) if cex.get("splits") else None,
        )

    sn = config.get("scaffold_novelty", {})
    if sn.get("enabled", True):
        results["Scaffold novelty bins"] = _safe_run(
            "Scaffold novelty bins",
            g_scaffold_novelty.run,
            panel_root=panel_root,
            out_dir=out_dir,
            activities_path=sn.get("activities_path",
                                   "data/processed/v1/activities.parquet"),
            ad_per_molecule_csv=sn.get("ad_per_molecule_csv"),
        )

    ht = config.get("halting_toggle", {})
    if ht.get("enabled", True):
        results["Halting ON vs OFF"] = _safe_run(
            "Halting ON vs OFF",
            g_halting_toggle.run,
            panel_root=panel_root,
            out_dir=out_dir,
            lf_model_dir=ht.get("lf_model_dir", "radiant"),
            split=ht.get("split", "scaffold"),
            loops=tuple(ht.get("loops", [1, 2, 4, 8, 12, 16])),
        )

    sc = config.get("smiles_consistency", {})
    if sc.get("enabled", True):
        results["SMILES augmentation consistency"] = _safe_run(
            "SMILES augmentation consistency",
            g_smiles_consistency.run,
            panel_root=panel_root,
            out_dir=out_dir,
            config_path=sc.get("config_path", "configs/radiant_75m.json"),
            vocab_path=sc.get("vocab_path", "data/zinc20/smiles_vocab.json"),
            lf_model_dir=sc.get("lf_model_dir", "radiant"),
            split=sc.get("split", "scaffold"),
            n_augmentations=sc.get("n_augmentations", 5),
            device=sc.get("device", "cuda"),
            batch_size=sc.get("batch_size", 64),
            max_cells=sc.get("max_cells"),
        )

    acm = config.get("activity_cliff_all_models", {})
    if acm.get("enabled", True):
        results["Activity-cliff (all models)"] = _safe_run(
            "Activity-cliff (all models)",
            g_activity_cliff_sar.run,
            panel_root=panel_root,
            out_dir=out_dir,
            model=None,  # iterate all 5 models
            split=acm.get("split", "activity_cliff"),
            tanimoto_threshold=acm.get("tanimoto_threshold", 0.55),
            activity_delta_threshold=acm.get("activity_delta_threshold", 1.0),
            max_pairs_per_cell=acm.get("max_pairs_per_cell", 2000),
            seed=acm.get("seed", 0),
        )

    report_path = assemble_phase_g_report(out_dir, results)
    logger.info("Phase G report assembled at %s", report_path)
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all Phase G analyses end-to-end")
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config; see run_phase_g.__doc__ for the schema")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    results = run(args.config)
    n_ok = sum(1 for r in results.values() if r is not None)
    logger.info("Done. %d / %d analyses completed.", n_ok, len(results))
    if n_ok == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
