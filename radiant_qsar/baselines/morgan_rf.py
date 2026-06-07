"""Morgan-fingerprint + Random Forest regression baseline.

Classical, surprisingly strong, and zero-GPU. Plugs into the same
single-target evaluation harness as :mod:`finetune.single_task` so the
result tables come out shape-compatible.

Outputs (per ``--out`` directory)
---------------------------------

* ``result.json``   -- metrics in canonical schema (``train``/``val``/``test`` blocks).
* ``model.joblib``  -- bundle: trained ``RandomForestRegressor`` + FP config
                       + dataset/target/split metadata + library versions.
* ``predictions.csv`` -- per-test-molecule (id, true, pred) rows for downstream
                       calibration / OOD analyses.

Inference utility
-----------------
:func:`predict_smiles_from_ckpt` loads ``model.joblib`` and scores a list
of SMILES; this is what :class:`radiant_qsar.screening.filters.ml_scoring.MorganRFPotency`
uses to plug a trained RF baseline into the screening pipeline.

Usage::

    python -m radiant_qsar.baselines.morgan_rf \\
        --activities data/processed/v1/activities.parquet \\
        --target CHEMBL279 \\
        --out runs/morgan_rf/CHEMBL279/scaffold \\
        --split scaffold --n-estimators 500 --n-jobs -1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


MODEL_FILENAME = "model.joblib"


@dataclass
class MorganRFConfig:
    activities: Path
    target_chembl_id: str
    out: Path
    split_kind: str = "scaffold"
    radius: int = 2
    n_bits: int = 2048
    n_estimators: int = 500
    max_features: str | float = "sqrt"
    n_jobs: int = -1
    seed: int = 1337
    splits_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
def _morgan_fp_matrix(smiles: Sequence[str], radius: int, n_bits: int) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    X = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def _split(sub, kind: str, ratios, seed):
    """Cache-aware split (see ``radiant_qsar.splits.cache``). The first
    call for a (target, split, seed) triple computes; subsequent calls
    across baselines read from ``data/splits/v1/<target>/<split>__seed<N>.json``."""
    from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split

    target = sub["target_chembl_id"].iloc[0]
    cfg = SplitCacheConfig(seed=seed, ratios=tuple(ratios))
    return load_or_compute_split(target, kind, sub, cfg)


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------
def _library_versions() -> dict[str, str]:
    versions = {}
    for mod_name in ("sklearn", "rdkit", "numpy", "joblib"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not-installed"
    return versions


def save_bundle(model, cfg: MorganRFConfig, dest: Path) -> Path:
    """Persist the trained model + everything needed to score new SMILES."""
    import joblib

    bundle = {
        "model": model,
        "feature_extractor": "morgan",
        "fp_radius": cfg.radius,
        "fp_n_bits": cfg.n_bits,
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "seed": cfg.seed,
        "sklearn_versions": _library_versions(),
        "build_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "schema_version": 1,
    }
    joblib.dump(bundle, dest)
    return dest


def load_bundle(path: Path | str) -> dict:
    """Load a Morgan/RF bundle written by :func:`save_bundle`."""
    import joblib

    return joblib.load(path)


def predict_smiles_from_ckpt(
    ckpt_path: Path | str,
    smiles: Sequence[str],
) -> np.ndarray:
    """Score a list of SMILES with a previously-trained Morgan/RF bundle.

    Returns an ``(N,)`` ndarray of predicted pchembl values. Unparseable
    SMILES get a NaN prediction (the FP row is all zeros, which the RF
    will still score, but the caller should treat NaN as "skip").
    """
    bundle = load_bundle(ckpt_path)
    X = _morgan_fp_matrix(list(smiles), bundle["fp_radius"], bundle["fp_n_bits"])
    preds = bundle["model"].predict(X)
    # Mark unparseable inputs as NaN so screening can drop them.
    bad = np.all(X == 0, axis=1)
    preds = preds.astype(float).copy()
    preds[bad] = np.nan
    return preds


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_morgan_rf(cfg: MorganRFConfig) -> dict:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    cfg.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.activities)
    sub = df[df["target_chembl_id"] == cfg.target_chembl_id].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"no rows for target {cfg.target_chembl_id}")
    train_idx, val_idx, test_idx = _split(sub, cfg.split_kind, cfg.splits_ratios, cfg.seed)

    smi = sub["standard_smiles"].tolist()
    pch = sub["pchembl"].astype(float).values

    t0 = time.time()
    X = _morgan_fp_matrix(smi, cfg.radius, cfg.n_bits)

    Xtr, ytr = X[train_idx], pch[train_idx]
    Xva, yva = X[val_idx], pch[val_idx]
    Xte, yte = X[test_idx], pch[test_idx]

    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        max_features=cfg.max_features,
        n_jobs=cfg.n_jobs,
        random_state=cfg.seed,
    )
    rf.fit(Xtr, ytr)

    from radiant_qsar.finetune.single_task import regression_metrics

    train_m = regression_metrics(rf.predict(Xtr), ytr)
    val_m   = regression_metrics(rf.predict(Xva), yva)
    test_m  = regression_metrics(rf.predict(Xte), yte)
    elapsed = time.time() - t0

    logger.info(
        "Morgan/RF %s [%s]: val MAE=%.3f rho=%.3f | test MAE=%.3f rho=%.3f (n_test=%d) in %.1fs",
        cfg.target_chembl_id, cfg.split_kind,
        val_m["mae"], val_m["pearson"], test_m["mae"], test_m["pearson"],
        test_m["n"], elapsed,
    )

    # 1. save model bundle
    model_path = cfg.out / MODEL_FILENAME
    save_bundle(rf, cfg, model_path)

    # 2. write per-test-molecule predictions for downstream calibration / OOD /
    # complexity / interpretability analyses (Phase G sub-claims C1-C5).
    from radiant_qsar.eval.predictions import write_predictions

    test_smi = [smi[i] for i in test_idx]
    test_inchikeys = sub["inchikey14"].iloc[test_idx].tolist()
    test_pred = rf.predict(Xte)
    write_predictions(
        cfg.out,
        indices=test_idx,
        inchikeys=test_inchikeys,
        smiles=test_smi,
        true_pchembl=yte.tolist(),
        pred_pchembl=test_pred.tolist(),
        target_chembl_id=cfg.target_chembl_id,
        split_kind=cfg.split_kind,
    )

    # 3. result.json in canonical schema (train / val / test blocks; no "_metrics" suffix)
    result = {
        "model": "morgan_rf",
        "target_chembl_id": cfg.target_chembl_id,
        "split_kind": cfg.split_kind,
        "radius": cfg.radius,
        "n_bits": cfg.n_bits,
        "n_estimators": cfg.n_estimators,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        # Canonical metric blocks consumed by sweep.aggregate_results.
        "train": train_m,
        "val": val_m,
        "test": test_m,
        # Legacy keys retained briefly for any pre-existing analyses; will be removed
        # in v0.2 once the sweep aggregator is universally on the new schema.
        "val_metrics": val_m,
        "test_metrics": test_m,
        "model_path": MODEL_FILENAME,
        "predictions_path": "predictions.csv",
        "elapsed_s": round(elapsed, 1),
        "library_versions": _library_versions(),
    }
    (cfg.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="scaffold",
                   choices=("random", "scaffold", "time", "cluster", "activity_cliff"))
    p.add_argument("--radius", type=int, default=2,
                   help="Morgan FP radius (default 2 = ECFP4-equivalent)")
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="parallelism for RF fit / predict (-1 = all cores)")
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_morgan_rf(
        MorganRFConfig(
            activities=args.activities,
            target_chembl_id=args.target,
            out=args.out,
            split_kind=args.split,
            radius=args.radius,
            n_bits=args.n_bits,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            max_features=args.max_features,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    _main()
