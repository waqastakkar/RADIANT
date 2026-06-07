"""Leakage audit for RADIANT-QSAR panel splits.

This report is meant for reviewer-facing evidence. It checks each
target/split for exact molecule overlap, scaffold overlap, temporal leakage,
and sampled nearest-neighbour Tanimoto overlap between train and test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from radiant_qsar.finetune.sweep import VALID_SPLITS
from radiant_qsar.splits.cache import SplitCacheConfig, load_or_compute_split


def _read_panel(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("entries", data if isinstance(data, list) else [])


def _murcko(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or ""
    except Exception:
        return ""


def _sampled_max_tanimoto(train_smiles: list[str], test_smiles: list[str], *, max_train: int, max_test: int, seed: int) -> dict:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except Exception:
        return {"sampled_max_tanimoto_mean": np.nan, "sampled_max_tanimoto_p95": np.nan, "sampled_test_n": 0}

    rng = np.random.default_rng(seed)
    if len(train_smiles) > max_train:
        train_smiles = [train_smiles[i] for i in rng.choice(len(train_smiles), max_train, replace=False)]
    if len(test_smiles) > max_test:
        test_smiles = [test_smiles[i] for i in rng.choice(len(test_smiles), max_test, replace=False)]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(s: str):
        mol = Chem.MolFromSmiles(s)
        return gen.GetFingerprint(mol) if mol is not None else None

    train_fps = [x for x in (fp(s) for s in train_smiles) if x is not None]
    test_fps = [x for x in (fp(s) for s in test_smiles) if x is not None]
    if not train_fps or not test_fps:
        return {"sampled_max_tanimoto_mean": np.nan, "sampled_max_tanimoto_p95": np.nan, "sampled_test_n": len(test_fps)}

    max_sims = []
    for q in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(q, train_fps)
        max_sims.append(max(sims) if sims else np.nan)
    arr = np.asarray(max_sims, dtype=float)
    return {
        "sampled_max_tanimoto_mean": float(np.nanmean(arr)),
        "sampled_max_tanimoto_p95": float(np.nanpercentile(arr, 95)),
        "sampled_test_n": int(np.isfinite(arr).sum()),
    }


def audit_panel(
    *,
    panel: Path,
    activities: Path,
    out_dir: Path,
    splits: list[str],
    seed: int = 1337,
    max_train_tanimoto: int = 5000,
    max_test_tanimoto: int = 500,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_panel(panel)
    acts = pd.read_parquet(activities)
    rows: list[dict] = []
    cfg = SplitCacheConfig(seed=seed)

    for entry in entries:
        target = entry["target_chembl_id"]
        sub = acts[acts["target_chembl_id"] == target].reset_index(drop=True)
        if sub.empty:
            continue
        for split in splits:
            train_idx, val_idx, test_idx = load_or_compute_split(target, split, sub, cfg)
            train = sub.iloc[train_idx]
            val = sub.iloc[val_idx]
            test = sub.iloc[test_idx]

            train_keys = set(train["inchikey14"].astype(str))
            val_keys = set(val["inchikey14"].astype(str))
            test_keys = set(test["inchikey14"].astype(str))
            train_scaf = set(_murcko(s) for s in train["standard_smiles"].astype(str))
            test_scaf = set(_murcko(s) for s in test["standard_smiles"].astype(str))
            train_scaf.discard("")
            test_scaf.discard("")

            years_train = pd.to_numeric(train.get("doc_year_max"), errors="coerce")
            years_test = pd.to_numeric(test.get("doc_year_max"), errors="coerce")
            temporal_leak = False
            if split == "time" and years_train.notna().any() and years_test.notna().any():
                temporal_leak = bool(years_train.max() > years_test.min())

            sim = _sampled_max_tanimoto(
                train["standard_smiles"].astype(str).tolist(),
                test["standard_smiles"].astype(str).tolist(),
                max_train=max_train_tanimoto,
                max_test=max_test_tanimoto,
                seed=seed,
            )
            exact_overlap = train_keys & test_keys
            val_test_overlap = val_keys & test_keys
            scaf_overlap = train_scaf & test_scaf

            rows.append({
                "target_chembl_id": target,
                "target_class": entry.get("target_class", ""),
                "split": split,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "exact_train_test_overlap_n": len(exact_overlap),
                "exact_val_test_overlap_n": len(val_test_overlap),
                "scaffold_train_test_overlap_n": len(scaf_overlap),
                "scaffold_test_overlap_fraction": len(scaf_overlap) / max(len(test_scaf), 1),
                "train_year_max": float(years_train.max()) if years_train.notna().any() else np.nan,
                "test_year_min": float(years_test.min()) if years_test.notna().any() else np.nan,
                "temporal_order_violation": temporal_leak,
                **sim,
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "leakage_report.csv"
    df.to_csv(csv_path, index=False)
    summary = {
        "n_rows": int(len(df)),
        "exact_overlap_cells": int((df["exact_train_test_overlap_n"] > 0).sum()) if not df.empty else 0,
        "temporal_violation_cells": int(df["temporal_order_violation"].sum()) if not df.empty else 0,
        "high_similarity_cells_p95_ge_0_90": int((df["sampled_max_tanimoto_p95"] >= 0.90).sum()) if not df.empty else 0,
    }
    (out_dir / "leakage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return df


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--panel", required=True, type=Path)
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=list(VALID_SPLITS), choices=VALID_SPLITS)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-train-tanimoto", type=int, default=5000)
    p.add_argument("--max-test-tanimoto", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = _parse()
    df = audit_panel(
        panel=args.panel,
        activities=args.activities,
        out_dir=args.out_dir,
        splits=args.splits,
        seed=args.seed,
        max_train_tanimoto=args.max_train_tanimoto,
        max_test_tanimoto=args.max_test_tanimoto,
    )
    print(f"wrote {args.out_dir / 'leakage_report.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
