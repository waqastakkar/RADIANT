"""ChemBERTa-2 baseline.

Loads a ZINC-pretrained ChemBERTa from HuggingFace Hub and fine-tunes a
regression head on a single ChEMBL target. The default checkpoint is the
77M-parameter MLM variant; smaller variants (10M / 100M) can be selected
via ``--model-id``.

Citation
--------
Chithrananda et al., 'ChemBERTa: Large-Scale Self-Supervised Pretraining
for Molecular Property Prediction'.

CLI::

    python -m radiant_qsar.baselines.chemberta \\
        --activities data/processed/v1/activities.parquet \\
        --target CHEMBL279 --out runs/chemberta/CHEMBL279/scaffold \\
        --split scaffold --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from radiant_qsar.baselines._hf_shared import HFBaselineConfig, train_hf_baseline


DEFAULT_MODEL_ID = "DeepChem/ChemBERTa-77M-MLM"


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--activities", required=True, type=Path)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help=f"HF model id (default: {DEFAULT_MODEL_ID!r})")
    p.add_argument("--split", default="scaffold",
                   choices=("random", "scaffold", "time", "cluster", "activity_cliff"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--pooling", default="cls", choices=("cls", "mean"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_hf_baseline(HFBaselineConfig(
        activities=args.activities,
        target_chembl_id=args.target,
        out=args.out,
        model_id=args.model_id,
        baseline_name="chemberta",
        split_kind=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        pooling=args.pooling,
        device=args.device,
        seed=args.seed,
        trust_remote_code=False,   # ChemBERTa uses standard RoBERTa tokenizer
    ))


if __name__ == "__main__":
    _main()
