"""MolFormer baseline.

Loads IBM's MolFormer (a transformer pretrained on ~1.1B molecules from
PubChem + ZINC) from HuggingFace Hub and fine-tunes a regression head on
a single ChEMBL target. MolFormer's tokenizer ships custom code, so we
pass ``trust_remote_code=True``.

Citation
--------
Ross et al., 'Large-scale chemical language representations capture
molecular structure and properties', *Nature Machine Intelligence* 2022.

CLI::

    python -m radiant_qsar.baselines.molformer \\
        --activities data/processed/v1/activities.parquet \\
        --target CHEMBL279 --out runs/molformer/CHEMBL279/scaffold \\
        --split scaffold --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from radiant_qsar.baselines._hf_shared import HFBaselineConfig, train_hf_baseline


# The 10pct-data XL variant is the smallest / fastest official release.
DEFAULT_MODEL_ID = "ibm/MoLFormer-XL-both-10pct"


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
    p.add_argument("--lr", type=float, default=3e-5,
                   help="MolFormer is a bigger model -- a slightly lower LR than ChemBERTa")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--pooling", default="mean", choices=("cls", "mean"),
                   help="MolFormer was pretrained without a CLS objective; mean pooling is the canonical choice")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_hf_baseline(HFBaselineConfig(
        activities=args.activities,
        target_chembl_id=args.target,
        out=args.out,
        model_id=args.model_id,
        baseline_name="molformer",
        split_kind=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        pooling=args.pooling,
        device=args.device,
        seed=args.seed,
        trust_remote_code=True,   # MolFormer ships a custom tokenizer
    ))


if __name__ == "__main__":
    _main()
