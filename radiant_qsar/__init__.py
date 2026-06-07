"""RADIANT-QSAR: a publication-grade QSAR pipeline built on RADIANT.

This package wraps the architecture-agnostic core (`radiant`) and chem
variant (`radiant_chem`) into an end-to-end QSAR study targeting
Nature Machine Intelligence-level rigor:

* `data/`       -- ChEMBL 36 SQL extraction, standardization, descriptors.
* `splits/`     -- five reproducible split strategies.
* `pretrain/`   -- masked-atom + contrastive pretraining drivers.
* `finetune/`   -- single-task and multi-task fine-tuning protocols.
* `baselines/`  -- re-trained-at-parity baselines.
* `eval/`       -- metrics, calibration, OOD aggregation, statistics.
* `analyses/`   -- the five sub-claim figures (C1-C5).
* `screening/`  -- retrospective virtual-screening case study.
* `reproduce/`  -- Docker, manifest, and end-to-end re-run script.

Designed to be additive: nothing in `radiant/` or `radiant_chem/`
needs to change to support this package.
"""

__version__ = "0.1.0"
