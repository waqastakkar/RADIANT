# RADIANT

**Recurrent Activity-Directed Iterative Architecture for Neural QSAR Training**

A recurrent-depth transformer for molecular property prediction, targeting publication in *Nature Machine Intelligence*. RADIANT processes SMILES through a weight-shared iterative refinement core with adaptive halting, attention pooling, and depth-adaptive representations, pretrained on 870M ZINC20 + ChEMBL molecules and evaluated across a 20-target panel against Morgan/RF, ChemBERTa-2, MolFormer, and GIN baselines.

---

## Architecture

```
input_ids ──► StemEncoder ──► e ──► h₀ = e
                                      │
                 ┌────► for t in range(n_loops):
                 │       sₜ        = IterationSignal(t)
                 │       pre_core   = norm(h + sₜ)
                 │       core_out   = SharedCore(pre_core)         # weight-shared
                 │       core_out   = IterationAdapter(core_out, t)
                 │       h          = h + βₜ · core_out + γₜ · Anchor(e)
                 │       (ConfidenceHalting freezes converged tokens)
                 └─────────────────────────────────────────────
                                      │
                             h ──► ExitDecoder ──► RMSNorm
                                      │
                          ┌───────────┴───────────┐
                     AttentionPool            DepthAdaptivePool
                          │                       │
                       TaskHead               TaskHead
```

**Key components:**

| Module | Role |
|---|---|
| `StemEncoder` | Token + positional embedding (once) |
| `IterativeRefinementCore` | Weight-shared transformer blocks applied N times |
| `StateAnchorUpdate` | Stable recurrence: `h_{t+1} = h_t + β·Core(h_t) + γ·Anchor(e)` |
| `ConfidenceHalting` | PonderNet-style adaptive depth per token |
| `AttentionPooling` | Learnable query-based pooling over pharmacophore-relevant atoms |
| `DepthAdaptivePool` | Weights intermediate hidden states by halting probabilities |
| `ExitDecoder` | Projects to output space (once) |

The same checkpoint can be evaluated at any loop count, trading compute for accuracy at inference time. Sinusoidal iteration signals enable extrapolation beyond the training loop count.

---

## 3-Stage Training Pipeline

```
Stage 1: SMILES MLM + Contrastive pretrain (870M ZINC20 + ChEMBL)
    └── Learns chemistry: full SMILES, scaffolds, R-groups, functional groups
         ↓
Stage 2: Multi-target activity pretrain (1.57M activities, 7,913 targets)
    └── Learns bioactivity: target-conditioned SAR plus Murcko R-group views
         ↓
Stage 3: Single-target fine-tune (per-target, head warmup + cosine schedule)
    └── Specializes to a single endpoint with pretrained backbone
```

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate              # Linux / macOS

pip install -e ".[all]"                   # everything
# pip install -e .                       # core only
# pip install -e ".[data]"              # + data curation
# pip install -e ".[baselines]"         # + all baselines
# pip install -e ".[dev]"              # + pytest

pytest -q                                # smoke test
```

CUDA is optional for the test suite but required for training.

---

## Repository Layout

```
RADIANT/
├── radiant/                     # Core architecture (domain-agnostic)
│   ├── config.py                  # RadiantConfig (frozen dataclass)
│   ├── model.py                   # RadiantModel (top-level forward)
│   ├── stem_encoder.py            # StemEncoder
│   ├── refinement_core.py         # IterativeRefinementCore
│   ├── state_anchor.py            # StateAnchorUpdate
│   ├── confidence_halting.py      # ConfidenceHalting + HaltingTrace
│   ├── exit_decoder.py            # ExitDecoder
│   ├── attention.py               # GQA with RoPE
│   ├── feedforward.py             # SwiGLU + optional MoE
│   ├── heads.py                   # LMHead, RegressionHead, PoolingHead
│   └── ...
│
├── radiant_chem/                # Chemistry-specific wrapper
│   ├── config.py                  # RadiantChemConfig
│   ├── model_chem.py              # RadiantChemModel + task heads
│   ├── tokenizer.py               # SMILES tokenizer
│   ├── depth_pool.py              # DepthAdaptivePool
│   ├── augment.py                 # SMILES randomization
│   └── ...
│
├── radiant_qsar/                # QSAR pipeline
│   ├── data/                      # Phase A: ChEMBL 36 data curation
│   ├── splits/                    # Phase B: random, scaffold, time, cluster, activity_cliff
│   ├── pretrain/                  # Stage 1 + 2: MLM, contrastive, activity pretrain
│   ├── finetune/                  # Stage 3: single-target + panel sweep
│   ├── baselines/                 # Morgan/RF, ChemBERTa-2, MolFormer, GIN
│   ├── analyses/                  # Phase G: depth-vs-complexity, calibration, attribution
│   ├── screening/                 # Phase H: 28 filters, 10 profiles, ML scoring
│   ├── eval/                      # Prediction writer, halting extras
│   └── reproduce/                 # Reproducibility bundle
│
├── training/                    # Generic Trainer + callbacks
├── configs/                     # JSON architecture presets
├── examples/                    # 7 runnable demos (no data needed)
├── tests/                       # 230+ tests, CPU-only
└── data/                        # Produced by pipeline (not committed)
```

---

## Pipeline Commands

All paths are relative to the repo root.

### Step 0: Place ChEMBL 36 SQLite

Download from [EBI](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/) and place as `chembl_36.db` at the repo root.

### Step 1: Data Curation

```bash
python -m radiant_qsar.data.run_phase_a \
    --db             chembl_36.db \
    --raw-dir        data/raw \
    --processed-dir  data/processed/v1 \
    --n-jobs         4
```

Produces 1,567,570 curated activities across 7,913 targets and 953,701 compounds.

### Step 2: Build Tokenizer Vocab

```bash
python -m radiant_qsar.data.build_vocab \
    --compounds   data/processed/v1/compounds.parquet \
    --out         data/processed/v1/smiles_vocab.json \
    --kind        smiles \
    --min-count   3
```

### Step 3: Stage 1 — SMILES Pretrain (ZINC20 + ChEMBL, 870M molecules)

Build the merged corpus, rebuild vocab for new tokens, then pretrain:

```bash
# Build corpus
python -m radiant_qsar.pretrain.zinc_corpus build \
    --zinc-dir data/zinc20 \
    --chembl data/processed/v1/compounds.parquet \
    --out data/zinc20/corpus.txt \
    --jobs 8 \
    --resume

# Rebuild vocab (preserves old token IDs for checkpoint compatibility)
python -m radiant_qsar.pretrain.zinc_corpus vocab \
    --corpus data/zinc20/corpus.txt \
    --old-vocab data/processed/v1/smiles_vocab.json \
    --out data/zinc20/smiles_vocab.json \
    --sample 5000000

# Pretrain
python -m radiant_qsar.pretrain.pretrain_loop \
    --corpus            data/zinc20/corpus.txt \
    --vocab             data/zinc20/smiles_vocab.json \
    --config            configs/radiant_75m.json \
    --out               checkpoints/pretrain \
    --steps             500000 \
    --batch-size        64 \
    --lr                1.5e-4 \
    --warmup-steps      2000 \
    --contrastive-temperature 0.1 \
    --scaffold-contrastive-weight 0.05 \
    --rgroup-contrastive-weight 0.05 \
    --rgroup-mlm-weight 0.25 \
    --n-loops-train     8 \
    --loop-sampling     range \
    --min-loops-train   2 \
    --bf16 \
    --device            cuda
```

Put all ZINC20 tranche files directly under `data/zinc20/`; the builder discovers `.smi`, `.smiles`, `.txt`, `.csv`, and `.gz` files. For large ZINC20 folders, prefer `--jobs N --resume` without `--deduplicate`: it writes resumable per-file clean chunks under `data/zinc20/.corpus.txt.state/` and reuses completed chunks if the run is interrupted. Use `--deduplicate` only for smaller builds with enough RAM, because global deduplication keeps a SMILES set in memory and therefore runs sequentially.

The 75M-parameter model (`d_model=640, 8 heads, 4 KV heads, 4/4/4 blocks`) trains MLM + InfoNCE on randomized SMILES pairs. By default it also derives Murcko scaffold and side-chain/R-group SMILES views and trains scaffold/R-group contrastive plus R-group MLM auxiliary losses. For corpora over 2GB, the loader automatically streams with a 500K shuffle buffer.

#### Training stability — avoiding loop-depth collapse

The recurrent core is **fixed-point stable by construction** (the `ExitDecoder` RMSNorm pins the output magnitude, and the iterated state converges as `n_loops` grows — verified empirically out to 32 loops). The historical instability was a *training-dynamics* failure, **not** an architectural blow-up:

- **Cause.** Training one loop depth per step (single-value curriculum) made the model brittle at other depths ("only works up to ~3 loops"). When the curriculum advanced (e.g. 3→4), it ran a loop index whose per-loop parameters (`StateAnchorUpdate` gates, `IterationSignal` tokens) were still at init — a cold-depth gradient shock at peak LR. The InfoNCE heads collapse first (all losses saturate at `ln(batch_size)` — e.g. 3.466 for batch 32) and, being an absorbing state, never recover; the shared trunk then drags MLM down and the run diverges.

- **Fix.** Sample loop depth from a **range** every step so the model learns a depth-robust recurrence and no loop index is ever cold-started:

  | Flag | Default | Purpose |
  |---|---|---|
  | `--loop-sampling {range,fixed}` | `range` | `range` samples `n_loops ∈ [min_loops_train, ceiling]` per step. `fixed` is the legacy single-depth behavior (brittle). |
  | `--min-loops-train` | `2` | Lower bound for range sampling. |
  | `--curriculum-loops` | off | Optional — anneals only the *upper* bound `min→n_loops_train` over the first third of training. Redundant with `range`; safe to omit. |

  The driver also **clips gradients at 1.0, skips any non-finite step** (so one bad batch can't poison the weights/optimizer moments), **logs `grad_norm`** each interval, and on `--resume-from` **restores the LR/loop schedule from the checkpoint** (mismatched CLI flags on resume were the "restart makes it worse" footgun — they now warn instead of silently shifting the schedule).

- **Recommended settings** (reflected in the command above): `--lr 1.5e-4` (3e-4 is hot for a 75M recurrent-depth model + InfoNCE), `--contrastive-temperature 0.1` (0.07 sharpens the collapse basin), and the larger the `--batch-size` the better (more in-batch negatives raise the `ln(batch)` collapse floor).

- **What to watch.** `grad_norm` should sit in the low tens without trending up; the three contrastive losses should stay **well below `ln(batch_size)`** and keep falling. A contrastive loss creeping toward `ln(batch)` is the earliest collapse warning — it appears ~40k steps before total loss visibly breaks. Collapsed checkpoints cannot recover; restart from a pre-collapse checkpoint or from scratch.

**Smoke test first** (5 min, CPU):

```bash
python -m radiant_qsar.pretrain.pretrain_loop \
    --compounds data/processed/v1/compounds.parquet \
    --vocab     data/processed/v1/smiles_vocab.json \
    --config    configs/radiant_tiny.json \
    --out       checkpoints/pretrain_smoke \
    --steps     500 --batch-size 16 --device cpu --num-workers 0
```

### Step 4: Stage 2 — Multi-Target Activity Pretrain

Trains on all 1.57M ChEMBL activities with target-conditioned regression, bridging chemistry knowledge to bioactivity understanding.
By default this stage also derives Murcko scaffold side-chain/R-group SMILES and adds an auxiliary activity loss on that R-group view, so the shared backbone is pushed to learn substituent-level SAR rather than only full-molecule strings.

```bash
python -m radiant_qsar.pretrain.activity_pretrain \
    --activities   data/processed/v1/activities.parquet \
    --vocab        data/zinc20/smiles_vocab.json \
    --config       configs/radiant_75m.json \
    --stage1-ckpt  checkpoints/pretrain/latest.pt \
    --out          checkpoints/activity_pretrain \
    --epochs       10 \
    --batch-size   128 \
    --lr           5e-5 \
    --rgroup-aux-weight 0.25 \
    --device       cuda
```

### Step 5: Stage 3 — Single-Target Fine-Tune

Fine-tuning uses head warmup (freeze backbone for 5 epochs, train head at 1e-3), attention pooling, depth-adaptive pooling, SMILES augmentation, and cosine scheduling with layer-wise LR decay.

```bash
python -m radiant_qsar.finetune.single_task \
    --activities    data/processed/v1/activities.parquet \
    --target        CHEMBL279 \
    --vocab         data/zinc20/smiles_vocab.json \
    --config        configs/radiant_75m.json \
    --pretrain-ckpt checkpoints/activity_pretrain/backbone_for_finetune.pt \
    --out           runs/panel/radiant/CHEMBL279/scaffold \
    --split         scaffold \
    --epochs        100 \
    --batch-size    16 \
    --lr            2e-5 \
    --pooling-kind  attention \
    --use-depth-pool \
    --head-warmup-epochs 5 \
    --smiles-augment-prob 0.50 \
    --device        cuda
```

NOTE on `--disable-halting-loss`: do NOT pass this flag during fine-tune.
Earlier sweeps that disabled halting loss caused `halt_step`, `effective_depth`,
and `confidence_var` to collapse to a constant in every test prediction
(documented as `radiant_no_halt_loss` ablation arm; see Step 6b). The
default config (`configs/radiant_75m.json`) sets `halting_loss_weight: 0.05`
with a 20-epoch warmup, which is the recommended setting.

### Step 6: Panel Sweep — 20 Targets x 5 Splits

```bash
# Select panel (20 targets across 7 ChEMBL classes)
python -m radiant_qsar.finetune.select_panel \
    --activities data/processed/v1/activities.parquet \
    --targets    data/processed/v1/targets.parquet \
    --out        data/processed/v1/panel.json

# Pre-compute splits (optional but recommended)
python -m radiant_qsar.splits.precompute \
    --panel      data/processed/v1/panel.json \
    --activities data/processed/v1/activities.parquet \
    --splits     random scaffold time cluster activity_cliff

# Run sweep (all models, all splits)
python -m radiant_qsar.finetune.sweep \
    --panel         data/processed/v1/panel.json \
    --activities    data/processed/v1/activities.parquet \
    --vocab         data/zinc20/smiles_vocab.json \
    --config        configs/radiant_75m.json \
    --pretrain-ckpt checkpoints/activity_pretrain/backbone_for_finetune.pt \
    --out           runs/panel \
    --splits        random scaffold time cluster activity_cliff \
    --models        radiant morgan_rf chemberta molformer gin \
    --head-warmup-epochs 5 \
    --device cuda
```

The sweep is resumable: existing cells are skipped on restart.

### Step 6b: Halt-Loss Ablation Arm (free with retraining)

If you previously trained with `--disable-halting-loss` (now removed),
your existing `runs/panel/radiant/` cells have a collapsed halting head
(`effective_depth = 2.0` constant, `confidence_var = 0`). Re-running Step
6 will overwrite them. **Preserve them first as a free ablation arm:**

```bash
# Preserve the existing collapsed-halt runs as a renamed model dir.
# Phase G will auto-discover this as a separate model row in every
# heatmap / rank table / win matrix.
mv runs/panel/radiant runs/panel/radiant_no_halt_loss

# Now re-run Step 6 (radiant only). Halt loss will be inherited from
# the config (0.05) since --disable-halting-loss is gone.
python -m radiant_qsar.finetune.sweep \
    --panel         data/processed/v1/panel.json \
    --activities    data/processed/v1/activities.parquet \
    --vocab         data/zinc20/smiles_vocab.json \
    --config        configs/radiant_75m.json \
    --pretrain-ckpt checkpoints/activity_pretrain/backbone_for_finetune.pt \
    --out           runs/panel \
    --splits        random scaffold time cluster activity_cliff \
    --models        radiant \
    --head-warmup-epochs 5 \
    --device cuda
```

After this you have **both rows** for free:
- `runs/panel/radiant/` (halt loss = 0.05, working halting) — main results
- `runs/panel/radiant_no_halt_loss/` (halt loss = 0.0, collapsed) — ablation arm

Every auto-iterating Phase G module (`g0_validation_metrics`, `g_ranks`,
`g_stat_tests`, `g_pairwise_wins`, `g_per_split_winrate`, `g_target_family`,
`g_applicability_domain`, `g_confidence_filter`, `g_calibration_extensions`,
`g_compute_parity`, `g_hard_splits`) will include both as separate rows
with no code change.

For the radiant-specific deep-dive modules (which default to `model="radiant"`),
run them once with the new model and once with the ablation arm:

```bash
# Re-populate G.4 loop_sweep/ for the NEW radiant (loop_sweep is per-cell)
python -m radiant_qsar.analyses.g4_test_time_loop_sweep \
    --mode panel --panel-root runs/panel \
    --config configs/radiant_75m.json \
    --vocab  data/zinc20/smiles_vocab.json \
    --lf-model-dir radiant \
    --out-dir runs/phase_g \
    --split scaffold --loops 1 2 4 8 12 16 \
    --descriptors data/processed/v1/descriptors.parquet \
    --device cuda --batch-size 64

# Now run each radiant-specific module twice -- once per model.
# Pattern: --model <name> for table-style modules,
#          --lf-model-dir <name> for model-loading modules.
for MODEL in radiant radiant_no_halt_loss; do
  python -m radiant_qsar.analyses.g_training_curves   --panel-root runs/panel --out-dir runs/phase_g --model "$MODEL"
  python -m radiant_qsar.analyses.g_failure_modes     --panel-root runs/panel --out-dir runs/phase_g --model "$MODEL"
  python -m radiant_qsar.analyses.g_rgroup_sar        --panel-root runs/panel --out-dir runs/phase_g --model "$MODEL" --split random
  python -m radiant_qsar.analyses.g_smiles_consistency --panel-root runs/panel --out-dir runs/phase_g --lf-model-dir "$MODEL"
  python -m radiant_qsar.analyses.g_halting_toggle    --panel-root runs/panel --out-dir runs/phase_g --lf-model-dir "$MODEL"
  python -m radiant_qsar.analyses.g5_atom_attribution --panel-root runs/panel --out-dir runs/phase_g --lf-model-dir "$MODEL"
done
```

(`g_activity_cliff_sar` already runs in all-models mode by default --
no per-model loop needed.)

### Step 7: Phase G — Publication Analyses

```bash
python -m radiant_qsar.analyses.run_phase_g --config configs/phase_g.yaml
```

Publication analyses covering the five compute-aware claims plus reviewer-facing
SAR / statistics / pretrain / interpretability modules. Every module emits
publication-grade PNG + SVG figures, CSV + TSV tables, and a `summary.md`
that the orchestrator stitches into `runs/phase_g/PHASE_G_REPORT.md`.

| Analysis | Module | What it produces |
|---|---|---|
| G.0 Validation metrics | `g0_validation_metrics` | MAE/RMSE/R²/Pearson/Spearman per cell, model-summary heatmap, parity plots |
| G.1 Depth vs complexity | `g1_depth_vs_complexity` | Effective-depth correlation with 6 complexity descriptors; depth distribution diagnostic |
| G.3 Calibration | `g3_calibration` | Sigma vs error, calibration_long.csv builder |
| G.4 Test-time loop sweep | `g4_test_time_loop_sweep` | MAE / Pearson vs n_loops in {1, 2, 4, 8, 12, 16}; per-target sensitivity curves |
| G.5 Atom attribution | `g5_atom_attribution` | Per-atom halt heatmaps + 40-tile combined case-study grid labelled by target name |
| R-group SAR | `g_rgroup_sar` | Same-scaffold substituent Δ-pChEMBL / sign-accuracy / pairwise rank |
| Activity-cliff SAR | `g_activity_cliff_sar` | All-models cliff Δ-MAE, sign accuracy, rank correlation; per-model comparison bars |
| Failure modes | `g_failure_modes` | Worst cells / scaffolds + RDKit structure thumbnails (PNG + SVG) for the 24 worst predictions |
| Pairwise wins | `g_pairwise_wins` | Head-to-head win rates per complexity bin + overall heatmap |
| Compute parity | `g_compute_parity` | Holm-corrected paired bootstrap + label-permutation sanity |
| Training curves | `g_training_curves` | Per-epoch train+val MAE/Pearson grids across 20 targets × 5 splits |
| **Avg rank + CD diagram** | `g_ranks` | Demšar critical-difference diagram per metric + per-split heatmaps |
| **Friedman + Nemenyi** | `g_stat_tests` | Omnibus + pairwise post-hoc p-values, significant-pairs CSV |
| **Hard-split summary** | `g_hard_splits` | Excludes `random` — mean MAE / R² / win-rate vs reference across scaffold/time/cluster/activity_cliff |
| **Applicability domain** | `g_applicability_domain` | Max-Tanimoto-to-train + 4-bin MAE comparison per model |
| **Confidence filter** | `g_confidence_filter` | Top-k retention MAE using Tanimoto-NN as confidence proxy |
| **Per-split win matrix** | `g_per_split_winrate` | K×K winrate heatmap per split (random / scaffold / time / cluster / activity_cliff) |
| **Target family** | `g_target_family` | MAE / Pearson per target_class (joined from panel.json) |
| **Pretrain curves** | `g_pretrain_curves` | ZINC20 loss curves + ChEMBL stage-2 val MAE/Pearson + 3-stage overview |
| **Calibration extensions** | `g_calibration_extensions` | ECE, reliability diagram, top-k MAE, parity overlay |
| **Scaffold novelty** | `g_scaffold_novelty` | Test scaffold Tanimoto vs train scaffolds → bin MAE |
| **Halting on/off toggle** | `g_halting_toggle` | Reads default predictions vs G.4 loop_sweep; compares halting-ON vs fixed-K MAE |
| **SMILES augmentation consistency** | `g_smiles_consistency` | Predicts each test molecule under K=5 random SMILES; stdev as confidence proxy |

The orchestrator config (`configs/phase_g.yaml`) controls which modules
run; each block accepts `enabled: false` to skip. Modules that depend on
other modules' outputs (e.g. `g_confidence_filter` reads
`g_applicability_domain/tables/ad_per_molecule.csv`) check the upstream
artifact and skip cleanly with a `FileNotFoundError` if it's missing.

### Step 8: Virtual Screening

```bash
python -m radiant_qsar.screening.prepare_library \
    --input    library.smi \
    --output   filtered.smi \
    --profile  kinase \
    --summary  summary.json
```

28 filters across 7 categories, 10 built-in profiles. RADIANT integrates as an ML scoring filter:

```python
from radiant_qsar.screening import Pipeline, get_filter
from radiant_qsar.screening.filters.ml_scoring import RADIANTPotency

pipe = Pipeline([
    get_filter("lipinski"),
    get_filter("pains"),
    RADIANTPotency(
        checkpoint_path="runs/panel/radiant/CHEMBL279/scaffold/best.pt",
        vocab_path="data/zinc20/smiles_vocab.json",
        min_pchembl=7.0,
        n_loops=8,
    ),
])
pipe.run("library.smi", "hits.smi")
```

### Production Orchestrator

For the pure-RADIANT path, use the resumable reproduction driver. It keeps
long jobs opt-in and gates screening on a selected validated RADIANT checkpoint:

```bash
# Stage 1 + Stage 2 + RADIANT-only 20-target panel + selection
STAGES=d,d2,panel,select bash radiant_qsar/reproduce/run_all.sh

# Pure-RADIANT ablations for the mechanistic claims
STAGES=ablate bash radiant_qsar/reproduce/run_all.sh

# Add baselines only for comparison, not fusion
STAGES=f bash radiant_qsar/reproduce/run_all.sh

# Reviewer-facing internal controls
STAGES=leak,cal,stats bash radiant_qsar/reproduce/run_all.sh

# Run Phase G once runs/panel exists
STAGES=g bash radiant_qsar/reproduce/run_all.sh

# Run the extra SAR/error analyses individually
STAGES=rgsar,cliffsar,failures,rgroupabl bash radiant_qsar/reproduce/run_all.sh

# Screen with the selected RADIANT checkpoint
INPUT_LIBRARY=library.smi STAGES=h bash radiant_qsar/reproduce/run_all.sh
```

The selection manifest is written to
`runs/screening/selected_radiant_model.json` and is produced by:

```bash
python -m radiant_qsar.finetune.select_checkpoint \
    --panel-root runs/panel \
    --target CHEMBL279 \
    --split scaffold \
    --metric pearson \
    --vocab data/zinc20/smiles_vocab.json \
    --out runs/screening/selected_radiant_model.json
```

---

## Quick Demos (No Data Required)

```bash
python examples/forward_pass.py            # forward pass, loop metrics
python examples/train_lm_synthetic.py      # 1-min CPU LM training
python examples/analyze_loop_dynamics.py   # per-loop norms, halting
python examples/ablate_loop_count.py       # eval-time loop count ablation
python examples/baseline_transformer.py    # comparison architectures
python examples/train_chem_property.py     # toy regression
```

---

## Configs

| File | Params | d_model | Use |
|---|---:|---:|---|
| `configs/radiant_tiny.json` | ~130K | 64 | Tests, smoke runs |
| `configs/radiant_75m.json` | ~75M | 640 | Full training (all stages) |
| `configs/radiant_base.json` | ~30-50M | 512 | Intermediate scale |

Generate configs programmatically:

```python
from radiant import small_config
small_config(d_model=256, n_loops_train=6).to_json("configs/custom.json")
```

---

## Tests

```bash
pytest -q                            # full suite (~2 min, CPU only)
pytest -q tests/qsar/                # QSAR pipeline only
pytest -q tests/test_independence.py # originality scan
```

230+ tests covering core architecture, chem wrapper, training scaffold, data pipeline, all split strategies, baselines, screening, and analyses. No external services or GPU required.

---

## Baselines

| Model | Params | Type | Notes |
|---|---:|---|---|
| Morgan/RF | — | Fingerprint + Random Forest | CPU-only, offline |
| ChemBERTa-2 | 77M | Pretrained transformer | HuggingFace, ~150MB download |
| MolFormer | 110M | Pretrained transformer | HuggingFace, ~440MB download |
| GIN | 1-3M | Graph neural network | Pure PyTorch, no PyG/DGL |

All baselines share identical CLI interface, output schema (`result.json`, `predictions.csv`), and integrate into the panel sweep.

---

---

## Citation

```bibtex
@software{radiant2026,
  title  = {RADIANT: Recurrent Activity-Directed Iterative Architecture
            for Neural QSAR Training},
  year   = {2026},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
