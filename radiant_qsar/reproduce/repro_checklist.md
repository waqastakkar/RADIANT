# RADIANT-QSAR — Reproducibility & FAIR-data checklist

Tracks Nature Machine Intelligence's mandatory **Reporting Summary**
and **Code / Data Availability** requirements line-by-line. Each row
is either ✅ done, 🟡 partially done (with what's outstanding), or
🔲 pending. Update this file in lockstep with `manifest.yaml`.

> **Reviewer-friendly entry point:** clone the GitHub release matching
> the cited tag in [manifest.yaml](manifest.yaml), drop `chembl_36.db`
> in the repo root, then either
> `bash radiant_qsar/reproduce/run_all.sh` or
> `docker run … radiant-qsar:0.1.0` (see Dockerfile header). The
> default `STAGES=a,c,e,f,g` runs everything except multi-day
> pretraining and the screening case study, both of which are gated
> behind explicit opt-in flags.

---

## 1. Data

| Item | Status | Notes |
|---|:-:|---|
| Upstream data is public | ✅ | ChEMBL 36 (free public release) |
| Upstream version + URL recorded | ✅ | `manifest.yaml:data_source` |
| Curated release is versioned | ✅ | `data/processed/v1/` + `manifest.json` |
| Curated release deposited (Zenodo DOI) | 🔲 | Mint at Phase I.1 |
| Curated release file hashes published | ✅ | SHA-256 in `manifest.yaml:data_release.files` and re-checked in `data/processed/v1/manifest.json` |
| Reproducible build from upstream → curated | ✅ | `python -m radiant_qsar.data.run_phase_a` re-builds bit-identical files |
| Inclusion / exclusion criteria documented | ✅ | `docs/qsar_plan.md` Phase A; SQL embedded in `radiant_qsar/data/chembl_extract.py` |
| Data validity filters documented | ✅ | `activity_curate.py` (relation / unit / IQR), `standardize.py` (chembl_structure_pipeline pipeline) |
| Activity type + unit normalization documented | ✅ | pchembl conversion + bounds `[3.0, 12.0]` |
| Duplicate handling documented | ✅ | InChIKey-14 dedup; median pchembl on replicates with IQR ≤ 1 log |
| Split strategies pre-registered | ✅ | random / scaffold / time / cluster / activity_cliff / target_holdout; seeded |
| Per-target stratification by class | ✅ | `target_consolidate.py`; 8-class taxonomy |
| Time-split year used | ✅ | T = 2020 (val T+1, test T+2..max) |

## 2. Code

| Item | Status | Notes |
|---|:-:|---|
| Code public on GitHub | 🟡 | Local working copy; release tag minted at submission |
| Permissive license | ✅ | MIT (`LICENSE`) |
| Version pinned in `manifest.yaml` | ✅ | `study.version` = 0.1.0 |
| Dependency versions pinned | ✅ | `manifest.yaml:dependencies` mirrors the `qsar` conda env |
| Single-command reproduction script | ✅ | `radiant_qsar/reproduce/run_all.sh` |
| Docker image | ✅ | `radiant_qsar/reproduce/Dockerfile` (CUDA 12.6, conda-forge pins) |
| Tests pass on a clean checkout | ✅ | 236 passed, 1 skipped (see `manifest.yaml:tests`) |
| CI configuration | 🔲 | GitHub Actions workflow pending |
| All randomness seeded | ✅ | `--seed` flag on every driver; defaults logged per run |

## 3. Models

| Item | Status | Notes |
|---|:-:|---|
| Model architecture fully documented | ✅ | `docs/architecture.md`, `docs/recurrence.md`, `docs/halting.md` |
| Hyperparameters logged | ✅ | `configs/radiant_75m.json` checked in; per-run dumps under `runs/.../args.json` |
| Pretrain corpus identical to data release | ✅ | Built from `data/processed/v1/compounds.parquet` |
| Per-baseline parity protocol documented | ✅ | `docs/qsar_plan.md` Phase F + README Step 5 |
| Ablations published alongside headline | 🟡 | `radiant_no_anchor / no_adapter / no_halting` pending (see manifest `baselines.pending`) |
| Deep-ensemble baseline for calibration parity | 🟡 | `radiant_ensemble_5` pending |
| All published checkpoints uploaded to HF Hub | 🔲 | After pretrain completes |
| Param + FLOPs reported per model | ✅ | `radiant_qsar.eval.compute` + Phase G `g_compute_parity.py` |

## 4. Evaluation

| Item | Status | Notes |
|---|:-:|---|
| Per-(target, split) metrics dumped | ✅ | Canonical `predictions.csv` + `result.json` per cell |
| Bootstrap CIs reported | ✅ | 10K resamples in `g_compute_parity.py` |
| Multiple-comparison correction | ✅ | Holm in `common.holm_correction` |
| Calibration metrics (ECE / Brier / NLL) | ✅ | `g3_calibration.py` |
| Compute-equivalent uncertainty baseline | ✅ | Deep-ensemble-5 vs RADIANT halt-var + MC-loops |
| OOD evaluation | ✅ | scaffold / cluster / activity_cliff / target_holdout splits |
| Negative controls | ✅ | Label-permutation in `g_compute_parity.py` |
| Per-atom attribution | ✅ | `g5_atom_attribution.py` (halt + grad×input + MMP overlap) |
| Heat-map case studies | ✅ | 6-12 molecules rendered by `g5` |
| Test-time loop scaling sweep | ✅ | `g4_test_time_loop_sweep.py` over `n_loops ∈ {1,2,4,8,12,16,24}` |
| Honest negative results acceptable | ✅ | Per-claim verdicts emitted in each `summary.md` |

## 5. Statistics

| Item | Status | Notes |
|---|:-:|---|
| Test statistics + sample sizes reported | ✅ | `n` column in every Phase G table |
| Pre-registered claim falsification thresholds | ✅ | `docs/qsar_plan.md` Phase G subsections |
| 95 % CIs over molecules / seeds | ✅ | Paired bootstrap (mol-level) + 5 seeds for headline tables |
| Seeds per experiment | ✅ | 5 seeds for headline; 3 minimum elsewhere; per-run logged |
| Effect-size reporting | ✅ | ΔMAE / ΔPearson / ΔRMSE in `g_pairwise_wins.py` with margin categories |

## 6. Compute

| Item | Status | Notes |
|---|:-:|---|
| Hardware reported | ✅ | `manifest.yaml:environment` |
| GPU model + driver | 🟡 | Driver class noted (570.x); exact card per submission run pending |
| Wallclock per phase reported | 🟡 | Phase A ≈ 75 min captured; pretrain / fine-tune updated at completion |
| FLOPs reported | ✅ | `radiant_qsar.eval.compute` counter; values pulled into `g_compute_parity.py` |
| Same-compute baseline matching | ✅ | Documented in `manifest.yaml:baselines.parity_rule` |

## 7. Submission artefacts

| Item | Status | Notes |
|---|:-:|---|
| Pre-print on ChemRxiv | 🔲 | I.2 |
| Pre-print on arXiv | 🔲 | I.2 (cs.LG) |
| Zenodo bundle minted | 🔲 | I.1 |
| GitHub release tag minted | 🔲 | I.1 |
| HF Hub model release | 🔲 | I.1 |
| Cover letter drafted | 🔲 | I.4 |
| Response-to-reviewers template ready | 🔲 | I.4 |
| Backup-journal transfer letters drafted | 🔲 | I.3 |

---

## How this checklist is enforced

* `tests/qsar/test_reproduce_manifest.py` parses
  [manifest.yaml](manifest.yaml) and asserts every referenced **local
  path** exists (so a stale manifest fails CI).
* `radiant_qsar/reproduce/run_all.sh` exits non-zero on any phase
  failure, so a green run guarantees the pipeline is end-to-end
  executable.
* The Dockerfile re-runs `pytest -q tests` during the image build, so
  the published image cannot ship in a broken state.

## Open items by phase

* **Phase I.1 (bundle):** Zenodo deposit + GitHub release + HF Hub
  releases for pretrain + multi-task hub.
* **Phase I.2 (pre-print):** ChemRxiv + arXiv submissions before
  journal submission for visibility (NMI editors track ChemRxiv).
* **Phase I.3 (journal):** primary NMI submission; backup chain
  Nat Comm → Nat Comp Sci → JCIM with pre-written transfer letters.
* **Phase I.4 (cover letter):** lead with C4 (test-time loop scaling
  for chemistry), C1 second, C5 third — architecture is framed as the
  tool, not the headline.
