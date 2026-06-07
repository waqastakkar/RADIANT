#!/usr/bin/env bash
# End-to-end driver for RADIANT-QSAR.
#
# Stages are opt-in so long GPU jobs do not start accidentally:
#   STAGES=a,c,d,d2,panel,select,g,h bash radiant_qsar/reproduce/run_all.sh
#
# Main stage names:
#   a      curate ChEMBL into data/processed/v1
#   c      build tokenizer vocab
#   d      Stage 1 SMILES pretraining
#   d2     Stage 2 multi-target activity pretraining
#   e      one RADIANT single-target fine-tune
#   panel  RADIANT-only 20 target x split sweep
#   ablate pure-RADIANT ablation sweep
#   f      Morgan/RF baseline for the headline cell
#   select select a validated RADIANT checkpoint for screening
#   leak   split leakage audit
#   cal    build Phase G.3 calibration input
#   stats  matched-cell statistical significance from panel_results.csv
#   g      Phase G analyses
#   rgsar  R-group SAR analysis
#   cliffsar activity-cliff SAR analysis
#   failures failure-mode analysis
#   stage1probe Stage-1 chemistry representation probe from embeddings CSV
#   rgroupabl R-group chemistry ablation comparison
#   h      library filtering + RADIANT potency screening

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

DB_PATH="${DB_PATH:-chembl_36.db}"
DATA_DIR="${DATA_DIR:-data/processed/v1}"
RUNS_DIR="${RUNS_DIR:-runs}"
PRETRAIN_DIR="${PRETRAIN_DIR:-checkpoints/pretrain_75m}"
ACTIVITY_PRETRAIN_DIR="${ACTIVITY_PRETRAIN_DIR:-checkpoints/activity_pretrain_75m}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-configs/radiant_75m.json}"
PHASE_G_CONFIG="${PHASE_G_CONFIG:-configs/phase_g.yaml}"
PHASE_G_OUT="${PHASE_G_OUT:-runs/phase_g}"
PANEL_ROOT="${PANEL_ROOT:-runs/panel_75m}"
ZINC_CORPUS="${ZINC_CORPUS:-data/zinc20/corpus.txt}"
ZINC_VOCAB="${ZINC_VOCAB:-data/zinc20/smiles_vocab.json}"
HEADLINE_TARGET="${HEADLINE_TARGET:-CHEMBL279}"
HEADLINE_SPLIT="${HEADLINE_SPLIT:-scaffold}"
SELECTED_MODEL_MANIFEST="${SELECTED_MODEL_MANIFEST:-runs/screening/selected_radiant_model.json}"
SCREEN_PROFILE="${SCREEN_PROFILE:-kinase}"
SCREENING_MIN_PCHEMBL="${SCREENING_MIN_PCHEMBL:-7.0}"
SEED="${SEED:-1337}"

STAGES="${STAGES:-c}"

want() { [[ ",${STAGES}," == *",$1,"* ]]; }
hdr() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

require_file() {
    [[ -f "$1" ]] || die "Missing required file: $1 ($2)"
}

py() { PYTHONPATH=. python "$@"; }

detect_device() {
    python - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    print("cpu")
PY
}

DEVICE="${DEVICE:-$(detect_device)}"
hdr "Detected device: ${DEVICE}"
hdr "Active stages: ${STAGES}"
hdr "Repo root: ${ROOT}"

if want a; then
    hdr "[Phase A] Curating ChEMBL ${DB_PATH} -> ${DATA_DIR}/"
    require_file "${DB_PATH}" "ChEMBL 36 SQLite"
    py -m radiant_qsar.data.run_phase_a \
        --db "${DB_PATH}" \
        --raw-dir data/raw \
        --processed-dir "${DATA_DIR}" \
        --n-jobs "${N_JOBS:-4}"
fi

if want c; then
    hdr "[Phase C] Building SMILES tokenizer vocab"
    require_file "${DATA_DIR}/compounds.parquet" "Phase A compounds"
    py -m radiant_qsar.data.build_vocab \
        --compounds "${DATA_DIR}/compounds.parquet" \
        --out "${DATA_DIR}/smiles_vocab.json" \
        --kind smiles \
        --min-count "${VOCAB_MIN_COUNT:-3}"
fi

if want d; then
    hdr "[Stage 1] SMILES pretraining"
    [[ "${DEVICE}" == "cuda" ]] || die "Stage 1 pretraining requires CUDA; got DEVICE=${DEVICE}"
    require_file "${PRETRAIN_CONFIG}" "RADIANT config"
    if [[ -f "${ZINC_CORPUS}" && -f "${ZINC_VOCAB}" ]]; then
        py -m radiant_qsar.pretrain.pretrain_loop \
            --corpus "${ZINC_CORPUS}" \
            --vocab "${ZINC_VOCAB}" \
            --config "${PRETRAIN_CONFIG}" \
            --out "${PRETRAIN_DIR}" \
            --steps "${PRETRAIN_STEPS:-500000}" \
            --batch-size "${PRETRAIN_BATCH_SIZE:-64}" \
            --lr "${PRETRAIN_LR:-3e-4}" \
            --warmup-steps "${PRETRAIN_WARMUP_STEPS:-2000}" \
            --scaffold-contrastive-weight "${SCAFFOLD_CONTRASTIVE_WEIGHT:-0.05}" \
            --rgroup-contrastive-weight "${RGROUP_CONTRASTIVE_WEIGHT:-0.05}" \
            --rgroup-mlm-weight "${RGROUP_MLM_WEIGHT:-0.25}" \
            --n-loops-train "${PRETRAIN_N_LOOPS:-8}" \
            --curriculum-loops \
            --bf16 \
            --device "${DEVICE}" \
            --seed "${SEED}"
    else
        hdr "[Stage 1] ${ZINC_CORPUS} not found; running ChEMBL-only source"
        require_file "${DATA_DIR}/smiles_vocab.json" "tokenizer vocab"
        py -m radiant_qsar.pretrain.pretrain_loop \
            --compounds "${DATA_DIR}/compounds.parquet" \
            --vocab "${DATA_DIR}/smiles_vocab.json" \
            --config "${PRETRAIN_CONFIG}" \
            --out "${PRETRAIN_DIR}" \
            --steps "${PRETRAIN_STEPS:-200000}" \
            --batch-size "${PRETRAIN_BATCH_SIZE:-64}" \
            --lr "${PRETRAIN_LR:-3e-4}" \
            --scaffold-contrastive-weight "${SCAFFOLD_CONTRASTIVE_WEIGHT:-0.05}" \
            --rgroup-contrastive-weight "${RGROUP_CONTRASTIVE_WEIGHT:-0.05}" \
            --rgroup-mlm-weight "${RGROUP_MLM_WEIGHT:-0.25}" \
            --n-loops-train "${PRETRAIN_N_LOOPS:-8}" \
            --curriculum-loops \
            --bf16 \
            --device "${DEVICE}" \
            --seed "${SEED}"
    fi
fi

if want d2; then
    hdr "[Stage 2] Multi-target activity pretraining"
    [[ "${DEVICE}" == "cuda" ]] || die "Stage 2 activity pretraining requires CUDA; got DEVICE=${DEVICE}"
    require_file "${DATA_DIR}/activities.parquet" "curated activities"
    require_file "${PRETRAIN_DIR}/latest.pt" "Stage 1 latest checkpoint"
    VOCAB_FOR_ACTIVITY="${ZINC_VOCAB}"
    [[ -f "${VOCAB_FOR_ACTIVITY}" ]] || VOCAB_FOR_ACTIVITY="${DATA_DIR}/smiles_vocab.json"
    require_file "${VOCAB_FOR_ACTIVITY}" "tokenizer vocab"
    py -m radiant_qsar.pretrain.activity_pretrain \
        --activities "${DATA_DIR}/activities.parquet" \
        --vocab "${VOCAB_FOR_ACTIVITY}" \
        --config "${PRETRAIN_CONFIG}" \
        --stage1-ckpt "${PRETRAIN_DIR}/latest.pt" \
        --out "${ACTIVITY_PRETRAIN_DIR}" \
        --epochs "${ACTIVITY_EPOCHS:-10}" \
        --batch-size "${ACTIVITY_BATCH_SIZE:-128}" \
        --lr "${ACTIVITY_LR:-5e-5}" \
        --n-loops-train "${ACTIVITY_N_LOOPS:-8}" \
        --rgroup-aux-weight "${RGROUP_AUX_WEIGHT:-0.25}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
fi

if want e; then
    hdr "[Stage 3] One RADIANT fine-tune target=${HEADLINE_TARGET} split=${HEADLINE_SPLIT}"
    [[ "${DEVICE}" == "cuda" ]] || die "Fine-tuning requires CUDA; got DEVICE=${DEVICE}"
    FINETUNE_CKPT="${FINETUNE_CKPT:-${ACTIVITY_PRETRAIN_DIR}/backbone_for_finetune.pt}"
    require_file "${FINETUNE_CKPT}" "Stage 2 backbone checkpoint"
    require_file "${DATA_DIR}/smiles_vocab.json" "tokenizer vocab"
    py -m radiant_qsar.finetune.single_task \
        --activities "${DATA_DIR}/activities.parquet" \
        --target "${HEADLINE_TARGET}" \
        --vocab "${DATA_DIR}/smiles_vocab.json" \
        --config "${PRETRAIN_CONFIG}" \
        --pretrain-ckpt "${FINETUNE_CKPT}" \
        --split "${HEADLINE_SPLIT}" \
        --out "${PANEL_ROOT}/radiant/${HEADLINE_TARGET}/${HEADLINE_SPLIT}" \
        --epochs "${FINETUNE_EPOCHS:-100}" \
        --batch-size "${FINETUNE_BATCH_SIZE:-16}" \
        --lr "${FINETUNE_LR:-2e-5}" \
        --head-warmup-epochs "${HEAD_WARMUP_EPOCHS:-5}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
fi

if want panel; then
    hdr "[Stage 3.panel] RADIANT-only panel sweep"
    [[ "${DEVICE}" == "cuda" ]] || die "RADIANT panel sweep requires CUDA; got DEVICE=${DEVICE}"
    FINETUNE_CKPT="${FINETUNE_CKPT:-${ACTIVITY_PRETRAIN_DIR}/backbone_for_finetune.pt}"
    require_file "${FINETUNE_CKPT}" "Stage 2 backbone checkpoint"
    py -m radiant_qsar.finetune.sweep \
        --panel "${DATA_DIR}/panel.json" \
        --activities "${DATA_DIR}/activities.parquet" \
        --vocab "${DATA_DIR}/smiles_vocab.json" \
        --config "${PRETRAIN_CONFIG}" \
        --pretrain-ckpt "${FINETUNE_CKPT}" \
        --out "${PANEL_ROOT}" \
        --splits ${PANEL_SPLITS:-random scaffold time cluster activity_cliff} \
        --models radiant \
        --epochs "${FINETUNE_EPOCHS:-100}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
fi

if want ablate; then
    hdr "[Ablations] Pure-RADIANT ablation sweep"
    [[ "${DEVICE}" == "cuda" ]] || die "Ablation sweep requires CUDA; got DEVICE=${DEVICE}"
    FINETUNE_CKPT="${FINETUNE_CKPT:-${ACTIVITY_PRETRAIN_DIR}/backbone_for_finetune.pt}"
    require_file "${FINETUNE_CKPT}" "Stage 2 backbone checkpoint"
    py -m radiant_qsar.finetune.sweep \
        --panel "${DATA_DIR}/panel.json" \
        --activities "${DATA_DIR}/activities.parquet" \
        --vocab "${DATA_DIR}/smiles_vocab.json" \
        --config "${PRETRAIN_CONFIG}" \
        --pretrain-ckpt "${FINETUNE_CKPT}" \
        --out "${PANEL_ROOT}" \
        --splits ${ABLATION_SPLITS:-scaffold activity_cliff} \
        --models ${ABLATION_MODELS:-radiant_no_halting radiant_no_anchor radiant_no_adapter radiant_no_depth_pool radiant_fixed_loops radiant_no_smiles_aug radiant_linear_head} \
        --epochs "${FINETUNE_EPOCHS:-100}" \
        --device "${DEVICE}" \
        --seed "${SEED}"
fi

if want f; then
    hdr "[Phase F] Morgan/RF baseline headline cell"
    py -m radiant_qsar.baselines.morgan_rf \
        --activities "${DATA_DIR}/activities.parquet" \
        --target "${HEADLINE_TARGET}" \
        --split "${HEADLINE_SPLIT}" \
        --out "${PANEL_ROOT}/morgan_rf/${HEADLINE_TARGET}/${HEADLINE_SPLIT}" \
        --n-estimators "${RF_N_ESTIMATORS:-500}" \
        --n-jobs "${RF_N_JOBS:--1}" \
        --seed "${SEED}"
fi

if want select; then
    hdr "[Model selection] Choosing validated RADIANT checkpoint for screening"
    py -m radiant_qsar.finetune.select_checkpoint \
        --panel-root "${PANEL_ROOT}" \
        --target "${HEADLINE_TARGET}" \
        --split "${HEADLINE_SPLIT}" \
        --metric pearson \
        --vocab "${DATA_DIR}/smiles_vocab.json" \
        --out "${SELECTED_MODEL_MANIFEST}"
fi

if want leak; then
    hdr "[Leakage audit] Exact/scaffold/time/similarity checks"
    py -m radiant_qsar.analyses.leakage_audit \
        --panel "${DATA_DIR}/panel.json" \
        --activities "${DATA_DIR}/activities.parquet" \
        --out-dir "${RUNS_DIR}/leakage_audit" \
        --splits ${PANEL_SPLITS:-random scaffold time cluster activity_cliff} \
        --seed "${SEED}"
fi

if want cal; then
    hdr "[Calibration] Building Phase G.3 long CSV"
    CAL_ARGS=(
        -m radiant_qsar.analyses.build_calibration_input
        --panel-root "${PANEL_ROOT}"
        --out "${PHASE_G_OUT}/_inputs/calibration_long.csv"
        --model radiant
        --split "${CALIBRATION_SPLIT:-scaffold}"
    )
    if [[ -n "${CALIBRATION_ENSEMBLE_PREFIX:-}" ]]; then
        CAL_ARGS+=(--ensemble-prefix "${CALIBRATION_ENSEMBLE_PREFIX}")
    fi
    py "${CAL_ARGS[@]}"
fi

if want stats; then
    hdr "[Statistics] Matched-cell significance tests"
    require_file "${PANEL_ROOT}/panel_results.csv" "panel sweep aggregate"
    py -m radiant_qsar.analyses.statistical_significance \
        --panel-results "${PANEL_ROOT}/panel_results.csv" \
        --out-dir "${RUNS_DIR}/statistics" \
        --primary radiant
fi

if want rgsar; then
    hdr "[Analysis] R-group SAR"
    py -m radiant_qsar.analyses.g_rgroup_sar \
        --panel-root "${PANEL_ROOT}" \
        --out-dir "${PHASE_G_OUT}" \
        --model "${RGROUP_SAR_MODEL:-radiant}" \
        --split "${RGROUP_SAR_SPLIT:-scaffold}" \
        --min-abs-true-delta "${RGROUP_SAR_MIN_DELTA:-0.3}" \
        --max-pairs-per-scaffold "${RGROUP_SAR_MAX_PAIRS:-250}" \
        --seed "${SEED}"
fi

if want cliffsar; then
    hdr "[Analysis] Activity-cliff SAR"
    py -m radiant_qsar.analyses.g_activity_cliff_sar \
        --panel-root "${PANEL_ROOT}" \
        --out-dir "${PHASE_G_OUT}" \
        --model "${CLIFF_SAR_MODEL:-radiant}" \
        --split "${CLIFF_SAR_SPLIT:-activity_cliff}" \
        --tanimoto-threshold "${CLIFF_SAR_TANIMOTO:-0.55}" \
        --activity-delta-threshold "${CLIFF_SAR_DELTA:-1.0}" \
        --max-pairs-per-cell "${CLIFF_SAR_MAX_PAIRS:-2000}" \
        --seed "${SEED}"
fi

if want failures; then
    hdr "[Analysis] Failure modes"
    py -m radiant_qsar.analyses.g_failure_modes \
        --panel-root "${PANEL_ROOT}" \
        --out-dir "${PHASE_G_OUT}" \
        --model "${FAILURE_MODEL:-radiant}" \
        --top-n "${FAILURE_TOP_N:-100}" \
        --min-scaffold-n "${FAILURE_MIN_SCAFFOLD_N:-3}"
fi

if want stage1probe; then
    hdr "[Analysis] Stage-1 representation probe"
    require_file "${STAGE1_EMBEDDINGS_CSV:?set STAGE1_EMBEDDINGS_CSV=runs/phase_g/_inputs/stage1_embeddings.csv}" "Stage-1 embedding CSV"
    py -m radiant_qsar.analyses.g_stage1_representation_probe \
        --embeddings-csv "${STAGE1_EMBEDDINGS_CSV}" \
        --out-dir "${PHASE_G_OUT}" \
        --k "${STAGE1_PROBE_K:-5}" \
        --seed "${SEED}"
fi

if want rgroupabl; then
    hdr "[Analysis] R-group chemistry ablation"
    require_file "${PANEL_ROOT}/panel_results.csv" "panel sweep aggregate"
    py -m radiant_qsar.analyses.g_rgroup_ablation \
        --panel-results "${PANEL_ROOT}/panel_results.csv" \
        --out-dir "${PHASE_G_OUT}" \
        --primary "${RGROUP_ABL_PRIMARY:-radiant}" \
        --ablations ${RGROUP_ABLATIONS:-radiant_no_stage1_rgroup radiant_no_stage2_rgroup radiant_no_rgroup}
fi

if want g; then
    hdr "[Phase G] Compute-aware analyses -> ${PHASE_G_OUT}"
    require_file "${PHASE_G_CONFIG}" "Phase G config"
    py -m radiant_qsar.analyses.run_phase_g --config "${PHASE_G_CONFIG}"
fi

if want h; then
    hdr "[Phase H] Library filtering + RADIANT potency screening"
    require_file "${INPUT_LIBRARY:?set INPUT_LIBRARY=path/to/library.smi}" "screening library"
    require_file "${SELECTED_MODEL_MANIFEST}" "selected RADIANT model manifest"
    mkdir -p "${RUNS_DIR}/screening"
    py -m radiant_qsar.screening.prepare_library \
        --input "${INPUT_LIBRARY}" \
        --output "${RUNS_DIR}/screening/prepared.smi" \
        --profile "${SCREEN_PROFILE}" \
        --rejects "${RUNS_DIR}/screening/rejected.csv" \
        --audit "${RUNS_DIR}/screening/audit.csv" \
        --summary "${RUNS_DIR}/screening/summary.json"
    SELECTED_MODEL_MANIFEST="${SELECTED_MODEL_MANIFEST}" \
    SCREENING_MIN_PCHEMBL="${SCREENING_MIN_PCHEMBL}" \
    RUNS_DIR="${RUNS_DIR}" \
    py - <<'PY'
import json
import os
from pathlib import Path

from radiant_qsar.screening import Pipeline
from radiant_qsar.screening.filters.ml_scoring import RADIANTPotency

manifest = json.loads(Path(os.environ["SELECTED_MODEL_MANIFEST"]).read_text())
runs_dir = Path(os.environ["RUNS_DIR"])
vocab = manifest.get("vocab_path") or "data/processed/v1/smiles_vocab.json"
pipe = Pipeline([
    RADIANTPotency(
        checkpoint_path=manifest["checkpoint_path"],
        vocab_path=vocab,
        min_pchembl=float(os.environ["SCREENING_MIN_PCHEMBL"]),
        n_loops=8,
        task_name=manifest.get("task_name", "pchembl"),
    )
])
pipe.run(
    runs_dir / "screening" / "prepared.smi",
    runs_dir / "screening" / "radiant_hits.smi",
    rejects_path=runs_dir / "screening" / "radiant_rejected.csv",
    audit_path=runs_dir / "screening" / "radiant_audit.csv",
    summary_path=runs_dir / "screening" / "radiant_summary.json",
)
PY
fi

hdr "Done. See ${RUNS_DIR}/, ${PANEL_ROOT}/, and ${PHASE_G_OUT}/ for outputs."
