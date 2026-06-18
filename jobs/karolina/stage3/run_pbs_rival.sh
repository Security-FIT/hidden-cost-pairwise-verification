#!/bin/bash
#
# Helper script to submit MLAAD Stage 3 rival curriculum training jobs with different seeds.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_seed_template.sh"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

# Default seed if not provided per job entry.
DEFAULT_SEED=42
# Default extractor fine-tuning mode (0 = frozen XLSR, 1 = finetune XLSR).
DEFAULT_FINETUNE_SSL=0
# Optional extractor LR when fine-tuning; empty => parser default (1e-6).
DEFAULT_EXTRACTOR_LR=
# Cap training/eval samples to fixed segments for this rival run.
DEFAULT_SEGMENT_SECONDS=4
DEFAULT_SAMPLE_RATE=16000

# DATASET | PROCESSOR | CLASSIFIER | WALLTIME | CHECKPOINT(optional) | START_EPOCH(optional) | SEED(optional) | FINETUNE_SSL(optional 0/1) | EXTRACTOR_LR(optional)
JOBS=(
  # "MLAADCurriculumRivalFFCosineDataset_pair|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed42_20251204_172405/FFCosine_25.pt|26|42|1|1e-6"
  "MLAADCurriculumRivalFFCosineDataset_pair_s123|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed123_20251204_172405/FFCosine_10.pt|11|123|1|1e-6"
  "MLAADCurriculumRivalFFCosineDataset_pair_s222|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed222_20251204_172404/FFCosine_30.pt|31|222|1|1e-6"
)

for ENTRY in "${JOBS[@]}"; do
  IFS="|" read -r DATASET PROCESSOR CLASSIFIER WALLTIME CHECKPOINT START_EPOCH_OVERRIDE SEED_OVERRIDE FINETUNE_SSL_OVERRIDE EXTRACTOR_LR_OVERRIDE <<< "${ENTRY}"
  if [[ -n "${SEED_OVERRIDE}" ]]; then
    SEED="${SEED_OVERRIDE}"
  else
    SEED="${DEFAULT_SEED}"
  fi
  if [[ -n "${FINETUNE_SSL_OVERRIDE}" ]]; then
    FINETUNE_SSL="${FINETUNE_SSL_OVERRIDE}"
  else
    FINETUNE_SSL="${DEFAULT_FINETUNE_SSL}"
  fi
  if [[ "${FINETUNE_SSL}" != "0" && "${FINETUNE_SSL}" != "1" ]]; then
    echo "Invalid FINETUNE_SSL value '${FINETUNE_SSL}' in JOBS entry '${ENTRY}'. Expected 0 or 1." >&2
    exit 1
  fi
  if [[ -n "${EXTRACTOR_LR_OVERRIDE}" ]]; then
    EXTRACTOR_LR="${EXTRACTOR_LR_OVERRIDE}"
  else
    EXTRACTOR_LR="${DEFAULT_EXTRACTOR_LR}"
  fi
  JOB_SCRIPT="${SCRIPT_DIR}/train_seed_rival_${PROCESSOR}_${CLASSIFIER}_${SEED}.sh"
  sed -e "s/{{SEED}}/${SEED}/g" \
      -e "s/{{DATASET}}/${DATASET}/g" \
      -e "s/{{WALLTIME}}/${WALLTIME}/g" \
      -e "s/{{PROCESSOR}}/${PROCESSOR}/g" \
      -e "s/{{CLASSIFIER}}/${CLASSIFIER}/g" \
      -e "s/{{MODE}}/curriculum_rival/g" \
      "${TEMPLATE}" > "${JOB_SCRIPT}"
  chmod +x "${JOB_SCRIPT}"
  XLSR_MODE="frozen"
  if [[ "${FINETUNE_SSL}" -eq 1 ]]; then
    if [[ -n "${EXTRACTOR_LR}" ]]; then
      XLSR_MODE="finetune (extractor_lr=${EXTRACTOR_LR})"
    else
      XLSR_MODE="finetune (extractor_lr=default)"
    fi
  fi
  echo "Submitting Stage 3 rival curriculum on Karolina (${PROCESSOR}+${CLASSIFIER}) with seed ${SEED} (walltime ${WALLTIME}; XLSR ${XLSR_MODE})..."
  if [[ -n "${CHECKPOINT}" ]]; then
    if [[ -n "${START_EPOCH_OVERRIDE}" ]]; then
      START_EPOCH="${START_EPOCH_OVERRIDE}"
    else
      START_EPOCH=26
    fi
    EXPORT_VARS="ALL,CHECKPOINT=${CHECKPOINT},NUM_EPOCHS=1,START_EPOCH=${START_EPOCH},LR_EPOCH_MULTS=1:0.01,VAL_INTERVAL=1,RUN_EVAL_AFTER_TRAIN=0,FINETUNE_SSL=${FINETUNE_SSL},SEGMENT_SECONDS=${DEFAULT_SEGMENT_SECONDS},SAMPLE_RATE=${DEFAULT_SAMPLE_RATE}"
    if [[ -n "${EXTRACTOR_LR}" ]]; then
      EXPORT_VARS="${EXPORT_VARS},EXTRACTOR_LR=${EXTRACTOR_LR}"
    fi
    sbatch --export="${EXPORT_VARS}" "${JOB_SCRIPT}"
  else
    EXPORT_VARS="ALL,NUM_EPOCHS=1,START_EPOCH=26,LR_EPOCH_MULTS=1:0.01,VAL_INTERVAL=1,RUN_EVAL_AFTER_TRAIN=0,FINETUNE_SSL=${FINETUNE_SSL},SEGMENT_SECONDS=${DEFAULT_SEGMENT_SECONDS},SAMPLE_RATE=${DEFAULT_SAMPLE_RATE}"
    if [[ -n "${EXTRACTOR_LR}" ]]; then
      EXPORT_VARS="${EXPORT_VARS},EXTRACTOR_LR=${EXTRACTOR_LR}"
    fi
    sbatch --export="${EXPORT_VARS}" "${JOB_SCRIPT}"
  fi
done
