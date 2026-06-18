#!/bin/bash
#
# Helper script to submit MLAAD Stage 3 directional training jobs with different seeds.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_seed_template.sh"
LOG_DIR="${SCRIPT_DIR}/../logs"
mkdir -p "${LOG_DIR}"

# Default seed if not provided per job entry.
DEFAULT_SEED=42

# DATASET | PROCESSOR | CLASSIFIER | WALLTIME | CHECKPOINT(optional) | START_EPOCH(optional) | SEED(optional)
JOBS=(
  # "MLAADCurriculumDirectionalFFCosineDataset_pair|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed42_20251204_172405/FFCosine_25.pt"
  "MLAADCurriculumDirectionalFFCosineDataset_pair_s123|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed123_20251204_172405/FFCosine_10.pt|11|123"
  "MLAADCurriculumDirectionalFFCosineDataset_pair_s222|MHFA|FFCosine|30:00:00|/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_MHFA_FFCosine_seed222_20251204_172404/FFCosine_30.pt|31|222"
)

for ENTRY in "${JOBS[@]}"; do
  IFS="|" read -r DATASET PROCESSOR CLASSIFIER WALLTIME CHECKPOINT START_EPOCH_OVERRIDE SEED_OVERRIDE <<< "${ENTRY}"
  if [[ -n "${SEED_OVERRIDE}" ]]; then
    SEED="${SEED_OVERRIDE}"
  else
    SEED="${DEFAULT_SEED}"
  fi
  JOB_SCRIPT="${SCRIPT_DIR}/train_seed_directional_${PROCESSOR}_${CLASSIFIER}_${SEED}.sh"
  sed -e "s/{{SEED}}/${SEED}/g" \
      -e "s/{{DATASET}}/${DATASET}/g" \
      -e "s/{{WALLTIME}}/${WALLTIME}/g" \
      -e "s/{{PROCESSOR}}/${PROCESSOR}/g" \
      -e "s/{{CLASSIFIER}}/${CLASSIFIER}/g" \
      -e "s/{{MODE}}/curriculum_directional/g" \
      "${TEMPLATE}" > "${JOB_SCRIPT}"
  chmod +x "${JOB_SCRIPT}"
  echo "Submitting Stage 3 directional on Karolina (${PROCESSOR}+${CLASSIFIER}) with seed ${SEED} (walltime ${WALLTIME})..."
  if [[ -n "${CHECKPOINT}" ]]; then
    if [[ -n "${START_EPOCH_OVERRIDE}" ]]; then
      START_EPOCH="${START_EPOCH_OVERRIDE}"
    else
      START_EPOCH=26
    fi
    sbatch --export=ALL,CHECKPOINT="${CHECKPOINT}",NUM_EPOCHS=1,START_EPOCH="${START_EPOCH}",LR_EPOCH_MULTS="1:0.01",RUN_EVAL_AFTER_TRAIN=1 "${JOB_SCRIPT}"
  else
    sbatch --export=ALL,NUM_EPOCHS=1,START_EPOCH=26,LR_EPOCH_MULTS="1:0.01",RUN_EVAL_AFTER_TRAIN=1 "${JOB_SCRIPT}"
  fi
done
