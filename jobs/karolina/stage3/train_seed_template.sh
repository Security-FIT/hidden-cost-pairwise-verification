#!/bin/bash
#SBATCH --job-name=mlaad_stage3_{{MODE}}_{{PROCESSOR}}_{{CLASSIFIER}}_seed{{SEED}}
#SBATCH --account=dd-25-3
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time={{WALLTIME}}

set -euo pipefail

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16} # 4 for attn models
DEV_BATCH_SIZE=${DEV_BATCH_SIZE:-4} # 3 for attn models
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
DEV_NUM_WORKERS=${DEV_NUM_WORKERS:-8}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
DEV_PREFETCH_FACTOR=${DEV_PREFETCH_FACTOR:-4}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
AMP_EVAL=${AMP_EVAL:-1}
AMP_TRAIN=${AMP_TRAIN:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
MAX_TRAIN_BATCHES=${MAX_TRAIN_BATCHES:-0}
NUM_EPOCHS=${NUM_EPOCHS:-30}
START_EPOCH=${START_EPOCH:-1}
LR_EPOCH_MULTS=${LR_EPOCH_MULTS:-1:1.0}
VAL_INTERVAL=${VAL_INTERVAL:-5}
LR_RAMP=${LR_RAMP:-0}
LR_RAMP_START_MULT=${LR_RAMP_START_MULT:-0.05}
LR_RAMP_TARGET_MULT=${LR_RAMP_TARGET_MULT:-1.0}
LR_RAMP_STEPS=${LR_RAMP_STEPS:-500}
CURRICULUM_EASY_CSV=${CURRICULUM_EASY_CSV:-}
CURRICULUM_HARD_CSV=${CURRICULUM_HARD_CSV:-}
CURRICULUM_STEPS_PER_EPOCH=${CURRICULUM_STEPS_PER_EPOCH:-0}
CURRICULUM_THREE_STREAM=${CURRICULUM_THREE_STREAM:-1}
CURRICULUM_HARD_NEG_RATIO=${CURRICULUM_HARD_NEG_RATIO:-0.5}
CURRICULUM_PAIRS_PER_EPOCH=${CURRICULUM_PAIRS_PER_EPOCH:-44000}
CHECKPOINT=${CHECKPOINT:-}
RUN_EVAL_AFTER_TRAIN=${RUN_EVAL_AFTER_TRAIN:-0}
FINETUNE_SSL=${FINETUNE_SSL:-0}
EXTRACTOR_LR=${EXTRACTOR_LR:-}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-}
SAMPLE_RATE=${SAMPLE_RATE:-16000}

if [[ "${PIN_MEMORY}" -eq 0 ]]; then
  PIN_MEMORY_FLAG="--no-pin-memory"
else
  PIN_MEMORY_FLAG="--pin-memory"
fi

if [[ "${PERSISTENT_WORKERS}" -eq 0 ]]; then
  PERSISTENT_WORKERS_FLAG="--no-persistent-workers"
else
  PERSISTENT_WORKERS_FLAG="--persistent-workers"
fi

if [[ "${AMP_EVAL}" -eq 1 ]]; then
  AMP_EVAL_FLAG="--amp-eval"
else
  AMP_EVAL_FLAG="--no-amp-eval"
fi
if [[ "${AMP_TRAIN}" -eq 1 ]]; then
  AMP_TRAIN_FLAG="--amp-train"
else
  AMP_TRAIN_FLAG="--no-amp-train"
fi
AMP_DTYPE_FLAG=(--amp-dtype "${AMP_DTYPE}")

if [[ "${MAX_TRAIN_BATCHES}" -gt 0 ]]; then
  MAX_BATCHES_FLAG=(--max-train-batches "${MAX_TRAIN_BATCHES}")
else
  MAX_BATCHES_FLAG=()
fi

if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_FLAG=(--checkpoint "${CHECKPOINT}")
else
  CHECKPOINT_FLAG=()
fi

if [[ "${FINETUNE_SSL}" -eq 1 ]]; then
  FINETUNE_SSL_FLAG=(--finetune-ssl)
else
  FINETUNE_SSL_FLAG=()
fi

if [[ -n "${EXTRACTOR_LR}" ]]; then
  EXTRACTOR_LR_FLAG=(--extractor-lr "${EXTRACTOR_LR}")
else
  EXTRACTOR_LR_FLAG=()
fi

if [[ -n "${SEGMENT_SECONDS}" ]]; then
  SEGMENT_FLAG=(--segment-seconds "${SEGMENT_SECONDS}" --sample-rate "${SAMPLE_RATE}")
else
  SEGMENT_FLAG=()
fi

if [[ "${LR_RAMP}" -eq 1 ]]; then
  LR_RAMP_FLAG=(
    --lr-ramp
    --lr-ramp-start-mult "${LR_RAMP_START_MULT}"
    --lr-ramp-target-mult "${LR_RAMP_TARGET_MULT}"
    --lr-ramp-steps "${LR_RAMP_STEPS}"
  )
else
  LR_RAMP_FLAG=()
fi

CURRICULUM_FLAG=()
if [[ -n "${CURRICULUM_EASY_CSV}" || -n "${CURRICULUM_HARD_CSV}" ]]; then
  if [[ -z "${CURRICULUM_EASY_CSV}" || -z "${CURRICULUM_HARD_CSV}" ]]; then
    echo "Both CURRICULUM_EASY_CSV and CURRICULUM_HARD_CSV must be set." >&2
    exit 1
  fi
  CURRICULUM_FLAG=(
    --curriculum-easy-csv "${CURRICULUM_EASY_CSV}"
    --curriculum-hard-csv "${CURRICULUM_HARD_CSV}"
  )
  if [[ "${CURRICULUM_THREE_STREAM}" -eq 1 ]]; then
    CURRICULUM_FLAG+=(--curriculum-three-stream)
  fi
  if [[ "${CURRICULUM_STEPS_PER_EPOCH}" -gt 0 ]]; then
    CURRICULUM_FLAG+=(--curriculum-steps-per-epoch "${CURRICULUM_STEPS_PER_EPOCH}")
  fi
  if [[ -n "${CURRICULUM_HARD_NEG_RATIO}" ]]; then
    CURRICULUM_FLAG+=(--curriculum-hard-neg-ratio "${CURRICULUM_HARD_NEG_RATIO}")
  fi
  if [[ "${CURRICULUM_PAIRS_PER_EPOCH}" -gt 0 ]]; then
    CURRICULUM_FLAG+=(--curriculum-pairs-per-epoch "${CURRICULUM_PAIRS_PER_EPOCH}")
  fi
fi

module purge
ml CUDA
ml Anaconda3/2024.02-1

if [ -n "${ANACONDA_DIR:-}" ] && [ -f "${ANACONDA_DIR}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${ANACONDA_DIR}/etc/profile.d/conda.sh"
elif [ -n "${EBROOTANACONDA3:-}" ] && [ -f "${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${EBROOTANACONDA3}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(command conda 'shell.bash' 'hook')"
else
  echo "Unable to locate conda.sh to activate environment." >&2
  exit 1
fi

conda activate inf_st

PROJECT_ROOT="/scratch/project/dd-25-3/DFS-Detection-Framework"
LOG_DIR="${PROJECT_ROOT}/jobs/karolina/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/karolina_stage3_{{MODE}}_{{PROCESSOR}}_{{CLASSIFIER}}_seed{{SEED}}_${RUN_TIMESTAMP}.train.log"
OUTPUT_DIR="runs/karolina_stage3_{{MODE}}_{{PROCESSOR}}_{{CLASSIFIER}}_seed{{SEED}}_${RUN_TIMESTAMP}"

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

echo "[$(date)] Starting MLAAD Stage 3 {{MODE}} job on Karolina for {{PROCESSOR}} + {{CLASSIFIER}} (seed {{SEED}})" | tee -a "${LOG_FILE}"
if [[ "${LR_RAMP}" -eq 1 ]]; then
  echo "[$(date)] LR ramp enabled: ${LR_RAMP_START_MULT}x -> ${LR_RAMP_TARGET_MULT}x over ${LR_RAMP_STEPS} optimizer steps." | tee -a "${LOG_FILE}"
else
  echo "[$(date)] LR ramp disabled." | tee -a "${LOG_FILE}"
fi
if [[ "${FINETUNE_SSL}" -eq 1 ]]; then
  if [[ -n "${EXTRACTOR_LR}" ]]; then
    echo "[$(date)] XLSR fine-tuning enabled (extractor LR ${EXTRACTOR_LR})." | tee -a "${LOG_FILE}"
  else
    echo "[$(date)] XLSR fine-tuning enabled (extractor LR default)." | tee -a "${LOG_FILE}"
  fi
else
  echo "[$(date)] XLSR fine-tuning disabled." | tee -a "${LOG_FILE}"
fi
if [[ -n "${SEGMENT_SECONDS}" ]]; then
  echo "[$(date)] Segment cap enabled: ${SEGMENT_SECONDS}s @ ${SAMPLE_RATE} Hz." | tee -a "${LOG_FILE}"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun -u python -u train_and_eval.py \
  --karolina \
  -d {{DATASET}} \
  -e XLSR_300M \
  -p {{PROCESSOR}} \
  -c {{CLASSIFIER}} \
  --num_epochs "${NUM_EPOCHS}" \
  --start-epoch "${START_EPOCH}" \
  --val_interval "${VAL_INTERVAL}" \
  --skip_eval \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --dev-batch-size "${DEV_BATCH_SIZE}" \
  --train-num-workers "${TRAIN_NUM_WORKERS}" \
  --dev-num-workers "${DEV_NUM_WORKERS}" \
  --train-prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  --dev-prefetch-factor "${DEV_PREFETCH_FACTOR}" \
  ${PIN_MEMORY_FLAG} \
  ${PERSISTENT_WORKERS_FLAG} \
  ${AMP_TRAIN_FLAG} \
  ${AMP_EVAL_FLAG} \
  "${AMP_DTYPE_FLAG[@]}" \
  "${MAX_BATCHES_FLAG[@]}" \
  "${CHECKPOINT_FLAG[@]}" \
  "${FINETUNE_SSL_FLAG[@]}" \
  "${EXTRACTOR_LR_FLAG[@]}" \
  "${SEGMENT_FLAG[@]}" \
  "${LR_RAMP_FLAG[@]}" \
  --lr-epoch-mults "${LR_EPOCH_MULTS}" \
  "${CURRICULUM_FLAG[@]}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed {{SEED}} |& tee -a "${LOG_FILE}"

if [[ "${RUN_EVAL_AFTER_TRAIN}" -eq 1 ]]; then
  LAST_EPOCH=$((START_EPOCH + NUM_EPOCHS - 1))
  EVAL_CHECKPOINT="${OUTPUT_DIR}/{{CLASSIFIER}}_${LAST_EPOCH}.pt"
  if [[ -f "${EVAL_CHECKPOINT}" ]]; then
    echo "[$(date)] Running eval_pair_model.py on ${EVAL_CHECKPOINT}" | tee -a "${LOG_FILE}"
    srun -u python -u eval_pair_model.py \
      --karolina \
      -d {{DATASET}} \
      -e XLSR_300M \
      -p {{PROCESSOR}} \
      -c {{CLASSIFIER}} \
      --checkpoint "${EVAL_CHECKPOINT}" \
      --output_dir "${OUTPUT_DIR}" \
      --train-batch-size "${TRAIN_BATCH_SIZE}" \
      --dev-batch-size "${DEV_BATCH_SIZE}" \
      --train-num-workers "${TRAIN_NUM_WORKERS}" \
      --dev-num-workers "${DEV_NUM_WORKERS}" \
      --train-prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
      --dev-prefetch-factor "${DEV_PREFETCH_FACTOR}" \
      ${PIN_MEMORY_FLAG} \
      ${PERSISTENT_WORKERS_FLAG} \
      ${AMP_EVAL_FLAG} \
      "${AMP_DTYPE_FLAG[@]}" \
      "${SEGMENT_FLAG[@]}" \
      --seed {{SEED}} |& tee -a "${LOG_FILE}"
  else
    echo "[$(date)] Warning: checkpoint ${EVAL_CHECKPOINT} not found; skipping eval." | tee -a "${LOG_FILE}"
  fi
fi

echo "[$(date)] Job finished." | tee -a "${LOG_FILE}"
