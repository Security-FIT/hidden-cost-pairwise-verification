#!/bin/bash
#SBATCH --job-name=mlaad_baseline_ff_multiclass_ft_xlsr_s222
#SBATCH --account=dd-25-3
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SEED=${SEED:-222}
NUM_EPOCHS=${NUM_EPOCHS:-5}
VAL_INTERVAL=${VAL_INTERVAL:-1}
PATIENCE=${PATIENCE:-10}
HEAD_LR=${HEAD_LR:-1e-5}
EXTRACTOR_LR=${EXTRACTOR_LR:-1e-6}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-4}
SAMPLE_RATE=${SAMPLE_RATE:-16000}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
DEV_BATCH_SIZE=${DEV_BATCH_SIZE:-64}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-4}
DEV_NUM_WORKERS=${DEV_NUM_WORKERS:-4}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
DEV_PREFETCH_FACTOR=${DEV_PREFETCH_FACTOR:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
AMP_EVAL=${AMP_EVAL:-1}
AMP_TRAIN=${AMP_TRAIN:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}

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
LOG_FILE="${LOG_DIR}/karolina_baseline_ff_multiclass_s_${SEED}_finetune_xlsr_${RUN_TIMESTAMP}.train.log"

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

DEFAULT_CHECKPOINT="/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_baseline_ff_multiclass_s_222_20260202_210158/best_model.pth"
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}
SOURCE_RUN_DIR=${SOURCE_RUN_DIR:-}

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -z "${SOURCE_RUN_DIR}" ]]; then
    RUN_PATTERN="runs/karolina_baseline_ff_multiclass_s_${SEED}_*"
    SOURCE_RUN_DIR=$(ls -td ${RUN_PATTERN} 2>/dev/null | head -n 1 || true)
  fi

  if [[ -z "${SOURCE_RUN_DIR}" ]]; then
    echo "Unable to find source run directory. Set SOURCE_RUN_DIR or CHECKPOINT explicitly." >&2
    exit 1
  fi

  if [[ -f "${SOURCE_RUN_DIR}/best_model.pth" ]]; then
    CHECKPOINT="${SOURCE_RUN_DIR}/best_model.pth"
  else
    CHECKPOINT=$(ls -t "${SOURCE_RUN_DIR}/FF_"*.pt 2>/dev/null | head -n 1 || true)
  fi
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found. Resolved value: '${CHECKPOINT}'" >&2
  exit 1
fi

echo "[$(date)] Starting MLAAD multiclass FF fine-tuning with XLSR unfrozen." | tee -a "${LOG_FILE}"
echo "[$(date)] Using checkpoint: ${CHECKPOINT}" | tee -a "${LOG_FILE}"
echo "[$(date)] Seed=${SEED}, epochs=${NUM_EPOCHS}, val_interval=${VAL_INTERVAL}, head_lr=${HEAD_LR}, extractor_lr=${EXTRACTOR_LR}" | tee -a "${LOG_FILE}"
echo "[$(date)] Train batch=${TRAIN_BATCH_SIZE}, grad_accum=${GRAD_ACCUM_STEPS}, effective batch=$((TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS)), dev batch=${DEV_BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "[$(date)] Segment cap: ${SEGMENT_SECONDS}s @ ${SAMPLE_RATE} Hz" | tee -a "${LOG_FILE}"

# Guard against conflicting SLURM vars leaked from submit environment
# (common when sbatch is launched from an interactive allocation).
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && -n "${SLURM_TRES_PER_TASK:-}" && "${SLURM_TRES_PER_TASK}" == *"cpu:"* ]]; then
  TRES_CPU="${SLURM_TRES_PER_TASK#*cpu:}"
  TRES_CPU="${TRES_CPU%%,*}"
  if [[ -n "${TRES_CPU}" && "${TRES_CPU}" != "${SLURM_CPUS_PER_TASK}" ]]; then
    echo "[$(date)] Detected conflicting SLURM CPU vars (${SLURM_CPUS_PER_TASK} vs ${SLURM_TRES_PER_TASK}); unsetting leaked values." | tee -a "${LOG_FILE}"
    unset SLURM_CPUS_PER_TASK
    unset SLURM_TRES_PER_TASK
  fi
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun -u python -u train_and_eval.py \
  --karolina \
  -d MLAADDataset_single \
  -e XLSR_300M \
  -p MHFA \
  -c FF \
  --checkpoint "${CHECKPOINT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --val_interval "${VAL_INTERVAL}" \
  --stop_on_plateau \
  --patience "${PATIENCE}" \
  --skip_eval \
  --segment-seconds "${SEGMENT_SECONDS}" \
  --sample-rate "${SAMPLE_RATE}" \
  --finetune-ssl \
  --extractor-lr "${EXTRACTOR_LR}" \
  --lr "${HEAD_LR}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --dev-batch-size "${DEV_BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --train-num-workers "${TRAIN_NUM_WORKERS}" \
  --dev-num-workers "${DEV_NUM_WORKERS}" \
  --train-prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  --dev-prefetch-factor "${DEV_PREFETCH_FACTOR}" \
  ${PIN_MEMORY_FLAG} \
  ${PERSISTENT_WORKERS_FLAG} \
  ${AMP_TRAIN_FLAG} \
  ${AMP_EVAL_FLAG} \
  "${AMP_DTYPE_FLAG[@]}" \
  --output_dir "runs/karolina_baseline_ff_multiclass_s_${SEED}_finetune_xlsr_${RUN_TIMESTAMP}" \
  --seed "${SEED}" |& tee -a "${LOG_FILE}"

echo "[$(date)] Job finished." | tee -a "${LOG_FILE}"
