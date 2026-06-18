#!/bin/bash
#SBATCH --job-name=mlaad_ff_embbneck13_s222
#SBATCH --account=dd-25-3
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
DEV_BATCH_SIZE=${DEV_BATCH_SIZE:-128}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}
DEV_NUM_WORKERS=${DEV_NUM_WORKERS:-8}
TRAIN_PREFETCH_FACTOR=${TRAIN_PREFETCH_FACTOR:-4}
DEV_PREFETCH_FACTOR=${DEV_PREFETCH_FACTOR:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
AMP_EVAL=${AMP_EVAL:-1}
AMP_TRAIN=${AMP_TRAIN:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
EMBEDDING_DIM=${EMBEDDING_DIM:-13}

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
LOG_FILE="${LOG_DIR}/karolina_baseline_ff_multiclass_embbneck13_s222_${RUN_TIMESTAMP}.train.log"

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

echo "[$(date)] Starting MLAAD multiclass FF baseline training." | tee -a "${LOG_FILE}"

echo "Using seed 222 for reproducibility." | tee -a "${LOG_FILE}"
echo "Using FF embedding_dim=${EMBEDDING_DIM}." | tee -a "${LOG_FILE}"

export PYTHONUNBUFFERED=1
srun -u python -u train_and_eval.py \
  --karolina \
  -d MLAADDataset_single \
  -e XLSR_300M \
  -p MHFA \
  -c FF \
  --embedding_dim "${EMBEDDING_DIM}" \
  --num_epochs 50 \
  --val_interval 1 \
  --stop_on_plateau \
  --patience 10 \
  --skip_eval \
  --segment-seconds 4 \
  --sample-rate 16000 \
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
  --output_dir runs/karolina_baseline_ff_multiclass_embbneck13_s222_${RUN_TIMESTAMP} \
  --seed 222 |& tee -a "${LOG_FILE}"

echo "[$(date)] Job finished." | tee -a "${LOG_FILE}"
