#!/bin/bash
#SBATCH --job-name=mlaad_baseline_refset
#SBATCH --account=dd-25-3
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the multiclass FF checkpoint}
REF_SIZE=${REF_SIZE:-5}
NEG_PER_TEST=${NEG_PER_TEST:-1}
BATCH_SIZE=${BATCH_SIZE:-12}
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
AMP_EVAL=${AMP_EVAL:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-4}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
OUTPUT_DIR="runs/karolina_baseline_refset_${RUN_TIMESTAMP}"
OUTPUT_CSV="${OUTPUT_DIR}/scores.csv"

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
LOG_FILE="${LOG_DIR}/karolina_baseline_refset_${RUN_TIMESTAMP}.eval.log"

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

echo "[$(date)] Starting MLAAD reference-set verification." | tee -a "${LOG_FILE}"

export PYTHONUNBUFFERED=1
srun -u python -u baselines/baseline_mlaad_multiclass_cosine.py \
  --karolina \
  -d MLAADCuratedDataset_pair \
  -e XLSR_300M \
  -p MHFA \
  -c FF \
  --checkpoint "${CHECKPOINT_PATH}" \
  --reference-size "${REF_SIZE}" \
  --negatives-per-test "${NEG_PER_TEST}" \
  --score-reduction max \
  --segment-seconds "${SEGMENT_SECONDS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  ${PIN_MEMORY_FLAG} \
  ${PERSISTENT_WORKERS_FLAG} \
  ${AMP_EVAL_FLAG} \
  "${AMP_DTYPE_FLAG[@]}" \
  --output-csv "${OUTPUT_CSV}" |& tee -a "${LOG_FILE}"

echo "[$(date)] Running eval_pair_model.py on ${OUTPUT_CSV}." | tee -a "${LOG_FILE}"
srun -u python -u eval_pair_model.py \
  --karolina \
  -d MLAADCuratedDataset_pair \
  -e XLSR_300M \
  -p MHFA \
  -c FF \
  --scores-in "${OUTPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed 42 |& tee -a "${LOG_FILE}"

echo "[$(date)] Job finished." | tee -a "${LOG_FILE}"
