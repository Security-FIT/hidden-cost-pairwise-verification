#!/bin/bash
#
# Submit dev-only evaluation jobs on Karolina for specific Stage 2 checkpoints.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/scratch/project/dd-25-3/DFS-Detection-Framework"
LOG_DIR="${PROJECT_ROOT}/jobs/karolina/logs"
mkdir -p "${LOG_DIR}"

DATASET="MLAADIntermediateDataset_pair"
EXTRACTOR="XLSR_300M"
DEV_BATCH_SIZE=2        # attn models are memory hungry; keep micro-batch small
DEV_NUM_WORKERS=4       # stay below system recommendation to avoid excess workers
TRAIN_NUM_WORKERS=0     # avoid spawning unused train workers in dev-only runs
TRAIN_PREFETCH_FACTOR=0
DEV_PREFETCH_FACTOR=4
AMP_DTYPE="bf16"
WALLTIME="04:00:00"

declare -A CHECKPOINTS=(
  [AASIST_FFAttn3]="/scratch/project/dd-25-3/DFS-Detection-Framework/runs/karolina_stage2_intermediate_AASIST_FFAttn3_seed123_20251210_101222/FFAttn3_30.pt"
)

for KEY in "${!CHECKPOINTS[@]}"; do
  IFS="_" read -r PROCESSOR CLASSIFIER <<<"${KEY}"
  CHECKPOINT="${CHECKPOINTS[$KEY]}"
  RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
  JOB_SCRIPT="${SCRIPT_DIR}/dev_eval_${PROCESSOR}_${CLASSIFIER}_${RUN_TIMESTAMP}.sh"
  LOG_FILE="${LOG_DIR}/karolina_stage2_dev_${PROCESSOR}_${CLASSIFIER}_${RUN_TIMESTAMP}.log"
  OUTPUT_DIR="runs/dev_eval_${PROCESSOR}_${CLASSIFIER}_${RUN_TIMESTAMP}"

  cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash
#SBATCH --job-name=dev_eval_${PROCESSOR}_${CLASSIFIER}
#SBATCH --account=dd-25-3
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=${WALLTIME}

set -euo pipefail

module purge
ml CUDA
ml Anaconda3/2024.02-1

if [ -n "\${ANACONDA_DIR:-}" ] && [ -f "\${ANACONDA_DIR}/etc/profile.d/conda.sh" ]; then
  source "\${ANACONDA_DIR}/etc/profile.d/conda.sh"
elif [ -n "\${EBROOTANACONDA3:-}" ] && [ -f "\${EBROOTANACONDA3}/etc/profile.d/conda.sh" ]; then
  source "\${EBROOTANACONDA3}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "\$(command conda 'shell.bash' 'hook')"
else
  echo "Unable to locate conda.sh to activate environment." >&2
  exit 1
fi

conda activate inf_st

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

echo "[\$(date)] Running dev-only eval for ${PROCESSOR}+${CLASSIFIER} checkpoint ${CHECKPOINT}" | tee "${LOG_FILE}"

export PYTHONUNBUFFERED=1
srun -u python -u train_and_eval.py \
  --karolina \
  --dev-only \
  --checkpoint "${CHECKPOINT}" \
  -d "${DATASET}" \
  -e "${EXTRACTOR}" \
  -p "${PROCESSOR}" \
  -c "${CLASSIFIER}" \
  --dev-batch-size ${DEV_BATCH_SIZE} \
  --train-num-workers ${TRAIN_NUM_WORKERS} \
  --dev-num-workers ${DEV_NUM_WORKERS} \
  --train-prefetch-factor ${TRAIN_PREFETCH_FACTOR} \
  --dev-prefetch-factor ${DEV_PREFETCH_FACTOR} \
  --amp-eval \
  --amp-dtype ${AMP_DTYPE} \
  --output_dir "${OUTPUT_DIR}" |& tee -a "${LOG_FILE}"

echo "[\$(date)] Job finished." | tee -a "${LOG_FILE}"
EOF

  chmod +x "${JOB_SCRIPT}"
  echo "Submitting dev-only eval for ${PROCESSOR}+${CLASSIFIER} (checkpoint ${CHECKPOINT})..."
  sbatch "${JOB_SCRIPT}"
done
