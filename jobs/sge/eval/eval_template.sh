#!/bin/bash
#$ -N eval_{{JOB_NAME}}
#$ -l gpu=1,gpu_ram=20G
#$ -q long.q
# -l hostname='supergpu11|supergpu12|supergpu13|supergpu15|supergpu10'
#$ -o /mnt/strade/ifirc/DFS-Detection-Framework/jobs/sge/logs/eval_{{JOB_NAME}}.$JOB_ID.log
#$ -e /mnt/strade/ifirc/DFS-Detection-Framework/jobs/sge/logs/eval_{{JOB_NAME}}.$JOB_ID.log

set -euo pipefail

DEV_BATCH_SIZE=1
DEV_NUM_WORKERS=${DEV_NUM_WORKERS:-1}
DEV_PREFETCH_FACTOR=${DEV_PREFETCH_FACTOR:-1}
TRAIN_NUM_WORKERS=0
TRAIN_PREFETCH_FACTOR=0
AMP_EVAL=${AMP_EVAL:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
CALIBRATE=${CALIBRATE:-0}
CALIBRATE_FROM=${CALIBRATE_FROM:-}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
BOTTLENECK_DIM="{{BOTTLENECK_DIM}}"

if [[ "${AMP_EVAL}" -eq 1 ]]; then
  AMP_EVAL_FLAG="--amp-eval"
else
  AMP_EVAL_FLAG="--no-amp-eval"
fi

calibrate_args=()
if [[ -n "${CALIBRATE_FROM}" ]]; then
  calibrate_args=(--calibrate-from "${CALIBRATE_FROM}")
elif [[ "${CALIBRATE}" -eq 1 ]]; then
  calibrate_args=(--calibrate)
fi

bottleneck_args=()
if [[ -z "${BOTTLENECK_DIM}" ]]; then
  if [[ "{{CHECKPOINT}}" =~ _B([0-9]+)_ ]]; then
    BOTTLENECK_DIM="${BASH_REMATCH[1]}"
  elif [[ "${RUN_ID}" =~ _B([0-9]+)_ ]]; then
    BOTTLENECK_DIM="${BASH_REMATCH[1]}"
  fi
fi
if [[ -n "${BOTTLENECK_DIM}" ]]; then
  bottleneck_args=(--bottleneck-dim "${BOTTLENECK_DIM}")
fi

PROJECT_ROOT="/mnt/strade/ifirc/DFS-Detection-Framework"

cd "${PROJECT_ROOT}" || { echo "Failed to cd into project root ${PROJECT_ROOT}"; exit 1; }

# Activate conda environment without relying on interactive shell init
CONDA_BASE="/mnt/strade/ifirc/miniconda3"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  export PATH="${CONDA_BASE}/bin:${PATH}"
fi
conda activate inf_st

echo "[$(date)] Starting eval for {{PROCESSOR}} + {{CLASSIFIER}} on {{DATASET}} (ckpt: {{CHECKPOINT}})"

# 240 hrs limit
ulimit -t 864000

useGPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | head -n1 | awk -F', ' '{print $1}')

RUN_ID="$(basename "$(dirname "{{CHECKPOINT}}")")"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_runs/${RUN_ID}}"
CKPT_BASENAME="$(basename "{{CHECKPOINT}}")"
CKPT_STEM="${CKPT_BASENAME%.*}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
DEFAULT_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-${DEFAULT_ALLOC_CONF}}}"
# PyTorch allocator parser is case-sensitive: expects True/False (not true/false).
ALLOC_CONF="${ALLOC_CONF//expandable_segments:true/expandable_segments:True}"
ALLOC_CONF="${ALLOC_CONF//expandable_segments:false/expandable_segments:False}"
export PYTORCH_ALLOC_CONF="${ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}"
CUDA_VISIBLE_DEVICES=$useGPU stdbuf -oL -eL python -u evaluations/scenarios/pair_source_verification/eval_pair_model.py \
  --sge \
  --checkpoint {{CHECKPOINT}} \
  "${calibrate_args[@]}" \
  -d {{DATASET}} \
  -e {{EXTRACTOR}} \
  -p {{PROCESSOR}} \
  -c {{CLASSIFIER}} \
  "${bottleneck_args[@]}" \
  --dev-batch-size "${DEV_BATCH_SIZE}" \
  --dev-num-workers "${DEV_NUM_WORKERS}" \
  --dev-prefetch-factor "${DEV_PREFETCH_FACTOR}" \
  --train-num-workers "${TRAIN_NUM_WORKERS}" \
  --train-prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  ${AMP_EVAL_FLAG} \
  --amp-dtype "${AMP_DTYPE}" \
  --output_dir "${OUTPUT_DIR}" \
  --output-tag "${CKPT_STEM}" \
  2>&1

echo "[$(date)] Job finished."
