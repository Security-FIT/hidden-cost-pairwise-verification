#!/bin/bash
#
# Generic runner for SGE pairwise refset baselines.
# Intended to be invoked from a qsub script (and optionally run multiple checkpoints sequentially).
#

set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

sanitize_tag() {
  # Keep job/output tags filesystem/SGE-friendly.
  echo "$1" | tr ' /:,' '____' | tr -cd '[:alnum:]_.-'
}

infer_classifier_from_checkpoint() {
  local ckpt_path="$1"
  local base tag inferred
  base="$(basename "${ckpt_path}")"
  tag="${base%.pt}"
  inferred="${tag%_*}"
  if [[ -z "${inferred}" || "${inferred}" == "${tag}" ]]; then
    echo "${tag}"
  else
    echo "${inferred}"
  fi
}

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
RUN_ID=${RUN_ID:-${JOB_ID:-${RUN_TIMESTAMP}}}

# ----------------------------
# What to run (edit this list)
# ----------------------------
#
# If RUNS is non-empty, this script will ignore CHECKPOINT_PATH and run each entry sequentially.
#
# Entry format (pipe-separated; later fields are optional):
#   exp_tag|checkpoint_path|classifier|reference_size|score_field|calibrate|calibrate_from|eval_score_column
#
# Examples:
#   RUNS=(
#     "rival_R1|/mnt/strade/.../FFCosineRaw2_1.pt|||cos_sim|0||score_raw"
#     "intermediate_R5|/mnt/strade/.../FFCosine_25.pt||5|logit_margin|1|/mnt/strade/.../scores_dev.csv|score"
#   )
#
# RUNS=(
#   "FFCosine_rival|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed42_20260112_220645/FFCosine_26.pt|FFCosine|1|logit_margin|0||score_raw"
#   "FFCosineRaw2_ep5|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_rival_MHFA_FFCosineRaw2_seed42_20260128_180119/best_model.pth|FFCosineRaw2|1|cos_sim|0||cos_sim_raw"
#   "FFCosine1_more_aug|/mnt/strade/ifirc/DFS-Detection-Framework/runs/ffcosine1_rival_aug_MHFA_B256_seed42_20260126_152812/best_model.pth|FFCosine1|1|logit_margin|0||score_raw"
#   "FFCosine1_less_aug|/mnt/strade/ifirc/DFS-Detection-Framework/runs/ffcosine1_rival_aug_MHFA_B256_seed42_20260126_155731/best_model.pth|FFCosine1|1|logit_margin|0||score_raw"
#   "FFCosine3_intermediate_ep1|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260123_161446/FFCosine3_1.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_noaug_ep1|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260124_181950/FFCosine3_1.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_aug_ep1|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260125_133207/FFCosine3_1.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_intermediate_ep5|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260123_161446/FFCosine3_5.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_noaug_ep5|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260124_181950/FFCosine3_5.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_aug_ep5|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260125_133207/FFCosine3_5.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_intermediate_ep10|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260123_161446/FFCosine3_10.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_noaug_ep10|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260124_181950/FFCosine3_10.pt|FFCosine3|1|logit_margin|0||score_raw"
#   "FFCosine3_rival_aug_ep10|/mnt/strade/ifirc/DFS-Detection-Framework/runs/antigravity_step2_MHFA_B256_seed42_20260125_133207/FFCosine3_10.pt|FFCosine3|1|logit_margin|0||score_raw"
#   )
RUNS=(
  # "FFCosine_intermediate_s42|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed42_20251204_172405/FFCosine_25.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_intermediate_s123|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed123_20251204_172405/FFCosine_10.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_intermediate_s222|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed222_20251204_172404/FFCosine_30.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_rival_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed123_20260203_142303/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_rival_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed222_20260203_142308/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_hardmined_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_hardmined_MHFA_FFCosine_seed123_20260205_173924/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_hardmined_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_hardmined_MHFA_FFCosine_seed222_20260205_173925/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_directional_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_directional_MHFA_FFCosine_seed123_20260205_173922/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_directional_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_directional_MHFA_FFCosine_seed222_20260205_173920/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_rival_finetune_xlsr_29|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_finetune_xlsr_seed42_20260209_075603/FFCosine_29.pt|FFCosine|1|logit_margin|0||score_raw"
    "FFCosine_rival_finetune_xlsr_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed123_20260209_100636/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
    "FFCosine_rival_finetune_xlsr_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed222_20260209_100636/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  )
  
MULTI_RUN=1

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/strade/ifirc/DFS-Detection-Framework}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/jobs/sge/logs}
mkdir -p "${LOG_DIR}"

EXTRACTOR=${EXTRACTOR:-XLSR_300M}
PROCESSOR=${PROCESSOR:-MHFA}

REFERENCE_SIZE=${REFERENCE_SIZE:-1}
AGGREGATION=${AGGREGATION:-max}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-4}
SAMPLE_RATE=${SAMPLE_RATE:-16000}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-4}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
AMP_EVAL=${AMP_EVAL:-1}
AMP_DTYPE=${AMP_DTYPE:-bf16}
SKIP_BAD_PATHS=${SKIP_BAD_PATHS:-0}
USE_EMBEDDINGS_CACHE=${USE_EMBEDDINGS_CACHE:-1}
SCORE_FIELD=${SCORE_FIELD:-}

P_TARGET=${P_TARGET:-0.5}
C_MISS=${C_MISS:-1.0}
C_FA=${C_FA:-1.0}
FIXED_FPRS=${FIXED_FPRS:-"0.0001 0.001 0.01 0.05"}
CALIBRATE=${CALIBRATE:-0}
CALIBRATE_FROM=${CALIBRATE_FROM:-}

EVAL_DATASET=${EVAL_DATASET:-MLAADCuratedDataset_pair}
EVAL_SCORE_COLUMN=${EVAL_SCORE_COLUMN:-}
EVAL_SEED=${EVAL_SEED:-42}

DATA_ROOT=${DATA_ROOT:-/mnt/strade/ifirc/Datasets/MLAAD}
BASELINE_PROTOCOL_ROOT=${BASELINE_PROTOCOL_ROOT:-/mnt/strade/ifirc/Datasets/MLAAD/mlaad4sourcetracing/baseline-protocols/protocols}

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

if [[ "${SKIP_BAD_PATHS}" -eq 1 ]]; then
  SKIP_BAD_FLAG="--skip-bad-paths"
else
  SKIP_BAD_FLAG="--no-skip-bad-paths"
fi

if [[ "${CALIBRATE}" -eq 1 ]]; then
  CALIBRATE_FLAG="--calibrate"
else
  CALIBRATE_FLAG="--no-calibrate"
fi
CALIBRATE_FROM_ARGS=()
if [[ "${CALIBRATE}" -eq 1 && -n "${CALIBRATE_FROM}" ]]; then
  CALIBRATE_FROM_ARGS=(--calibrate-from "${CALIBRATE_FROM}")
fi

EMBEDDINGS_CACHE_ARGS=()

cd "${PROJECT_ROOT}" || die "Failed to cd into project root ${PROJECT_ROOT}"

# Activate conda environment without relying on interactive shell init
CONDA_BASE=${CONDA_BASE:-/mnt/strade/ifirc/miniconda3}
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  export PATH="${CONDA_BASE}/bin:${PATH}"
fi
conda activate inf_st

ulimit -t 864000

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
DEFAULT_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-${DEFAULT_ALLOC_CONF}}}"
ALLOC_CONF="${ALLOC_CONF//expandable_segments:true/expandable_segments:True}"
ALLOC_CONF="${ALLOC_CONF//expandable_segments:false/expandable_segments:False}"
export PYTORCH_ALLOC_CONF="${ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}"

run_one() {
  local exp_tag="$1"
  local checkpoint_path="$2"
  local classifier_in="$3"
  local reference_size_in="$4"
  local score_field_in="$5"
  local calibrate_in="$6"
  local calibrate_from_in="$7"
  local eval_score_column_in="$8"

  [[ -n "${checkpoint_path}" ]] || die "Empty checkpoint path in RUNS entry."
  [[ -f "${checkpoint_path}" ]] || die "Checkpoint not found: ${checkpoint_path}"

  local classifier="${classifier_in:-${CLASSIFIER:-$(infer_classifier_from_checkpoint "${checkpoint_path}")}}"
  local reference_size="${reference_size_in:-${REFERENCE_SIZE}}"
  local score_field="${score_field_in:-${SCORE_FIELD}}"
  if [[ -z "${score_field}" ]]; then
    if [[ "${classifier}" == FFCosineRaw* ]]; then
      score_field=cos_sim
    else
      score_field=logit_margin
    fi
  fi

  local calibrate="${calibrate_in:-${CALIBRATE}}"
  local calibrate_from="${calibrate_from_in:-${CALIBRATE_FROM}}"
  local eval_score_column="${eval_score_column_in:-${EVAL_SCORE_COLUMN}}"
  if [[ -z "${eval_score_column}" ]]; then
    if [[ "${calibrate}" -eq 1 ]]; then
      eval_score_column="score"
    else
      eval_score_column="score_raw"
    fi
  fi

  local eval_protocol_tag="mlaad_eval_R${reference_size}"
  local reference_protocol="${BASELINE_PROTOCOL_ROOT}/${eval_protocol_tag}/references.csv"
  local test_protocol="${BASELINE_PROTOCOL_ROOT}/${eval_protocol_tag}/trials.csv"

  local checkpoint_tag
  checkpoint_tag="$(basename "${checkpoint_path}")"
  checkpoint_tag="${checkpoint_tag%.pt}"
  local segment_tag="${SEGMENT_SECONDS//./p}"

  local output_tag
  if [[ "${MULTI_RUN}" -eq 0 && -n "${OUTPUT_TAG:-}" ]]; then
    output_tag="${OUTPUT_TAG}"
  elif [[ -n "${exp_tag}" ]]; then
    output_tag="baseline_pairwise_refset_${exp_tag}_R${reference_size}_${PROCESSOR}_${classifier}_${checkpoint_tag}"
  else
    output_tag="baseline_pairwise_refset_R${reference_size}_${PROCESSOR}_${classifier}_${checkpoint_tag}"
  fi
  output_tag="$(sanitize_tag "${output_tag}")"

  local output_csv
  if [[ "${MULTI_RUN}" -eq 0 && -n "${OUTPUT_CSV:-}" ]]; then
    output_csv="${OUTPUT_CSV}"
  else
    output_csv="runs/${output_tag}_${RUN_ID}.scores.csv"
  fi
  local report_dir
  if [[ "${MULTI_RUN}" -eq 0 && -n "${REPORT_DIR:-}" ]]; then
    report_dir="${REPORT_DIR}"
  else
    report_dir="tmp/${output_tag}_${RUN_ID}.report"
  fi
  local embeddings_cache
  if [[ "${MULTI_RUN}" -eq 0 && -n "${EMBEDDINGS_CACHE:-}" ]]; then
    embeddings_cache="${EMBEDDINGS_CACHE}"
  else
    embeddings_cache="tmp/${output_tag}_embs_${EXTRACTOR}_${PROCESSOR}_${classifier}_${checkpoint_tag}_seg${segment_tag}.npz"
  fi

  mkdir -p "$(dirname "${output_csv}")"
  mkdir -p "$(dirname "${embeddings_cache}")"
  mkdir -p "${report_dir}"

  local log_file="${LOG_DIR}/${output_tag}_${RUN_ID}.log"
  echo "[$(date)] Starting pairwise refset baseline." | tee -a "${log_file}"
  echo "  checkpoint=${checkpoint_path}" | tee -a "${log_file}"
  echo "  extractor=${EXTRACTOR} processor=${PROCESSOR} classifier=${classifier}" | tee -a "${log_file}"
  echo "  R=${reference_size} aggregation=${AGGREGATION} score_field=${score_field}" | tee -a "${log_file}"
  echo "  output_csv=${output_csv}" | tee -a "${log_file}"
  echo "  embeddings_cache=${embeddings_cache} (enabled=${USE_EMBEDDINGS_CACHE})" | tee -a "${log_file}"

  local embeddings_cache_args=()
  if [[ "${USE_EMBEDDINGS_CACHE}" -eq 1 ]]; then
    embeddings_cache_args=(--embeddings-cache "${embeddings_cache}")
  fi

  local calibrate_flag
  if [[ "${calibrate}" -eq 1 ]]; then
    calibrate_flag="--calibrate"
  else
    calibrate_flag="--no-calibrate"
  fi
  local calibrate_from_args=()
  if [[ "${calibrate}" -eq 1 && -n "${calibrate_from}" ]]; then
    calibrate_from_args=(--calibrate-from "${calibrate_from}")
  fi

  useGPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | head -n1 | awk -F', ' '{print $1}')

  echo "Using GPU ${useGPU}" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=$useGPU stdbuf -oL -eL python -u baselines/baseline_pairwise_refset.py \
    --sge \
    --checkpoint "${checkpoint_path}" \
    --reference-protocol "${reference_protocol}" \
    --test-protocol "${test_protocol}" \
    --data-root "${DATA_ROOT}" \
    -e "${EXTRACTOR}" \
    -p "${PROCESSOR}" \
    -c "${classifier}" \
    --reference-size "${reference_size}" \
    --aggregation "${AGGREGATION}" \
    --score-field "${score_field}" \
    --segment-seconds "${SEGMENT_SECONDS}" \
    --sample-rate "${SAMPLE_RATE}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    ${PIN_MEMORY_FLAG} \
    ${PERSISTENT_WORKERS_FLAG} \
    ${AMP_EVAL_FLAG} \
    "${AMP_DTYPE_FLAG[@]}" \
    ${SKIP_BAD_FLAG} \
    "${embeddings_cache_args[@]}" \
    --report-dir "${report_dir}" \
    --p-target "${P_TARGET}" \
    --c-miss "${C_MISS}" \
    --c-fa "${C_FA}" \
    --fixed-fprs ${FIXED_FPRS} \
    ${calibrate_flag} \
    "${calibrate_from_args[@]}" \
    --output-csv "${output_csv}" |& tee -a "${log_file}"

  local fixed_fprs_csv="${FIXED_FPRS// /,}"
  local eval_out_dir
  eval_out_dir="$(dirname "${output_csv}")"
  echo "[$(date)] Running eval_pair_model.py (score_column=${eval_score_column})" | tee -a "${log_file}"
  python -u eval_pair_model.py \
    --sge \
    -d "${EVAL_DATASET}" \
    -e "${EXTRACTOR}" \
    -p "${PROCESSOR}" \
    -c "${classifier}" \
    --scores-in "${output_csv}" \
    --scores-in-score-column "${eval_score_column}" \
    --output_dir "${eval_out_dir}" \
    --fixed-fprs "${fixed_fprs_csv}" \
    --seed "${EVAL_SEED}" |& tee -a "${log_file}"

  echo "[$(date)] Finished: ${output_tag}" | tee -a "${log_file}"
}

if [[ ${#RUNS[@]} -gt 0 ]]; then
  MULTI_RUN=1
  for entry in "${RUNS[@]}"; do
    IFS="|" read -r exp_tag checkpoint_path classifier reference_size score_field calibrate calibrate_from eval_score_column <<< "${entry}"
    run_one "${exp_tag:-}" "${checkpoint_path:-}" "${classifier:-}" "${reference_size:-}" "${score_field:-}" "${calibrate:-}" "${calibrate_from:-}" "${eval_score_column:-}"
  done
else
  MULTI_RUN=0
  CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
  if [[ -z "${CHECKPOINT_PATH}" ]]; then
    die "CHECKPOINT_PATH must be set (or populate RUNS=())."
  fi
  run_one "${EXP_TAG:-}" "${CHECKPOINT_PATH}" "${CLASSIFIER:-}" "${REFERENCE_SIZE}" "${SCORE_FIELD}" "${CALIBRATE}" "${CALIBRATE_FROM}" "${EVAL_SCORE_COLUMN}"
fi

echo "[$(date)] All runs finished." | tee -a "${LOG_DIR}/pairwise_refset_eval_runner_${RUN_ID}.log"
