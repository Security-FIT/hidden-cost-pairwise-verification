#!/bin/bash
#
# Runner for STOPA pairwise refset baselines (nc-1 trials).
# Generates reference/test protocols from STOPA TEE/Trials protocols,
# precomputes embeddings (if the model supports it), then scores trials.
#

set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

sanitize_tag() {
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
# Entry format (pipe-separated; later fields are optional):
#   exp_tag|checkpoint_path|classifier|reference_size|score_field|eval_score_column
#
# RUNS=(
#   "ffcosine_s42|/mnt/strade/.../FFCosine_25.pt|FFCosine|1|logit_margin"
# )
RUNS=(
  # "FFCosine_intermediate_s42|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed42_20251204_172405/FFCosine_25.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_intermediate_s123|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed123_20251204_172405/FFCosine_10.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_intermediate_s222|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage2_intermediate_MHFA_FFCosine_seed222_20251204_172404/FFCosine_30.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_rival_s42|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed42_20260112_220645/FFCosine_26.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_rival_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed123_20260203_142303/FFCosine_11.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_rival_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed222_20260203_142308/FFCosine_31.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_hardmined_s42|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage3_curriculum_hardmined_MHFA_FFCosine_seed42_20260112_223751/FFCosine_26.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosine_directional_s42|/mnt/strade/ifirc/DFS-Detection-Framework/best_runs/karolina_stage3_curriculum_directional_MHFA_FFCosine_seed42_20260112_223750/FFCosine_26.pt|FFCosine|1|logit_margin|score_raw"
  # "FFCosineRaw2|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_rival_MHFA_FFCosineRaw2_seed42_20260128_180119/best_model.pth|FFCosineRaw2|1|cos_sim|cos_sim_raw"
  # "FFCosine1_less_aug|/mnt/strade/ifirc/DFS-Detection-Framework/runs/ffcosine1_rival_aug_MHFA_B256_seed42_20260126_155731/best_model.pth|FFCosine1|1|logit_margin|score_raw"
  # "FFCosineRaw|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_intermediate_MHFA_FFCosineRaw_seed42_20260128_135042/best_model.pth|FFCosineRaw|1|cos_sim|cos_sim_raw"
  # "FFCosine_hardmined_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_hardmined_MHFA_FFCosine_seed123_20260205_173924/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_hardmined_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_hardmined_MHFA_FFCosine_seed222_20260205_173925/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_directional_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_directional_MHFA_FFCosine_seed123_20260205_173922/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
  # "FFCosine_directional_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_directional_MHFA_FFCosine_seed222_20260205_173920/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
    # "FFCosine_rival_finetune_xlsr|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_finetune_xlsr_seed42_20260209_075603/FFCosine_26.pt|FFCosine|1|logit_margin|0||score_raw"
    "FFCosine_rival_finetune_xlsr_s123|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed123_20260209_100636/FFCosine_11.pt|FFCosine|1|logit_margin|0||score_raw"
    "FFCosine_rival_finetune_xlsr_s222|/mnt/strade/ifirc/DFS-Detection-Framework/runs/karolina_stage3_curriculum_rival_MHFA_FFCosine_seed222_20260209_100636/FFCosine_31.pt|FFCosine|1|logit_margin|0||score_raw"
  )
MULTI_RUN=1

PROJECT_ROOT=${PROJECT_ROOT:-/mnt/strade/ifirc/DFS-Detection-Framework}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/jobs/sge/logs}
mkdir -p "${LOG_DIR}"

EXTRACTOR=${EXTRACTOR:-XLSR_300M}
PROCESSOR=${PROCESSOR:-MHFA}
CLASSIFIER=${CLASSIFIER:-}

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
RUN_AMVM=${RUN_AMVM:-0}

P_TARGET=${P_TARGET:-0.5}
C_MISS=${C_MISS:-1.0}
C_FA=${C_FA:-1.0}
FIXED_FPRS=${FIXED_FPRS:-"0.0001 0.001 0.01 0.05"}
SEED=${SEED:-42}

DATA_ROOT=${DATA_ROOT:-/mnt/strade/ifirc/Datasets/STOPA}
TEE_PROTOCOLS=${TEE_PROTOCOLS:-${DATA_ROOT}/TEE/protocols}
TEE_SUFFIX=${TEE_SUFFIX:--nc-1.txt}
TRIALS_PROTOCOLS=${TRIALS_PROTOCOLS:-${DATA_ROOT}/Trials/protocols}
TRIALS_SUFFIX=${TRIALS_SUFFIX:--nc-1_trials.txt}
METADATA_PATH=${METADATA_PATH:-${DATA_ROOT}/metadata/attacks.json}
KNOWN_ATTACKS=${KNOWN_ATTACKS:-AA01,AA03,AA05,AA07,AA10}

PROTOCOL_CACHE_DIR=${PROTOCOL_CACHE_DIR:-${PROJECT_ROOT}/tmp/stopa_refset_protocols_${RUN_ID}}
PROTOCOL_REBUILD=${PROTOCOL_REBUILD:-0}
REF_PROTOCOL="${PROTOCOL_CACHE_DIR}/references.csv"
TEST_PROTOCOL="${PROTOCOL_CACHE_DIR}/trials.csv"

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

if [[ ! -d "${TEE_PROTOCOLS}" && -d "${DATA_ROOT}/TEE/TEE_protocols" ]]; then
  TEE_PROTOCOLS="${DATA_ROOT}/TEE/TEE_protocols"
fi

prepare_stopa_protocols() {
  if [[ -f "${REF_PROTOCOL}" && -f "${TEST_PROTOCOL}" && "${PROTOCOL_REBUILD}" -eq 0 ]]; then
    echo "Using cached STOPA protocols in ${PROTOCOL_CACHE_DIR}"
    return
  fi

  mkdir -p "${PROTOCOL_CACHE_DIR}"
  echo "Building STOPA protocols in ${PROTOCOL_CACHE_DIR}"
  REF_PROTOCOL="${REF_PROTOCOL}" \
  TEST_PROTOCOL="${TEST_PROTOCOL}" \
  DATA_ROOT="${DATA_ROOT}" \
  TEE_PROTOCOLS="${TEE_PROTOCOLS}" \
  TEE_SUFFIX="${TEE_SUFFIX}" \
  TRIALS_PROTOCOLS="${TRIALS_PROTOCOLS}" \
  TRIALS_SUFFIX="${TRIALS_SUFFIX}" \
  METADATA_PATH="${METADATA_PATH}" \
  KNOWN_ATTACKS="${KNOWN_ATTACKS}" \
  REFERENCE_SIZE="${REFERENCE_SIZE}" \
  SEED="${SEED}" \
  python - <<'PY'
import os
import re
from pathlib import Path
import pandas as pd

from datasets.STOPA import (
    KNOWN_ATTACKS_DEFAULT,
    _ATTACK_RE,
    _coerce_label,
    _iter_protocol_files,
    _load_attack_metadata,
    _normalize_tee_path,
    _normalize_trial_path,
    _read_protocol_df,
)

ref_csv = Path(os.environ["REF_PROTOCOL"]).expanduser()
test_csv = Path(os.environ["TEST_PROTOCOL"]).expanduser()

tee_protocols = os.environ["TEE_PROTOCOLS"]
tee_suffix = os.environ.get("TEE_SUFFIX") or ""
trials_protocols = os.environ["TRIALS_PROTOCOLS"]
trials_suffix = os.environ.get("TRIALS_SUFFIX") or "-nc-1_trials.txt"
metadata_path = os.environ.get("METADATA_PATH") or ""
known_attacks_raw = os.environ.get("KNOWN_ATTACKS", "")
reference_size = int(os.environ.get("REFERENCE_SIZE", "1"))
seed = int(os.environ.get("SEED", "42"))

known_attacks = [a.strip() for a in known_attacks_raw.split(",") if a.strip()]
if not known_attacks:
    known_attacks = list(KNOWN_ATTACKS_DEFAULT)

if reference_size != 1:
    raise SystemExit("STOPA nc-1 protocols require REFERENCE_SIZE=1.")

def _find_attack_id(text: str):
    match = _ATTACK_RE.search(text)
    return match.group(0) if match else None

def _iter_trials_protocol_files(path: str, suffix: str | None):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if suffix:
                if "nc-1" in suffix and re.search(r"nc-1\\d", name):
                    continue
                if not name.endswith(suffix):
                    continue
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                yield candidate
    elif os.path.isfile(path):
        yield path

if tee_suffix and os.path.isdir(tee_protocols):
    names = os.listdir(tee_protocols)
    if not any(name.endswith(tee_suffix) for name in names):
        if tee_suffix == "-nc-1" and any(name.endswith("-nc-1.txt") for name in names):
            tee_suffix = "-nc-1.txt"

entries = []
for proto_path in _iter_protocol_files(tee_protocols):
    if tee_suffix:
        name = os.path.basename(proto_path)
        if "nc-1" in tee_suffix and re.search(r"nc-1\\d", name):
            continue
        if not name.endswith(tee_suffix):
            continue
    df, has_header = _read_protocol_df(proto_path, expect_header=True)
    if has_header:
        cols = {c.lower(): c for c in df.columns}
        path_col = None
        for cand in ("path", "wav", "utt", "audio", "filename"):
            if cand in cols:
                path_col = cols[cand]
                break
        attack_col = None
        for cand in ("attack_id", "attack", "hyp_attack_id", "atk"):
            if cand in cols:
                attack_col = cols[cand]
                break
        for _, row in df.iterrows():
            path_val = row[path_col] if path_col else None
            attack_val = row[attack_col] if attack_col else None
            path_str = str(path_val) if path_val is not None else ""
            attack_id = str(attack_val) if attack_val is not None else _find_attack_id(path_str or "")
            if not attack_id:
                attack_id = _find_attack_id(os.path.basename(proto_path))
            if not attack_id or not path_str:
                continue
            entries.append((attack_id, _normalize_tee_path(path_str)))
    else:
        for _, row in df.iterrows():
            tokens = [str(x) for x in row.to_list() if str(x) != "nan"]
            joined = " ".join(tokens)
            attack_id = _find_attack_id(joined)
            wav_path = next((t for t in tokens if t.lower().endswith((".wav", ".flac"))), None)
            if attack_id and wav_path:
                entries.append((attack_id, _normalize_tee_path(wav_path)))

reference_sets = {}
for attack_id in known_attacks:
    candidates = sorted({path for atk, path in entries if atk == attack_id})
    if len(candidates) != 1:
        raise SystemExit(
            f"TEE protocols for {attack_id} must contain exactly 1 ref (got {len(candidates)})."
        )
    reference_sets[attack_id] = candidates

attack_metadata = _load_attack_metadata(metadata_path)
known_set = set(known_attacks)

trial_rows = []
pair_count = 0
for proto_path in _iter_trials_protocol_files(trials_protocols, trials_suffix):
    df, has_header = _read_protocol_df(proto_path, expect_header=True)
    if not has_header:
        raise SystemExit(
            f"Trials protocol {proto_path} has no header; please provide a CSV/TSV with column names."
        )
    cols = {c.lower(): c for c in df.columns}
    def _pick_col(candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    hyp_attack_col = _pick_col(("hyp_attack_id", "hyp_attack", "enroll_attack", "attack_id_hyp"))
    trial_attack_col = _pick_col(("trial_attack_id", "trial_attack", "attack_id_trial"))
    attack_col = _pick_col(("attack_id", "attack", "atk"))
    hyp_path_col = _pick_col(("hyp_path", "hyp_wav", "enroll_path", "ref_path", "path_a", "patha"))
    trial_path_col = _pick_col(("trial_path", "trial_wav", "path_b", "pathb", "path", "filename"))
    label_atk_col = _pick_col(("same_atk", "label_atk", "same_attack", "istargetatk"))
    label_am_col = _pick_col(("same_am", "label_am", "istargetacousticmodel"))
    label_vm_col = _pick_col(("same_vm", "label_vm", "istargetvocodermodel"))
    unknown_col = _pick_col(("trial_is_unknown", "unknown_attack", "is_unknown"))

    for _, row in df.iterrows():
        trial_path = str(row[trial_path_col]) if trial_path_col else ""
        hyp_path = str(row[hyp_path_col]) if hyp_path_col else ""

        hyp_attack = None
        if hyp_attack_col:
            hyp_attack = str(row[hyp_attack_col])
        if not hyp_attack and hyp_path:
            hyp_attack = _find_attack_id(hyp_path)
        if not hyp_attack:
            hyp_attack = _find_attack_id(os.path.basename(proto_path))
        if not hyp_attack and attack_col:
            candidate = str(row[attack_col])
            if candidate in known_set:
                hyp_attack = candidate
        if not hyp_attack:
            raise SystemExit(f"Missing hypothesis attack id in {proto_path}.")
        if hyp_attack not in known_set:
            continue

        trial_attack = None
        if trial_attack_col:
            trial_attack = str(row[trial_attack_col])
        if not trial_attack and attack_col and attack_col != hyp_attack_col:
            trial_attack = str(row[attack_col])
        if not trial_attack and trial_path:
            trial_attack = _find_attack_id(trial_path)

        if hyp_path:
            hyp_rel = _normalize_tee_path(hyp_path)
            ref_path = reference_sets[hyp_attack][0]
            if hyp_rel != ref_path:
                raise SystemExit(
                    f"Trials hyp_path {hyp_rel} does not match nc-1 ref {ref_path} for {hyp_attack}."
                )

        if not trial_path:
            raise SystemExit(f"Missing trial wav path in {proto_path}.")

        label_atk = _coerce_label(row[label_atk_col]) if label_atk_col else None
        label_am = _coerce_label(row[label_am_col]) if label_am_col else None
        label_vm = _coerce_label(row[label_vm_col]) if label_vm_col else None

        trial_is_unknown = None
        if unknown_col:
            trial_is_unknown = _coerce_label(row[unknown_col])
        if trial_is_unknown is None and trial_attack:
            trial_is_unknown = int(trial_attack not in known_set)

        if label_atk is None or label_am is None or label_vm is None:
            hyp_meta = attack_metadata.get(hyp_attack, {})
            trial_meta = attack_metadata.get(trial_attack or "", {})
            hyp_am = hyp_meta.get("am")
            hyp_vm = hyp_meta.get("vm")
            trial_am = trial_meta.get("am")
            trial_vm = trial_meta.get("vm")

            if label_atk is None and trial_attack:
                label_atk = int(trial_attack == hyp_attack)
            if label_am is None and hyp_am and trial_am:
                label_am = int(trial_am == hyp_am)
            if label_vm is None and hyp_vm and trial_vm:
                label_vm = int(trial_vm == hyp_vm)

        if label_atk is None or label_am is None or label_vm is None:
            raise SystemExit(
                f"Unable to derive labels for trial {trial_path} (hyp {hyp_attack})."
            )

        trial_rel = _normalize_trial_path(trial_path)
        trial_rows.append(
            {
                "query_utt_id": f"trial_{pair_count}",
                "claim_id": hyp_attack,
                "query_path": trial_rel,
                "query_model_id": trial_attack or "unknown",
                "label": int(label_atk),
            }
        )
        pair_count += 1

ref_rows = []
for attack_id, refs in reference_sets.items():
    ref_rows.append({"claim_id": attack_id, "path": refs[0], "rank": 0})
if not ref_rows:
    raise SystemExit("No reference rows generated; check TEE protocols.")
pd.DataFrame(ref_rows).to_csv(ref_csv, index=False)

ref_rows = []
for attack_id, refs in reference_sets.items():
    for rank, path in enumerate(refs):
        ref_rows.append({"claim_id": attack_id, "path": path, "rank": rank})
if not ref_rows:
    raise SystemExit("No reference rows generated; check TEE protocols.")
pd.DataFrame(ref_rows).to_csv(ref_csv, index=False)

if not trial_rows:
    raise SystemExit("No trial rows generated; check Trials protocols/suffix.")
pd.DataFrame(trial_rows).to_csv(test_csv, index=False)
print(f"Wrote {len(ref_rows)} references to {ref_csv}")
print(f"Wrote {len(trial_rows)} trials to {test_csv}")
PY
}

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

IS_BATCH=0
if [[ -n "${JOB_ID:-}" || -n "${PBS_JOBID:-}" || -n "${SLURM_JOB_ID:-}" ]]; then
  IS_BATCH=1
fi
if ! ulimit -t 864000 2>/dev/null; then
  if [[ "${IS_BATCH}" -eq 1 ]]; then
    echo "error: unable to set CPU time limit (ulimit -t) in batch environment" >&2
    exit 1
  else
    echo "warning: unable to set CPU time limit (ulimit -t) in this environment"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
DEFAULT_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-${DEFAULT_ALLOC_CONF}}}"
ALLOC_CONF="${ALLOC_CONF//expandable_segments:true/expandable_segments:True}"
ALLOC_CONF="${ALLOC_CONF//expandable_segments:false/expandable_segments:False}"
export PYTORCH_ALLOC_CONF="${ALLOC_CONF}"
export PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF}"

prepare_stopa_protocols

run_one() {
  local exp_tag="$1"
  local checkpoint_path="$2"
  local classifier_in="$3"
  local reference_size_in="$4"
  local score_field_in="$5"
  local eval_score_column_in="$6"

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
  local eval_score_column="${eval_score_column_in:-}"
  if [[ -z "${eval_score_column}" ]]; then
    eval_score_column="score"
  fi

  local checkpoint_tag
  checkpoint_tag="$(basename "${checkpoint_path}")"
  checkpoint_tag="${checkpoint_tag%.pt}"
  local segment_tag="${SEGMENT_SECONDS//./p}"

  local output_tag
  if [[ "${MULTI_RUN}" -eq 0 && -n "${OUTPUT_TAG:-}" ]]; then
    output_tag="${OUTPUT_TAG}"
  elif [[ -n "${exp_tag}" ]]; then
    output_tag="stopa_pairwise_refset_${exp_tag}_R${reference_size}_${PROCESSOR}_${classifier}_${checkpoint_tag}"
  else
    output_tag="stopa_pairwise_refset_R${reference_size}_${PROCESSOR}_${classifier}_${checkpoint_tag}"
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
  echo "[$(date)] Starting STOPA pairwise refset baseline." | tee -a "${log_file}"
  echo "  checkpoint=${checkpoint_path}" | tee -a "${log_file}"
  echo "  extractor=${EXTRACTOR} processor=${PROCESSOR} classifier=${classifier}" | tee -a "${log_file}"
  echo "  R=${reference_size} aggregation=${AGGREGATION} score_field=${score_field} eval_score_column=${eval_score_column}" | tee -a "${log_file}"
  echo "  output_csv=${output_csv}" | tee -a "${log_file}"
  echo "  embeddings_cache=${embeddings_cache} (enabled=${USE_EMBEDDINGS_CACHE})" | tee -a "${log_file}"

  local embeddings_cache_args=()
  if [[ "${USE_EMBEDDINGS_CACHE}" -eq 1 ]]; then
    embeddings_cache_args=(--embeddings-cache "${embeddings_cache}")
  fi

  useGPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -n -k2 | head -n1 | awk -F', ' '{print $1}')
  echo "Using GPU ${useGPU}" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=$useGPU stdbuf -oL -eL python -u baselines/baseline_pairwise_refset.py \
    --sge \
    --checkpoint "${checkpoint_path}" \
    --reference-protocol "${REF_PROTOCOL}" \
    --test-protocol "${TEST_PROTOCOL}" \
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
    --output-csv "${output_csv}" |& tee -a "${log_file}"

  if [[ "${RUN_AMVM}" -eq 1 ]]; then
    echo "[$(date)] Running STOPA AM/VM evaluation." | tee -a "${log_file}"
    python -u evaluate_stopa_pair_model.py \
      --sge \
      -d STOPADataset_pair \
      -e "${EXTRACTOR}" \
      -p "${PROCESSOR}" \
      -c "${classifier}" \
      --checkpoint "${checkpoint_path}" \
      --scores-in "${output_csv}" \
      --scores-in-score-column "${eval_score_column}" \
      --eer-ci-bootstrap 0 \
      --output_dir "${report_dir}" |& tee -a "${log_file}"
  else
    echo "[$(date)] Skipping STOPA AM/VM evaluation (RUN_AMVM=${RUN_AMVM})." | tee -a "${log_file}"
  fi

  echo "[$(date)] Finished: ${output_tag}" | tee -a "${log_file}"
}

if [[ ${#RUNS[@]} -gt 0 ]]; then
  MULTI_RUN=1
  for entry in "${RUNS[@]}"; do
    IFS="|" read -r -a parts <<< "${entry}"
    exp_tag="${parts[0]:-}"
    checkpoint_path="${parts[1]:-}"
    classifier="${parts[2]:-}"
    reference_size="${parts[3]:-}"
    score_field="${parts[4]:-}"
    eval_score_column=""
    if (( ${#parts[@]} > 5 )); then
      last_idx=$((${#parts[@]} - 1))
      eval_score_column="${parts[$last_idx]}"
    fi
    run_one "${exp_tag:-}" "${checkpoint_path:-}" "${classifier:-}" "${reference_size:-}" "${score_field:-}" "${eval_score_column:-}"
  done
else
  MULTI_RUN=0
  CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
  if [[ -z "${CHECKPOINT_PATH}" ]]; then
    die "CHECKPOINT_PATH must be set (or populate RUNS=())."
  fi
  run_one "${EXP_TAG:-}" "${CHECKPOINT_PATH}" "${CLASSIFIER:-}" "${REFERENCE_SIZE}" "${SCORE_FIELD}" "${EVAL_SCORE_COLUMN:-}"
fi

echo "[$(date)] All runs finished." | tee -a "${LOG_DIR}/stopa_pairwise_refset_eval_runner_${RUN_ID}.log"
