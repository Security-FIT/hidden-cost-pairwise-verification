# Pair Source Verification (Default)

## What this evaluates
This scenario evaluates pairwise source verification: given two synthetic
utterances, decide whether they come from the same synthesiser (target) or from
different synthesisers (non-target). It is the default evaluation aligned with
the training setup (same/different generator trials).

## Why it exists
This matches the operational use case for forensics and intelligence workflows.
The evaluation reports both discrimination (EER, minDCF) and deployment behavior
(actDCF, fixed-FPR operating points), including calibrated scoring when enabled.

## How to run
The main script is:
- `evaluations/scenarios/pair_source_verification/eval_pair_model.py`

Example (calibrated eval):

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u \
    evaluations/scenarios/pair_source_verification/eval_pair_model.py \
    --local \
    --checkpoint /path/to/ckpt.pt \
    --calibrate-from /path/to/scores_dev.csv \
    -d MLAADIntermediateDataset_pair \
    -e XLSR_300M \
    -p MHFA \
    -c FFCosine

Job launchers in `jobs/` already call this script.

## Metrics reported
- EER (+ 95% CI)
- minDCF / actDCF for the primary profile
- Normalized DCFs (minDCF/actDCF)
- C_llr (calibration quality)
- TPR at fixed FPR targets (uses FPR <= target when possible)
- Per-scenario summaries (if scenario_group is available)
- Optional extra DCF profiles (e.g., forensics, intel)
- Additional label-level summaries (model_type_same, model_family_same, architecture_same) when available; use `--label-protocol` if evaluating from `--scores-in`.

## Thresholding modes
- Default `actDCF` uses a sweep-based Bayes-optimal threshold on the evaluated set (`--act-threshold-mode sweep`).
- Use `--act-threshold-mode bayes` only when scores are calibrated posteriors or LLRs (set `--scores-are-llr` if applicable).
- Calibration (`--calibrate` or `--calibrate-from`) produces LLRs for valid Bayes thresholding.

## S-norm (optional)
Enable cohort S-norm scoring with:
- `--snorm --snorm-cohort-embeddings /path/to/cohort_prototypes.npz`
- Optional `--snorm-eval-embeddings /path/to/eval_embeddings.npz` (required if using `--scores-in`).
S-norm uses cosine scores in the processor embedding space; results are reported in the summary as an extra block.

## How to read results
- EER reflects separation only; it is not a deployable operating point.
- minDCF is optimistic (best achievable threshold on this set).
- actDCF uses a fixed threshold (Bayes-optimal for the chosen prior/costs).
  Use calibrated scores to make this meaningful.
- Fixed-FPR points show how much recall you get when controlling false links.

## Outputs and where to find them
Outputs are written under the eval run directory (default `eval_runs/<run_id>` or
`--output_dir`). Expected files:
- `scores.csv` (per-pair scores)
- `scores_dev.csv` (if calibration was run on dev)
- `summary.json` (full metrics)
- `summary.txt` (human-readable report)
Use `--output-tag` to append a suffix to these filenames (e.g., `scores_FFCosine_5.csv`).

The job scripts use `OUTPUT_ROOT` to override the default run location.
