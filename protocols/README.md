# MLAAD protocols used in the paper

Protocol artifacts for *The Hidden Cost of Pairwise Verification in Synthetic Speech
Source Tracing*. These extend the MLAAD source-tracing protocols from
<https://zenodo.org/records/11593133>. The `path` columns are relative to the MLAAD
audio root; place (or symlink) this `mlaad4sourcetracing/` tree under your MLAAD
`data_dir` so the paths in [`../config.py`](../config.py) resolve.

## Contents

1. **Single-utterance metadata** (`path, model_name, …, model_architecture, model_type, model_family`)
   - `mlaad4sourcetracing/train_meta.csv`, `dev_meta.csv`, `eval_meta.csv`
   - Used for global (CE) training, metadata-guided rival mining, and the
     architecture/family error breakdown (Table 2).

2. **Stage-1 pairwise training protocols** (`path_A, model_name_A, path_B, model_name_B, same_model`)
   - `pair-protocols-stage1/train_pairs_stage1_{minimal,intermediate,curated}.csv`

3. **Stage-3 pairwise hard-regime protocols** (3 seeds: 42 / 123 / 222)
   - `pair-protocols-stage3/train_pairs_stage3_{directional,hardmined,rival}_s{42,123,222}_FFCosine.csv`

4. **Pairwise dev trial list** (the paper's tuning list)
   - `pair-protocols-dev/dev_pairs.csv` — **97,203 trials = 20,000 target + 77,203 non-target**

5. **Claim-based MLAAD evaluation protocols**
   - `eval-protocols/mlaad_eval_R{1,5}/references.csv` (enrolment sets) — included.
   - `eval-protocols/mlaad_eval_R{1,5}/trials.csv` — **NOT included** (~475 MB each, exceeds
     GitHub's 100 MB file limit). Regenerate from `references.csv` + the MLAAD eval set:
     ```bash
     python scripts/build_mlaad_eval_source_verification_protocols.py --help
     ```

6. **Digital-twin separability micro protocols** (`path, model_name, label`)
   - `digital-twins-separability/micro_experiment_*.csv` — the same-architecture
     generator pairs probed in Table 3 (VITS/VITS-Neon, Parler, Bark/Suno-Bark, MMS).

## Notes

- Stage-1/3 manifests are deterministic given the documented `--seed`; the generators
  in [`../scripts/`](../scripts/) reproduce them. The CSVs here are the exact ones used.
- `config.py` references some manifests by per-seed filename; confirm the names match
  (or adjust the config paths) for the seed you are running.
