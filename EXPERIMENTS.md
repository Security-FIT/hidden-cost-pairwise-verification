# Experiment recipes (authoritative)

This file documents the **exact configurations used to produce the paper**, transcribed
from the SLURM job scripts that launched the runs (under [`jobs/`](jobs/)). The job
scripts are the ground truth; the commands below are their resolved form with the
defaults inlined. Cluster-specific bits (paths under `/scratch/...`, `conda activate
inf_st`, `srun`, SLURM headers) are not needed to reproduce the science — only the
`python ... train_and_eval.py` / eval arguments are.

**Common to all runs:** backbone `XLSR_300M` (frozen unless noted), pooling `MHFA`,
AMP `bf16` (`--amp-train --amp-eval --amp-dtype bf16`), `--segment-seconds 4
--sample-rate 16000`. Optimizer is `Adam`, head LR `1e-3` (`model.default_lr`),
`weight_decay=0` — see [`trainers/BaseFFTrainer.py`](trainers/BaseFFTrainer.py). All
numbers are the **mean over 3 seeds: 42, 123, 222**.

---

## Global anchoring (CE) — Table 1, "Global (CE)" block

A single-utterance multiclass FF over the training generators; verification is done by
cosine similarity of the penultimate embeddings. `num_classes` is **inferred** from the
training protocol (do **not** pass `--num_classes`).

Job script: [`jobs/karolina/baseline/train_multiclass_ff.sh`](jobs/karolina/baseline/train_multiclass_ff.sh)

```bash
python train_and_eval.py --karolina \
  -d MLAADDataset_single -e XLSR_300M -p MHFA -c FF \
  --num_epochs 50 --val_interval 1 --stop_on_plateau --patience 10 --skip_eval \
  --segment-seconds 4 --sample-rate 16000 \
  --train-batch-size 128 --dev-batch-size 128 \
  --amp-train --amp-eval --amp-dtype bf16 \
  --seed 42 --output_dir runs/global_ce_seed42
```

| Table 1 row | Difference | Job script |
|---|---|---|
| `Global (CE)` | as above | `baseline/train_multiclass_ff.sh` |
| `+ XLS-R finetune` | resume from the CE checkpoint, `--num_epochs 5 --finetune-ssl --extractor-lr 1e-6 --lr 1e-5 --train-batch-size 64 --grad-accum-steps 2` (effective batch 128) | [`baseline/finetune_multiclass_ff_xlsr.sh`](jobs/karolina/baseline/finetune_multiclass_ff_xlsr.sh) |
| `+ emb bottleneck (10)` | add `--embedding_dim 10` to the CE training | [`baseline/train_multiclass_ff_10dim.sh`](jobs/karolina/baseline/train_multiclass_ff_10dim.sh) |
| `+ emb bottleneck (13)` | add `--embedding_dim 13` to the CE training | [`baseline/train_multiclass_ff_13dim.sh`](jobs/karolina/baseline/train_multiclass_ff_13dim.sh) |

**Verification / scoring** (claim-based; R=5 for comparison with prior work, R=1 for the
main table) — [`jobs/karolina/baseline/verify_refset.sh`](jobs/karolina/baseline/verify_refset.sh):

```bash
python baselines/baseline_mlaad_multiclass_cosine.py --karolina \
  -d MLAADCuratedDataset_pair -e XLSR_300M -p MHFA -c FF \
  --checkpoint <ckpt>.pt --reference-size 5 --negatives-per-test 1 \
  --score-reduction max --segment-seconds 4 --output-csv <scores>.csv
python eval_pair_model.py --karolina -d MLAADCuratedDataset_pair \
  -e XLSR_300M -p MHFA -c FF --scores-in <scores>.csv --seed 42
```

---

## Pairwise verification (BCE) — Table 1, "Pairwise (BCE)" block

The pairwise systems use the **FFCosine** head (`s = w·cos(h_a,h_b)+b`) and are trained
in **two stages**:

1. **Stage 2 — Intermediate, from scratch** (this is the `Intermediate (scratch)` row).
   Job: [`jobs/karolina/stage2/train_seed_template.sh`](jobs/karolina/stage2/train_seed_template.sh)
   (submitted by [`run_pbs.sh`](jobs/karolina/stage2/run_pbs.sh) over seeds 42/123/222,
   processors {MHFA, AASIST}, the head sweep, MHFA+FFCosine selected as default).

   ```bash
   python train_and_eval.py --karolina \
     -d MLAADIntermediateDataset_pair -e XLSR_300M -p MHFA -c FFCosine \
     --num_epochs 30 --val_interval 5 --skip_eval \
     --train-batch-size 16 --dev-batch-size 12 --grad-accum-steps 1 \
     --amp-train --amp-eval --amp-dtype bf16 \
     --seed 42 --output_dir runs/stage2_intermediate_MHFA_FFCosine_seed42
   ```

2. **Stage 3 — mining regime, resumed from the Stage-2 checkpoint for 1 epoch** at
   0.01× LR (`--lr-epoch-mults 1:0.01`) with deterministic 3-stream curriculum mixing
   (`--curriculum-three-stream --curriculum-hard-neg-ratio 0.5
   --curriculum-pairs-per-epoch 44000`).
   Job: [`jobs/karolina/stage3/train_seed_template.sh`](jobs/karolina/stage3/train_seed_template.sh).

   ```bash
   python train_and_eval.py --karolina \
     -d <REGIME_DATASET> -e XLSR_300M -p MHFA -c FFCosine \
     --checkpoint <stage2_FFCosine_ckpt>.pt \
     --num_epochs 1 --start-epoch <N> --lr-epoch-mults 1:0.01 \
     --val_interval 5 --skip_eval \
     --curriculum-three-stream --curriculum-hard-neg-ratio 0.5 \
     --curriculum-pairs-per-epoch 44000 \
     --train-batch-size 16 --dev-batch-size 4 \
     --amp-train --amp-eval --amp-dtype bf16 \
     --segment-seconds 4 --sample-rate 16000 \
     --seed 42 --output_dir runs/stage3_<regime>_MHFA_FFCosine_seed42
   ```

| Table 1 row | `<REGIME_DATASET>` (per seed: `_s123` / `_s222` suffix) | `--finetune-ssl` | Submitter |
|---|---|---|---|
| `Intermediate (scratch)` | — (Stage-2 checkpoint itself, no Stage 3) | no | `stage2/run_pbs.sh` |
| `Hard-mined` | `MLAADCurriculumHardminedFFCosineDataset_pair` | no | [`stage3/run_pbs_hardmined.sh`](jobs/karolina/stage3/run_pbs_hardmined.sh) |
| `Directional` | `MLAADCurriculumDirectionalFFCosineDataset_pair` | no | [`stage3/run_pbs_directional.sh`](jobs/karolina/stage3/run_pbs_directional.sh) |
| `Rival mining` | `MLAADCurriculumRivalFFCosineDataset_pair` | no | [`stage3/run_pbs_rival.sh`](jobs/karolina/stage3/run_pbs_rival.sh) |
| `Rival + XLS-R finetune` | `MLAADCurriculumRivalFFCosineDataset_pair` | **yes** (`--finetune-ssl --extractor-lr 1e-6`) | `stage3/run_pbs_rival.sh` (FINETUNE_SSL=1) |

> `--start-epoch` continues the LR/epoch schedule from the Stage-2 checkpoint's epoch
> (e.g. 26/11/31 for the seed-42/123/222 checkpoints actually used — see the `JOBS=(...)`
> arrays in the stage-3 submitters). The exact Stage-2 checkpoint paths used per seed are
> listed there verbatim.

**Evaluation** (claim-based, R=1):

```bash
# MLAAD (in-domain)
python eval_pair_model.py --karolina \
  -d <REGIME_DATASET> -e XLSR_300M -p MHFA -c FFCosine \
  --checkpoint <stage3_ckpt>.pt --calibrate-from <dev_scores>.csv

# STOPA (out-of-domain)
python evaluate_stopa_pair_model.py --karolina \
  -d STOPADataset_pair -e XLSR_300M -p MHFA -c FFCosine \
  --checkpoint <stage3_ckpt>.pt
```

Eval job templates: [`jobs/sge/eval/eval_template.sh`](jobs/sge/eval/eval_template.sh),
[`jobs/sge/eval/eval_stopa_template.sh`](jobs/sge/eval/eval_stopa_template.sh),
and the refset runners under [`jobs/sge/baseline/`](jobs/sge/baseline/).

---

## Architecture / head sweep (default selection)

`jobs/karolina/stage2/run_pbs.sh` iterates `PROCESSORS=(AASIST MHFA)` ×
`CLASSIFIERS` over the six scoring heads (FFCosine, FFConcat3, FFDiff, FFDiffAbs,
FFDiffQuadratic, FFAttn3) × seeds 42/123/222. MHFA+FFCosine was the most stable and
became the default. Swap `-p` / `-c` in the Stage-2 command to reproduce any grid cell.

## Notes on exactness

- Paths (`/scratch/project/dd-25-3/...`), the conda env name (`inf_st`), and SLURM/`srun`
  wrappers are cluster-specific; replace them with your own. The science is fully
  determined by the `train_and_eval.py` / eval arguments above.
- Training manifests (the pair protocols referenced via the dataset names in
  [`config.py`](config.py)) are regenerated deterministically with the documented
  `--seed` by the scripts in [`scripts/`](scripts/) — see [`scripts/README.md`](scripts/README.md).
