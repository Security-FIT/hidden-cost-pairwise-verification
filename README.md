# The Hidden Cost of Pairwise Verification in Synthetic Speech Source Tracing

Code to reproduce the experiments in:

> A. Firc, Z. Lička, V. Staněk, K. Malinka. *The Hidden Cost of Pairwise
> Verification in Synthetic Speech Source Tracing.* Accepted at Interspeech.
> arXiv: <https://arxiv.org/abs/2606.11666>

This repository compares **global anchoring** (closed-set CE classification used as
a verification backbone) against **pairwise verification** (Siamese-style metric
learning) for open-set synthetic-speech source tracing, under matched backbones,
protocols, and a fixed data/epoch budget on **MLAAD** (in-domain) and **STOPA** (OOD).

It is a curated subset of a larger internal deepfake-detection framework, reduced to
exactly the components used in the paper: the **XLS-R** backbone, **MHFA** / **AASIST**
pooling, the **FF** (global) and **FFCosine** / **FF\*** (pairwise) heads, the **MLAAD**
and **STOPA** datasets, and the protocol-generation and analysis scripts that produce
the paper's tables and figures.

---

## 1. Installation

```bash
conda create -n hidden-cost python=3.10
conda activate hidden-cost
# Install PyTorch matching your CUDA (see https://pytorch.org). Example:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

The XLS-R 300M backbone is downloaded automatically by `torchaudio`
(`WAV2VEC2_XLSR_300M`).

## 2. Data and paths

Two datasets are required:

- **MLAAD v8** — <https://deepfake-total.com/mlaad> (used as `mlaad4sourcetracing/`).
- **STOPA** — <https://doi.org/10.5281/zenodo.15606628> (Firc et al., Interspeech 2025).

Edit the dataset roots in [`config.py`](config.py). The three profiles
(`local_config`, `sge_config`, `karolina_config`) differ only in `data_dir` and
batch sizes; pick one with the matching `--local` / `--sge` / `--karolina` flag.
Each config maps a logical dataset name to its pair/single protocol CSVs under the
dataset root.

**The protocol CSVs used in the paper are bundled** under
[`protocols/mlaad4sourcetracing/`](protocols/) — train metadata, the Stage-1 / Stage-3
pair manifests (3 seeds), the 97,203-trial dev list, claim-based eval references, and the
digital-twin micro protocols. Place (or symlink) this `mlaad4sourcetracing/` tree under
your MLAAD `data_dir` so the paths in `config.py` resolve. See
[`protocols/README.md`](protocols/README.md). The two ~475 MB claim-based eval `trials.csv`
files are **not** shipped (GitHub size limit) — regenerate them from the bundled
`references.csv` with `scripts/build_mlaad_eval_source_verification_protocols.py`.

All systems share the **XLS-R + MHFA** backbone (frozen unless `--finetune-ssl`).
All results in the paper are the **mean over 3 seeds: 42, 123, 222** (`--seed`).

> **Authoritative recipes:** the exact, as-run commands and hyperparameters for every
> Table 1 row — transcribed from the SLURM job scripts in [`jobs/`](jobs/) — are in
> **[`EXPERIMENTS.md`](EXPERIMENTS.md)**. The section below is a readable summary; when
> in doubt, `EXPERIMENTS.md` and `jobs/` are ground truth.

## 3. Reproducing Table 1 (Global vs. Pairwise)

### Global anchoring (CE) — the baseline

Train a single-utterance multiclass FF over the N=24 training generators, then verify
by cosine similarity of the penultimate embeddings (claim-based protocol). See
[`baselines/BASELINES.md`](baselines/BASELINES.md) for the full recipe.

```bash
# Train the global (CE) model — num_classes is INFERRED from the protocol (do not pass it)
python train_and_eval.py --karolina \
  -d MLAADDataset_single -e XLSR_300M -p MHFA -c FF \
  --num_epochs 50 --val_interval 1 --stop_on_plateau --patience 10 --skip_eval \
  --segment-seconds 4 --sample-rate 16000 \
  --train-batch-size 128 --dev-batch-size 128 --amp-train --amp-eval --amp-dtype bf16 \
  --seed 42 --output_dir runs/global_ce_seed42

# Claim-based verification (R=5 vs prior work; use --reference-size 1 for the R=1 table)
python baselines/baseline_mlaad_multiclass_cosine.py --karolina \
  -d MLAADCuratedDataset_pair -e XLSR_300M -p MHFA -c FF \
  --checkpoint runs/global_ce_seed42/<ckpt>.pt \
  --reference-size 5 --negatives-per-test 1 --score-reduction max --segment-seconds 4 \
  --output-csv eval_runs/global_ce_seed42.csv
python eval_pair_model.py --karolina -d MLAADCuratedDataset_pair \
  -e XLSR_300M -p MHFA -c FF --scores-in eval_runs/global_ce_seed42.csv --seed 42
```

| Table 1 row | Change to the training command |
|---|---|
| `Global (CE)` | as shown |
| `+ XLS-R finetune` | resume from CE ckpt: `--num_epochs 5 --finetune-ssl --extractor-lr 1e-6 --lr 1e-5 --train-batch-size 64 --grad-accum-steps 2` |
| `+ emb bottleneck (10)` | add `--embedding_dim 10` |
| `+ emb bottleneck (13)` | add `--embedding_dim 13` |

STOPA (OOD) evaluation uses
[`baselines/baseline_stopa_multiclass_cosine.py`](baselines/baseline_stopa_multiclass_cosine.py).

### Pairwise verification (BCE)

The FFCosine head (`s = w·cos(h_a, h_b) + b`) is trained in **two stages**: a Stage-2
*intermediate* run from scratch (30 epochs), then a Stage-3 *mining* run that resumes the
Stage-2 checkpoint for **1 epoch at 0.01× LR** with deterministic 3-stream curriculum
mixing. The `Intermediate (scratch)` row is the Stage-2 checkpoint itself.

```bash
# Stage 2 — intermediate, from scratch
python train_and_eval.py --karolina \
  -d MLAADIntermediateDataset_pair -e XLSR_300M -p MHFA -c FFCosine \
  --num_epochs 30 --val_interval 5 --skip_eval \
  --train-batch-size 16 --dev-batch-size 12 --amp-train --amp-eval --amp-dtype bf16 \
  --seed 42 --output_dir runs/stage2_intermediate_seed42

# Stage 3 — mining regime, resume 1 epoch at 0.01x LR with curriculum mixing
python train_and_eval.py --karolina \
  -d <REGIME_DATASET> -e XLSR_300M -p MHFA -c FFCosine \
  --checkpoint runs/stage2_intermediate_seed42/<FFCosine_N>.pt \
  --num_epochs 1 --start-epoch <N+1> --lr-epoch-mults 1:0.01 --skip_eval \
  --curriculum-three-stream --curriculum-hard-neg-ratio 0.5 --curriculum-pairs-per-epoch 44000 \
  --train-batch-size 16 --dev-batch-size 4 --amp-train --amp-eval --amp-dtype bf16 \
  --segment-seconds 4 --sample-rate 16000 \
  --seed 42 --output_dir runs/stage3_<regime>_seed42
```

| Table 1 row | `<REGIME_DATASET>` | `--finetune-ssl`? |
|---|---|---|
| `Intermediate (scratch)` | (Stage-2 checkpoint; no Stage 3) | no |
| `Hard-mined` | `MLAADCurriculumHardminedFFCosineDataset_pair` | no |
| `Directional` | `MLAADCurriculumDirectionalFFCosineDataset_pair` | no |
| `Rival mining` | `MLAADCurriculumRivalFFCosineDataset_pair` | no |
| `Rival + XLS-R finetune` | `MLAADCurriculumRivalFFCosineDataset_pair` | yes (`--finetune-ssl --extractor-lr 1e-6`) |

Evaluation (claim-based, R=1):

```bash
python eval_pair_model.py --karolina -d <REGIME_DATASET> \
  -e XLSR_300M -p MHFA -c FFCosine --checkpoint <stage3_ckpt>.pt --calibrate-from <dev_scores>.csv
python evaluate_stopa_pair_model.py --karolina -d STOPADataset_pair \
  -e XLSR_300M -p MHFA -c FFCosine --checkpoint <stage3_ckpt>.pt
```

See **[`EXPERIMENTS.md`](EXPERIMENTS.md)** for exact per-seed checkpoints/start-epochs and
the architecture/head sweep (XLS-R × {MHFA, AASIST} × six heads) that selected
MHFA+FFCosine. Metrics are documented in
[`evaluations/scenarios/pair_source_verification/README.md`](evaluations/scenarios/pair_source_verification/README.md).

## 4. Reproducing the figures and tables

| Paper item | Script |
|---|---|
| Fig. 1 — DET curves (MLAAD / STOPA) | [`scripts/plot_det_compare.py`](scripts/plot_det_compare.py) |
| Fig. 2 — cumulative variance / k99 (dimensionality collapse) | [`scripts/analysis/plot_manifold_collapse.py`](scripts/analysis/plot_manifold_collapse.py), [`scripts/compare_embedding_spaces.py`](scripts/compare_embedding_spaces.py) |
| Fig. 3 — score CDFs (target vs. non-target) | [`scripts/analysis/plot_score_distributions.py`](scripts/analysis/plot_score_distributions.py) |
| Table 2 — false-acceptance breakdown (impostor pairs) | [`scripts/analyze_eer_errors.py`](scripts/analyze_eer_errors.py), [`scripts/analyze_pair_tail_errors.py`](scripts/analyze_pair_tail_errors.py), [`scripts/analysis/analyze_confusing_pairs.py`](scripts/analysis/analyze_confusing_pairs.py) |
| Table 3 — binary-probe separability of generator variants | [`scripts/probe_embedding_space.py`](scripts/probe_embedding_space.py), [`scripts/analysis/vits_neon_layer_separability.py`](scripts/analysis/vits_neon_layer_separability.py) |

## 5. Reproducing the training protocols

The pair manifests for each regime are generated from the MLAAD single-utterance
protocol. Full command reference is in [`scripts/README.md`](scripts/README.md).

- Stage-1 regimes (minimal / intermediate / curated) and Stage-3 regimes
  (random / hard-mined / directional / rival): `scripts/generate_mlaad_train_pairs.py`
- Hard-negative mining (sample → score): `scripts/hard_negative_mining.py`
- Rival mining (confusion-guided negatives): `scripts/generate_rival_pairs.py`
- Embedding extraction (for mining/analysis): `scripts/extract_mlaad_embeddings.py`
- Merge positives with mined negatives: `scripts/combine_intermediate_hardneg.py`
- Claim-based eval protocols: `scripts/build_mlaad_eval_source_verification_protocols.py`
- Generator metadata / `configs/model_attributes.json`:
  `scripts/compile_mlaad_metadata.py`, `scripts/build_model_attributes.py`

## 6. Repository layout

```
train_and_eval.py        Main entry point: train + evaluate (FF / FFCosine families)
finetune.py              SSL-backbone fine-tuning entry point
eval_pair_model.py       Claim-based pairwise verification eval (MLAAD)  [wrapper]
evaluate_stopa_pair_model.py  Claim-based pairwise verification eval (STOPA)  [wrapper]
common.py                build_model / get_dataloaders + model/dataset registries
config.py                Dataset roots, protocol paths, batch/epoch settings
parse_arguments.py       CLI
extractors/              XLS-R backbone
feature_processors/      MHFA, AASIST pooling
classifiers/             FF (global) and FFCosine / FFConcat / FFDiff / FFAttn (pairwise) heads
trainers/                Training loops (global FFTrainer, pairwise FFPairTrainer, joint, raw-cosine)
datasets/                MLAAD, STOPA dataset classes + pair collation
augmentation/            Augmentation utilities (used by the dataset loaders)
evaluations/             Evaluation scenarios (pair source verification, STOPA)
baselines/               Global multiclass + cosine verification baselines
scripts/                 Protocol generation and figure/table analysis
jobs/                    Authoritative SLURM job scripts (the as-run recipes)
protocols/               MLAAD protocol CSVs used in the paper (see protocols/README.md)
detailed_results/        Result spreadsheets behind the tables/ablations
configs/                 Generator attribute metadata
EXPERIMENTS.md           Exact per-row commands/hyperparameters (transcribed from jobs/)
```
