# Scripts overview

Utilities to prepare MLAAD pair protocols and support hard-negative mining.

## Key entry points
- `generate_mlaad_train_pairs.py`: Stage-1 regimes (minimal/intermediate/curated) and Stage-3 regimes (random/directional/hardmined) for generating pair manifests in the standard `path_A, model_name_A, path_B, model_name_B, same_model` format.
- `hard_negative_mining.py`: Two-step helper for Stage-3 hard mining: `sample` builds random candidate pairs per anchor; `score` runs a trained checkpoint to add scores used by the `hardmined` mode above.
- `generate_rival_pairs.py`: Build a generator-to-rival mapping via centroid cosine similarity (confusion-guided rival mining).
- `analyze_rival_pairs.py`: t-SNE/PCA visualization + hardness CSV for rival (or other) pair manifests.
- `analyze_train_embeddings.py`: t-SNE/PCA visualization + centroid similarity stats for single-utterance training data.
- `validate_rival_pairs.py`: sanity checks for rival-mined negatives (leakage, collapse, rivalry map, hardness band).
- `analyze_eer_errors.py`: EER error attribution (FA/FR breakdowns, tail focus, similarity-bin contributions).
- Other helpers: `analyze_mlaad_metadata.py`, `analyze_pair_tail_errors.py`, `compile_mlaad_metadata.py`, `validate_pair_protocols.py`, `build_model_attributes.py`, `generate_mlaad_pair_protocols.py` (legacy).

## Stage-1 pair sampling (minimal / intermediate / curated)
```
python scripts/generate_mlaad_train_pairs.py \
  --method minimal|intermediate|curated|curated_balanced \
  --train-csv /path/to/train_single.csv \
  --path-column path --source-column model_name \
  --output-dir /path/to/out_dir
```
- Minimal: one random partner per anchor (balanced labels).
- Intermediate: bounded per-anchor positives/negatives (`--intermediate-max-same`, `--intermediate-max-diff`).
- Curated: per-source capped positives + quota-based negatives (`--curated-*` knobs).
- Curated_balanced: width-aware curated alternative (caps per-source + per-utterance depth, uses all anchors where possible; set `--curated-max-trials-per-utt 0` to disable the per-utterance cap).
- Deterministic with `--seed`; these regimes are unchanged from prior versions.

## Stage-3 pair sampling (random / hard-mined)
- Random (balanced, seed-stable):
```
python scripts/generate_mlaad_train_pairs.py --method random --total-pairs 44000 --seed 42 ...
```

### Rival mining (confusion-guided negatives)
1) Build a rivals map (centroid nearest neighbors per generator with an adaptive band, plus optional metadata constraints):
```
python scripts/generate_rival_pairs.py \
  --train-csv /path/to/train_single.csv \
  --embeddings /path/to/embeddings.npz \
  --samples-per-source 50 \
  --delta 0.10 \
  --output /path/to/rivals.csv
```
Rival rule: for each generator A, find its nearest neighbor with similarity `S_max`, then keep all generators
with similarity `>= (S_max - delta)` (no safety threshold; lone models still get a rival). Metadata constraints
are unioned with this list and flagged via `is_forced_meta` in the rivals CSV.
`--embeddings` should be an utterance embedding table covering the `path` values in `train_single.csv`.
You can create it directly from the train protocol:
```
python scripts/extract_mlaad_embeddings.py \
  --pairs-csv /path/to/train_single.csv \
  --path-columns path \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor MHFA --classifier FFCosine \
  --batch-size 8 --amp-eval \
  --output /path/to/train_embeddings_mhfa.npz
```
Optional metadata forcing (training metadata only). Constraints are built from
`meta_training_data::meta_language`, excluding `unknown`, `multi-dataset`, `none`, and `multilingual`.
Defaults to `data/train_meta.csv` if it exists:
```
python scripts/generate_rival_pairs.py \
  --train-csv /path/to/train_single.csv \
  --embeddings /path/to/embeddings.npz \
  --samples-per-source 50 \
  --delta 0.10 \
  --metadata-csv /path/to/train_meta.csv \
  --metadata-training-data-column meta_training_data \
  --metadata-language-column meta_language \
  --output /path/to/rivals.csv
```
If you do not have embeddings yet, generate them on the fly via a checkpoint:
```
python scripts/generate_rival_pairs.py \
  --train-csv /path/to/train_single.csv \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor MHFA --classifier FFCosine \
  --samples-per-source 50 \
  --delta 0.10 \
  --output /path/to/rivals.csv
```
2) Generate rival pairs (negative-only):
```
python scripts/generate_mlaad_train_pairs.py \
  --train-csv /path/to/train_single.csv \
  --method rival \
  --rivals-csv /path/to/rivals.csv \
  --rival-neg-per-anchor 2 \
  --total-pairs 44000 \
  --seed 42 \
  --output-dir /path/to/out_dir
```
If the rivals CSV includes `is_forced_meta`, the output pair CSV will carry it through. Override the column
name with `--rival-forced-column` if needed.

### Rival analysis (t-SNE + hardness CSV)
`analyze_rival_pairs.py` needs embeddings for every path in the pair CSV. You can reuse any precomputed embedding table that covers those paths, or generate one from the same checkpoint used for training:
```
python scripts/extract_mlaad_embeddings.py \
  --pairs-csv /path/to/out_dir/train_pairs_stage3_rival_B44000.csv \
  --path-columns path_A path_B \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor MHFA --classifier FFCosine \
  --batch-size 8 --amp-eval \
  --output tmp/rival_embeddings_mhfa.npz
```
Then run the analysis:
```
python scripts/analyze_rival_pairs.py \
  --pairs-csv /path/to/out_dir/train_pairs_stage3_rival_B44000.csv \
  --embeddings tmp/rival_embeddings_mhfa.npz \
  --output-dir tmp/rival_analysis
```
Outputs:
- `rival_tsne.png`: 2D scatter plot (t-SNE by default; use `--method pca` for a fast view).
- `rival_tsne_points.csv`: `path, model_name, x, y` for custom plotting.
- `rival_pair_hardness.csv`: per-generator-pair cosine stats (mean/median/p90/p95, etc.).

### Rival validation (leakage / collapse / hardness band)
Basic validation of rival-mined negatives:
```
python scripts/validate_rival_pairs.py \
  --pairs-csv /path/to/out_dir/train_pairs_stage3_rival_B44000.csv \
  --train-csv /path/to/train_single.csv \
  --centroid-similarity-csv tmp/train_centroid_similarity.csv \
  --sample-size 100 \
  --seed 42
```
Optional rivalry-map checks (dataset-specific anchors):
```
python scripts/validate_rival_pairs.py \
  --pairs-csv /path/to/out_dir/train_pairs_stage3_rival_B44000.csv \
  --train-csv /path/to/train_single.csv \
  --centroid-similarity-csv tmp/train_centroid_similarity.csv \
  --sibling-anchor suno/bark \
  --sibling-prefix suno/bark \
  --lone-anchor facebook/mms-tts-deu \
  --cross-anchor vits \
  --cross-allowed vits-neon,tacotron-lj
```

### Training embedding analysis (single-utterance view)
This visualizes the full training set in embedding space and summarizes generator differences using centroid cosine similarities.

1) Extract embeddings for the train protocol:
```
python scripts/extract_mlaad_embeddings.py \
  --pairs-csv /path/to/train_single.csv \
  --path-columns path \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor MHFA --classifier FFCosine \
  --batch-size 8 --amp-eval \
  --output tmp/train_embeddings_mhfa.npz
```
2) Analyze the full embedding field:
```
python scripts/analyze_train_embeddings.py \
  --train-csv /path/to/train_single.csv \
  --embeddings tmp/train_embeddings_mhfa.npz \
  --method tsne \
  --max-points 0 --max-per-source 0 \
  --output-dir tmp/train_embedding_analysis
```
Outputs:
- `train_tsne.png`: 2D scatter plot of all utterances (colored by generator).
- `train_tsne_points.csv`: `path, model_name, x, y` for custom plotting.
- `train_centroid_stats.csv`: per-generator centroid tightness (mean/median cosine to centroid).
- `train_centroid_similarity.csv`: pairwise centroid cosine similarities between generators.

### Embedding-space linear probe (dev protocol)
Train a small multinomial Logistic Regression classifier on extracted embeddings from a filtered dev protocol.

```
python scripts/probe_embedding_space.py \
  --checkpoint /path/to/ckpt.pt \
  --out-path tmp/dev_probe_report.json \
  --local
```

Outputs:
- `dev_probe_report.json`: filtering metadata, split info, embedding dim, and probe metrics.
- `dev_probe_report.predictions.csv`: per-item test predictions (`target_id`, `pred_id`, labels).

### Hard-negative mining (baseline)
1) Sample candidates:
```
python scripts/hard_negative_mining.py sample \
  --train-csv /path/to/train_single.csv \
  --sys-column model_name \
  --arch-column model_architecture \
  --seen-column model_seen \
  --max-neg-candidates-per-anchor 100 \
  --pos-per-anchor 0 \
  --seed 42 \
  --path-map /path/to/path_map.csv \
  --global-canon-cap 50 \
  --cohort-stats-out tmp/candidate_cohort_stats.csv \
  --output tmp/candidates.csv
```
- Per anchor: light stratification across metadata (default 40% cross-arch, 30% same-arch different sys, 30% from the remaining negatives). If `--seen-column` is provided, a minimum slice of the same-arch bucket is drawn from unseen. Counts scale with `--max-neg-candidates-per-anchor`. Use `--path-map` to de-duplicate path aliases, and `--cohort-stats-out` to write per-anchor cohort counts.
2) Score candidates:
```
python scripts/hard_negative_mining.py score \
  --pairs-csv tmp/candidates.csv \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor AASIST --classifier FFConcat3 \
  --batch-size 8 --amp-eval
```
3) Build hard-mined train pairs (hardest per anchor):
```
python scripts/generate_mlaad_train_pairs.py --method hardmined \
  --scored-pairs tmp/candidates_scored.csv \
  --hard-neg-per-anchor 2 \
  --hard-score-column score_same \
  --seed 42 \
  --output-dir /path/to/out_dir
```
- Hardness: highest `score_same` negatives per anchor. Fails fast if any anchor has fewer than `--hard-neg-per-anchor` negatives. Output is negative-only.

4)  Build final train pairs by mixing positives with these hard negatives:
```
python scripts/generate_mlaad_train_pairs.py --method intermediate \
  --train-csv /path/to/train_single.csv \
  --path-column path --source-column model_name \
  --intermediate-max-same 1 --intermediate-max-diff 1 \
  --seed 42 --output-dir tmp/outs
# -> tmp/outs/train_pairs_stage1_intermediate.csv

python scripts/combine_intermediate_hardneg.py \
  --intermediate-csv tmp/outs/train_pairs_stage1_intermediate.csv \
  --hardneg-csv /path/to/out_dir/train_pairs_stage3_hardmined_B44000.csv \
  --shuffle-seed 42 \
  --output tmp/outs/train_pairs_final_intermediate_pos_hardneg.csv
```
Adjust seeds/paths/caps (`--max-neg-candidates-per-anchor`, `--hard-neg-per-anchor`) as needed.

### Directional hard-mining (coverage + hardness)
1) Sample candidates (include positives if you want directional positives):
```
python scripts/hard_negative_mining.py sample \
  --train-csv /path/to/train_single.csv \
  --sys-column model_name \
  --arch-column model_architecture \
  --seen-column model_seen \
  --max-neg-candidates-per-anchor 100 \
  --pos-per-anchor 4 \
  --seed 42 \
  --path-map /path/to/path_map.csv \
  --global-canon-cap 50 \
  --cohort-stats-out tmp/candidate_cohort_stats.csv \
  --output tmp/candidates.csv
```
2) Score candidates and extract MHFA embeddings (same checkpoint):
```
python scripts/hard_negative_mining.py score \
  --pairs-csv tmp/candidates.csv \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor AASIST --classifier FFConcat3 \
  --batch-size 8 --amp-eval \
  --output tmp/candidates_scored.csv
```
```
python scripts/extract_mlaad_embeddings.py \
  --pairs-csv tmp/candidates_scored.csv \
  --path-columns path_A path_B \
  --data-root /data/MLAAD \
  --checkpoint /path/to/ckpt.pt \
  --extractor XLSR_300M --processor MHFA --classifier FFCosine \
  --batch-size 8 --amp-eval \
  --output tmp/train_embeddings_mhfa.npz
```
Note: the embedding extractor must match the checkpoint (use an MHFA-trained checkpoint here).
3) Build directional pairs (negative-only, uneven per-anchor budgets + directional coverage):
```
python scripts/generate_mlaad_train_pairs.py --method directional \
  --scored-pairs tmp/candidates_scored.csv \
  --embeddings tmp/train_embeddings_mhfa.npz \
  --hard-neg-per-anchor 2 --hard-score-column score_same \
  --seed 42 \
  --output-dir /path/to/out_dir
```
4) Merge the directional negatives with intermediate positives:
```
python scripts/combine_intermediate_hardneg.py \
  --intermediate-csv tmp/outs/train_pairs_stage1_intermediate.csv \
  --hardneg-csv /path/to/out_dir/train_pairs_stage3_directional_B44000.csv \
  --shuffle-seed 42 \
  --output tmp/outs/train_pairs_final_intermediate_pos_dirneg.csv
```
- Per-anchor selection: compute per-anchor risk (max - q95), allocate budgets (1/2/3) with p=0.15, cap tail pool at q99, and select directional tail sculptors plus a stabilizer guided by global system coverage.
- Output: negative-only, total `hard-neg-per-anchor * num_anchors` (22k when `--hard-neg-per-anchor 2`), with uneven per-anchor budgets.

### Directional diagnostics (visual + metrics)
Visualize one anchor (all candidates vs selected):
```
python scripts/plot_directional_anchor.py \
  --scored-csv tmp/candidates_scored.csv \
  --selected-csv /path/to/out_dir/train_pairs_stage3_directional_B44000.csv \
  --embeddings tmp/train_embeddings_mhfa.npz \
  --method tsne \
  --space direction
```
Overlay directional vs hardmined:
```
python scripts/plot_directional_anchor.py \
  --scored-csv tmp/candidates_scored.csv \
  --selected-csv /path/to/out_dir/train_pairs_stage3_directional_B44000.csv \
  --compare-selected-csv /path/to/out_dir/train_pairs_stage3_hardmined_B44000.csv \
  --embeddings tmp/train_embeddings_mhfa.npz \
  --method tsne \
  --space direction
```
Validate selection quality and diversity:
```
python scripts/validate_hardmined_q95.py \
  --scored-csv tmp/candidates_scored.csv \
  --selected-csv /path/to/out_dir/train_pairs_stage3_directional_B44000.csv \
  --score-column score_same \
  --expected-per-anchor 2 \
  --embeddings tmp/train_embeddings_mhfa.npz
```
Directional protocol checklist (directional-only, includes coverage + diversity plots):
```
python scripts/validate_directional_protocol.py \
  --scored-csv tmp/candidates_scored.csv \
  --selected-csv /path/to/out_dir/train_pairs_stage3_directional_B44000.csv \
  --embeddings tmp/train_embeddings_mhfa.npz \
  --score-column score_same \
  --expected-per-anchor 2
```

## Notes
- All generators preserve balanced labels where applicable and shuffle output deterministically when `--seed` is fixed.
- Stage-1 regimes (minimal/intermediate/curated) remain unchanged from prior versions and are seed-stable.
- Inputs expected by default: single-utterance CSV columns `path` and `model_name`; for stratified sampling also pass `--sys-column` / `--arch-column` (e.g., `model_architecture`). Use `--path-map` to de-duplicate path aliases and `--global-canon-cap` to limit reuse across anchors. Audio rooted at `--data-root` for scoring.
- Outputs are standard pair protocols (`path_A, model_name_A, path_B, model_name_B, same_model`) suitable for MLAAD pair datasets. Override filenames with `--output-name`; append seeds with `--seed-in-filename`.

## Tail error attribution (FA/FR)
Analyze false accepts/rejects at fixed FPR and attribute the tail:
```
python scripts/analyze_pair_tail_errors.py \
  --dev-scores /path/to/dev_scores.csv \
  --eval-scores /path/to/eval_scores.csv \
  --metadata /path/to/eval_meta.csv \
  --score-column score_same \
  --label-column same_model \
  --output-dir tmp/tail_analysis
```
- Works with `eval_pair_model.py` scores (`pathA,pathB,score,label`) or scored pair protocols (`path_A,path_B,score_same,same_model`).
- Writes FA/FR lists, per-claim FA/FR/TPR, top impostor pairs, and same-rate cohort stats when metadata is available.
- Optional embedding neighborhood analysis via `--embedding-npz` (expects `embeddings` + `utt_ids` arrays).

## EER error attribution (global)
Analyze which models and pairs dominate the EER point (FA/FR, tail errors, similarity bins):
```
python scripts/analyze_eer_errors.py \
  --eval-scores /path/to/eval_scores.csv \
  --score-column score_same \
  --label-column same_model \
  --centroid-similarity-csv tmp/train_centroid_similarity.csv \
  --output-dir tmp/eer_analysis
```
Outputs:
- `eer_summary.json`: EER threshold + global counts.
- `eer_fa_by_pair.csv`: FA counts/rates by model pair (negative pairs).
- `eer_fr_by_model.csv`: FR counts/rates by model.
- `eer_fa_by_model.csv`: FA counts/rates as claim and as query.
- `eer_fa_tail_by_pair.csv`: tail FA pairs (top quantile by margin).
- `eer_fr_tail_by_model.csv`: tail FR models.
- `eer_fa_by_similarity_bin.csv`: FA rate by centroid similarity bin (if provided).

## DET curve comparison (multi-system)
Plot one DET figure to compare multiple systems from their `scores.csv` files:
```
python scripts/plot_det_compare.py \
  eval_runs/runA/scores.csv eval_runs/runB/scores.csv \
  --name runA --name runB \
  -o tmp/det_compare.png
```
Notes:
- Defaults assume `score` and `label` columns and `pos_label=1` (target=1, non-target=0).
- Use `--score-column`, `--label-column`, `--pos-label`, or `--score-direction lower` for non-standard schemas.
- Use `--drawstyle steps-post` (and optionally `--no-antialias`) for a more "steppy" curve.
- Use `--xlim 1%,40% --ylim 0.1%,40%` (or omit to auto-zoom in `--style paper`) to focus on the interesting operating region.
- For B/W-friendly figures, use `--palette grayscale --markers --linewidth 1.6` (linestyle + marker per curve).
- If multiple curves nearly overlap, add an inset zoom: `--inset --inset-xlim 2%,20% --inset-ylim 2%,12%`.
