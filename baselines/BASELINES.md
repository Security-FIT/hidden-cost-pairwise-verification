# Baselines

Baselines aligned with MLAAD source-tracing: train a multiclass classifier over source models, then verify with cosine similarity of penultimate embeddings.

## Baseline 1 – MLAAD multiclass FF + cosine verifier

Recipe:
1) Train a single-utterance FF classifier on MLAAD (`path, model_name`) with `num_classes = #source_models`.
2) Remove the last linear layer and use penultimate embeddings.
3) L2-normalize embeddings and score either:
   - max/mean cosine to a reference set (R=5 by default), or
   - direct pairwise cosine (R=1).

### Train (multiclass FF)
```
python train_and_eval.py \
  --karolina \                 # or --local/--metacentrum/--sge
  -d MLAADDataset_single \
  -e XLSR_300M -p MHFA -c FF \
  --num-classes 80 \
  --num-epochs 50
```

### Source verification with reference sets (R=5)
```
python baselines/baseline_mlaad_multiclass_cosine.py \
  --karolina \                 # or --local/--metacentrum/--sge
  -d MLAADCuratedDataset_pair \
  -e XLSR_300M -p MHFA -c FF \
  --checkpoint /path/to/ff_multiclass.pt \
  --reference-size 5 \
  --score-reduction max \
  --segment-seconds 4 \
  --output-csv /tmp/mlaad_refset_scores.csv
```

### Pairwise cosine (R=1)
```
python baselines/baseline_mlaad_multiclass_cosine.py \
  --karolina \                 # or --local/--metacentrum/--sge
  -d MLAADCuratedDataset_pair \# any MLAAD pair dataset
  -e XLSR_300M -p MHFA -c FF \
  --checkpoint /path/to/ff_multiclass.pt \
  --pairs-csv /path/to/pairs.csv \
  --segment-seconds 4 \
  --output-csv /tmp/mlaad_multiclass_cosine_scores.csv
```

### Notes
- `--num-classes` should match the number of source models in the multiclass training protocol.
- The cosine script infers `num_classes` from the provided single-utterance protocol if not set.
- Reference-set mode uses `path, model_name` protocols and builds 1 target + N non-target trials per test sample.
- Pair protocols must contain `path_A, path_B, same_model`.
- The script reports EER and AUC, matching the paper’s metrics.
- `--output-csv` now includes `path_A`, `path_B`, and `claim_id`/`query_id` (when available) so you can run `scripts/analyze_pair_tail_errors.py` directly.

## Baseline 2 – Pairwise model with reference sets (R=1/R=5)

This baseline uses a *trained pairwise checkpoint* (e.g., `FFCosine`, `FFDiff`, `FFAttn*`) and scores each trial against an enrolled reference set.

For R=5, the script computes 5 scores per trial and aggregates with `--aggregation max` (keeps the best reference score and discards the other 4).

### Pairwise refset scoring (R=5)
```
python baselines/baseline_pairwise_refset.py \
  --karolina \                       # or --local/--metacentrum/--sge
  --checkpoint /path/to/FFCosine_26.pt \
  --reference-protocol /path/to/references.csv \
  --test-protocol /path/to/trials.csv \
  --data-root /path/to/MLAAD \
  -e XLSR_300M -p MHFA -c FFCosine \
  --reference-size 5 \
  --aggregation max \
  --segment-seconds 4 \
  --output-csv /tmp/pairwise_refset_scores.csv
```

### Embedding cache (speed vs. safety)
- For embedding-based pairwise classifiers (e.g., `FFCosine`, `FFDiff`, `FFConcat3`), the script can use `--embeddings-cache` to compute each utterance embedding once and then score trials cheaply.
- The cache is automatically invalidated and rebuilt when the checkpoint/config changes (you can override with `--allow-stale-embeddings-cache`, not recommended).
