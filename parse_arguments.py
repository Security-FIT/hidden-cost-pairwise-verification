import argparse

from common import CLASSIFIERS

try:
    from safe_gpu import safe_gpu  # type: ignore
except ModuleNotFoundError:
    safe_gpu = None


def parse_args():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating the classifiers.")

    # either --metacentrum, --sge or --local must be specified
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metacentrum", action="store_true", help="Flag for running on metacentrum.")
    group.add_argument("--sge", action="store_true", help="Flag for running on SGE on BUT FIT.")
    group.add_argument("--karolina", action="store_true", help="Flag for running on the Karolina cluster.")
    group.add_argument("--local", action="store_true", help="Flag for running locally.")

    # Add argument for loading a checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to a checkpoint to be loaded. If not specified, the model will be trained from scratch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for saving checkpoints and other artifacts. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training is finished.",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Skip training and only run validation on the dev split (useful for sanity-checking checkpoints).",
    )
    parser.add_argument(
        "--amp-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run validation/eval passes under torch.cuda.amp autocast (mixed precision inference).",
    )
    parser.add_argument(
        "--amp-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run training (forward/backward) passes under torch.cuda.amp autocast (mixed precision).",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=("bf16", "fp16"),
        default="bf16",
        help="Autocast dtype to use when AMP is enabled (bf16 or fp16).",
    )

    # dataset
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="MLAADIntermediateDataset_pair",
        help="Dataset to be used. See common.DATASETS for available datasets.",
        required=True,
    )

    protocol_group = parser.add_argument_group("Protocol overrides")
    protocol_group.add_argument(
        "--train-protocol",
        type=str,
        default=None,
        help=(
            "Override the training protocol file (relative to dataset root_dir or absolute path). "
            "Useful for micro-experiments with custom CSVs."
        ),
    )
    protocol_group.add_argument(
        "--dev-protocol",
        type=str,
        default=None,
        help="Override the dev/validation protocol file (relative to dataset root_dir or absolute path).",
    )
    protocol_group.add_argument(
        "--eval-protocol",
        type=str,
        default=None,
        help="Override the eval protocol file (relative to dataset root_dir or absolute path).",
    )

    # extractor
    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        default="XLSR_300M",
        help=f"Extractor to be used. See common.EXTRACTORS for available extractors.",
        required=True,
    )

    # feature processor
    feature_processors = ["MHFA", "AASIST", "Mean", "SLS"]
    parser.add_argument(
        "-p",
        "--processor",
        "--pooling",
        type=str,
        help=f"Feature processor to be used. One of: {', '.join(feature_processors)}",
        required=True,
    )
    # TODO: Allow for passing parameters to the feature processor (mainly MHFA)

    # classifier
    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        help=f"Classifier to be used. See common.CLASSIFIERS for available classifiers.",
        required=True,
    )

    # augmentations
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="Flag for whether to use augmentations during training. Does nothing during evaluation.",
    )

    augment_group = parser.add_argument_group("Augmentations")
    augment_group.add_argument(
        "--benign-aug-prob-min",
        type=float,
        default=None,
        help="Minimum probability to apply benign augmentation to positive pairs (MLAAD only).",
    )
    augment_group.add_argument(
        "--benign-aug-prob-max",
        type=float,
        default=None,
        help="Maximum probability to apply benign augmentation to positive pairs (MLAAD only).",
    )

    # Add arguments specific to each classifier
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    classifier_args = parser.add_argument_group("Classifier-specific arguments")
    added_args = set()
    for classifier, (classifier_class, args) in CLASSIFIERS.items():
        if args:  # if there are any arguments that can be passed to the classifier
            for arg, arg_type in args.items():
                if arg in added_args:
                    continue
                added_args.add(arg)
                
                if arg == "kernel":  # only for SVMDiff, display the possible kernels
                    classifier_args.add_argument(
                        f"--{arg}",
                        type=str,
                        help=f"{arg} for {classifier}. One of: {', '.join(kernels)}",
                    )
                    # TODO: Add parameters for the kernels (e.g. degree for poly, gamma for rbf, etc.)
                else:
                    if arg_type is bool:
                        classifier_args.add_argument(
                            f"--{arg}",
                            action=argparse.BooleanOptionalAction,
                            default=None,
                            help=f"{arg} for {classifier}",
                        )
                    else:
                        classifier_args.add_argument(f"--{arg}", type=arg_type, help=f"{arg} for {classifier}")

    # maybe TODO: add flag for enabling/disabling evaluation after training

    # region Optional arguments
    # training
    classifier_args.add_argument(
        "-ep",
        "--num_epochs",
        type=int,
        help="Number of epochs to train for. Does not concern SkLearn classifiers.",
        default=50,
    )
    classifier_args.add_argument(
        "--start-epoch",
        type=int,
        help="Epoch index to start from when resuming training (FF trainers).",
        default=1,
    )
    classifier_args.add_argument(
        "--val_interval",
        type=int,
        help="Validate on dev set every N epochs (FF trainers).",
        default=5,
    )
    classifier_args.add_argument(
        "--stop_on_plateau",
        action=argparse.BooleanOptionalAction,
        help="Enable dev EER plateau early stopping for FF trainers (disabled by default for Stage 1).",
        default=False,
    )
    classifier_args.add_argument(
        "--patience",
        type=int,
        help="Patience (in validations) for early stopping when enabled.",
        default=None,
    )

    # ArcFace & MLP arguments (legacy support)
    classifier_args.add_argument(
        "--use-arcface",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable ArcFace loss/architecture logic where supported (deprecated).",
    )
    classifier_args.add_argument(
        "--arcface-m",
        type=float,
        default=0.5,
        help="Angular margin (m) for ArcFace (deprecated).",
    )
    classifier_args.add_argument(
        "--arcface-s",
        type=float,
        default=30.0,
        help="Scale factor (s) for ArcFace (deprecated).",
    )
    classifier_args.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=2048,
        help="Deprecated (unused; retained for backward compatibility).",
    )
    classifier_args.add_argument(
        "--projection-dim",
        type=int,
        default=1024,
        help="Deprecated (unused; retained for backward compatibility).",
    )
    classifier_args.add_argument(
        "--bottleneck-dim",
        type=int,
        default=50,
        help="Dimension of the bottleneck layer (FFMulticlass/FFCosine3).",
    )

    perf_group = parser.add_argument_group("Performance tuning")
    perf_group.add_argument(
        "--train-batch-size",
        type=int,
        help="Override training batch size (defaults to config batch size or LSTM batch size).",
        default=None,
    )
    perf_group.add_argument(
        "--dev-batch-size",
        type=int,
        help="Override dev/eval batch size (defaults to a multiple of the train batch size).",
        default=None,
    )
    perf_group.add_argument(
        "--train-num-workers",
        type=int,
        help="Number of workers for the training DataLoader.",
        default=None,
    )
    perf_group.add_argument(
        "--dev-num-workers",
        type=int,
        help="Number of workers for validation/eval DataLoaders.",
        default=None,
    )
    perf_group.add_argument(
        "--train-prefetch-factor",
        type=int,
        help="Override DataLoader prefetch factor for training.",
        default=None,
    )
    perf_group.add_argument(
        "--dev-prefetch-factor",
        type=int,
        help="Override DataLoader prefetch factor for validation/eval.",
        default=None,
    )
    perf_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable pin_memory for DataLoaders (defaults to config).",
        default=None,
    )
    perf_group.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        help="Enable/disable persistent workers for non-zero worker counts.",
        default=None,
    )
    perf_group.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Number of gradient accumulation steps during training (FF trainers).",
        default=1,
    )
    perf_group.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit the number of training batches per epoch (useful for debugging).",
        default=None,
    )

    curriculum_group = parser.add_argument_group("Curriculum sampling")
    curriculum_group.add_argument(
        "--curriculum-easy-csv",
        type=str,
        default=None,
        help="Easy/intermediate training trials CSV for curriculum mixing.",
    )
    curriculum_group.add_argument(
        "--curriculum-hard-csv",
        type=str,
        default=None,
        help="Hard training trials CSV for curriculum mixing.",
    )
    curriculum_group.add_argument(
        "--curriculum-steps-per-epoch",
        type=int,
        default=None,
        help="Override number of batches per epoch when curriculum mixing is enabled.",
    )
    curriculum_group.add_argument(
        "--curriculum-three-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic 3-stream mixing (pos/easy-neg/hard-neg).",
    )
    curriculum_group.add_argument(
        "--curriculum-hard-neg-ratio",
        type=float,
        default=0.5,
        help="Hard-negative ratio within the negative portion of each batch (default: 0.5).",
    )
    curriculum_group.add_argument(
        "--curriculum-aug-csv",
        type=str,
        default=None,
        help="Protocol CSV used to sample anchors for augmented-negative pairs.",
    )
    curriculum_group.add_argument(
        "--curriculum-aug-ratio",
        type=float,
        default=None,
        help="Fraction of each batch from augmented-negative pairs (enables augmented curriculum mixing).",
    )
    curriculum_group.add_argument(
        "--curriculum-easy-ratio",
        type=float,
        default=None,
        help="Fraction of each batch from easy/intermediate pairs when augmented mixing is enabled.",
    )
    curriculum_group.add_argument(
        "--curriculum-hard-ratio",
        type=float,
        default=None,
        help="Fraction of each batch from hard/rival pairs when augmented mixing is enabled.",
    )
    curriculum_group.add_argument(
        "--curriculum-pairs-per-epoch",
        type=int,
        default=None,
        help="Target number of training pairs per epoch when curriculum mixing is enabled.",
    )

    lr_group = parser.add_argument_group("Learning rate schedule")
    lr_group.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override default Learning Rate (e.g. 1e-4 for fine-tuning).",
    )

    finetune_group = parser.add_argument_group("Fine-tuning (SSL extractors)")
    finetune_group.add_argument(
        "--finetune-ssl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable gradient updates for SSL extractor (e.g. XLSR/WavLM/Wav2Vec2/HuBERT). "
            "Use --extractor-lr to keep a much smaller LR for the extractor."
        ),
    )
    finetune_group.add_argument(
        "--extractor-lr",
        type=float,
        default=None,
        help=(
            "Learning rate for the SSL extractor when --finetune-ssl is enabled "
            "(defaults to 1e-6 if not provided)."
        ),
    )

    lr_group.add_argument(
        "--lr-ramp",
        action="store_true",
        help="Enable a linear LR ramp (per optimizer step) relative to the base LR (FF trainers).",
        default=False,
    )
    lr_group.add_argument(
        "--lr-ramp-start-mult",
        type=float,
        default=0.05,
        help="Start multiplier relative to base LR when --lr-ramp is enabled.",
    )
    lr_group.add_argument(
        "--lr-ramp-target-mult",
        type=float,
        default=0.2,
        help="Target multiplier relative to base LR after the ramp.",
    )
    lr_group.add_argument(
        "--lr-ramp-steps",
        type=int,
        default=500,
        help="Number of optimizer steps for the linear LR ramp.",
    )
    lr_group.add_argument(
        "--lr-epoch-mults",
        type=str,
        default=None,
        help="Piecewise epoch->LR multiplier schedule 'epoch:mult,epoch:mult' (FF trainers).",
    )

    segment_group = parser.add_argument_group("Audio segmenting")
    segment_group.add_argument(
        "--segment-seconds",
        type=float,
        default=None,
        help="Optional fixed segment length (seconds) for MLAAD audio.",
    )
    segment_group.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate for segmenting (default: 16000).",
    )

    filter_group = parser.add_argument_group("Class filtering (MLAAD)")
    filter_group.add_argument(
        "--allowed-classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of MLAAD model_name classes to keep. "
            "Supports space-separated and/or comma-separated values."
        ),
    )

    # region Eval/scoring options (used by eval scripts; harmless during training)
    eval_group = parser.add_argument_group("Evaluation/scoring")
    eval_group.add_argument(
        "--scores-out",
        type=str,
        help="Optional path to dump per-pair scores (CSV with pair_id, score, label).",
        default=None,
    )
    eval_group.add_argument(
        "--scores-in",
        type=str,
        help="Optional path to a precomputed scores CSV (pathA,pathB,score,label[,scenario_group]); skips model inference.",
        default=None,
    )
    eval_group.add_argument(
        "--scores-in-score-column",
        type=str,
        default="score",
        help="Column name to read as the score when using --scores-in (default: score).",
    )
    eval_group.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional suffix tag for outputs (scores/summary filenames).",
    )
    eval_group.add_argument(
        "--label-protocol",
        type=str,
        default=None,
        help=(
            "Optional pair protocol CSV used to source additional label columns "
            "(model_type_same, model_family_same, architecture_A/B) for multi-level eval. "
            "If omitted, the eval script will reuse the dataset protocol when available."
        ),
    )
    eval_group.add_argument(
        "--calibrate-from",
        type=str,
        help="Optional dev scores CSV (same columns as scores-in) to learn a Platt/logistic calibration (score -> posterior).",
        default=None,
    )
    eval_group.add_argument(
        "--calibrate-from-score-column",
        type=str,
        default="score",
        help="Column name to read as the score when using --calibrate-from (default: score).",
    )
    eval_group.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable score calibration (Platt). Use --calibrate-from to supply dev scores; otherwise dev will be scored if a checkpoint is provided.",
    )
    eval_group.add_argument(
        "--include-label",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print labels alongside scores when supported by the script.",
    )
    eval_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override for evaluation (e.g., cuda:0, cpu). Defaults to auto.",
    )
    eval_group.add_argument(
        "--pos-label",
        type=int,
        default=1,
        help="Label value treated as positive (target) when computing DET/DCFs (default: 1).",
    )
    eval_group.add_argument(
        "--p-target",
        type=float,
        default=0.5,
        help="Target prior for minDCF/actDCF.",
    )
    eval_group.add_argument(
        "--c-miss",
        type=float,
        default=1.0,
        help="Miss cost for minDCF/actDCF.",
    )
    eval_group.add_argument(
        "--c-fa",
        type=float,
        default=1.0,
        help="False-alarm cost for minDCF/actDCF.",
    )
    eval_group.add_argument(
        "--eval-profiles",
        type=str,
        default=None,
        help=(
            "Comma-separated DCF profile names to report (e.g., 'primary,forensics,intel'). "
            "In evaluations/scenarios/pair_source_verification/eval_pair_model.py, omitting this defaults to 'primary,forensics,intel'."
        ),
    )
    eval_group.add_argument(
        "--act-threshold",
        type=float,
        default=None,
        help=(
            "Score threshold override for actDCF; interpreted in the same domain as scores "
            "(probability for posteriors, LLR if --scores-are-llr or calibration is enabled)."
        ),
    )
    eval_group.add_argument(
        "--act-threshold-mode",
        type=str,
        default="sweep",
        choices=("sweep", "bayes"),
        help=(
            "How to select the actDCF threshold: "
            "'sweep' uses the empirical Bayes-optimal point on the DET curve (default), "
            "'bayes' uses the Bayes threshold (valid for calibrated posteriors/LLRs)."
        ),
    )
    eval_group.add_argument(
        "--fixed-fprs",
        type=str,
        default=None,
        help=(
            "Comma-separated FPR targets for reporting TPR@FPR (e.g., '0.0001,0.001,0.01' or '0.01%%,0.1%%,1%%'). "
            "If omitted, the eval script default is used."
        ),
    )
    eval_group.add_argument(
        "--eer-ci-bootstrap",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap samples for EER confidence intervals. "
            "Set to 0 to disable CI computation."
        ),
    )
    eval_group.add_argument(
        "--scores-are-llr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat input scores as log-likelihood ratios (enables Bayes thresholding in LLR space).",
    )
    eval_group.add_argument(
        "--snorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable S-norm scoring using a cohort of embeddings/prototypes.",
    )
    eval_group.add_argument(
        "--snorm-cohort-embeddings",
        type=str,
        default=None,
        help="NPZ with cohort embeddings (embeddings + utt_ids). Required when --snorm is enabled.",
    )
    eval_group.add_argument(
        "--snorm-eval-embeddings",
        type=str,
        default=None,
        help="Optional NPZ with eval embeddings (embeddings + utt_ids). If omitted, embeddings are extracted from the eval set.",
    )
    eval_group.add_argument(
        "--snorm-cohort-max",
        type=int,
        default=None,
        help="Optional max cohort size (subsample uniformly).",
    )
    eval_group.add_argument(
        "--snorm-cohort-seed",
        type=int,
        default=0,
        help="Random seed for cohort subsampling.",
    )
    eval_group.add_argument(
        "--snorm-batch-size",
        type=int,
        default=1024,
        help="Chunk size for S-norm cohort scoring.",
    )
    eval_group.add_argument(
        "--snorm-eps",
        type=float,
        default=1e-6,
        help="Stddev floor for S-norm normalization.",
    )
    eval_group.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Max samples for on-the-fly calibration (default: None/All).",
    )
    eval_group.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Optional dataset to use specifically for on-the-fly calibration (replaces the dev set of the main dataset).",
    )
    # endregion

    classifier_args.add_argument(
        "--sampling",
        type=str,
        help="Variant of sampling the data for training SkLearn mocels. One of: all, avg_pool, random_sample.",
        default="all",
    )
    # endregion

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for training and evaluation.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="If set, skip loading the eval set and running final evaluation after training.",
    )

    args = parser.parse_args()

    # Claim a GPU when running on SGE using safe_gpu
    # if args.sge:
    #     try:
    #         safe_gpu.claim_gpus()
    #         print(f"Claimed GPUs via safe_gpu: {safe_gpu.gpu_ids}")
    #     except RuntimeError as e:
    #         print(f"Failed to claim GPU via safe_gpu: {e}")
    #         exit(69)

    return args
