#!/usr/bin/env python3
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import build_model, get_dataloaders
from config import karolina_config, local_config, sge_config
from parse_arguments import parse_args

# trainers
from trainers.BaseFFTrainer import BaseFFTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_allowed_classes(raw: list[str] | None) -> list[str] | None:
    if raw is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        parts = str(item).split(",")
        for part in parts:
            value = part.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
    return normalized if normalized else None


def main():
    args = parse_args()

    if args.sge:
        config = sge_config
    elif args.karolina:
        config = karolina_config
    else:
        config = local_config

    if args.segment_seconds is not None:
        config = dict(config)
        config["segment_seconds"] = args.segment_seconds
        config["sample_rate"] = args.sample_rate

    args.allowed_classes = _normalize_allowed_classes(args.allowed_classes)
    if args.allowed_classes is not None:
        print(
            "Restricting MLAAD classes to: "
            + ", ".join(args.allowed_classes)
        )

    if (
        args.classifier in ["FF", "FFMulticlass"]
        and getattr(args, "num_classes", None) is None
        and "MLAAD" in args.dataset
        and "single" in args.dataset
    ):
        dataset_config = config.get("mlaad_single", config["mlaad"])
        train_protocol_name = args.train_protocol or dataset_config["train_protocol"]
        protocol_root = Path(config["data_dir"]) / dataset_config["train_subdir"]
        train_protocol = (
            Path(train_protocol_name)
            if Path(train_protocol_name).is_absolute()
            else protocol_root / train_protocol_name
        )
        df = pd.read_csv(train_protocol)
        if "model_name" not in df.columns:
            raise ValueError(f"MLAAD single protocol missing model_name: {train_protocol}")
        model_names = df["model_name"].dropna().astype(str)
        if args.allowed_classes is not None:
            allowed_set = set(args.allowed_classes)
            present = set(model_names.unique())
            missing = sorted(allowed_set - present)
            model_names = model_names[model_names.isin(allowed_set)]
            if missing:
                print(
                    "[num_classes] Warning: allowed_classes not present in train protocol: "
                    + ", ".join(missing)
                )
            if model_names.empty:
                raise ValueError("allowed_classes filter removed all MLAAD training samples.")
        args.num_classes = int(model_names.nunique())
        print(f"Inferred num_classes={args.num_classes} from {train_protocol}.")

    if (
        args.classifier == "FFCosineJoint"
        and getattr(args, "num_classes", None) is None
        and args.allowed_classes is not None
    ):
        args.num_classes = int(len(args.allowed_classes))
        print(f"Inferred num_classes={args.num_classes} from allowed_classes for FFCosineJoint.")

    set_seed(args.seed)
    print(f"Using random seed {args.seed}.")

    model, trainer = build_model(args)
    if args.lr is not None:
        print(f"Overriding optimizer LR with: {args.lr}")
        for param_group in trainer.optimizer.param_groups:
            if param_group.get("group") == "extractor":
                continue
            param_group["lr"] = args.lr

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.set_output_dir(output_dir)

    if isinstance(trainer, BaseFFTrainer):
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
        trainer.set_amp_eval(args.amp_eval, dtype=amp_dtype)
        trainer.set_amp_train(args.amp_train, dtype=amp_dtype)
        trainer.set_max_train_batches(args.max_train_batches)
        trainer.set_grad_accum_steps(args.grad_accum_steps)
        trainer.configure_lr_ramp(
            args.lr_ramp,
            start_mult=args.lr_ramp_start_mult,
            target_mult=args.lr_ramp_target_mult,
            ramp_steps=args.lr_ramp_steps,
        )
        trainer.configure_epoch_lr_multipliers(args.lr_epoch_mults)

    if args.checkpoint:
        trainer.load_model(args.checkpoint)
        print(f"Loaded model from checkpoint {args.checkpoint}.")

    load_eval_split = not args.skip_eval and not args.dev_only

    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
        load_eval=load_eval_split,
        train_protocol=args.train_protocol,
        dev_protocol=args.dev_protocol,
        eval_protocol=args.eval_protocol,
        allowed_classes=args.allowed_classes,
        curriculum_easy_csv=args.curriculum_easy_csv,
        curriculum_hard_csv=args.curriculum_hard_csv,
        curriculum_steps_per_epoch=args.curriculum_steps_per_epoch,
        curriculum_three_stream=args.curriculum_three_stream,
        curriculum_hard_neg_ratio=args.curriculum_hard_neg_ratio,
        curriculum_aug_csv=args.curriculum_aug_csv,
        curriculum_aug_ratio=args.curriculum_aug_ratio,
        curriculum_easy_ratio=args.curriculum_easy_ratio,
        curriculum_hard_ratio=args.curriculum_hard_ratio,
        curriculum_pairs_per_epoch=args.curriculum_pairs_per_epoch,
        curriculum_total_epochs=args.num_epochs,
        benign_aug_prob_min=args.benign_aug_prob_min,
        benign_aug_prob_max=args.benign_aug_prob_max,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        dev_batch_size=args.dev_batch_size,
        train_num_workers=args.train_num_workers,
        dev_num_workers=args.dev_num_workers,
        train_prefetch_factor=args.train_prefetch_factor,
        dev_prefetch_factor=args.dev_prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    # TODO: Implement training of MHFA and AASIST with SkLearn models

    print(f"Training on {type(train_dataloader.dataset).__name__} dataloader.")

    if args.dev_only:
        print("Dev-only flag enabled; skipping training.")
        if isinstance(trainer, BaseFFTrainer):
            trainer._clear_cuda_cache()
            val_loss, val_accuracy, eer = trainer.val(val_dataloader)
            trainer._clear_cuda_cache()
            eer_display = "None" if eer is None else f"{eer*100:.2f}%"
            print(f"[Dev] loss: {val_loss}, accuracy: {val_accuracy*100:.2f}%, EER: {eer_display}")
        else:
            raise ValueError("Unsupported trainer type for dev-only inference.")
        return

    # Train the model
    if isinstance(trainer, BaseFFTrainer):
        # Default value of numepochs = 50
        trainer.train(
            train_dataloader,
            val_dataloader,
            numepochs=args.num_epochs,
            start_epoch=args.start_epoch,
            validation_interval=args.val_interval,
            stop_on_plateau=args.stop_on_plateau,
            patience=args.patience,
        )
        if not args.skip_eval and eval_dataloader is not None:
            trainer.eval(eval_dataloader, subtitle=str(args.num_epochs))  # Eval after training

    else:
        # Should not happen, should inherit from BaseFFTrainer
        raise ValueError("Invalid trainer, should inherit from BaseFFTrainer.")


if __name__ == "__main__":
    main()
