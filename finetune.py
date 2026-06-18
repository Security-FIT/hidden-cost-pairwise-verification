#!/usr/bin/env python3
from pathlib import Path

from common import build_model, get_dataloaders
from config import local_config, sge_config
from parse_arguments import parse_args
from trainers.BaseFFTrainer import BaseFFTrainer


def main():
    args = parse_args()

    config = sge_config if args.sge else local_config

    model, trainer = build_model(args)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.set_output_dir(output_dir)

    if isinstance(trainer, BaseFFTrainer):
        trainer.set_amp_eval(args.amp_eval)
        trainer.set_max_train_batches(args.max_train_batches)
        trainer.set_grad_accum_steps(args.grad_accum_steps)

    print(f"Trainer: {type(trainer).__name__}")

    # Load the model from the checkpoint
    if args.checkpoint:
        trainer.load_model(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}.")
    else:
        raise ValueError("Checkpoint must be specified when only evaluating.")

    # Load the datasets
    train_dataloader, val_dataloader, eval_dataloader = get_dataloaders(
        dataset=args.dataset,
        config=config,
        lstm=True if "LSTM" in args.classifier else False,
        augment=args.augment,
        train_batch_size=args.train_batch_size,
        dev_batch_size=args.dev_batch_size,
        train_num_workers=args.train_num_workers,
        dev_num_workers=args.dev_num_workers,
        train_prefetch_factor=args.train_prefetch_factor,
        dev_prefetch_factor=args.dev_prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    print(f"Fine-tuning {type(model).__name__} on {type(train_dataloader.dataset).__name__} dataloader.")

    # Fine-tune the model
    if isinstance(trainer, BaseFFTrainer):
        trainer.finetune(train_dataloader, eval_dataloader, numepochs=8, finetune_ssl=True)
        # trainer.eval(eval_dataloader, subtitle="finetune")
    else:
        raise NotImplementedError("Fine-tuning is only implemented for FF models.")
    

if __name__ == "__main__":
    main()
