#!/usr/bin/env python3
"""Train or dry-run the matched LED generation-plus-triplet experiment."""

from __future__ import annotations

import argparse
import errno
import sys
from pathlib import Path
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_joint import JointTrainingConfig, run_joint_training  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--triplet-lambda", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-project", default="anlp-project")
    parser.add_argument("--wandb-entity", default="manasagrawal206-iiit-hyderabad")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--require-wandb", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> JointTrainingConfig:
    return JointTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        generation_max_new_tokens=args.generation_max_new_tokens,
        triplet_lambda=args.triplet_lambda,
        triplet_margin=args.triplet_margin,
        attention_window=args.attention_window,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        precision=args.precision,
        overwrite_output_dir=args.overwrite_output_dir,
        dry_run=args.dry_run,
        wandb_enabled=not args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_required=args.require_wandb,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run_joint_training(config_from_args(parse_args(argv)))
    except torch.cuda.OutOfMemoryError as exc:
        print(f"CUDA OUT OF MEMORY: {exc}", file=sys.stderr)
        print(
            "The fixed experiment uses microbatch size 1 and checkpointing. "
            "Report the OOM and available GPU memory; do not silently change settings.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"JOINT TRAINING ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, OSError) and exc.errno in (errno.EDQUOT, errno.ENOSPC):
            print(
                "Output storage is full or its quota is exhausted. Select an output "
                "directory on storage with sufficient quota (the launchers default "
                "to compute-node scratch). Existing partial checkpoints may be "
                "incomplete; use a fresh run directory.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
