#!/usr/bin/env python3
"""Run exactly one real-data joint optimizer update without saving checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.triplet_dataset import create_triplet_dataloaders  # noqa: E402
from src.models.led_triplet import load_led_triplet  # noqa: E402
from src.training.train_joint import (  # noqa: E402
    JointTrainingConfig,
    _model_inputs,
    _move_tensors,
    optimizer_parameter_groups,
    resolve_device,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--triplet-lambda", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=512)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        device = resolve_device(args.device)
        if device.type != "cuda":
            raise RuntimeError("the real-data smoke update requires an allocated CUDA GPU")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        bundle = create_triplet_dataloaders(
            args.data_dir,
            tokenizer,
            1,
            1,
            max_input_length=args.max_input_length,
            max_target_length=args.max_target_length,
            seed=args.seed,
            num_workers=0,
        )
        model = load_led_triplet(
            args.model_name, attention_window=args.attention_window
        ).to(device)
        model.led.gradient_checkpointing_enable()
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(model, 0.01),
            lr=3e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        torch.cuda.reset_peak_memory_stats(device)
        batch = _move_tensors(next(iter(bundle.dataloaders["train"])), device)
        output = model(
            **_model_inputs(batch),
            triplet_lambda=args.triplet_lambda,
            triplet_margin=args.triplet_margin,
        )
        output.total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("smoke update produced a non-finite gradient norm")
        optimizer.step()
        report = {
            "example_ids": batch["example_id"],
            "generation_loss": float(output.generation_loss.item()),
            "triplet_loss": float(output.triplet_loss.item()),
            "weighted_triplet_loss": args.triplet_lambda * float(output.triplet_loss.item()),
            "total_loss": float(output.total_loss.item()),
            "gradient_norm_before_clipping": float(gradient_norm),
            "valid_triplets": output.valid_count,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "optimizer_updates": 1,
            "max_input_length": args.max_input_length,
            "max_target_length": args.max_target_length,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "attention_window": args.attention_window,
            "artifacts_written": False,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        print("SMOKE TEST COMPLETE — ONE UPDATE, NO CHECKPOINT WRITTEN")
    except Exception as exc:
        print(f"JOINT SMOKE TEST ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
