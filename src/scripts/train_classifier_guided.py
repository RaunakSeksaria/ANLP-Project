#!/usr/bin/env python3
"""Two-GPU source-only LED training with frozen classifier self-critical reward."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.led_triplet import load_led_triplet  # noqa: E402
from src.training.classifier_guided_core import (  # noqa: E402
    ClassifierGuidedLED, FrozenPoliticsClassifier, SourceTripletDataset,
    collate_sources, file_sha256,
)
from src.training.overfit_baseline import seed_everything  # noqa: E402
from src.training.train_joint import optimizer_parameter_groups  # noqa: E402
from src.training.train_joint_ddp import distributed_group_size, is_real_sample  # noqa: E402


METRICS = (
    "total", "pg", "ranking", "attraction", "auxiliary", "positive_distance",
    "negative_distance", "sampled_reward", "greedy_reward", "sampled_nll",
    "greedy_nll", "sampled_length", "greedy_length", "sampled_empty",
    "greedy_empty", "sampled_truncated", "greedy_truncated", "valid_triplet", "examples",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", default="allenai/led-base-16384")
    parser.add_argument("--triplet-lambda", type=float, choices=(0.0, 0.35), required=True)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-ratio", type=float, default=.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--attention-window", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-one-update", action="store_true")
    parser.add_argument("--smoke-max-new-tokens", type=int, default=8)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="anlp-project")
    parser.add_argument("--wandb-entity", default="manasagrawal206-iiit-hyderabad")
    parser.add_argument("--wandb-run-name", required=True)
    args = parser.parse_args()
    if args.model_name != "allenai/led-base-16384":
        parser.error("both experiments must start from allenai/led-base-16384")
    if args.train_batch_size != 1 or args.eval_batch_size != 1 or args.gradient_accumulation_steps != 4:
        parser.error("matched two-GPU experiment requires microbatch=1 and accumulation=4")
    if args.max_input_length != 512 or args.generation_max_new_tokens != 512 or args.attention_window != 512:
        parser.error("matched experiment requires 512-token input, generation, and attention window")
    if args.max_epochs != 10 or args.seed != 42:
        parser.error("matched experiment requires 10 epochs and seed 42")
    if args.learning_rate != 3e-5 or args.weight_decay != .01 or args.warmup_ratio != .1:
        parser.error("matched optimizer requires LR=3e-5, weight decay=.01, warmup ratio=.1")
    if args.max_grad_norm != 1.0 or args.early_stopping_patience != 3 or args.num_workers != 0:
        parser.error("matched clipping, patience, and worker settings differ")
    if args.smoke_one_update and not args.disable_wandb:
        parser.error("bounded smoke checks require --disable-wandb")
    if args.require_wandb and args.disable_wandb:
        parser.error("--require-wandb conflicts with --disable-wandb")
    if args.smoke_max_new_tokens < 1:
        parser.error("--smoke-max-new-tokens must be positive")
    return args


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@torch.no_grad()
def rollouts(model: ClassifierGuidedLED, batch: dict, tokenizer, classifier,
             max_new_tokens: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[str], list[str]]:
    led = model.model.led
    inputs = {key: batch[key] for key in (
        "anchor_input_ids", "anchor_attention_mask", "anchor_global_attention_mask")}
    common = {"num_beams": 1, "max_new_tokens": max_new_tokens,
              "pad_token_id": led.config.pad_token_id, "eos_token_id": led.config.eos_token_id,
              "decoder_start_token_id": led.config.decoder_start_token_id, "use_cache": True}
    sampled = led.generate(**inputs, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, **common)
    greedy = led.generate(**inputs, do_sample=False, **common)
    # Existing neutral targets contain body text, without generated titles.
    # Treat the complete decoded rewrite as body text for classifier scoring.
    sampled_text = tokenizer.batch_decode(sampled, skip_special_tokens=True)
    greedy_text = tokenizer.batch_decode(greedy, skip_special_tokens=True)
    sr, snll, strunc = classifier.score(sampled_text)
    gr, gnll, gtrunc = classifier.score(greedy_text)
    diagnostics = {"sampled_reward": sr, "greedy_reward": gr,
                   "sampled_nll": snll, "greedy_nll": gnll,
                   "sampled_truncated": strunc, "greedy_truncated": gtrunc,
                   "sampled_empty": torch.tensor([not text.strip() for text in sampled_text],
                                                 dtype=torch.float32, device=sr.device),
                   "greedy_empty": torch.tensor([not text.strip() for text in greedy_text],
                                                dtype=torch.float32, device=sr.device),
                   "greedy_length": (greedy[:, 1:] != led.config.pad_token_id).sum(dim=1).float()}
    return sampled, (sr - gr).detach().to(sampled.device), diagnostics, sampled_text, greedy_text


def initialize_wandb(args, report):
    if args.disable_wandb:
        return None
    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     name=args.wandb_run_name, config=report)
    if args.require_wandb and run is None:
        raise RuntimeError("required W&B run did not start")
    return run


def save_checkpoint(path: Path, model: ClassifierGuidedLED, tokenizer, optimizer,
                    scheduler, state: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "classifier_guided_state.json", state)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "state": state}, path / "trainer_state.pt")


@torch.no_grad()
def evaluate(model: ClassifierGuidedLED, loader: DataLoader, tokenizer,
             classifier: FrozenPoliticsClassifier, device: torch.device,
             max_new_tokens: int, output: Path, split: str, epoch: int) -> dict:
    totals = {name: 0.0 for name in (
        "reward", "negative_log_centre_probability", "length", "empty", "truncated")}
    records = []
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        led = model.model.led
        generated = led.generate(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
            global_attention_mask=batch["anchor_global_attention_mask"],
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            pad_token_id=led.config.pad_token_id,
            eos_token_id=led.config.eos_token_id,
            decoder_start_token_id=led.config.decoder_start_token_id, use_cache=True,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        reward, nll, truncated = classifier.score(texts)
        lengths = (generated[:, 1:] != led.config.pad_token_id).sum(dim=1)
        for i, text in enumerate(texts):
            record = {"article_id": batch["article_id"][i], "event_id": batch["event_id"][i],
                      "ideology": batch["ideology"][i], "generated_body": text,
                      "centre_reward": float(reward[i]), "negative_log_centre_probability": float(nll[i]),
                      "generated_length": int(lengths[i]), "empty": not bool(text.strip()),
                      "classifier_truncated": bool(truncated[i])}
            records.append(record)
            for key, value in (("reward", reward[i]),
                               ("negative_log_centre_probability", nll[i]),
                               ("length", lengths[i]), ("empty", record["empty"]),
                               ("truncated", truncated[i])):
                totals[key] += float(value)
    write_jsonl(output / f"{split}_predictions_epoch_{epoch:03d}.jsonl", records)
    count = len(records)
    return {"examples": count, "centre_reward": totals["reward"] / count,
            "negative_log_centre_probability": totals["negative_log_centre_probability"] / count,
            "generated_length": totals["length"] / count,
            "empty_output_rate": totals["empty"] / count,
            "classifier_truncation_rate": totals["truncated"] / count}


def run(args) -> None:
    if int(os.environ.get("WORLD_SIZE", "0")) != 2 or not torch.cuda.is_available():
        raise RuntimeError("launch with torchrun --nproc-per-node=2 on two allocated GPUs")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(days=5))
    rank = dist.get_rank()
    try:
        seed_everything(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not args.classifier_checkpoint.is_file():
            raise FileNotFoundError(f"Required politics_media.pt is absent: {args.classifier_checkpoint}")
        if rank == 0 and not args.smoke_one_update:
            if args.output_dir.exists():
                raise FileExistsError(f"output directory already exists: {args.output_dir}")
            args.output_dir.mkdir(parents=True)
        dist.barrier()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # The reward model is frozen and only scores decoded strings. Keeping it
        # on CPU leaves the two 11 GB GPUs for LED and its optimizer states.
        classifier = FrozenPoliticsClassifier(args.classifier_checkpoint, torch.device("cpu"))
        classifier_before_smoke = classifier.parameter_sha256() if args.smoke_one_update else None
        datasets = {split: SourceTripletDataset(args.data_dir / split / "triplets.jsonl",
                                               tokenizer, args.max_input_length)
                    for split in ("train", "validation", "test")}
        sampler = DistributedSampler(datasets["train"], num_replicas=2, rank=rank,
                                     shuffle=True, seed=args.seed, drop_last=False)
        collate = lambda rows: collate_sources(rows, tokenizer.pad_token_id)
        train_loader = DataLoader(datasets["train"], batch_size=1, sampler=sampler,
                                  collate_fn=collate, num_workers=0)
        updates_per_epoch = math.ceil(len(datasets["train"]) / 8)
        planned_updates = updates_per_epoch * args.max_epochs
        warmup_updates = round(planned_updates * args.warmup_ratio)
        report = {"mode": "classifier_guided_self_critical", "pretrained_model": args.model_name,
                  "classifier_checkpoint": str(args.classifier_checkpoint.resolve()),
                  "classifier_sha256": classifier.checkpoint_sha256,
                  "classifier_model_and_tokenizer": "launch/POLITICS",
                  "classifier_device": "cpu",
                  "classifier_labels": list(classifier.label_order),
                  "classifier_scoring_policy": classifier.scoring_policy,
                  "source_files": {split: str((args.data_dir / split / "triplets.jsonl").resolve())
                                   for split in datasets},
                  "source_file_sha256": {split: file_sha256(args.data_dir / split / "triplets.jsonl")
                                         for split in datasets},
                  "source_examples": {split: len(dataset) for split, dataset in datasets.items()},
                  "valid_triplets": {split: sum(row.valid_triplet for row in dataset.rows)
                                     for split, dataset in datasets.items()},
                  "triplet_lambda": args.triplet_lambda, "margin": .2, "beta": .3, "tau": .5,
                  "pooling": "first_token", "first_token_global_attention": True,
                  "global_effective_batch": 8, "per_gpu_microbatch": 1, "accumulation": 4,
                  "optimizer": "AdamW", "learning_rate": args.learning_rate,
                  "weight_decay": args.weight_decay, "warmup_ratio": args.warmup_ratio,
                  "max_grad_norm": args.max_grad_norm, "precision": "fp32",
                  "source_limit": args.max_input_length,
                  "reference_target_limit": "512 in supervised runs; unused here",
                  "train_rollout_max_new_tokens": args.generation_max_new_tokens,
                  "classifier_limit": 512, "attention_window": args.attention_window,
                  "seed": args.seed, "max_epochs": args.max_epochs,
                  "early_stopping_patience": args.early_stopping_patience,
                  "planned_optimizer_updates": planned_updates, "warmup_updates": warmup_updates,
                  "dropout_policy": "eval mode for rollout and differentiable recomputation",
                  "sample_policy": "full categorical, temperature=1, top_k=0, top_p=1, one sample",
                  "baseline_policy": "greedy no_grad",
                  "checkpoint_selection": "maximum full-validation greedy Centre reward; factual quality unmeasured",
                  "neutral_target_fields_used": False,
                  "content_reward": False, "reconstruction_loss": False, "kl_penalty": False,
                  "test_use": "selected checkpoint only"}
        if rank == 0 and not args.smoke_one_update:
            write_json(args.output_dir / "training_config.json", report)
        wandb_run = initialize_wandb(args, report) if rank == 0 and not args.smoke_one_update else None
        dist.barrier()
        led = load_led_triplet(args.model_name, attention_window=args.attention_window,
                               pooling="first_token").to(device)
        led.led.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapped = ClassifierGuidedLED(led).to(device)
        wrapped.eval()  # Rollout and recomputation share a dropout-free policy.
        ddp = DistributedDataParallel(wrapped, device_ids=[local_rank], output_device=local_rank,
                                      broadcast_buffers=False)
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(wrapped, args.weight_decay),
                                      lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_updates, planned_updates)
        val_loader = test_loader = None
        if rank == 0 and not args.smoke_one_update:
            val_loader = DataLoader(datasets["validation"], batch_size=1, shuffle=False, collate_fn=collate)
            test_loader = DataLoader(datasets["test"], batch_size=1, shuffle=False, collate_fn=collate)
        best_reward, best_epoch, stale, global_step = -float("inf"), -1, 0, 0
        history = []
        if rank == 0 and not args.smoke_one_update:
            initial = evaluate(wrapped, val_loader, tokenizer, classifier, device,
                               args.generation_max_new_tokens, args.output_dir, "validation", 0)
            best_reward, best_epoch = initial["centre_reward"], 0
            state = {"epoch": 0, "global_optimizer_step": 0, "best_validation_centre_reward": best_reward,
                     "triplet_lambda": args.triplet_lambda, "classifier_sha256": classifier.checkpoint_sha256,
                     "planned_optimizer_updates": planned_updates, "configuration": report}
            save_checkpoint(args.output_dir / "best", wrapped, tokenizer, optimizer, scheduler, state)
            history.append({"epoch": 0, "validation": initial, "training": None})
            write_json(args.output_dir / "training_history.json", history)
            if wandb_run:
                wandb_run.log({f"validation/{k}": v for k, v in initial.items()}, step=0)
        dist.barrier()
        for epoch in range(1, args.max_epochs + 1):
            sampler.set_epoch(epoch)
            ddp.eval()
            optimizer.zero_grad(set_to_none=True)
            totals = dict.fromkeys(METRICS, 0.0)
            group = dict.fromkeys(METRICS, 0.0)
            started = time.perf_counter()
            for index, cpu_batch in enumerate(train_loader):
                batch = to_device(cpu_batch, device)
                real = is_real_sample(index, rank, 2, len(datasets["train"]))
                update_index = index // args.gradient_accumulation_steps
                group_size = distributed_group_size(len(datasets["train"]), update_index, 8)
                boundary = ((index + 1) % args.gradient_accumulation_steps == 0
                            or index + 1 == len(train_loader))
                rollout_tokens = args.smoke_max_new_tokens if args.smoke_one_update else args.generation_max_new_tokens
                sampled, advantage, diagnostic, sampled_text, greedy_text = rollouts(
                    wrapped, batch, tokenizer, classifier, rollout_tokens
                )
                with nullcontext() if boundary else ddp.no_sync():
                    result = ddp(batch, sampled, advantage, args.triplet_lambda)
                    if not torch.isfinite(result["total"]):
                        raise FloatingPointError("non-finite classifier-guided objective")
                    scale = 2 / group_size if real else 0.0
                    (result["total"] * scale).backward()
                if real:
                    values = {
                        "total": result["total"], "pg": result["pg"],
                        "ranking": result["ranking"], "attraction": result["attraction"],
                        "auxiliary": result["auxiliary"],
                        "positive_distance": result["positive_distance"],
                        "negative_distance": result["negative_distance"],
                        "sampled_reward": diagnostic["sampled_reward"].mean(),
                        "greedy_reward": diagnostic["greedy_reward"].mean(),
                        "sampled_nll": diagnostic["sampled_nll"].mean(),
                        "greedy_nll": diagnostic["greedy_nll"].mean(),
                        "sampled_length": result["lengths"],
                        "greedy_length": diagnostic["greedy_length"].mean(),
                        "sampled_empty": diagnostic["sampled_empty"].mean(),
                        "greedy_empty": diagnostic["greedy_empty"].mean(),
                        "sampled_truncated": diagnostic["sampled_truncated"].mean(),
                        "greedy_truncated": diagnostic["greedy_truncated"].mean(),
                        "valid_triplet": batch["valid_triplet"].float().mean(),
                        "examples": torch.tensor(1., device=device),
                    }
                    for key, value in values.items():
                        group[key] += float(value.detach())
                        totals[key] += float(value.detach())
                if not boundary:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), args.max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                tensor = torch.tensor([group[key] for key in METRICS], dtype=torch.float64, device=device)
                dist.all_reduce(tensor)
                if rank == 0 and not args.smoke_one_update:
                    reduced = dict(zip(METRICS, tensor.tolist()))
                    count = reduced["examples"]
                    logged = {key: reduced[key] / count for key in METRICS if key != "examples"}
                    if args.triplet_lambda:
                        valid_count = reduced["valid_triplet"]
                        logged["positive_distance"] = reduced["positive_distance"] / valid_count if valid_count else None
                        logged["negative_distance"] = reduced["negative_distance"] / valid_count if valid_count else None
                    else:
                        logged["positive_distance"] = None
                        logged["negative_distance"] = None
                    logged["weighted_auxiliary"] = args.triplet_lambda * logged["auxiliary"]
                    logged.update(epoch=epoch, global_optimizer_step=global_step,
                                  learning_rate=optimizer.param_groups[0]["lr"],
                                  gradient_norm=float(norm))
                    with (args.output_dir / "step_metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(logged) + "\n")
                    if wandb_run:
                        wandb_run.log({f"train/{key}": value for key, value in logged.items()
                                       if key not in {"epoch", "global_optimizer_step"} and value is not None},
                                      step=global_step)
                group = dict.fromkeys(METRICS, 0.0)
                if args.smoke_one_update:
                    if any(parameter.grad is not None for parameter in classifier.model.parameters()):
                        raise AssertionError("frozen classifier received gradients")
                    if classifier.parameter_sha256() != classifier_before_smoke:
                        raise AssertionError("frozen classifier weights changed")
                    if rank == 0:
                        print(f"SMOKE PASS lambda={args.triplet_lambda} one optimizer update; "
                              f"PG={float(result['pg']):.5f} auxiliary={float(result['auxiliary']):.5f}; "
                              f"no checkpoint saved", flush=True)
                    return
            tensor = torch.tensor([totals[key] for key in METRICS], dtype=torch.float64, device=device)
            dist.all_reduce(tensor)
            epoch_totals = dict(zip(METRICS, tensor.tolist()))
            if epoch_totals["examples"] != len(datasets["train"]):
                raise ValueError("distributed epoch did not cover all real examples exactly once")
            if rank == 0:
                train_metrics = {key: epoch_totals[key] / epoch_totals["examples"]
                                 for key in METRICS if key != "examples"}
                if args.triplet_lambda:
                    valid_count = epoch_totals["valid_triplet"]
                    train_metrics["positive_distance"] = epoch_totals["positive_distance"] / valid_count if valid_count else None
                    train_metrics["negative_distance"] = epoch_totals["negative_distance"] / valid_count if valid_count else None
                else:
                    train_metrics["positive_distance"] = None
                    train_metrics["negative_distance"] = None
                train_metrics["weighted_auxiliary"] = args.triplet_lambda * train_metrics["auxiliary"]
                validation = evaluate(wrapped, val_loader, tokenizer, classifier, device,
                                      args.generation_max_new_tokens, args.output_dir,
                                      "validation", epoch)
                improved = validation["centre_reward"] > best_reward
                if improved:
                    best_reward, best_epoch, stale = validation["centre_reward"], epoch, 0
                    state = {"epoch": epoch, "global_optimizer_step": global_step,
                             "best_validation_centre_reward": best_reward,
                             "triplet_lambda": args.triplet_lambda,
                             "classifier_sha256": classifier.checkpoint_sha256,
                             "planned_optimizer_updates": planned_updates, "configuration": report}
                    save_checkpoint(args.output_dir / "best", wrapped, tokenizer,
                                    optimizer, scheduler, state)
                else:
                    stale += 1
                history.append({"epoch": epoch, "global_optimizer_step": global_step,
                                "training": train_metrics, "validation": validation,
                                "improved": improved, "duration_seconds": time.perf_counter() - started})
                write_json(args.output_dir / "training_history.json", history)
                if wandb_run:
                    wandb_run.log({f"train/epoch_{k}": v for k, v in train_metrics.items()
                                   if v is not None}, step=global_step)
                    wandb_run.log({f"validation/{k}": v for k, v in validation.items()}, step=global_step)
                print(f"epoch {epoch} reward={validation['centre_reward']:.5f} "
                      f"best={best_reward:.5f} PG={train_metrics['pg']:.5f}", flush=True)
            flag = torch.tensor([int(stale >= args.early_stopping_patience) if rank == 0 else 0],
                                device=device)
            dist.broadcast(flag, src=0)
            if flag.item():
                break
        if rank == 0:
            best = args.output_dir / "best"
            selected = ClassifierGuidedLED(load_led_triplet(
                best, attention_window=args.attention_window, pooling="first_token"
            ).to(device)).eval()
            test = evaluate(selected, test_loader, tokenizer, classifier, device,
                            args.generation_max_new_tokens, args.output_dir, "test", best_epoch)
            write_json(args.output_dir / "test_metrics_selected.json", test)
            write_json(args.output_dir / "run_summary.json", {
                "best_epoch": best_epoch, "best_validation_centre_reward": best_reward,
                "test": test, "selection_warning": "Centre reward does not assess factual quality"})
            if wandb_run:
                wandb_run.log({f"test/{key}": value for key, value in test.items()}, step=global_step)
                wandb_run.finish()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"CLASSIFIER-GUIDED TRAINING ERROR: {exc}", file=sys.stderr, flush=True)
        raise
