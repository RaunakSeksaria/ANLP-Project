"""Two-GPU training path; validation and artifacts remain rank-zero only."""

from __future__ import annotations

import gc
import itertools
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.baseline_dataset import LEDBaselineCollator, LEDBaselineDataset
from src.data.triplet_dataset import LEDTripletCollator, create_triplet_dataloaders
from src.models.led_triplet import load_led_triplet
from src.training.overfit_baseline import (
    prepare_output_directory, seed_everything, token_accuracy_counts,
)
from src.training.train_baseline import atomic_write_json
from src.training.train_joint import (
    JointEarlyStopping, JointTrainEpoch, JointTrainingConfig,
    VALIDATION_GENERATION_SUBSET_SEED,
    _build_baseline_evaluation, _cuda_memory,
    _free_generation_dict, _generation_dict, _model_inputs, _move_tensors,
    append_jsonl, build_reproducibility_report, calculate_joint_steps,
    evaluate_free_generation, evaluate_generation, evaluate_teacher_forced,
    evaluate_triplets, initialize_wandb, log_epoch_to_wandb,
    optimizer_parameter_groups, save_joint_checkpoint, write_jsonl,
)


# Sums are reduced across ranks before any denominator is applied.
METRIC_NAMES = (
    "generation", "triplet", "ranking", "positive", "weighted_positive",
    "correct_tokens", "target_tokens", "examples", "valid", "active",
    "attraction_active", "valid_triplet", "positive_distance", "negative_distance",
)


def distributed_group_size(train_examples: int, update_index: int, global_batch: int) -> int:
    """Real examples in a global accumulation group, including the short last group."""
    return min(global_batch, max(0, train_examples - update_index * global_batch))


def is_real_sample(local_index: int, rank: int, world_size: int, train_examples: int) -> bool:
    """Identify DistributedSampler's final padding item without training on it."""
    return local_index * world_size + rank < train_examples


def _reduce_metrics(values: dict[str, float], device: torch.device) -> dict[str, float]:
    tensor = torch.tensor([values[name] for name in METRIC_NAMES], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return dict(zip(METRIC_NAMES, tensor.tolist()))


def train_joint_epoch_distributed(
    model: DistributedDataParallel,
    loader: DataLoader,
    sampler: DistributedSampler,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: JointTrainingConfig,
    *,
    epoch: int,
    global_step: int,
    train_examples: int,
    step_log_path: Path | None,
    wandb_run: Any | None,
) -> JointTrainEpoch:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    sampler.set_epoch(epoch)
    model.train()
    model.module.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    totals = dict.fromkeys(METRIC_NAMES, 0.0)
    group = dict.fromkeys(METRIC_NAMES, 0.0)
    updates = 0
    total_microbatches = len(loader)
    global_batch = world_size * config.train_batch_size * config.gradient_accumulation_steps
    for index, cpu_batch in enumerate(loader):
        batch = _move_tensors(cpu_batch, device)
        real = is_real_sample(index, rank, world_size, train_examples)
        update_index = index // config.gradient_accumulation_steps
        group_size = distributed_group_size(train_examples, update_index, global_batch)
        boundary = (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == total_microbatches
        with nullcontext() if boundary else model.no_sync():
            output = model(
                **_model_inputs(batch),
                triplet_lambda=config.triplet_lambda,
                triplet_margin=config.triplet_margin,
                triplet_objective=config.triplet_objective,
                positive_distance_threshold=config.positive_distance_threshold,
                positive_beta=config.positive_beta,
            )
            if not bool(torch.isfinite(output.total_loss)):
                raise FloatingPointError("non-finite distributed objective")
            if not torch.allclose(
                output.total_loss,
                output.generation_loss + config.triplet_lambda * output.triplet_loss,
                rtol=1e-6, atol=1e-7,
            ):
                raise ValueError("distributed objective does not reconstruct its components")
            # DDP averages gradients across ranks; compensate once. The padded
            # sampler item runs a zero-weight forward solely to keep collectives aligned.
            scale = world_size / group_size if real else 0.0
            (output.total_loss * scale).backward()
        if real:
            valid_mask = batch["triplet_valid_mask"].bool()
            correct, _ = token_accuracy_counts(output.generation_logits, batch["labels"])
            counts = {
                "generation": float(output.generation_loss.item()),
                "triplet": float(output.triplet_loss.item()),
                "ranking": float(output.ranking_loss.item()),
                "positive": float(output.positive_loss.item()),
                "weighted_positive": float(output.weighted_positive_loss.item()),
                "correct_tokens": float(correct),
                "target_tokens": float(batch["labels"].ne(-100).sum().item()),
                "examples": 1.0,
                "valid": float(output.valid_count),
                "active": float(((output.per_example_triplet_loss > 0) & valid_mask).sum().item()),
                "attraction_active": float(((output.per_example_positive_loss > 0) & valid_mask).sum().item()),
                "valid_triplet": float(output.valid_triplet_mean.item()) * output.valid_count,
                "positive_distance": float(output.anchor_positive_distance[valid_mask].sum().item()),
                "negative_distance": float(output.anchor_negative_distance[valid_mask].sum().item()),
            }
            for name, value in counts.items():
                totals[name] += value
                group[name] += value
        if not boundary:
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("non-finite distributed gradient norm")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        reduced = _reduce_metrics(group, device)
        if rank == 0:
            count = reduced["examples"]
            valid = reduced["valid"]
            generation = reduced["generation"] / count
            triplet = reduced["triplet"] / count
            metrics = {
                "epoch": epoch,
                "global_optimizer_step": global_step + updates,
                "generation_loss": generation,
                "triplet_loss": triplet,
                "ranking_loss": reduced["ranking"] / count,
                "positive_loss": reduced["positive"] / count,
                "weighted_positive_loss": reduced["weighted_positive"] / count,
                "positive_beta": config.positive_beta if config.triplet_objective == "attraction" else 0.0,
                "weighted_triplet_loss": config.triplet_lambda * triplet,
                "total_loss": generation + config.triplet_lambda * triplet,
                "token_accuracy": reduced["correct_tokens"] / reduced["target_tokens"],
                "valid_triplet_fraction": valid / count,
                "active_triplet_fraction": reduced["active"] / valid if valid else 0.0,
                "attraction_active_fraction": reduced["attraction_active"] / valid if valid else 0.0,
                "mean_anchor_positive_distance": reduced["positive_distance"] / valid if valid else 0.0,
                "mean_anchor_negative_distance": reduced["negative_distance"] / valid if valid else 0.0,
                "gradient_norm_before_clipping": float(norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if step_log_path is not None:
                append_jsonl(step_log_path, metrics)
            if wandb_run is not None:
                wandb_run.log({f"train/step_{key}": value for key, value in metrics.items()
                               if key not in {"epoch", "global_optimizer_step"}},
                              step=global_step + updates)
        group = dict.fromkeys(METRIC_NAMES, 0.0)
    reduced = _reduce_metrics(totals, device)
    count = reduced["examples"]
    valid = reduced["valid"]
    if count != train_examples or reduced["target_tokens"] == 0:
        raise ValueError("distributed epoch did not cover each real training example exactly once")
    generation = reduced["generation"] / count
    triplet = reduced["triplet"] / count
    return JointTrainEpoch(
        generation_loss=generation,
        token_accuracy=reduced["correct_tokens"] / reduced["target_tokens"],
        triplet_loss=triplet,
        weighted_triplet_loss=config.triplet_lambda * triplet,
        valid_triplet_mean=reduced["valid_triplet"] / valid if valid else 0.0,
        total_loss=generation + config.triplet_lambda * triplet,
        valid_triplet_fraction=valid / count,
        active_triplet_fraction=reduced["active"] / valid if valid else 0.0,
        optimizer_steps=updates,
        ranking_loss=reduced["ranking"] / count,
        positive_loss=reduced["positive"] / count,
        weighted_positive_loss=reduced["weighted_positive"] / count,
        attraction_active_fraction=reduced["attraction_active"] / valid if valid else 0.0,
        mean_anchor_positive_distance=reduced["positive_distance"] / valid if valid else 0.0,
        mean_anchor_negative_distance=reduced["negative_distance"] / valid if valid else 0.0,
        positive_beta=config.positive_beta if config.triplet_objective == "attraction" else 0.0,
    )


def run_joint_training_ddp(config: JointTrainingConfig) -> dict[str, Any]:
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("two-GPU DDP requires CUDA")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 2:
        raise RuntimeError("launch two processes on one node using torchrun --nproc_per_node=2")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Rank 1 waits while rank 0 performs full validation generation and writes
    # checkpoints; those phases can exceed NCCL's default collective timeout.
    dist.init_process_group("nccl", init_method="env://", timeout=timedelta(days=5))
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("expected exactly two distributed processes")
        bounded_check = config.startup_check or config.smoke_one_update
        if rank == 0 and not bounded_check:
            prepare_output_directory(config.output_dir, config.overwrite_output_dir)
        dist.barrier()
        seed_everything(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        bundle = create_triplet_dataloaders(
            config.data_dir, tokenizer, config.train_batch_size, config.eval_batch_size,
            config.max_input_length, config.max_target_length, config.seed, config.num_workers,
        )
        train_examples = len(bundle.datasets["train"])
        steps = calculate_joint_steps(train_examples, config)
        sampler = DistributedSampler(
            bundle.datasets["train"], num_replicas=2, rank=rank,
            shuffle=True, seed=config.seed, drop_last=False,
        )
        train_loader = DataLoader(
            bundle.datasets["train"], batch_size=1, sampler=sampler,
            collate_fn=LEDTripletCollator(tokenizer), num_workers=config.num_workers,
            drop_last=False,
        )
        if math.ceil(len(train_loader) / config.gradient_accumulation_steps) != steps.optimizer_steps_per_epoch:
            raise ValueError("distributed loader changed the planned optimizer update count")
        wandb_run = None
        if rank == 0 and not bounded_check:
            truncation = {split: dataset.truncation_statistics()
                          for split, dataset in bundle.datasets.items()}
            report = build_reproducibility_report(config, bundle, steps, truncation)
            report["distributed_policy"] = {
                "training": "DistributedSampler; padding item has zero objective weight",
                "evaluation": "rank zero evaluates each full split exactly once",
                "global_batch_size": config.effective_anchor_batch_size,
            }
            atomic_write_json(config.output_dir / "training_config.json", report)
            for split, dataset in bundle.datasets.items():
                write_jsonl(config.output_dir / f"resolved_triplets_{split}.jsonl",
                            dataset.assignment_records())
            wandb_run = initialize_wandb(config, report)
        dist.barrier()
        model = load_led_triplet(
            config.model_name, attention_window=config.attention_window, pooling=config.pooling
        ).to(device)
        model.led.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False
        distributed_model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False,
        )
        if config.startup_check:
            sampler.set_epoch(1)
            distributed_model.train()
            batch = _move_tensors(next(iter(train_loader)), device)
            output = distributed_model(
                **_model_inputs(batch),
                triplet_lambda=config.triplet_lambda,
                triplet_margin=config.triplet_margin,
                triplet_objective=config.triplet_objective,
                positive_distance_threshold=config.positive_distance_threshold,
                positive_beta=config.positive_beta,
            )
            if not bool(torch.isfinite(output.total_loss)):
                raise FloatingPointError("startup check produced a non-finite loss")
            output.total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                distributed_model.parameters(), config.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("startup check produced non-finite gradients")
            result = {
                "rank": rank,
                "pooling": config.pooling,
                "objective": config.triplet_objective,
                "generation_loss": float(output.generation_loss.item()),
                "ranking_loss": float(output.ranking_loss.item()),
                "positive_loss": float(output.positive_loss.item()),
                "auxiliary_loss": float(output.triplet_loss.item()),
                "total_loss": float(output.total_loss.item()),
                "gradient_norm": float(grad_norm.item()),
                "gpu_peak_allocated_gib": _cuda_memory(device)[0],
                "optimizer_updates": 0,
            }
            print(f"STARTUP CHECK COMPLETE: {result}", flush=True)
            dist.barrier()
            return result
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(distributed_model, config.weight_decay),
            lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=steps.warmup_steps,
            num_training_steps=steps.maximum_optimizer_steps,
        )
        if config.smoke_one_update:
            sampler.set_epoch(1)
            first_group = list(itertools.islice(train_loader, config.gradient_accumulation_steps))
            trained = train_joint_epoch_distributed(
                distributed_model, first_group, sampler, optimizer, scheduler, device,
                config, epoch=1, global_step=0,
                train_examples=config.effective_anchor_batch_size,
                step_log_path=None, wandb_run=None,
            )
            if trained.optimizer_steps != 1:
                raise AssertionError("bounded check must make exactly one optimizer update")
            result = {"rank": rank, "optimizer_updates": trained.optimizer_steps,
                      "generation_loss": trained.generation_loss,
                      "auxiliary_loss": trained.triplet_loss,
                      "total_loss": trained.total_loss,
                      "decoding_strategy": config.decoding_strategy,
                      "artifacts_written": False}
            print(f"ONE-UPDATE SMOKE COMPLETE: {result}", flush=True)
            dist.barrier()
            return result
        early = JointEarlyStopping(config.early_stopping_patience, config.early_stopping_min_delta)
        history: list[dict[str, Any]] = []
        global_step = 0
        if rank == 0:
            (validation, validation_loader, validation_view, validation_generation_loader,
             train_view, fixed_train_loader, validation_indices, train_indices) = (
                _build_baseline_evaluation(config.data_dir, tokenizer, config)
            )
            atomic_write_json(config.output_dir / "fixed_validation_generation_subset.json", {
                "seed": VALIDATION_GENERATION_SUBSET_SEED,
                "size": len(validation_indices), "indices": validation_indices,
                "example_ids": [validation_view.record(i).example_id for i in range(len(validation_view))],
            })
            atomic_write_json(config.output_dir / "fixed_train_subset.json", {
                "indices": train_indices,
                "example_ids": [train_view.record(i).example_id for i in range(len(train_view))],
            })

            def evaluate_epoch(epoch: int, checkpoint_id: str):
                teacher = evaluate_teacher_forced(model, validation_loader, device)
                generated = evaluate_free_generation(
                    model, validation_view, validation_generation_loader, tokenizer, device,
                    config.generation_settings, epoch=epoch, checkpoint_id=checkpoint_id,
                )
                fixed = evaluate_generation(
                    model, train_view, fixed_train_loader, tokenizer, device,
                    config.generation_settings, epoch=epoch, checkpoint_id=checkpoint_id,
                )
                triplet = evaluate_triplets(
                    model, bundle.dataloaders["validation"], device, config.triplet_margin,
                    config.triplet_objective, config.positive_distance_threshold, config.positive_beta,
                )
                write_jsonl(config.output_dir / f"validation_generation_subset_predictions_epoch_{epoch:03d}.jsonl",
                            generated.predictions)
                write_jsonl(config.output_dir / f"fixed_train_predictions_epoch_{epoch:03d}.jsonl",
                            fixed.predictions)
                return teacher, generated, fixed, triplet

            teacher, generated, fixed, triplet = evaluate_epoch(0, "pretrained_epoch_0")
            early.update(teacher.generation_loss, 0)
            history.append({
                "epoch": 0, "global_optimizer_step": 0, "training": None,
                "validation_teacher_forced": asdict(teacher),
                "validation_generation_subset": _free_generation_dict(generated),
                "fixed_train_generation": _generation_dict(fixed),
                "validation_triplet": asdict(triplet), "improved": True,
                "patience_counter": 0, "learning_rate": 0.0,
            })
            atomic_write_json(config.output_dir / "training_history.json", history)
            log_epoch_to_wandb(
                wandb_run, epoch=0, global_step=0, trained=None,
                validation_teacher=teacher, validation_generated=generated,
                fixed_train=fixed, validation_triplet=triplet, improved=True,
                patience_counter=0, learning_rate=0.0, duration_seconds=0.0,
                memory=_cuda_memory(device),
            )
            save_joint_checkpoint(config.output_dir / "best", model, tokenizer, optimizer,
                                  scheduler, config, early, bundle.train_generator,
                                  epoch=0, global_step=0)
            print(f"Epoch 0/{config.max_epochs} | val_loss={teacher.generation_loss:.4f} | pretrained", flush=True)
        dist.barrier()
        stopped_early = False
        for epoch in range(1, config.max_epochs + 1):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            trained = train_joint_epoch_distributed(
                distributed_model, train_loader, sampler, optimizer, scheduler, device,
                config, epoch=epoch, global_step=global_step, train_examples=train_examples,
                step_log_path=config.output_dir / "step_metrics.jsonl" if rank == 0 else None,
                wandb_run=wandb_run,
            )
            global_step += trained.optimizer_steps
            stop = torch.zeros(1, dtype=torch.int64, device=device)
            if rank == 0:
                teacher, generated, fixed, triplet = evaluate_epoch(epoch, f"epoch_{epoch}")
                improved = early.update(teacher.generation_loss, epoch)
                duration = time.perf_counter() - started
                history.append({
                    "epoch": epoch, "global_optimizer_step": global_step,
                    "training": asdict(trained),
                    "validation_teacher_forced": asdict(teacher),
                    "validation_generation_subset": _free_generation_dict(generated),
                    "fixed_train_generation": _generation_dict(fixed),
                    "validation_triplet": asdict(triplet), "improved": improved,
                    "patience_counter": early.non_improving,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "epoch_duration_seconds": duration,
                    "gpu_peak_allocated_gib": _cuda_memory(device)[0],
                    "gpu_peak_reserved_gib": _cuda_memory(device)[1],
                })
                atomic_write_json(config.output_dir / "training_history.json", history)
                log_epoch_to_wandb(
                    wandb_run, epoch=epoch, global_step=global_step, trained=trained,
                    validation_teacher=teacher, validation_generated=generated,
                    fixed_train=fixed, validation_triplet=triplet, improved=improved,
                    patience_counter=early.non_improving,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    duration_seconds=duration, memory=_cuda_memory(device),
                )
                if improved:
                    save_joint_checkpoint(config.output_dir / "best", model, tokenizer,
                                          optimizer, scheduler, config, early,
                                          bundle.train_generator, epoch=epoch,
                                          global_step=global_step)
                print(f"Epoch {epoch}/{config.max_epochs} | train_total_loss={trained.total_loss:.4f} "
                      f"val_loss={teacher.generation_loss:.4f} | improved={improved}", flush=True)
                stop[0] = int(early.should_stop)
            dist.broadcast(stop, src=0)
            if bool(stop.item()):
                stopped_early = True
                break
        if rank == 0:
            save_joint_checkpoint(config.output_dir / "last", model, tokenizer, optimizer,
                                  scheduler, config, early, bundle.train_generator,
                                  epoch=len(history) - 1, global_step=global_step)
        dist.barrier()
        del distributed_model, model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        summary: dict[str, Any] = {}
        if rank == 0:
            best_checkpoint = config.output_dir / "best"
            best_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
            best_validation = LEDBaselineDataset(
                config.data_dir / "validation" / "baseline.jsonl", best_tokenizer,
                config.max_input_length, config.max_target_length,
            )
            best_loader = DataLoader(
                best_validation, batch_size=config.eval_batch_size, shuffle=False,
                collate_fn=LEDBaselineCollator(best_tokenizer), num_workers=config.num_workers,
            )
            best_model = load_led_triplet(best_checkpoint, attention_window=config.attention_window).to(device)
            if best_model.pooling != config.pooling:
                raise ValueError("selected checkpoint did not preserve pooling method")
            best_result = evaluate_generation(
                best_model, best_validation, best_loader, best_tokenizer, device,
                config.generation_settings, epoch=early.best_epoch or 0,
                checkpoint_id=str(best_checkpoint.resolve()),
                description="Selected best · complete validation generation",
            )
            write_jsonl(config.output_dir / "validation_predictions_best.jsonl", best_result.predictions)
            atomic_write_json(config.output_dir / "validation_metrics_best.json", {
                **_generation_dict(best_result), "checkpoint": str(best_checkpoint.resolve()),
                "selected_epoch": early.best_epoch,
                "generation_settings": config.generation_settings,
            })
            summary = {
                "best_epoch": early.best_epoch,
                "best_validation_generation_loss": early.best_loss,
                "global_optimizer_steps": global_step,
                "early_stopping_occurred": stopped_early,
                "complete_validation_generation_examples": len(best_validation),
                "test_evaluated": False,
            }
            atomic_write_json(config.output_dir / "run_summary.json", summary)
            if wandb_run is not None:
                wandb_run.summary.update(summary)
                wandb_run.finish()
        dist.barrier()
        return summary
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
