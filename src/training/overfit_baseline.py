"""Deliberate eight-example overfitting check for the pretrained LED baseline."""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ContextManager, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, LEDForConditionalGeneration

from src.data.baseline_dataset import (
    LEDBaselineCollator,
    LEDBaselineDataset,
    PaddedBatch,
    TokenizerLike,
    UnpaddedExample,
)
from src.models.led_baseline import forward_led_baseline, load_led_baseline


SUBSET_SIZE = 8
GENERATION_SETTINGS = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 160,
    "early_stopping": False,
}
REQUIRED_RESULT_FIELDS = frozenset(
    {
        "model_name",
        "seed",
        "subset_size",
        "batch_size",
        "precision",
        "device",
        "learning_rate",
        "optimizer_settings",
        "completed_optimizer_steps",
        "initial_loss",
        "final_loss",
        "initial_token_accuracy",
        "final_token_accuracy",
        "initial_overlap_f1",
        "final_overlap_f1",
        "initial_exact_match_count",
        "final_exact_match_count",
        "early_stopping",
        "passed",
    }
)


@dataclass(frozen=True)
class OverfitConfig:
    data_dir: Path
    model_name: str = "allenai/led-base-16384"
    output_dir: Path = Path("outputs/overfit_baseline")
    batch_size: int = 1
    max_input_length: int = 256
    max_target_length: int = 256
    max_steps: int = 200
    eval_every: int = 10
    learning_rate: float = 1e-4
    seed: int = 42
    device: str = "auto"
    precision: str = "fp32"
    gradient_checkpointing: bool = False
    overwrite_output_dir: bool = False


@dataclass
class EvaluationMetrics:
    mean_loss: float
    token_accuracy: float
    overlap_f1: float
    exact_match_count: int
    predictions: list[dict[str, str]] = field(repr=False)
    generated_token_ids: list[list[int]] = field(repr=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_first_eight(dataset: LEDBaselineDataset) -> Subset[UnpaddedExample]:
    """Select exactly the first eight preserved-order training records."""
    if dataset.jsonl_path.parent.name != "train":
        raise ValueError("The overfitting subset must come from train/baseline.jsonl")
    if len(dataset) < SUBSET_SIZE:
        raise ValueError(f"Overfitting requires at least 8 training records; found {len(dataset)}")
    indices = list(range(SUBSET_SIZE))
    subset = Subset(dataset, indices)
    if len(subset) != SUBSET_SIZE:
        raise AssertionError("Exactly 8 examples must be selected")
    return subset


def prepare_output_directory(output_dir: Path, overwrite: bool) -> None:
    """Create an empty output directory while protecting existing artifacts."""
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; pass --overwrite-output-dir to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def validate_precision(device: torch.device, precision: str) -> None:
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise RuntimeError("fp16 is supported only on CUDA for this overfitting test")
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but this CUDA device does not support it")
        if device.type == "cpu" and not torch.amp.autocast_mode.is_autocast_available("cpu"):
            raise RuntimeError("bf16 was requested but CPU autocast is unavailable")


def autocast_context(device: torch.device, precision: str) -> ContextManager[Any]:
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def move_batch(batch: PaddedBatch, device: torch.device) -> PaddedBatch:
    return {name: tensor.to(device) for name, tensor in batch.items()}  # type: ignore[return-value]


def token_accuracy_counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int]:
    """Return correct and total teacher-forced tokens, ignoring -100 labels."""
    predictions = logits.argmax(dim=-1)
    if predictions.shape != labels.shape:
        raise ValueError("Prediction and label shapes differ")
    mask = labels.ne(-100)
    correct = int(((predictions == labels) & mask).sum().item())
    total = int(mask.sum().item())
    return correct, total


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def token_overlap_f1(prediction: str, reference: str) -> float:
    """Compute whitespace-token multiset overlap F1 after required normalization."""
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def passes_overfit_criteria(initial: EvaluationMetrics, current: EvaluationMetrics) -> bool:
    return (
        math.isfinite(current.mean_loss)
        and current.mean_loss <= 0.5 * initial.mean_loss
        and current.token_accuracy > initial.token_accuracy
    )


def decode_reference(tokenizer: Any, labels: torch.Tensor) -> str:
    """Decode labels after restoring ignored padding to the tokenizer pad ID."""
    copied = labels.detach().cpu().clone()
    copied[copied == -100] = tokenizer.pad_token_id
    return tokenizer.decode(copied, skip_special_tokens=True)


def generate_with_cache(
    model: LEDForConditionalGeneration,
    batch: PaddedBatch,
    device: torch.device,
    precision: str,
    generation_settings: Mapping[str, int | bool],
) -> torch.Tensor:
    """Generate without labels while temporarily enabling generation cache."""
    previous_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    try:
        with autocast_context(device, precision):
            return model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                global_attention_mask=batch["global_attention_mask"],
                **generation_settings,
            )
    finally:
        model.config.use_cache = previous_use_cache


def _generate(
    model: LEDForConditionalGeneration,
    batch: PaddedBatch,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    return generate_with_cache(model, batch, device, precision, GENERATION_SETTINGS)


def evaluate_subset(
    model: LEDForConditionalGeneration,
    dataset: LEDBaselineDataset,
    tokenizer: Any,
    device: torch.device,
    precision: str,
) -> EvaluationMetrics:
    """Evaluate all eight records with teacher forcing and greedy generation."""
    subset = select_first_eight(dataset)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=LEDBaselineCollator(tokenizer),
        drop_last=False,
    )
    model.eval()
    losses: list[float] = []
    correct_tokens = 0
    total_tokens = 0
    f1_values: list[float] = []
    exact_matches = 0
    predictions: list[dict[str, str]] = []
    generated_ids: list[list[int]] = []
    with torch.no_grad():
        for index, cpu_batch in enumerate(loader):
            batch = move_batch(cpu_batch, device)
            with autocast_context(device, precision):
                output = forward_led_baseline(model, **batch)
            losses.append(float(output.loss.detach().float().item()))
            correct, total = token_accuracy_counts(output.logits, batch["labels"])
            correct_tokens += correct
            total_tokens += total
            generated = _generate(model, batch, device, precision)
            generated_row = generated[0].detach().cpu().tolist()
            generated_ids.append(generated_row)
            source_ids = batch["input_ids"][0][batch["attention_mask"][0].bool()].detach().cpu()
            source_text = tokenizer.decode(source_ids, skip_special_tokens=True)
            reference_text = decode_reference(tokenizer, batch["labels"][0])
            generated_text = tokenizer.decode(generated_row, skip_special_tokens=True)
            f1_values.append(token_overlap_f1(generated_text, reference_text))
            exact_matches += int(normalize_text(generated_text) == normalize_text(reference_text))
            record = dataset.record(index)
            predictions.append(
                {
                    "example_id": record.example_id,
                    "event_id": record.event_id,
                    "input_ideology": record.input_ideology,
                    "source_text": source_text,
                    "reference_text": reference_text,
                    "generated_text": generated_text,
                }
            )
    if len(losses) != SUBSET_SIZE:
        raise AssertionError(f"Expected 8 evaluation examples, observed {len(losses)}")
    return EvaluationMetrics(
        mean_loss=sum(losses) / len(losses),
        token_accuracy=correct_tokens / total_tokens if total_tokens else 0.0,
        overlap_f1=sum(f1_values) / len(f1_values),
        exact_match_count=exact_matches,
        predictions=predictions,
        generated_token_ids=generated_ids,
    )


def preview(text: str, limit: int = 280) -> str:
    safe = " ".join(text.split())
    return safe if len(safe) <= limit else safe[: limit - 1] + "…"


def print_previews(label: str, metrics: EvaluationMetrics) -> None:
    print(f"{label} generation previews:")
    for prediction in metrics.predictions[:2]:
        print(f"  {prediction['example_id']} source: {preview(prediction['source_text'])}")
        print(f"  {prediction['example_id']} reference: {preview(prediction['reference_text'])}")
        print(f"  {prediction['example_id']} generated: {preview(prediction['generated_text'])}")


def print_evaluation(step: int, metrics: EvaluationMetrics) -> None:
    print(
        f"evaluation step={step} mean_loss={metrics.mean_loss:.6f} "
        f"token_accuracy={metrics.token_accuracy:.6f} overlap_f1={metrics.overlap_f1:.6f} "
        f"exact_match={metrics.exact_match_count}/8"
    )


def build_results(
    config: OverfitConfig,
    device: torch.device,
    completed_steps: int,
    initial: EvaluationMetrics,
    final: EvaluationMetrics,
    passed: bool,
    early_stopping: bool,
) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "seed": config.seed,
        "subset_size": SUBSET_SIZE,
        "batch_size": config.batch_size,
        "precision": config.precision,
        "device": str(device),
        "learning_rate": config.learning_rate,
        "optimizer_settings": {
            "name": "torch.optim.AdamW",
            "weight_decay": 0.0,
            "gradient_clip_max_norm": 1.0,
            "scheduler": None,
        },
        "completed_optimizer_steps": completed_steps,
        "initial_loss": initial.mean_loss,
        "final_loss": final.mean_loss,
        "initial_token_accuracy": initial.token_accuracy,
        "final_token_accuracy": final.token_accuracy,
        "initial_overlap_f1": initial.overlap_f1,
        "final_overlap_f1": final.overlap_f1,
        "initial_exact_match_count": initial.exact_match_count,
        "final_exact_match_count": final.exact_match_count,
        "early_stopping": early_stopping,
        "passed": passed,
    }


def write_results_json(path: Path, results: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_RESULT_FIELDS - results.keys())
    if missing:
        raise ValueError(f"Results are missing required fields: {', '.join(missing)}")
    path.write_text(json.dumps(dict(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_predictions(path: Path, predictions: Iterable[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            handle.write(json.dumps(dict(prediction), ensure_ascii=False, sort_keys=True) + "\n")


def _generate_first_two_ids(
    model: LEDForConditionalGeneration,
    dataset: LEDBaselineDataset,
    tokenizer: Any,
    device: torch.device,
    precision: str,
) -> list[list[int]]:
    collator = LEDBaselineCollator(tokenizer)
    results: list[list[int]] = []
    model.eval()
    with torch.no_grad():
        for index in range(2):
            batch = move_batch(collator([dataset[index]]), device)
            generated = _generate(model, batch, device, precision)
            results.append(generated[0].detach().cpu().tolist())
    return results


def validate_config(config: OverfitConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def run_overfit_test(config: OverfitConfig) -> dict[str, Any]:
    """Run the complete eight-example pretrained-LED correctness test."""
    validate_config(config)
    prepare_output_directory(config.output_dir, config.overwrite_output_dir)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    validate_precision(device, config.precision)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = LEDBaselineDataset(
        config.data_dir / "train" / "baseline.jsonl",
        tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )
    subset = select_first_eight(dataset)
    print(f"selected device={device}, precision={config.precision}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"model: {config.model_name}")
    print(f"selected example IDs: {[dataset.record(index).example_id for index in range(8)]}")

    model = load_led_baseline(config.model_name)
    model.to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"parameters: total={total_parameters}, trainable={trainable_parameters}")
    if total_parameters != trainable_parameters:
        raise AssertionError("The complete pretrained LED model must be trainable")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    collator = LEDBaselineCollator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    training_loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        drop_last=False,
    )
    initial_cpu_batch = next(iter(training_loader))
    initial_batch = move_batch(initial_cpu_batch, device)
    model.eval()
    with torch.no_grad(), autocast_context(device, config.precision):
        initial_output = forward_led_baseline(model, **initial_batch)
    print("initial forward pass:")
    for name, tensor in initial_batch.items():
        print(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
    print(f"  logits: shape={list(initial_output.logits.shape)}, dtype={initial_output.logits.dtype}")
    print(f"  loss={float(initial_output.loss.detach().float().item()):.6f}")

    initial_metrics = evaluate_subset(model, dataset, tokenizer, device, config.precision)
    final_metrics = initial_metrics
    print_evaluation(0, initial_metrics)
    print_previews("before training", initial_metrics)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    completed_steps = 0
    last_evaluated_step = 0
    passed = False
    while completed_steps < config.max_steps and not passed:
        for cpu_batch in training_loader:
            if completed_steps >= config.max_steps:
                break
            model.train()
            model.config.use_cache = False
            optimizer.zero_grad(set_to_none=True)
            batch = move_batch(cpu_batch, device)
            with autocast_context(device, config.precision):
                output = forward_led_baseline(model, **batch)
                loss = output.loss
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Training loss is not finite")
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not bool(torch.isfinite(gradient_norm).all()):
                raise FloatingPointError("Gradient norm is not finite")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            completed_steps += 1
            print(
                f"step={completed_steps} loss={float(loss.detach().float().item()):.6f} "
                f"gradient_norm={float(gradient_norm.detach().float().item()):.6f}"
            )
            if completed_steps % config.eval_every == 0:
                final_metrics = evaluate_subset(model, dataset, tokenizer, device, config.precision)
                last_evaluated_step = completed_steps
                print_evaluation(completed_steps, final_metrics)
                passed = passes_overfit_criteria(initial_metrics, final_metrics)
                if passed:
                    break

    if last_evaluated_step != completed_steps:
        final_metrics = evaluate_subset(model, dataset, tokenizer, device, config.precision)
        print_evaluation(completed_steps, final_metrics)
        passed = passes_overfit_criteria(initial_metrics, final_metrics)
    early_stopping = passed and completed_steps < config.max_steps
    print_previews("after training", final_metrics)
    results = build_results(
        config,
        device,
        completed_steps,
        initial_metrics,
        final_metrics,
        passed,
        early_stopping,
    )
    write_results_json(config.output_dir / "overfit_results.json", results)
    write_predictions(config.output_dir / "overfit_predictions.jsonl", final_metrics.predictions)

    if not passed:
        print("OVERFIT TEST FAILED")
        return results

    expected_first_two = final_metrics.generated_token_ids[:2]
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    del optimizer, model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reloaded_tokenizer = AutoTokenizer.from_pretrained(config.output_dir)
    reloaded_dataset = LEDBaselineDataset(
        config.data_dir / "train" / "baseline.jsonl",
        reloaded_tokenizer,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
    )
    reloaded_model = load_led_baseline(config.output_dir)
    reloaded_model.to(device)
    reloaded_model.eval()
    actual_first_two = _generate_first_two_ids(
        reloaded_model,
        reloaded_dataset,
        reloaded_tokenizer,
        device,
        config.precision,
    )
    if actual_first_two != expected_first_two:
        raise AssertionError("Reloaded checkpoint generation differs from pre-save generation")
    print("CHECKPOINT RELOAD VERIFIED")
    print(f"checkpoint: {config.output_dir.resolve()}")
    print("OVERFIT TEST PASSED")
    return results
