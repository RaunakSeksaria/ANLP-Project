"""Complete single-device supervised training pipeline for the LED baseline."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from src.data.baseline_dataset import (
    BaselineDataBundle,
    LEDBaselineDataset,
    PaddedBatch,
    create_baseline_dataloaders,
)
from src.models.led_baseline import forward_led_baseline, load_led_baseline
from src.training.decoding import decoding_settings
from src.training.overfit_baseline import (
    autocast_context,
    decode_reference,
    generate_with_cache,
    move_batch,
    normalize_text,
    prepare_output_directory,
    resolve_device,
    seed_everything,
    token_accuracy_counts,
    token_overlap_f1,
    validate_precision,
)


DEFAULT_MODEL_NAME = "allenai/led-base-16384"
GENERATION_DEFAULTS = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 160,
    "early_stopping": False,
}
REQUIRED_HISTORY_FIELDS = frozenset(
    {
        "epoch",
        "global_optimizer_step",
        "training_loss",
        "validation_loss",
        "validation_token_accuracy",
        "validation_token_overlap_f1",
        "validation_exact_match_count",
        "validation_exact_match_rate",
        "learning_rate",
        "epoch_duration_seconds",
        "gpu_peak_allocated_gib",
        "gpu_peak_reserved_gib",
        "improved_best_checkpoint",
        "patience_counter",
    }
)
REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "example_id",
        "event_id",
        "input_article_id",
        "input_ideology",
        "target_article_id",
        "source_text",
        "reference_text",
        "generated_text",
        "token_overlap_f1",
        "exact_match",
    }
)
REQUIRED_TEST_METRIC_FIELDS = frozenset(
    {
        "best_checkpoint_path",
        "best_validation_loss",
        "selected_epoch",
        "test_teacher_forced_loss",
        "test_token_accuracy",
        "mean_test_token_overlap_f1",
        "test_exact_match_count",
        "test_exact_match_rate",
        "test_examples",
        "total_training_optimizer_steps",
        "early_stopping_occurred",
        "total_training_duration_seconds",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
        "generation_settings",
    }
)


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: Path = Path("outputs/baseline")
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    max_input_length: int = 256
    max_target_length: int = 256
    generation_max_new_tokens: int = 160
    decoding_strategy: str = "greedy"
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    precision: str = "fp16"
    gradient_checkpointing: bool = False
    overwrite_output_dir: bool = False
    dry_run: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps

    @property
    def generation_settings(self) -> dict[str, int | bool]:
        return decoding_settings(
            {**GENERATION_DEFAULTS, "max_new_tokens": self.generation_max_new_tokens},
            self.decoding_strategy,
        )


@dataclass(frozen=True)
class StepCounts:
    train_examples: int
    training_microbatches_per_epoch: int
    optimizer_steps_per_epoch: int
    maximum_optimizer_steps: int
    warmup_steps: int


@dataclass(frozen=True)
class EpochTrainingResult:
    token_weighted_loss: float
    optimizer_steps: int


@dataclass
class EvaluationResult:
    teacher_forced_loss: float
    token_accuracy: float
    token_overlap_f1: float
    exact_match_count: int
    exact_match_rate: float
    examples: int
    predictions: list[dict[str, Any]] = field(repr=False)


@dataclass
class EarlyStoppingState:
    patience: int
    minimum_delta: float
    best_loss: float = math.inf
    best_epoch: int | None = None
    non_improving_evaluations: int = 0

    def update(self, validation_loss: float, epoch: int) -> bool:
        """Update from validation loss only and return whether it improved."""
        if validation_loss < self.best_loss - self.minimum_delta:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.non_improving_evaluations = 0
            return True
        self.non_improving_evaluations += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.non_improving_evaluations >= self.patience


def validate_training_config(config: TrainingConfig) -> None:
    decoding_settings(GENERATION_DEFAULTS, config.decoding_strategy)
    positive_integers = {
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_epochs": config.max_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "max_input_length": config.max_input_length,
        "max_target_length": config.max_target_length,
        "generation_max_new_tokens": config.generation_max_new_tokens,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if not 0.0 <= config.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {config.device}")
    if config.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {config.precision}")


def calculate_step_counts(train_examples: int, config: TrainingConfig) -> StepCounts:
    if train_examples <= 0:
        raise ValueError("Training split must contain at least one example")
    microbatches = math.ceil(train_examples / config.train_batch_size)
    optimizer_steps = math.ceil(microbatches / config.gradient_accumulation_steps)
    maximum_steps = optimizer_steps * config.max_epochs
    warmup_steps = round(maximum_steps * config.warmup_ratio)
    return StepCounts(
        train_examples=train_examples,
        training_microbatches_per_epoch=microbatches,
        optimizer_steps_per_epoch=optimizer_steps,
        maximum_optimizer_steps=maximum_steps,
        warmup_steps=warmup_steps,
    )


def count_jsonl_records(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline file not found: {path}")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            count += 1
    return count


def dataset_counts(data_dir: Path) -> dict[str, int]:
    return {
        split: count_jsonl_records(data_dir / split / "baseline.jsonl")
        for split in ("train", "validation", "test")
    }


def resolved_configuration(
    config: TrainingConfig,
    counts: Mapping[str, int],
    steps: StepCounts,
    resolved_device: str,
) -> dict[str, Any]:
    values = asdict(config)
    values["data_dir"] = str(config.data_dir.resolve())
    values["output_dir"] = str(config.output_dir.resolve())
    values["resolved_device"] = resolved_device
    values["single_device_only"] = True
    values["effective_batch_size"] = config.effective_batch_size
    values["dataset_counts"] = dict(counts)
    values["step_counts"] = asdict(steps)
    values["optimizer"] = {
        "name": "torch.optim.AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_max_norm": config.max_grad_norm,
    }
    values["scheduler"] = {
        "name": "linear_decay",
        "warmup_steps": steps.warmup_steps,
        "training_steps": steps.maximum_optimizer_steps,
    }
    values["generation_settings"] = config.generation_settings
    values["checkpoint_selection_metric"] = "lowest validation teacher-forced loss"
    values["determinism_note"] = (
        "Seeds and deterministic cuDNN settings are configured, but complete bitwise "
        "determinism is not guaranteed for every CUDA kernel."
    )
    return values


def run_dry_run(config: TrainingConfig) -> dict[str, Any]:
    """Validate/count using CPU file I/O only; never inspect or initialize CUDA."""
    validate_training_config(config)
    counts = dataset_counts(config.data_dir)
    steps = calculate_step_counts(counts["train"], config)
    if config.device == "cpu":
        device_description = "cpu"
    elif config.device == "cuda":
        device_description = "cuda:0 (availability checked only for a real run)"
    else:
        device_description = "auto (cuda:0 when available, otherwise cpu; deferred in dry-run)"
    resolved = resolved_configuration(config, counts, steps, device_description)
    print(json.dumps(resolved, indent=2, sort_keys=True))
    print("single-device execution confirmed; CUDA access is deferred")
    print("DRY RUN COMPLETE — NO MODEL LOADED, NO TRAINING PERFORMED")
    return resolved


def configure_reproducibility(seed: int) -> None:
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_single_device(requested: str) -> torch.device:
    """Resolve the run device, explicitly pinning all CUDA work to cuda:0."""
    device = resolve_device(requested)
    return torch.device("cuda:0") if device.type == "cuda" else torch.device("cpu")


def place_model_on_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move the unwrapped model to exactly one device."""
    moved = model.to(device)
    if moved is not model:
        raise AssertionError("Model.to unexpectedly returned a different model instance")
    return model


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON metadata file in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_prediction_jsonl(path: Path, predictions: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prediction in predictions:
            missing = sorted(REQUIRED_PREDICTION_FIELDS - prediction.keys())
            if missing:
                raise ValueError(f"Prediction is missing fields: {', '.join(missing)}")
            handle.write(json.dumps(dict(prediction), ensure_ascii=False, sort_keys=True) + "\n")


def gradients_are_finite(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[PaddedBatch],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    precision: str,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    global_optimizer_step: int = 0,
    epoch: int = 1,
    scaler: Any | None = None,
    show_progress: bool = False,
) -> EpochTrainingResult:
    """Train one complete epoch, including a final partial accumulation group."""
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    weighted_loss_sum = 0.0
    target_tokens = 0
    steps_this_epoch = 0
    microbatch_count = len(dataloader)
    progress = tqdm(
        dataloader,
        total=microbatch_count,
        desc=f"Epoch {epoch} · train",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        position=0,
        mininterval=1.0,
        maxinterval=5.0,
        disable=not (show_progress and sys.stderr.isatty()),
    )
    for microbatch_index, cpu_batch in enumerate(progress, 1):
        batch = move_batch(cpu_batch, device)
        with autocast_context(device, precision):
            output = forward_led_baseline(model, **batch)
            loss = output.loss
        unscaled_loss = float(loss.detach().float().item())
        token_count = int(batch["labels"].ne(-100).sum().item())
        weighted_loss_sum += unscaled_loss * token_count
        target_tokens += token_count
        backward_loss = loss / gradient_accumulation_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        progress.set_postfix(
            loss=f"{unscaled_loss:.4f}",
            accumulation=f"{((microbatch_index - 1) % gradient_accumulation_steps) + 1}/{gradient_accumulation_steps}",
            optimizer_step=global_optimizer_step + steps_this_epoch,
            refresh=False,
        )

        accumulation_boundary = microbatch_index % gradient_accumulation_steps == 0
        final_partial_group = microbatch_index == microbatch_count
        if not (accumulation_boundary or final_partial_group):
            continue
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        if not gradients_are_finite(model):
            raise FloatingPointError("Model gradients contain NaN or infinity")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if not bool(torch.isfinite(gradient_norm).all()):
            raise FloatingPointError("Gradient norm is not finite")
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        steps_this_epoch += 1
        current_step = global_optimizer_step + steps_this_epoch
        learning_rate = float(optimizer.param_groups[0]["lr"])
        progress.set_postfix(
            loss=f"{unscaled_loss:.4f}",
            grad_norm=f"{float(gradient_norm):.3f}",
            lr=f"{learning_rate:.3e}",
            optimizer_step=current_step,
            refresh=False,
        )
    if target_tokens == 0:
        raise ValueError("Training epoch contained no target tokens")
    return EpochTrainingResult(weighted_loss_sum / target_tokens, steps_this_epoch)


def evaluate_split(
    model: torch.nn.Module,
    dataset: LEDBaselineDataset,
    dataloader: DataLoader[PaddedBatch],
    tokenizer: Any,
    device: torch.device,
    precision: str,
    generation_settings: Mapping[str, int | bool],
    progress_description: str | None = None,
) -> EvaluationResult:
    """Evaluate an ordered split with teacher forcing and aligned generation."""
    model.eval()
    weighted_loss_sum = 0.0
    target_tokens = 0
    correct_tokens = 0
    f1_sum = 0.0
    exact_matches = 0
    predictions: list[dict[str, Any]] = []
    record_index = 0
    with torch.no_grad():
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=progress_description or "evaluation",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            position=0,
            mininterval=1.0,
            maxinterval=5.0,
            disable=progress_description is None or not sys.stderr.isatty(),
        )
        for cpu_batch in progress:
            batch = move_batch(cpu_batch, device)
            with autocast_context(device, precision):
                output = forward_led_baseline(model, **batch)
            batch_target_tokens = int(batch["labels"].ne(-100).sum().item())
            weighted_loss_sum += float(output.loss.detach().float().item()) * batch_target_tokens
            target_tokens += batch_target_tokens
            correct, total = token_accuracy_counts(output.logits, batch["labels"])
            correct_tokens += correct
            generated = generate_with_cache(
                model, batch, device, precision, generation_settings
            ).detach().cpu()
            for row in range(batch["input_ids"].shape[0]):
                record = dataset.record(record_index)
                reference_text = decode_reference(tokenizer, batch["labels"][row])
                generated_text = tokenizer.decode(generated[row], skip_special_tokens=True)
                overlap = token_overlap_f1(generated_text, reference_text)
                exact = normalize_text(generated_text) == normalize_text(reference_text)
                predictions.append(
                    {
                        "example_id": record.example_id,
                        "event_id": record.event_id,
                        "input_article_id": record.input_article_id,
                        "input_ideology": record.input_ideology,
                        "target_article_id": record.target_article_id,
                        "source_text": dataset.source_text(record_index),
                        "reference_text": reference_text,
                        "generated_text": generated_text,
                        "token_overlap_f1": overlap,
                        "exact_match": exact,
                    }
                )
                f1_sum += overlap
                exact_matches += int(exact)
                record_index += 1
            progress.set_postfix(
                loss=f"{weighted_loss_sum / target_tokens:.4f}",
                token_acc=f"{correct_tokens / target_tokens:.3f}",
                examples=record_index,
                refresh=False,
            )
    if record_index != len(dataset):
        raise AssertionError(
            f"Evaluation order/count mismatch: observed {record_index}, expected {len(dataset)}"
        )
    if target_tokens == 0 or record_index == 0:
        raise ValueError("Evaluation split contains no examples or target tokens")
    return EvaluationResult(
        teacher_forced_loss=weighted_loss_sum / target_tokens,
        token_accuracy=correct_tokens / target_tokens,
        token_overlap_f1=f1_sum / record_index,
        exact_match_count=exact_matches,
        exact_match_rate=exact_matches / record_index,
        examples=record_index,
        predictions=predictions,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    *,
    epoch: int,
    global_optimizer_step: int,
    early_stopping: EarlyStoppingState,
    config: TrainingConfig,
    steps: StepCounts,
) -> None:
    prepare_output_directory(checkpoint_dir, overwrite=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    metadata = {
        "completed_epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "best_validation_loss": early_stopping.best_loss,
        "patience_counter": early_stopping.non_improving_evaluations,
        "optimizer_configuration": {
            "name": "torch.optim.AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "gradient_clip_max_norm": config.max_grad_norm,
        },
        "scheduler_configuration": {
            "name": "linear_decay",
            "warmup_steps": steps.warmup_steps,
            "training_steps": steps.maximum_optimizer_steps,
        },
        "precision": config.precision,
        "seed": config.seed,
        "decoding_strategy": config.decoding_strategy,
        "generation_settings": config.generation_settings,
    }
    atomic_write_json(checkpoint_dir / "training_state.json", metadata)
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "metadata": metadata,
    }
    torch.save(state, checkpoint_dir / "trainer_state.pt")


def _memory_gib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    divisor = 1024**3
    return (
        torch.cuda.max_memory_allocated(device) / divisor,
        torch.cuda.max_memory_reserved(device) / divisor,
    )


def _start_epoch_timing(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _finish_epoch_timing(device: torch.device, started: float) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _history_entry(
    epoch: int,
    global_step: int,
    training: EpochTrainingResult,
    validation: EvaluationResult,
    learning_rate: float,
    duration: float,
    memory: tuple[float | None, float | None],
    improved: bool,
    patience_counter: int,
) -> dict[str, Any]:
    entry = {
        "epoch": epoch,
        "global_optimizer_step": global_step,
        "training_loss": training.token_weighted_loss,
        "validation_loss": validation.teacher_forced_loss,
        "validation_token_accuracy": validation.token_accuracy,
        "validation_token_overlap_f1": validation.token_overlap_f1,
        "validation_exact_match_count": validation.exact_match_count,
        "validation_exact_match_rate": validation.exact_match_rate,
        "learning_rate": learning_rate,
        "epoch_duration_seconds": duration,
        "gpu_peak_allocated_gib": memory[0],
        "gpu_peak_reserved_gib": memory[1],
        "improved_best_checkpoint": improved,
        "patience_counter": patience_counter,
    }
    assert REQUIRED_HISTORY_FIELDS <= entry.keys()
    return entry


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run supervised LED training and final best-checkpoint evaluation."""
    validate_training_config(config)
    if config.dry_run:
        return run_dry_run(config)
    # Hugging Face otherwise creates a second progress display while loading weights.
    transformers_logging.disable_progress_bar()
    counts = dataset_counts(config.data_dir)
    steps = calculate_step_counts(counts["train"], config)
    device = resolve_single_device(config.device)
    validate_precision(device, config.precision)
    prepare_output_directory(config.output_dir, config.overwrite_output_dir)
    configure_reproducibility(config.seed)
    resolved = resolved_configuration(config, counts, steps, str(device))
    atomic_write_json(config.output_dir / "training_config.json", resolved)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bundle: BaselineDataBundle = create_baseline_dataloaders(
        config.data_dir,
        tokenizer,
        config.train_batch_size,
        config.eval_batch_size,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    actual_counts = {split: len(dataset) for split, dataset in bundle.datasets.items()}
    if actual_counts != counts:
        raise AssertionError(f"Dataset counts changed after loading: {actual_counts} != {counts}")

    model = load_led_baseline(config.model_name)
    place_model_on_device(model, device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps.warmup_steps,
        num_training_steps=steps.maximum_optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    early_stopping = EarlyStoppingState(
        config.early_stopping_patience, config.early_stopping_min_delta
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    stopped_early = False
    peak_allocated: float | None = None
    peak_reserved: float | None = None
    training_started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = _start_epoch_timing(device)
        training_result = train_one_epoch(
            model,
            bundle.dataloaders["train"],
            optimizer,
            scheduler,
            device,
            config.precision,
            config.gradient_accumulation_steps,
            config.max_grad_norm,
            global_optimizer_step=global_step,
            epoch=epoch,
            scaler=scaler,
            show_progress=True,
        )
        global_step += training_result.optimizer_steps
        validation = evaluate_split(
            model,
            bundle.datasets["validation"],
            bundle.dataloaders["validation"],
            tokenizer,
            device,
            config.precision,
            config.generation_settings,
            progress_description=f"Epoch {epoch} · validation",
        )
        improved = early_stopping.update(validation.teacher_forced_loss, epoch)
        duration = _finish_epoch_timing(device, epoch_started)
        memory = _memory_gib(device)
        if memory[0] is not None:
            peak_allocated = max(peak_allocated or 0.0, memory[0])
            peak_reserved = max(peak_reserved or 0.0, memory[1] or 0.0)
        entry = _history_entry(
            epoch,
            global_step,
            training_result,
            validation,
            float(optimizer.param_groups[0]["lr"]),
            duration,
            memory,
            improved,
            early_stopping.non_improving_evaluations,
        )
        history.append(entry)
        atomic_write_json(config.output_dir / "training_history.json", history)
        if improved:
            save_checkpoint(
                config.output_dir / "best",
                model,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                global_optimizer_step=global_step,
                early_stopping=early_stopping,
                config=config,
                steps=steps,
            )
        print(json.dumps(entry, sort_keys=True))
        if early_stopping.should_stop:
            stopped_early = True
            break

    completed_epoch = len(history)
    save_checkpoint(
        config.output_dir / "last",
        model,
        tokenizer,
        optimizer,
        scheduler,
        scaler,
        epoch=completed_epoch,
        global_optimizer_step=global_step,
        early_stopping=early_stopping,
        config=config,
        steps=steps,
    )
    total_duration = time.perf_counter() - training_started
    del optimizer, scheduler, scaler, model, tokenizer, bundle
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_dir = config.output_dir / "best"
    best_tokenizer = AutoTokenizer.from_pretrained(best_dir)
    final_bundle = create_baseline_dataloaders(
        config.data_dir,
        best_tokenizer,
        config.train_batch_size,
        config.eval_batch_size,
        max_input_length=config.max_input_length,
        max_target_length=config.max_target_length,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    best_model = load_led_baseline(best_dir)
    place_model_on_device(best_model, device)
    best_model.config.use_cache = False
    final_validation = evaluate_split(
        best_model,
        final_bundle.datasets["validation"],
        final_bundle.dataloaders["validation"],
        best_tokenizer,
        device,
        config.precision,
        config.generation_settings,
        progress_description="Final validation",
    )
    write_prediction_jsonl(
        config.output_dir / "validation_predictions.jsonl", final_validation.predictions
    )
    test_result = evaluate_split(
        best_model,
        final_bundle.datasets["test"],
        final_bundle.dataloaders["test"],
        best_tokenizer,
        device,
        config.precision,
        config.generation_settings,
        progress_description="Final test",
    )
    write_prediction_jsonl(config.output_dir / "test_predictions.jsonl", test_result.predictions)
    test_metrics = {
        "best_checkpoint_path": str(best_dir.resolve()),
        "best_validation_loss": early_stopping.best_loss,
        "selected_epoch": early_stopping.best_epoch,
        "test_teacher_forced_loss": test_result.teacher_forced_loss,
        "test_token_accuracy": test_result.token_accuracy,
        "mean_test_token_overlap_f1": test_result.token_overlap_f1,
        "test_exact_match_count": test_result.exact_match_count,
        "test_exact_match_rate": test_result.exact_match_rate,
        "test_examples": test_result.examples,
        "total_training_optimizer_steps": global_step,
        "early_stopping_occurred": stopped_early,
        "total_training_duration_seconds": total_duration,
        "peak_gpu_allocated_gib": peak_allocated,
        "peak_gpu_reserved_gib": peak_reserved,
        "generation_settings": config.generation_settings,
    }
    assert REQUIRED_TEST_METRIC_FIELDS <= test_metrics.keys()
    atomic_write_json(config.output_dir / "test_metrics.json", test_metrics)
    return test_metrics
