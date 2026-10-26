"""Joint neutral generation and ideology-triplet training."""

from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.baseline_dataset import (
    LEDBaselineCollator,
    LEDBaselineDataset,
    PaddedBatch,
)
from src.data.triplet_dataset import (
    LEDTripletDataset,
    TripletBatch,
    TripletDataBundle,
    create_triplet_dataloaders,
)
from src.models.led_baseline import forward_led_baseline
from src.training.decoding import decoding_settings
from src.models.led_triplet import (
    LEDTripletModel, load_led_triplet, masked_positive_attraction_loss,
    masked_triplet_margin_loss,
)
from src.training.overfit_baseline import (
    decode_reference,
    generate_with_cache,
    normalize_text,
    prepare_output_directory,
    seed_everything,
    token_accuracy_counts,
    token_overlap_f1,
)
from src.training.train_baseline import atomic_write_json


DEFAULT_MODEL_NAME = "allenai/led-base-16384"
GENERATION_SETTINGS = {
    "num_beams": 1,
    "do_sample": False,
    "max_new_tokens": 512,
    "early_stopping": False,
}
VALIDATION_GENERATION_SUBSET_SIZE = 256
VALIDATION_GENERATION_SUBSET_SEED = 42
FIXED_TRAIN_SUBSET_SIZE = 64


@dataclass(frozen=True)
class JointTrainingConfig:
    data_dir: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    max_input_length: int = 512
    max_target_length: int = 512
    generation_max_new_tokens: int = 512
    decoding_strategy: str = "greedy"
    triplet_lambda: float = 0.1
    triplet_margin: float = 0.2
    triplet_objective: str = "standard"
    positive_distance_threshold: float = 0.5
    positive_beta: float = 0.3
    pooling: str = "masked_mean"
    world_size: int = 1
    attention_window: int = 512
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    precision: str = "fp32"
    gradient_checkpointing: bool = True
    overwrite_output_dir: bool = False
    dry_run: bool = False
    startup_check: bool = False
    smoke_one_update: bool = False
    wandb_enabled: bool = True
    wandb_project: str = "anlp-project"
    wandb_entity: str = "manasagrawal206-iiit-hyderabad"
    wandb_run_name: str | None = None
    wandb_required: bool = False

    @property
    def effective_anchor_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps * self.world_size

    @property
    def generation_settings(self) -> dict[str, int | bool]:
        return decoding_settings(
            {**GENERATION_SETTINGS, "max_new_tokens": self.generation_max_new_tokens},
            self.decoding_strategy,
        )


@dataclass(frozen=True)
class JointStepCounts:
    train_examples: int
    microbatches_per_epoch: int
    optimizer_steps_per_epoch: int
    maximum_optimizer_steps: int
    warmup_steps: int


@dataclass
class GenerationEvaluation:
    generation_loss: float
    token_accuracy: float
    token_overlap_f1: float
    exact_match_count: int
    exact_match_rate: float
    mean_word_length_ratio: float
    fraction_reaching_cap_without_eos: float
    eos_termination_rate: float
    mean_repeated_trigram_fraction: float
    examples: int
    predictions: list[dict[str, Any]] = field(repr=False)


@dataclass(frozen=True)
class TeacherForcedEvaluation:
    generation_loss: float
    token_accuracy: float
    examples: int
    target_tokens: int


@dataclass
class FreeGenerationEvaluation:
    token_overlap_f1: float
    exact_match_count: int
    exact_match_rate: float
    mean_word_length_ratio: float
    fraction_reaching_cap_without_eos: float
    eos_termination_rate: float
    mean_repeated_trigram_fraction: float
    examples: int
    predictions: list[dict[str, Any]] = field(repr=False)


@dataclass
class TripletEvaluation:
    mean_loss: float
    mean_anchor_positive_distance: float
    mean_anchor_negative_distance: float
    mean_distance_gap: float
    fraction_positive_closer: float
    fraction_margin_satisfied: float
    fraction_nonzero_hinge: float
    valid_count: int
    invalid_count: int
    mean_unnormalized_embedding_norm: float
    embedding_variance: float
    mean_positive_loss: float = 0.0
    mean_weighted_positive_loss: float = 0.0
    mean_auxiliary_loss: float = 0.0
    fraction_attraction_active: float = 0.0


@dataclass
class JointTrainEpoch:
    generation_loss: float
    token_accuracy: float
    triplet_loss: float
    weighted_triplet_loss: float
    valid_triplet_mean: float
    total_loss: float
    valid_triplet_fraction: float
    active_triplet_fraction: float
    optimizer_steps: int
    ranking_loss: float = 0.0
    positive_loss: float = 0.0
    weighted_positive_loss: float = 0.0
    attraction_active_fraction: float = 0.0
    mean_anchor_positive_distance: float = 0.0
    mean_anchor_negative_distance: float = 0.0
    positive_beta: float = 0.0


@dataclass
class JointEarlyStopping:
    patience: int
    minimum_delta: float
    best_loss: float = math.inf
    best_epoch: int | None = None
    non_improving: int = 0

    def update(self, loss: float, epoch: int) -> bool:
        if loss < self.best_loss - self.minimum_delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.non_improving = 0
            return True
        self.non_improving += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.non_improving >= self.patience


class EvaluationDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...
    def record(self, index: int) -> Any: ...
    def source_text(self, index: int) -> str: ...


class BaselineSubsetView(Dataset[Any]):
    """A stable index view retaining evaluation metadata methods."""

    def __init__(self, dataset: LEDBaselineDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def record(self, index: int) -> Any:
        return self.dataset.record(self.indices[index])

    def source_text(self, index: int) -> str:
        return self.dataset.source_text(self.indices[index])


def validate_joint_config(config: JointTrainingConfig) -> None:
    decoding_settings(GENERATION_SETTINGS, config.decoding_strategy)
    positive = {
        "train_batch_size": config.train_batch_size,
        "eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_epochs": config.max_epochs,
        "max_input_length": config.max_input_length,
        "max_target_length": config.max_target_length,
        "generation_max_new_tokens": config.generation_max_new_tokens,
        "attention_window": config.attention_window,
        "early_stopping_patience": config.early_stopping_patience,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.precision != "fp32":
        raise ValueError("the first joint experiment requires precision=fp32")
    if config.triplet_lambda < 0 or config.triplet_margin <= 0:
        raise ValueError("triplet lambda must be nonnegative and margin must be positive")
    if config.triplet_objective not in {"standard", "attraction"}:
        raise ValueError("triplet objective must be standard or attraction")
    if config.pooling not in {"masked_mean", "first_token"}:
        raise ValueError("pooling must be masked_mean or first_token")
    if config.world_size not in {1, 2}:
        raise ValueError("joint training supports one process or two GPU processes")
    if config.world_size == 2 and config.train_batch_size != 1:
        raise ValueError("two-GPU training currently requires per-GPU microbatch size 1")
    if config.startup_check and config.smoke_one_update:
        raise ValueError("choose only one bounded training check")
    if (config.startup_check or config.smoke_one_update) and (
        config.world_size != 2 or config.wandb_enabled
    ):
        raise ValueError("bounded checks require two GPU processes and --disable-wandb")
    if not math.isfinite(config.positive_distance_threshold) or config.positive_distance_threshold < 0:
        raise ValueError("positive distance threshold must be finite and nonnegative")
    if not math.isfinite(config.positive_beta) or config.positive_beta < 0:
        raise ValueError("positive beta must be finite and nonnegative")
    if config.learning_rate <= 0 or config.max_grad_norm <= 0:
        raise ValueError("learning rate and max gradient norm must be positive")
    if config.weight_decay < 0 or not 0 <= config.warmup_ratio <= 1:
        raise ValueError("invalid weight decay or warmup ratio")
    if config.num_workers < 0 or config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("invalid workers or device")
    if config.wandb_enabled and (
        not config.wandb_project.strip() or not config.wandb_entity.strip()
    ):
        raise ValueError("W&B project and entity must be non-empty when enabled")
    if config.wandb_required and not config.wandb_enabled:
        raise ValueError("--require-wandb cannot be combined with --disable-wandb")
    if not config.gradient_checkpointing:
        raise ValueError("gradient checkpointing must remain enabled for this experiment")


def calculate_joint_steps(train_examples: int, config: JointTrainingConfig) -> JointStepCounts:
    if train_examples <= 0:
        raise ValueError("training data is empty")
    per_rank_examples = math.ceil(train_examples / config.world_size)
    microbatches = math.ceil(per_rank_examples / config.train_batch_size)
    steps = math.ceil(microbatches / config.gradient_accumulation_steps)
    maximum = steps * config.max_epochs
    return JointStepCounts(
        train_examples,
        microbatches,
        steps,
        maximum,
        round(maximum * config.warmup_ratio),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes(data_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(data_dir)): file_sha256(path)
        for split in ("train", "validation", "test")
        for path in (
            data_dir / split / "baseline.jsonl",
            data_dir / split / "triplets.jsonl",
        )
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def repeated_trigram_fraction(text: str) -> float:
    words = normalize_text(text).split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def _move_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: batch[name]
        for name in (
            "anchor_input_ids",
            "anchor_attention_mask",
            "anchor_global_attention_mask",
            "positive_input_ids",
            "positive_attention_mask",
            "positive_global_attention_mask",
            "negative_input_ids",
            "negative_attention_mask",
            "negative_global_attention_mask",
            "labels",
            "triplet_valid_mask",
        )
    }


def evaluate_teacher_forced(
    model: LEDTripletModel,
    dataloader: DataLoader[PaddedBatch],
    device: torch.device,
    *,
    description: str | None = None,
) -> TeacherForcedEvaluation:
    """Evaluate loss and token accuracy without autoregressive decoding."""
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    target_tokens = correct_tokens = examples = 0
    try:
        with torch.inference_mode():
            progress = tqdm(
                dataloader,
                total=len(dataloader),
                desc=description,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=description is None or not sys.stderr.isatty(),
            )
            for cpu_batch in progress:
                batch = _move_tensors(cpu_batch, device)
                output = forward_led_baseline(model.led, **batch)
                batch_examples = int(batch["input_ids"].shape[0])
                count = int(batch["labels"].ne(-100).sum().item())
                weighted_loss += float(output.loss.item()) * count
                target_tokens += count
                correct, _ = token_accuracy_counts(output.logits, batch["labels"])
                correct_tokens += correct
                examples += batch_examples
                progress.set_postfix(
                    loss=f"{weighted_loss / target_tokens:.4f}",
                    accuracy=f"{correct_tokens / target_tokens:.4f}",
                    refresh=True,
                )
    finally:
        model.train(was_training)
    if examples == 0 or target_tokens == 0:
        raise AssertionError("teacher-forced evaluation contained no examples or targets")
    return TeacherForcedEvaluation(
        generation_loss=weighted_loss / target_tokens,
        token_accuracy=correct_tokens / target_tokens,
        examples=examples,
        target_tokens=target_tokens,
    )


def evaluate_free_generation(
    model: LEDTripletModel,
    dataset: EvaluationDataset,
    dataloader: DataLoader[PaddedBatch],
    tokenizer: Any,
    device: torch.device,
    generation_settings: Mapping[str, int | bool],
    *,
    epoch: int,
    checkpoint_id: str,
    description: str | None = None,
) -> FreeGenerationEvaluation:
    """Autoregressively decode an aligned, unshuffled evaluation dataset."""
    was_training = model.training
    model.eval()
    examples = exact = capped = eos_terminated = 0
    f1_sum = ratio_sum = trigram_sum = 0.0
    predictions: list[dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id
    try:
        with torch.inference_mode():
            progress = tqdm(
                dataloader,
                total=len(dataloader),
                desc=description,
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                disable=description is None or not sys.stderr.isatty(),
            )
            for cpu_batch in progress:
                batch = _move_tensors(cpu_batch, device)
                generated = generate_with_cache(
                    model.led,
                    batch,
                    device,
                    "fp32",
                    generation_settings,
                ).cpu()
                for row in range(generated.shape[0]):
                    record = dataset.record(examples)
                    reference = decode_reference(tokenizer, batch["labels"][row])
                    generated_ids = generated[row].tolist()
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    overlap = token_overlap_f1(generated_text, reference)
                    is_exact = normalize_text(generated_text) == normalize_text(reference)
                    generated_words = len(generated_text.split())
                    reference_words = max(1, len(reference.split()))
                    ratio = generated_words / reference_words
                    repeat = repeated_trigram_fraction(generated_text)
                    new_ids = generated_ids[1:]
                    has_eos = eos_id is not None and eos_id in new_ids
                    hit_cap = (
                        len(new_ids) >= int(generation_settings["max_new_tokens"])
                        and not has_eos
                    )
                    predictions.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "epoch": epoch,
                            "example_id": record.example_id,
                            "event_id": record.event_id,
                            "input_article_id": record.input_article_id,
                            "input_ideology": record.input_ideology,
                            "target_article_id": record.target_article_id,
                            "source_text": dataset.source_text(examples),
                            "reference_text": reference,
                            "generated_text": generated_text,
                            "token_overlap_f1": overlap,
                            "exact_match": is_exact,
                            "word_length_ratio": ratio,
                            "repeated_trigram_fraction": repeat,
                            "terminated_with_eos": has_eos,
                            "reached_token_cap_without_eos": hit_cap,
                        }
                    )
                    f1_sum += overlap
                    exact += int(is_exact)
                    ratio_sum += ratio
                    trigram_sum += repeat
                    capped += int(hit_cap)
                    eos_terminated += int(has_eos)
                    examples += 1
                progress.set_postfix(
                    overlap_f1=f"{f1_sum / examples:.4f}",
                    repeat_3gram=f"{trigram_sum / examples:.4f}",
                    eos_rate=f"{eos_terminated / examples:.4f}",
                    refresh=True,
                )
    finally:
        model.train(was_training)
    if examples != len(dataset):
        raise AssertionError("free-generation evaluation lost dataset alignment")
    return FreeGenerationEvaluation(
        token_overlap_f1=f1_sum / examples,
        exact_match_count=exact,
        exact_match_rate=exact / examples,
        mean_word_length_ratio=ratio_sum / examples,
        fraction_reaching_cap_without_eos=capped / examples,
        eos_termination_rate=eos_terminated / examples,
        mean_repeated_trigram_fraction=trigram_sum / examples,
        examples=examples,
        predictions=predictions,
    )


def evaluate_generation(
    model: LEDTripletModel,
    dataset: EvaluationDataset,
    dataloader: DataLoader[PaddedBatch],
    tokenizer: Any,
    device: torch.device,
    generation_settings: Mapping[str, int | bool],
    *,
    epoch: int,
    checkpoint_id: str,
    description: str | None = None,
) -> GenerationEvaluation:
    """Run matched teacher-forced and free-generation evaluation on one dataset."""
    teacher = evaluate_teacher_forced(
        model,
        dataloader,
        device,
        description=None if description is None else f"{description} · teacher forced",
    )
    generated = evaluate_free_generation(
        model,
        dataset,
        dataloader,
        tokenizer,
        device,
        generation_settings,
        epoch=epoch,
        checkpoint_id=checkpoint_id,
        description=None if description is None else f"{description} · decoding",
    )
    if teacher.examples != generated.examples:
        raise AssertionError("teacher-forced and generation example counts differ")
    return GenerationEvaluation(
        generation_loss=teacher.generation_loss,
        token_accuracy=teacher.token_accuracy,
        token_overlap_f1=generated.token_overlap_f1,
        exact_match_count=generated.exact_match_count,
        exact_match_rate=generated.exact_match_rate,
        mean_word_length_ratio=generated.mean_word_length_ratio,
        fraction_reaching_cap_without_eos=generated.fraction_reaching_cap_without_eos,
        eos_termination_rate=generated.eos_termination_rate,
        mean_repeated_trigram_fraction=generated.mean_repeated_trigram_fraction,
        examples=generated.examples,
        predictions=generated.predictions,
    )


def evaluate_triplets(
    model: LEDTripletModel,
    dataloader: DataLoader[TripletBatch],
    device: torch.device,
    margin: float,
    objective: str = "standard",
    threshold: float = 0.5,
    beta: float = 0.3,
) -> TripletEvaluation:
    was_training = model.training
    model.eval()
    valid = invalid = active = attraction_active = closer = satisfied = 0
    loss_sum = positive_sum = dap_sum = dan_sum = norm_sum = 0.0
    embedding_count = 0
    vector_sum: torch.Tensor | None = None
    vector_square_sum: torch.Tensor | None = None
    try:
        with torch.inference_mode():
            for cpu_batch in dataloader:
                batch = _move_tensors(cpu_batch, device)
                mask = batch["triplet_valid_mask"].bool()
                invalid += int((~mask).sum().item())
                if not bool(mask.any()):
                    continue
                anchor, anchor_raw = model._encode(
                    batch["anchor_input_ids"],
                    batch["anchor_attention_mask"],
                    batch["anchor_global_attention_mask"],
                )
                positive, positive_raw = model._encode(
                    batch["positive_input_ids"],
                    batch["positive_attention_mask"],
                    batch["positive_global_attention_mask"],
                )
                negative, negative_raw = model._encode(
                    batch["negative_input_ids"],
                    batch["negative_attention_mask"],
                    batch["negative_global_attention_mask"],
                )
                _, _, per_example, d_ap, d_an = masked_triplet_margin_loss(
                    anchor, positive, negative, mask, margin
                )
                selected = mask.nonzero(as_tuple=False).flatten()
                if objective == "attraction":
                    _, per_positive = masked_positive_attraction_loss(d_ap, mask, threshold)
                    positive_sum += float(per_positive[selected].sum().item())
                    attraction_active += int((per_positive[selected] > 0).sum().item())
                valid += len(selected)
                loss_sum += float(per_example[selected].sum().item())
                dap_sum += float(d_ap[selected].sum().item())
                dan_sum += float(d_an[selected].sum().item())
                active += int((per_example[selected] > 0).sum().item())
                closer += int((d_ap[selected] < d_an[selected]).sum().item())
                satisfied += int((d_ap[selected] + margin <= d_an[selected]).sum().item())
                raw = torch.cat(
                    (anchor_raw[selected], positive_raw[selected], negative_raw[selected]), dim=0
                ).double()
                norm_sum += float(torch.linalg.vector_norm(raw, dim=-1).sum().item())
                normed_anchor = anchor[selected].double().cpu()
                batch_sum = normed_anchor.sum(dim=0)
                batch_square = (normed_anchor * normed_anchor).sum(dim=0)
                vector_sum = batch_sum if vector_sum is None else vector_sum + batch_sum
                vector_square_sum = (
                    batch_square
                    if vector_square_sum is None
                    else vector_square_sum + batch_square
                )
                embedding_count += len(selected)
    finally:
        model.train(was_training)
    if valid == 0:
        variance = 0.0
    else:
        assert vector_sum is not None and vector_square_sum is not None
        coordinate_variance = vector_square_sum / embedding_count - (
            vector_sum / embedding_count
        ).square()
        variance = float(coordinate_variance.clamp_min(0).mean().item())
    return TripletEvaluation(
        mean_loss=loss_sum / valid if valid else 0.0,
        mean_anchor_positive_distance=dap_sum / valid if valid else 0.0,
        mean_anchor_negative_distance=dan_sum / valid if valid else 0.0,
        mean_distance_gap=(dan_sum - dap_sum) / valid if valid else 0.0,
        fraction_positive_closer=closer / valid if valid else 0.0,
        fraction_margin_satisfied=satisfied / valid if valid else 0.0,
        fraction_nonzero_hinge=active / valid if valid else 0.0,
        valid_count=valid,
        invalid_count=invalid,
        mean_unnormalized_embedding_norm=norm_sum / (3 * valid) if valid else 0.0,
        embedding_variance=variance,
        mean_positive_loss=positive_sum / valid if valid else 0.0,
        mean_weighted_positive_loss=beta * positive_sum / valid if valid and objective == "attraction" else 0.0,
        mean_auxiliary_loss=(loss_sum + (beta * positive_sum if objective == "attraction" else 0.0)) / valid if valid else 0.0,
        fraction_attraction_active=attraction_active / valid if valid else 0.0,
    )


def optimizer_parameter_groups(
    model: torch.nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        lowered = name.lower()
        if name.endswith(".bias") or "layernorm" in lowered or "layer_norm" in lowered or ".norm" in lowered:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def train_joint_epoch(
    model: LEDTripletModel,
    dataloader: DataLoader[TripletBatch],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: JointTrainingConfig,
    *,
    epoch: int,
    global_step: int,
    step_log_path: Path | None = None,
    show_progress: bool = False,
    wandb_run: Any | None = None,
) -> JointTrainEpoch:
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    generation_sum = 0.0
    generation_tokens = correct_tokens = 0
    examples = valid_examples = active = attraction_active = 0
    triplet_sum = valid_triplet_sum = ranking_sum = positive_sum = weighted_positive_sum = 0.0
    positive_distance_sum = negative_distance_sum = 0.0
    microbatches = 0
    optimizer_steps = 0
    group_logs: list[dict[str, float]] = []
    total_microbatches = len(dataloader)
    progress = tqdm(
        dataloader,
        total=total_microbatches,
        desc=f"Epoch {epoch} · joint train",
        unit="anchor",
        leave=False,
        dynamic_ncols=True,
        disable=not (show_progress and sys.stderr.isatty()),
    )
    for microbatch_index, cpu_batch in enumerate(progress, 1):
        batch = _move_tensors(cpu_batch, device)
        ids = list(batch["example_id"])
        output = model(
            **_model_inputs(batch),
            triplet_lambda=config.triplet_lambda,
            triplet_margin=config.triplet_margin,
            triplet_objective=config.triplet_objective,
            positive_distance_threshold=config.positive_distance_threshold,
            positive_beta=config.positive_beta,
        )
        if not bool(torch.isfinite(output.total_loss)):
            raise FloatingPointError(f"non-finite loss for examples {ids}")
        expected_total = (
            output.generation_loss + config.triplet_lambda * output.triplet_loss
        )
        if not torch.allclose(output.total_loss, expected_total, rtol=1e-6, atol=1e-7):
            raise ValueError(
                "joint objective mismatch before gradient-accumulation scaling: "
                f"total_loss={output.total_loss.item():.8g}, "
                f"generation_loss={output.generation_loss.item():.8g}, "
                f"triplet_lambda={config.triplet_lambda:.8g}, "
                f"triplet_loss={output.triplet_loss.item():.8g}"
            )
        if output.ranking_loss is not None and output.weighted_positive_loss is not None:
            if not torch.allclose(
                output.triplet_loss,
                output.ranking_loss + output.weighted_positive_loss,
                rtol=1e-6, atol=1e-7,
            ):
                raise ValueError("auxiliary loss does not reconstruct ranking plus attraction")
        remaining = total_microbatches - microbatch_index + 1
        position = (microbatch_index - 1) % config.gradient_accumulation_steps
        group_size = min(config.gradient_accumulation_steps, position + remaining)
        (output.total_loss / group_size).backward()
        token_count = int(batch["labels"].ne(-100).sum().item())
        correct_count, _ = token_accuracy_counts(
            output.generation_logits, batch["labels"]
        )
        batch_size = int(batch["labels"].shape[0])
        valid_count = output.valid_count
        active_count = int(
            (
                (output.per_example_triplet_loss > 0)
                & batch["triplet_valid_mask"].bool()
            ).sum().item()
        )
        valid_mask = batch["triplet_valid_mask"].bool()
        attraction_count = int(
            ((output.per_example_positive_loss > 0) & valid_mask).sum().item()
        ) if output.per_example_positive_loss is not None else 0
        ranking = output.ranking_loss if output.ranking_loss is not None else output.triplet_loss
        positive = output.positive_loss if output.positive_loss is not None else output.triplet_loss.new_zeros(())
        weighted_positive = (
            output.weighted_positive_loss if output.weighted_positive_loss is not None
            else output.triplet_loss.new_zeros(())
        )
        d_ap_sum = float(output.anchor_positive_distance[valid_mask].sum().item())
        d_an_sum = float(output.anchor_negative_distance[valid_mask].sum().item())
        generation_sum += float(output.generation_loss.item())
        generation_tokens += token_count
        correct_tokens += correct_count
        triplet_sum += float(output.triplet_loss.item())
        ranking_sum += float(ranking.item())
        positive_sum += float(positive.item())
        weighted_positive_sum += float(weighted_positive.item())
        positive_distance_sum += d_ap_sum
        negative_distance_sum += d_an_sum
        valid_triplet_sum += float(output.valid_triplet_mean.item()) * valid_count
        microbatches += 1
        examples += batch_size
        valid_examples += valid_count
        active += active_count
        attraction_active += attraction_count
        group_logs.append(
            {
                "generation_loss": float(output.generation_loss.item()),
                "correct_tokens": float(correct_count),
                "target_tokens": float(token_count),
                "triplet_loss": float(output.triplet_loss.item()),
                "ranking_loss": float(ranking.item()),
                "positive_loss": float(positive.item()),
                "weighted_positive_loss": float(weighted_positive.item()),
                "positive_distance_sum": d_ap_sum,
                "negative_distance_sum": d_an_sum,
                "attraction_active": float(attraction_count),
                "valid_triplet_mean": float(output.valid_triplet_mean.item()),
                "valid": float(valid_count),
                "active": float(active_count),
                "examples": float(batch_size),
            }
        )
        boundary = microbatch_index % config.gradient_accumulation_steps == 0
        final = microbatch_index == total_microbatches
        if not (boundary or final):
            continue
        finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        if not finite:
            raise FloatingPointError(f"non-finite gradients near examples {ids}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm near examples {ids}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        global_optimizer_step = global_step + optimizer_steps
        group_examples = sum(value["examples"] for value in group_logs)
        group_valid = sum(value["valid"] for value in group_logs)
        group_active = sum(value["active"] for value in group_logs)
        group_correct = sum(value["correct_tokens"] for value in group_logs)
        group_target_tokens = sum(value["target_tokens"] for value in group_logs)
        generation_mean = sum(value["generation_loss"] for value in group_logs) / len(
            group_logs
        )
        triplet_mean = sum(value["triplet_loss"] for value in group_logs) / len(
            group_logs
        )
        ranking_mean = sum(value["ranking_loss"] for value in group_logs) / len(group_logs)
        positive_mean = sum(value["positive_loss"] for value in group_logs) / len(group_logs)
        weighted_positive_mean = sum(value["weighted_positive_loss"] for value in group_logs) / len(group_logs)
        group_attraction_active = sum(value["attraction_active"] for value in group_logs)
        group_d_ap = sum(value["positive_distance_sum"] for value in group_logs)
        group_d_an = sum(value["negative_distance_sum"] for value in group_logs)
        beta = config.positive_beta if config.triplet_objective == "attraction" else 0.0
        weighted_triplet_mean = config.triplet_lambda * triplet_mean
        total_mean = generation_mean + weighted_triplet_mean
        current_lr = float(optimizer.param_groups[0]["lr"])
        progress.set_postfix(
            gen_loss=f"{generation_mean:.4f}",
            accuracy=f"{group_correct / group_target_tokens:.4f}",
            triplet=f"{triplet_mean:.4f}",
            total=f"{total_mean:.4f}",
            lr=f"{current_lr:.2e}",
            step=global_optimizer_step,
            refresh=True,
        )
        step_metrics = {
            "epoch": epoch,
            "global_optimizer_step": global_optimizer_step,
            "generation_loss": generation_mean,
            "token_accuracy": group_correct / group_target_tokens,
            "triplet_loss": triplet_mean,
            "ranking_loss": ranking_mean,
            "positive_loss": positive_mean,
            "weighted_positive_loss": weighted_positive_mean,
            "auxiliary_loss": triplet_mean,
            "positive_beta": beta,
            "attraction_active_fraction": group_attraction_active / group_valid if group_valid else 0.0,
            "mean_anchor_positive_distance": group_d_ap / group_valid if group_valid else 0.0,
            "mean_anchor_negative_distance": group_d_an / group_valid if group_valid else 0.0,
            "weighted_triplet_loss": weighted_triplet_mean,
            "total_loss": total_mean,
            "gradient_norm_before_clipping": float(grad_norm),
            "learning_rate": current_lr,
            "valid_triplet_fraction": group_valid / group_examples,
            "active_triplet_fraction": group_active / group_valid if group_valid else 0.0,
        }
        if step_log_path is not None:
            append_jsonl(step_log_path, step_metrics)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/step_generation_loss": generation_mean,
                    "train/step_token_accuracy": group_correct / group_target_tokens,
                    "train/step_triplet_loss": triplet_mean,
                    "train/step_ranking_loss": ranking_mean,
                    "train/step_positive_loss": positive_mean,
                    "train/step_weighted_positive_loss": weighted_positive_mean,
                    "train/step_auxiliary_loss": triplet_mean,
                    "train/step_positive_beta": beta,
                    "train/step_attraction_active_fraction": group_attraction_active / group_valid if group_valid else 0.0,
                    "train/step_mean_anchor_positive_distance": group_d_ap / group_valid if group_valid else 0.0,
                    "train/step_mean_anchor_negative_distance": group_d_an / group_valid if group_valid else 0.0,
                    "train/step_weighted_triplet_loss": weighted_triplet_mean,
                    "train/step_total_loss": total_mean,
                    "train/gradient_norm_before_clipping": float(grad_norm),
                    "train/learning_rate": current_lr,
                    "train/valid_triplet_fraction": group_valid / group_examples,
                    "train/active_triplet_fraction": (
                        group_active / group_valid if group_valid else 0.0
                    ),
                    "epoch": epoch,
                },
                step=global_optimizer_step,
            )
        group_logs.clear()
    if generation_tokens == 0 or examples == 0:
        raise ValueError("training epoch contained no supervised tokens")
    generation_mean = generation_sum / microbatches
    triplet_mean = triplet_sum / microbatches
    weighted_triplet_mean = config.triplet_lambda * triplet_mean
    return JointTrainEpoch(
        generation_loss=generation_mean,
        token_accuracy=correct_tokens / generation_tokens,
        triplet_loss=triplet_mean,
        weighted_triplet_loss=weighted_triplet_mean,
        valid_triplet_mean=valid_triplet_sum / valid_examples if valid_examples else 0.0,
        total_loss=generation_mean + weighted_triplet_mean,
        valid_triplet_fraction=valid_examples / examples,
        active_triplet_fraction=active / valid_examples if valid_examples else 0.0,
        optimizer_steps=optimizer_steps,
        ranking_loss=ranking_sum / microbatches,
        positive_loss=positive_sum / microbatches,
        weighted_positive_loss=weighted_positive_sum / microbatches,
        attraction_active_fraction=attraction_active / valid_examples if valid_examples else 0.0,
        mean_anchor_positive_distance=positive_distance_sum / valid_examples if valid_examples else 0.0,
        mean_anchor_negative_distance=negative_distance_sum / valid_examples if valid_examples else 0.0,
        positive_beta=config.positive_beta if config.triplet_objective == "attraction" else 0.0,
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_joint_checkpoint(
    path: Path,
    model: LEDTripletModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: JointTrainingConfig,
    early: JointEarlyStopping,
    train_generator: torch.Generator,
    *,
    epoch: int,
    global_step: int,
) -> None:
    prepare_output_directory(path, overwrite=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    metadata = {
        "epoch": epoch,
        "global_optimizer_step": global_step,
        "best_validation_generation_loss": early.best_loss,
        "best_epoch": early.best_epoch,
        "patience_counter": early.non_improving,
        "pooling": config.pooling,
        "world_size": config.world_size,
        "effective_anchor_batch_size": config.effective_anchor_batch_size,
        "triplet_lambda": config.triplet_lambda,
        "triplet_margin": config.triplet_margin,
        "triplet_objective": config.triplet_objective,
        "positive_distance_threshold": config.positive_distance_threshold,
        "positive_beta": config.positive_beta,
        "max_input_length": config.max_input_length,
        "max_target_length": config.max_target_length,
        "attention_window": config.attention_window,
        "generation_settings": config.generation_settings,
        "decoding_strategy": config.decoding_strategy,
        "validation_generation_subset_size": VALIDATION_GENERATION_SUBSET_SIZE,
        "validation_generation_subset_seed": VALIDATION_GENERATION_SUBSET_SEED,
        "precision": config.precision,
        "seed": config.seed,
    }
    atomic_write_json(path / "joint_state.json", metadata)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": _rng_state(),
            "train_dataloader_generator_state": train_generator.get_state(),
            "metadata": metadata,
        },
        path / "trainer_state.pt",
    )


def _generation_dict(value: GenerationEvaluation) -> dict[str, Any]:
    result = asdict(value)
    result.pop("predictions")
    return result


def _free_generation_dict(value: FreeGenerationEvaluation) -> dict[str, Any]:
    result = asdict(value)
    predictions = result.pop("predictions")
    result["prediction_previews"] = [
        {
            "example_id": prediction["example_id"],
            "reference_preview": prediction["reference_text"][:160],
            "generated_preview": prediction["generated_text"][:160],
        }
        for prediction in predictions[:3]
    ]
    return result


def log_epoch_to_wandb(
    wandb_run: Any | None,
    *,
    epoch: int,
    global_step: int,
    trained: JointTrainEpoch | None,
    validation_teacher: TeacherForcedEvaluation,
    validation_generated: FreeGenerationEvaluation,
    fixed_train: GenerationEvaluation,
    validation_triplet: TripletEvaluation,
    improved: bool,
    patience_counter: int,
    learning_rate: float,
    duration_seconds: float,
    memory: tuple[float | None, float | None],
) -> None:
    if wandb_run is None:
        return
    metrics: dict[str, Any] = {
        "epoch": epoch,
        "validation/loss": validation_teacher.generation_loss,
        "validation/token_accuracy": validation_teacher.token_accuracy,
        "validation/target_tokens": validation_teacher.target_tokens,
        "validation_generation_subset/token_overlap_f1": (
            validation_generated.token_overlap_f1
        ),
        "validation_generation_subset/exact_match_rate": (
            validation_generated.exact_match_rate
        ),
        "validation_generation_subset/word_length_ratio": (
            validation_generated.mean_word_length_ratio
        ),
        "validation_generation_subset/repeated_trigram_fraction": (
            validation_generated.mean_repeated_trigram_fraction
        ),
        "validation_generation_subset/eos_termination_rate": (
            validation_generated.eos_termination_rate
        ),
        "validation_generation_subset/reached_cap_without_eos": (
            validation_generated.fraction_reaching_cap_without_eos
        ),
        "fixed_train/loss": fixed_train.generation_loss,
        "fixed_train/token_accuracy": fixed_train.token_accuracy,
        "fixed_train/token_overlap_f1": fixed_train.token_overlap_f1,
        "validation_triplet/loss": validation_triplet.mean_loss,
        "validation_triplet/ranking_loss": validation_triplet.mean_loss,
        "validation_triplet/positive_loss": validation_triplet.mean_positive_loss,
        "validation_triplet/weighted_positive_loss": validation_triplet.mean_weighted_positive_loss,
        "validation_triplet/auxiliary_loss": validation_triplet.mean_auxiliary_loss,
        "validation_triplet/attraction_active_fraction": validation_triplet.fraction_attraction_active,
        "validation_triplet/anchor_positive_distance": (
            validation_triplet.mean_anchor_positive_distance
        ),
        "validation_triplet/anchor_negative_distance": (
            validation_triplet.mean_anchor_negative_distance
        ),
        "validation_triplet/distance_gap": validation_triplet.mean_distance_gap,
        "validation_triplet/margin_satisfied_fraction": (
            validation_triplet.fraction_margin_satisfied
        ),
        "run/improved": int(improved),
        "run/patience_counter": patience_counter,
        "run/epoch_learning_rate": learning_rate,
        "run/epoch_duration_seconds": duration_seconds,
    }
    if trained is not None:
        metrics.update(
            {
                "train/epoch_generation_loss": trained.generation_loss,
                "train/epoch_token_accuracy": trained.token_accuracy,
                "train/epoch_triplet_loss": trained.triplet_loss,
                "train/epoch_ranking_loss": trained.ranking_loss,
                "train/epoch_positive_loss": trained.positive_loss,
                "train/epoch_weighted_positive_loss": trained.weighted_positive_loss,
                "train/epoch_auxiliary_loss": trained.triplet_loss,
                "train/epoch_positive_beta": trained.positive_beta,
                "train/epoch_attraction_active_fraction": trained.attraction_active_fraction,
                "train/epoch_mean_anchor_positive_distance": trained.mean_anchor_positive_distance,
                "train/epoch_mean_anchor_negative_distance": trained.mean_anchor_negative_distance,
                "train/epoch_weighted_triplet_loss": trained.weighted_triplet_loss,
                "train/epoch_valid_triplet_mean": trained.valid_triplet_mean,
                "train/epoch_total_loss": trained.total_loss,
                "train/epoch_valid_triplet_fraction": trained.valid_triplet_fraction,
                "train/epoch_active_triplet_fraction": trained.active_triplet_fraction,
            }
        )
    if memory[0] is not None:
        metrics["system/gpu_peak_allocated_gib"] = memory[0]
    if memory[1] is not None:
        metrics["system/gpu_peak_reserved_gib"] = memory[1]
    wandb_run.log(metrics, step=global_step)


def _fixed_subset_indices(length: int, seed: int, count: int, name: str) -> list[int]:
    if length < count:
        raise ValueError(f"{name} dataset has only {length} examples; need {count}")
    return sorted(random.Random(seed).sample(range(length), count))


def _build_baseline_evaluation(
    data_dir: Path,
    tokenizer: Any,
    config: JointTrainingConfig,
) -> tuple[
    LEDBaselineDataset,
    DataLoader[PaddedBatch],
    BaselineSubsetView,
    DataLoader[PaddedBatch],
    BaselineSubsetView,
    DataLoader[PaddedBatch],
    list[int],
    list[int],
]:
    validation = LEDBaselineDataset(
        data_dir / "validation" / "baseline.jsonl",
        tokenizer,
        config.max_input_length,
        config.max_target_length,
    )
    train = LEDBaselineDataset(
        data_dir / "train" / "baseline.jsonl",
        tokenizer,
        config.max_input_length,
        config.max_target_length,
    )
    collator = LEDBaselineCollator(tokenizer)
    validation_loader = DataLoader(
        validation,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    validation_indices = _fixed_subset_indices(
        len(validation),
        VALIDATION_GENERATION_SUBSET_SEED,
        VALIDATION_GENERATION_SUBSET_SIZE,
        "validation",
    )
    validation_view = BaselineSubsetView(validation, validation_indices)
    validation_generation_loader = DataLoader(
        validation_view,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    train_indices = _fixed_subset_indices(
        len(train), config.seed, FIXED_TRAIN_SUBSET_SIZE, "training"
    )
    train_view = BaselineSubsetView(train, train_indices)
    train_loader = DataLoader(
        train_view,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return (
        validation,
        validation_loader,
        validation_view,
        validation_generation_loader,
        train_view,
        train_loader,
        validation_indices,
        train_indices,
    )


def build_reproducibility_report(
    config: JointTrainingConfig,
    bundle: TripletDataBundle,
    steps: JointStepCounts,
    truncation: Mapping[str, Any],
) -> dict[str, Any]:
    previews = []
    for index in range(min(3, len(bundle.datasets["train"]))):
        record = bundle.datasets["train"].record(index)
        previews.append(
            {
                "triplet_id": record.triplet_id,
                "target_article_id": record.target_article_id,
                "target_preview": record.target_text[:160],
            }
        )
    return {
        "configuration": {
            **asdict(config),
            "data_dir": str(config.data_dir.resolve()),
            "output_dir": str(config.output_dir.resolve()),
            "effective_anchor_batch_size": config.effective_anchor_batch_size,
            "generation_settings": config.generation_settings,
            "pooling": config.pooling,
        },
        "resolved_paths": {
            split: {
                "baseline": str((config.data_dir / split / "baseline.jsonl").resolve()),
                "triplets": str((config.data_dir / split / "triplets.jsonl").resolve()),
            }
            for split in ("train", "validation", "test")
        },
        "file_sha256": data_hashes(config.data_dir),
        "library_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "dataset_counts": {split: len(dataset) for split, dataset in bundle.datasets.items()},
        "triplet_resolution": {
            split: dataset.resolution_statistics for split, dataset in bundle.datasets.items()
        },
        "truncation_statistics": dict(truncation),
        "evaluation_policy": {
            "validation_teacher_forced_examples_each_epoch": "complete split",
            "validation_generation_subset_size": VALIDATION_GENERATION_SUBSET_SIZE,
            "validation_generation_subset_seed": VALIDATION_GENERATION_SUBSET_SEED,
            "full_validation_generation": "selected best checkpoint only",
            "test_evaluation": "separate explicit command only",
        },
        "step_counts": asdict(steps),
        "target_previews": previews,
    }


def run_joint_dry_run(config: JointTrainingConfig) -> dict[str, Any]:
    validate_joint_config(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bundle = create_triplet_dataloaders(
        config.data_dir,
        tokenizer,
        config.train_batch_size,
        config.eval_batch_size,
        config.max_input_length,
        config.max_target_length,
        config.seed,
        config.num_workers,
    )
    steps = calculate_joint_steps(len(bundle.datasets["train"]), config)
    truncation = {
        split: dataset.truncation_statistics()
        for split, dataset in bundle.datasets.items()
    }
    report = build_reproducibility_report(config, bundle, steps, truncation)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("DRY RUN COMPLETE — NO MODEL LOADED, NO OPTIMIZATION PERFORMED")
    return report


def _cuda_memory(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    divisor = 1024**3
    return (
        torch.cuda.max_memory_allocated(device) / divisor,
        torch.cuda.max_memory_reserved(device) / divisor,
    )


def initialize_wandb(config: JointTrainingConfig, report: Mapping[str, Any]) -> Any | None:
    """Start online W&B tracking only for a real training run."""
    if not config.wandb_enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is unavailable; install it with "
            "`python -m pip install wandb` or pass --disable-wandb"
        ) from exc
    wandb_config = json.loads(json.dumps(dict(report), default=str))
    try:
        return wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=wandb_config,
            dir=str(config.output_dir),
            resume="never",
            job_type="joint-led-training",
            tags=[
                "led",
                "lambda-zero" if config.triplet_lambda == 0 else "triplet",
                f"lambda-{config.triplet_lambda:g}",
            ],
        )
    except Exception as exc:
        message = (
            f"W&B initialization failed for {config.wandb_entity}/"
            f"{config.wandb_project}: {exc}"
        )
        if config.wandb_required:
            raise RuntimeError(message) from exc
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
        print(
            "WARNING: Continuing training without W&B; local JSON/JSONL metrics "
            "remain enabled.",
            file=sys.stderr,
            flush=True,
        )
        try:
            wandb.finish(exit_code=1, quiet=True)
        except Exception:
            pass
        return None


def run_joint_training(config: JointTrainingConfig) -> dict[str, Any]:
    validate_joint_config(config)
    if config.dry_run:
        return run_joint_dry_run(config)
    if config.world_size == 2:
        from src.training.train_joint_ddp import run_joint_training_ddp

        return run_joint_training_ddp(config)
    device = resolve_device(config.device)
    prepare_output_directory(config.output_dir, config.overwrite_output_dir)
    seed_everything(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    bundle = create_triplet_dataloaders(
        config.data_dir,
        tokenizer,
        config.train_batch_size,
        config.eval_batch_size,
        config.max_input_length,
        config.max_target_length,
        config.seed,
        config.num_workers,
    )
    steps = calculate_joint_steps(len(bundle.datasets["train"]), config)
    truncation = {
        split: dataset.truncation_statistics()
        for split, dataset in bundle.datasets.items()
    }
    report = build_reproducibility_report(config, bundle, steps, truncation)
    atomic_write_json(config.output_dir / "training_config.json", report)
    for split, dataset in bundle.datasets.items():
        write_jsonl(
            config.output_dir / f"resolved_triplets_{split}.jsonl",
            dataset.assignment_records(),
        )
    wandb_run = initialize_wandb(config, report)

    model = load_led_triplet(
        config.model_name, attention_window=config.attention_window, pooling=config.pooling
    )
    model.to(device)
    model.led.gradient_checkpointing_enable()
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=steps.warmup_steps,
        num_training_steps=steps.maximum_optimizer_steps,
    )
    (
        validation,
        validation_loader,
        validation_generation_view,
        validation_generation_loader,
        train_view,
        train_loader,
        validation_indices,
        train_indices,
    ) = _build_baseline_evaluation(config.data_dir, tokenizer, config)
    atomic_write_json(
        config.output_dir / "fixed_validation_generation_subset.json",
        {
            "seed": VALIDATION_GENERATION_SUBSET_SEED,
            "size": len(validation_indices),
            "indices": validation_indices,
            "example_ids": [
                validation_generation_view.record(i).example_id
                for i in range(len(validation_generation_view))
            ],
        },
    )
    atomic_write_json(
        config.output_dir / "fixed_train_subset.json",
        {
            "indices": train_indices,
            "example_ids": [train_view.record(i).example_id for i in range(len(train_view))],
        },
    )
    early = JointEarlyStopping(
        config.early_stopping_patience, config.early_stopping_min_delta
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    generation_settings = config.generation_settings

    def evaluate_epoch(
        epoch: int,
        checkpoint_id: str,
    ) -> tuple[
        TeacherForcedEvaluation,
        FreeGenerationEvaluation,
        GenerationEvaluation,
        TripletEvaluation,
    ]:
        validation_teacher = evaluate_teacher_forced(
            model,
            validation_loader,
            device,
            description=f"Epoch {epoch} · full validation teacher forced",
        )
        validation_generated = evaluate_free_generation(
            model,
            validation_generation_view,
            validation_generation_loader,
            tokenizer,
            device,
            generation_settings,
            epoch=epoch,
            checkpoint_id=checkpoint_id,
            description=f"Epoch {epoch} · validation generation subset",
        )
        train_result = evaluate_generation(
            model,
            train_view,
            train_loader,
            tokenizer,
            device,
            generation_settings,
            epoch=epoch,
            checkpoint_id=checkpoint_id,
            description=f"Epoch {epoch} · fixed train generation",
        )
        triplet_result = evaluate_triplets(
            model, bundle.dataloaders["validation"], device, config.triplet_margin,
            config.triplet_objective, config.positive_distance_threshold, config.positive_beta,
        )
        write_jsonl(
            config.output_dir
            / f"validation_generation_subset_predictions_epoch_{epoch:03d}.jsonl",
            validation_generated.predictions,
        )
        write_jsonl(
            config.output_dir / f"fixed_train_predictions_epoch_{epoch:03d}.jsonl",
            train_result.predictions,
        )
        return validation_teacher, validation_generated, train_result, triplet_result

    val_teacher0, val_generated0, train0, triplet0 = evaluate_epoch(
        0, "pretrained_epoch_0"
    )
    early.update(val_teacher0.generation_loss, 0)
    print(
        f"Epoch 0/{config.max_epochs} | "
        f"train_fixed_loss={train0.generation_loss:.4f} "
        f"train_fixed_acc={train0.token_accuracy:.4f} | "
        f"val_loss={val_teacher0.generation_loss:.4f} "
        f"val_acc={val_teacher0.token_accuracy:.4f} "
        f"val_subset_f1={val_generated0.token_overlap_f1:.4f} "
        f"val_subset_repeat3={val_generated0.mean_repeated_trigram_fraction:.4f} "
        f"val_subset_len_ratio={val_generated0.mean_word_length_ratio:.4f} "
        f"val_subset_eos={val_generated0.eos_termination_rate:.4f} | "
        f"val_triplet_loss={triplet0.mean_loss:.4f} | pretrained",
        flush=True,
    )
    history.append(
        {
            "epoch": 0,
            "global_optimizer_step": 0,
            "training": None,
            "validation_teacher_forced": asdict(val_teacher0),
            "validation_generation_subset": _free_generation_dict(val_generated0),
            "fixed_train_generation": _generation_dict(train0),
            "validation_triplet": asdict(triplet0),
            "improved": True,
            "patience_counter": 0,
            "learning_rate": 0.0,
            "epoch_duration_seconds": 0.0,
            "gpu_peak_allocated_gib": _cuda_memory(device)[0],
            "gpu_peak_reserved_gib": _cuda_memory(device)[1],
        }
    )
    atomic_write_json(config.output_dir / "training_history.json", history)
    log_epoch_to_wandb(
        wandb_run,
        epoch=0,
        global_step=0,
        trained=None,
        validation_teacher=val_teacher0,
        validation_generated=val_generated0,
        fixed_train=train0,
        validation_triplet=triplet0,
        improved=True,
        patience_counter=0,
        learning_rate=0.0,
        duration_seconds=0.0,
        memory=_cuda_memory(device),
    )
    save_joint_checkpoint(
        config.output_dir / "best",
        model,
        tokenizer,
        optimizer,
        scheduler,
        config,
        early,
        bundle.train_generator,
        epoch=0,
        global_step=0,
    )

    stopped_early = False
    for epoch in range(1, config.max_epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        trained = train_joint_epoch(
            model,
            bundle.dataloaders["train"],
            optimizer,
            scheduler,
            device,
            config,
            epoch=epoch,
            global_step=global_step,
            step_log_path=config.output_dir / "step_metrics.jsonl",
            show_progress=True,
            wandb_run=wandb_run,
        )
        global_step += trained.optimizer_steps
        (
            validation_teacher,
            validation_generated,
            train_result,
            triplet_result,
        ) = evaluate_epoch(epoch, f"epoch_{epoch}")
        improved = early.update(validation_teacher.generation_loss, epoch)
        print(
            f"Epoch {epoch}/{config.max_epochs} | "
            f"train_gen_loss={trained.generation_loss:.4f} "
            f"train_acc={trained.token_accuracy:.4f} "
            f"train_triplet_loss={trained.triplet_loss:.4f} "
            f"train_total_loss={trained.total_loss:.4f} | "
            f"val_loss={validation_teacher.generation_loss:.4f} "
            f"val_acc={validation_teacher.token_accuracy:.4f} "
            f"val_subset_f1={validation_generated.token_overlap_f1:.4f} "
            f"val_subset_repeat3={validation_generated.mean_repeated_trigram_fraction:.4f} "
            f"val_subset_len_ratio={validation_generated.mean_word_length_ratio:.4f} "
            f"val_subset_eos={validation_generated.eos_termination_rate:.4f} | "
            f"val_triplet_loss={triplet_result.mean_loss:.4f} | "
            f"improved={'yes' if improved else 'no'} "
            f"patience={early.non_improving}/{early.patience}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        memory = _cuda_memory(device)
        entry = {
            "epoch": epoch,
            "global_optimizer_step": global_step,
            "training": asdict(trained),
            "validation_teacher_forced": asdict(validation_teacher),
            "validation_generation_subset": _free_generation_dict(validation_generated),
            "fixed_train_generation": _generation_dict(train_result),
            "validation_triplet": asdict(triplet_result),
            "improved": improved,
            "patience_counter": early.non_improving,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_duration_seconds": duration,
            "gpu_peak_allocated_gib": memory[0],
            "gpu_peak_reserved_gib": memory[1],
        }
        history.append(entry)
        atomic_write_json(config.output_dir / "training_history.json", history)
        log_epoch_to_wandb(
            wandb_run,
            epoch=epoch,
            global_step=global_step,
            trained=trained,
            validation_teacher=validation_teacher,
            validation_generated=validation_generated,
            fixed_train=train_result,
            validation_triplet=triplet_result,
            improved=improved,
            patience_counter=early.non_improving,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            duration_seconds=duration,
            memory=memory,
        )
        if improved:
            save_joint_checkpoint(
                config.output_dir / "best",
                model,
                tokenizer,
                optimizer,
                scheduler,
                config,
                early,
                bundle.train_generator,
                epoch=epoch,
                global_step=global_step,
            )
        if early.should_stop:
            stopped_early = True
            break
    save_joint_checkpoint(
        config.output_dir / "last",
        model,
        tokenizer,
        optimizer,
        scheduler,
        config,
        early,
        bundle.train_generator,
        epoch=len(history) - 1,
        global_step=global_step,
    )
    # Release the training instance, then reload the selected checkpoint and
    # perform the only complete-validation autoregressive decode in the run.
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    best_checkpoint = config.output_dir / "best"
    best_tokenizer = AutoTokenizer.from_pretrained(best_checkpoint)
    best_validation = LEDBaselineDataset(
        config.data_dir / "validation" / "baseline.jsonl",
        best_tokenizer,
        config.max_input_length,
        config.max_target_length,
    )
    best_validation_loader = DataLoader(
        best_validation,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=LEDBaselineCollator(best_tokenizer),
        num_workers=config.num_workers,
        drop_last=False,
    )
    best_model = load_led_triplet(
        best_checkpoint, attention_window=config.attention_window
    ).to(device)
    expected_window = [config.attention_window] * best_model.config.encoder_layers
    if list(best_model.config.attention_window) != expected_window:
        raise ValueError("best checkpoint attention-window configuration was not preserved")
    best_validation_result = evaluate_generation(
        best_model,
        best_validation,
        best_validation_loader,
        best_tokenizer,
        device,
        generation_settings,
        epoch=early.best_epoch if early.best_epoch is not None else -1,
        checkpoint_id=str(best_checkpoint.resolve()),
        description="Selected best · complete validation generation",
    )
    write_jsonl(
        config.output_dir / "validation_predictions_best.jsonl",
        best_validation_result.predictions,
    )
    atomic_write_json(
        config.output_dir / "validation_metrics_best.json",
        {
            **_generation_dict(best_validation_result),
            "checkpoint": str(best_checkpoint.resolve()),
            "selected_epoch": early.best_epoch,
            "generation_settings": generation_settings,
        },
    )
    print(
        f"Best checkpoint epoch {early.best_epoch} | "
        f"full_val_loss={best_validation_result.generation_loss:.4f} "
        f"full_val_acc={best_validation_result.token_accuracy:.4f} "
        f"full_val_f1={best_validation_result.token_overlap_f1:.4f} "
        f"full_val_repeat3={best_validation_result.mean_repeated_trigram_fraction:.4f} "
        f"full_val_len_ratio={best_validation_result.mean_word_length_ratio:.4f} "
        f"full_val_eos={best_validation_result.eos_termination_rate:.4f}",
        flush=True,
    )
    del best_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
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
        wandb_run.summary.update(
            {
                **summary,
                "selected_best/full_validation_loss": (
                    best_validation_result.generation_loss
                ),
                "selected_best/full_validation_token_accuracy": (
                    best_validation_result.token_accuracy
                ),
                "selected_best/full_validation_token_overlap_f1": (
                    best_validation_result.token_overlap_f1
                ),
                "selected_best/full_validation_eos_termination_rate": (
                    best_validation_result.eos_termination_rate
                ),
            }
        )
        wandb_run.finish()
    return summary


def evaluate_joint_checkpoint(
    *,
    data_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    split: str,
    eval_batch_size: int = 1,
    max_input_length: int = 512,
    max_target_length: int = 512,
    generation_max_new_tokens: int = 512,
    decoding_strategy: str = "greedy",
    attention_window: int = 512,
    device_name: str = "auto",
    overwrite_output_dir: bool = False,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("explicit checkpoint evaluation supports validation or test")
    state_path = checkpoint / "joint_state.json"
    if state_path.is_file():
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        expected_settings = {
            "max_input_length": max_input_length,
            "max_target_length": max_target_length,
            "attention_window": attention_window,
        }
        for name, requested in expected_settings.items():
            saved = state.get(name)
            if saved is not None and saved != requested:
                raise ValueError(
                    f"checkpoint {name}={saved} does not match requested {requested}"
                )
        saved_generation = state.get("generation_settings", {})
        saved_max_new = saved_generation.get("max_new_tokens")
        if saved_max_new is not None and saved_max_new != generation_max_new_tokens:
            raise ValueError(
                "checkpoint generation max_new_tokens="
                f"{saved_max_new} does not match requested {generation_max_new_tokens}"
            )
    prepare_output_directory(output_dir, overwrite_output_dir)
    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    dataset = LEDBaselineDataset(
        data_dir / split / "baseline.jsonl",
        tokenizer,
        max_input_length,
        max_target_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=LEDBaselineCollator(tokenizer),
        drop_last=False,
    )
    model = load_led_triplet(checkpoint, attention_window=attention_window).to(device)
    expected = [attention_window] * model.config.encoder_layers
    if list(model.config.attention_window) != expected:
        raise ValueError("checkpoint attention-window configuration was not preserved")
    settings = decoding_settings(
        {**GENERATION_SETTINGS, "max_new_tokens": generation_max_new_tokens},
        decoding_strategy,
    )
    result = evaluate_generation(
        model,
        dataset,
        loader,
        tokenizer,
        device,
        settings,
        epoch=-1,
        checkpoint_id=str(checkpoint.resolve()),
        description=f"Explicit {split} evaluation",
    )
    metrics = _generation_dict(result)
    metrics.update(
        {
            "checkpoint": str(checkpoint.resolve()),
            "split": split,
            "generation_settings": settings,
            "decoding_strategy": decoding_strategy,
        }
    )
    write_jsonl(output_dir / f"{split}_predictions.jsonl", result.predictions)
    atomic_write_json(output_dir / f"{split}_metrics.json", metrics)
    return metrics
