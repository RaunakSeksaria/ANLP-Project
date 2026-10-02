from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.training.overfit_baseline import prepare_output_directory
from src.training.train_baseline import (
    REQUIRED_HISTORY_FIELDS,
    REQUIRED_PREDICTION_FIELDS,
    REQUIRED_TEST_METRIC_FIELDS,
    EarlyStoppingState,
    EpochTrainingResult,
    EvaluationResult,
    TrainingConfig,
    _history_entry,
    atomic_write_json,
    calculate_step_counts,
    evaluate_split,
    place_model_on_device,
    resolve_single_device,
    run_training,
    train_one_epoch,
)


class CountingOptimizer:
    def __init__(self, parameters: list[torch.nn.Parameter]) -> None:
        self.inner = torch.optim.SGD(parameters, lr=0.0)
        self.step_count = 0
        self.gradients_at_step: list[float] = []

    @property
    def param_groups(self):  # type: ignore[no-untyped-def]
        return self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        parameter = self.inner.param_groups[0]["params"][0]
        self.gradients_at_step.append(float(parameter.grad.item()))
        self.step_count += 1
        self.inner.step()


class CountingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


class TinySeq2SeqModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=False)
        self.devices: list[str] = []

    def to(self, device):  # type: ignore[no-untyped-def]
        self.devices.append(str(device))
        return super().to(device)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask, global_attention_mask
        loss = self.weight * input_ids[:, 0].float().mean()
        vocabulary_size = 16
        logits = torch.zeros(
            labels.shape[0], labels.shape[1], vocabulary_size, device=labels.device
        )
        for row in range(labels.shape[0]):
            for column in range(labels.shape[1]):
                label = int(labels[row, column])
                if label >= 0:
                    logits[row, column, label] = 10.0
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, *, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        marker = int(input_ids[0, 0])
        return torch.tensor([[marker]], dtype=torch.long, device=input_ids.device)


class TinyTokenizer:
    pad_token_id = 0

    def decode(self, values, skip_special_tokens: bool = True) -> str:  # type: ignore[no-untyped-def]
        del skip_special_tokens
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        return " ".join(str(value) for value in values if value not in {0, -100})


class TinyDataset:
    def __init__(self) -> None:
        self.records = [
            SimpleNamespace(
                example_id="example-0",
                event_id="event-0",
                input_article_id="article-0",
                input_ideology="left",
                target_article_id="target-0",
            ),
            SimpleNamespace(
                example_id="example-1",
                event_id="event-1",
                input_article_id="article-1",
                input_ideology="right",
                target_article_id="target-1",
            ),
        ]

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int):  # type: ignore[no-untyped-def]
        return self.records[index]

    def source_text(self, index: int) -> str:
        return f"source-{index}"


def batch(marker: int, labels: list[int]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[marker]], dtype=torch.long),
        "attention_mask": torch.tensor([[1]], dtype=torch.long),
        "global_attention_mask": torch.tensor([[1]], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def create_count_files(root: Path, counts: dict[str, int]) -> None:
    for split, count in counts.items():
        path = root / split / "baseline.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join("{}\n" for _ in range(count)), encoding="utf-8")


class TrainBaselineTests(unittest.TestCase):
    def test_default_configuration_values(self) -> None:
        config = TrainingConfig(data_dir=Path("prepared"))
        self.assertEqual(config.model_name, "allenai/led-base-16384")
        self.assertEqual(config.max_epochs, 10)
        self.assertEqual(config.train_batch_size, 4)
        self.assertEqual(config.eval_batch_size, 4)
        self.assertEqual(config.gradient_accumulation_steps, 2)
        self.assertEqual(config.learning_rate, 3e-5)
        self.assertEqual(config.weight_decay, 0.01)
        self.assertEqual(config.warmup_ratio, 0.10)
        self.assertEqual(config.precision, "fp16")
        self.assertEqual(config.generation_settings["num_beams"], 1)
        self.assertFalse(config.generation_settings["do_sample"])

    def test_step_counts_and_effective_batch_size(self) -> None:
        config = TrainingConfig(data_dir=Path("prepared"))
        counts = calculate_step_counts(512, config)
        self.assertEqual(config.effective_batch_size, 8)
        self.assertEqual(counts.training_microbatches_per_epoch, 128)
        self.assertEqual(counts.optimizer_steps_per_epoch, 64)
        self.assertEqual(counts.maximum_optimizer_steps, 640)
        self.assertEqual(counts.warmup_steps, 64)

    def test_gradient_accumulation_divides_loss_and_steps_on_boundaries(self) -> None:
        model = TinySeq2SeqModel()
        optimizer = CountingOptimizer(list(model.parameters()))
        scheduler = CountingScheduler()
        result = train_one_epoch(
            model,
            [batch(4, [1]), batch(4, [1])],  # type: ignore[arg-type]
            optimizer,  # type: ignore[arg-type]
            scheduler,
            torch.device("cpu"),
            "fp32",
            2,
            100.0,
        )
        self.assertEqual(result.optimizer_steps, 1)
        self.assertEqual(optimizer.step_count, 1)
        self.assertEqual(scheduler.step_count, 1)
        # Each loss is 4 and each backward contribution is 4 / 2, totalling 4.
        self.assertAlmostEqual(optimizer.gradients_at_step[0], 4.0)

    def test_final_partial_group_steps_optimizer_and_scheduler_once(self) -> None:
        model = TinySeq2SeqModel()
        optimizer = CountingOptimizer(list(model.parameters()))
        scheduler = CountingScheduler()
        result = train_one_epoch(
            model,
            [batch(2, [1]), batch(2, [1]), batch(2, [1])],  # type: ignore[arg-type]
            optimizer,  # type: ignore[arg-type]
            scheduler,
            torch.device("cpu"),
            "fp32",
            2,
            100.0,
        )
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(optimizer.step_count, 2)
        self.assertEqual(scheduler.step_count, 2)

    def test_training_loss_is_token_weighted(self) -> None:
        model = TinySeq2SeqModel()
        optimizer = CountingOptimizer(list(model.parameters()))
        result = train_one_epoch(
            model,
            [batch(2, [1, -100, -100]), batch(4, [1, 2, 3])],  # type: ignore[arg-type]
            optimizer,  # type: ignore[arg-type]
            CountingScheduler(),
            torch.device("cpu"),
            "fp32",
            2,
            100.0,
        )
        self.assertAlmostEqual(result.token_weighted_loss, 3.5)

    def test_validation_is_token_weighted_accurate_and_ordered(self) -> None:
        model = TinySeq2SeqModel()
        dataset = TinyDataset()
        result = evaluate_split(
            model,
            dataset,  # type: ignore[arg-type]
            [batch(2, [2, -100, -100]), batch(4, [4, 3, 2])],  # type: ignore[arg-type]
            TinyTokenizer(),
            torch.device("cpu"),
            "fp32",
            {"num_beams": 1, "do_sample": False, "max_new_tokens": 4, "early_stopping": False},
        )
        self.assertAlmostEqual(result.teacher_forced_loss, 3.5)
        self.assertEqual(result.token_accuracy, 1.0)
        self.assertEqual([item["example_id"] for item in result.predictions], ["example-0", "example-1"])
        self.assertEqual(result.predictions[0]["generated_text"], "2")
        self.assertEqual(result.predictions[1]["generated_text"], "4")

    def test_early_stopping_improvement_and_exact_patience(self) -> None:
        state = EarlyStoppingState(patience=3, minimum_delta=0.0)
        self.assertTrue(state.update(5.0, 1))
        self.assertEqual(state.best_epoch, 1)
        self.assertFalse(state.update(5.0, 2))
        self.assertFalse(state.should_stop)
        self.assertFalse(state.update(5.1, 3))
        self.assertFalse(state.should_stop)
        self.assertFalse(state.update(6.0, 4))
        self.assertTrue(state.should_stop)
        self.assertEqual(state.best_epoch, 1)

    def test_checkpoint_selection_uses_only_validation_loss(self) -> None:
        state = EarlyStoppingState(patience=3, minimum_delta=0.0)
        unrelated_test_losses = [0.01, 99.0, 0.001]
        for epoch, validation_loss in enumerate([3.0, 2.0, 2.5], 1):
            state.update(validation_loss, epoch)
        self.assertEqual(unrelated_test_losses, [0.01, 99.0, 0.001])
        self.assertEqual(state.best_loss, 2.0)
        self.assertEqual(state.best_epoch, 2)

    def test_output_directory_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output, overwrite=False)

    def test_atomic_json_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metadata.json"
            atomic_write_json(path, {"version": 1})
            atomic_write_json(path, {"version": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"version": 2})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_required_output_field_sets(self) -> None:
        evaluation = EvaluationResult(1.0, 0.5, 0.25, 1, 0.5, 2, [])
        entry = _history_entry(
            1,
            2,
            EpochTrainingResult(3.0, 2),
            evaluation,
            1e-5,
            4.0,
            (None, None),
            True,
            0,
        )
        self.assertTrue(REQUIRED_HISTORY_FIELDS <= entry.keys())
        self.assertEqual(len(REQUIRED_PREDICTION_FIELDS), 10)
        self.assertEqual(len(REQUIRED_TEST_METRIC_FIELDS), 15)

    def test_dry_run_has_no_loaders_optimizer_cuda_or_output_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "prepared"
            output_dir = root / "must-not-exist"
            create_count_files(data_dir, {"train": 8, "validation": 2, "test": 2})
            config = TrainingConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                dry_run=True,
            )
            with (
                patch("src.training.train_baseline.AutoTokenizer.from_pretrained") as tokenizer_loader,
                patch("src.training.train_baseline.load_led_baseline") as model_loader,
                patch("src.training.train_baseline.torch.optim.AdamW") as optimizer_loader,
                patch("src.training.train_baseline.get_linear_schedule_with_warmup") as scheduler_loader,
                patch("src.training.train_baseline.torch.cuda.is_available", side_effect=AssertionError("CUDA touched")),
            ):
                resolved = run_training(config)
            tokenizer_loader.assert_not_called()
            model_loader.assert_not_called()
            optimizer_loader.assert_not_called()
            scheduler_loader.assert_not_called()
            self.assertFalse(output_dir.exists())
            self.assertEqual(resolved["dataset_counts"]["train"], 8)

    def test_cuda_device_is_always_zero_and_model_is_not_wrapped(self) -> None:
        with patch("src.training.overfit_baseline.torch.cuda.is_available", return_value=True):
            device = resolve_single_device("cuda")
        self.assertEqual(str(device), "cuda:0")
        model = TinySeq2SeqModel()
        returned = place_model_on_device(model, torch.device("cpu"))
        self.assertIs(returned, model)
        self.assertNotIsInstance(returned, torch.nn.DataParallel)


if __name__ == "__main__":
    unittest.main()
