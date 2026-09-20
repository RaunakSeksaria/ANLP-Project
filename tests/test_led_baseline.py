from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.models.led_baseline import (
    forward_led_baseline,
    load_led_baseline,
    validate_forward_output,
)
from src.training.overfit_baseline import (
    REQUIRED_RESULT_FIELDS,
    EvaluationMetrics,
    passes_overfit_criteria,
    prepare_output_directory,
    select_first_eight,
    token_accuracy_counts,
    token_overlap_f1,
    write_results_json,
)


class RecordingModel:
    def __init__(self, output: SimpleNamespace) -> None:
        self.output = output
        self.kwargs: dict[str, torch.Tensor] | None = None

    def __call__(self, **kwargs: torch.Tensor) -> SimpleNamespace:
        self.kwargs = kwargs
        return self.output


class FakeDataset:
    jsonl_path = Path("fixture/train/baseline.jsonl")

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def metrics(loss: float, accuracy: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        mean_loss=loss,
        token_accuracy=accuracy,
        overlap_f1=0.0,
        exact_match_count=0,
        predictions=[],
        generated_token_ids=[],
    )


class LEDBaselineTests(unittest.TestCase):
    def test_loader_requests_provided_pretrained_identifier(self) -> None:
        fake_model = MagicMock()
        fake_model.parameters.return_value = [torch.nn.Parameter(torch.ones(1))]
        fake_config = SimpleNamespace(encoder_layers=6, attention_window=[1024] * 6)
        with (
            patch(
                "src.models.led_baseline.LEDConfig.from_pretrained",
                return_value=fake_config,
            ) as config_from_pretrained,
            patch(
                "src.models.led_baseline.LEDForConditionalGeneration.from_pretrained",
                return_value=fake_model,
            ) as model_from_pretrained,
        ):
            loaded = load_led_baseline("local-or-hub-identifier")
        config_from_pretrained.assert_called_once_with("local-or-hub-identifier")
        self.assertEqual(fake_config.attention_window, [512] * 6)
        model_from_pretrained.assert_called_once_with(
            "local-or-hub-identifier", config=fake_config, use_safetensors=True
        )
        self.assertIs(loaded, fake_model)

    def test_forward_helper_passes_exact_required_tensors(self) -> None:
        tensors = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.long),
            "global_attention_mask": torch.tensor([[1, 0, 0], [1, 0, 0]]),
            "labels": torch.ones((2, 4), dtype=torch.long),
        }
        output = SimpleNamespace(loss=torch.tensor(1.0), logits=torch.zeros((2, 4, 7)))
        model = RecordingModel(output)
        returned = forward_led_baseline(model, **tensors)  # type: ignore[arg-type]
        self.assertIs(returned, output)
        self.assertEqual(set(model.kwargs or {}), set(tensors))
        for name, tensor in tensors.items():
            self.assertIs((model.kwargs or {})[name], tensor)

    def test_missing_or_nonfinite_loss_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing loss"):
            validate_forward_output(SimpleNamespace(loss=None, logits=torch.zeros((1, 2, 3))))
        with self.assertRaisesRegex(FloatingPointError, "loss is not finite"):
            validate_forward_output(
                SimpleNamespace(loss=torch.tensor(float("nan")), logits=torch.zeros((1, 2, 3)))
            )

    def test_nonfinite_logits_are_rejected(self) -> None:
        logits = torch.zeros((1, 2, 3))
        logits[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(FloatingPointError, "logits contain"):
            validate_forward_output(SimpleNamespace(loss=torch.tensor(1.0), logits=logits))

    def test_first_eight_records_are_selected_deterministically(self) -> None:
        subset = select_first_eight(FakeDataset(12))  # type: ignore[arg-type]
        self.assertEqual(list(subset.indices), list(range(8)))
        self.assertEqual([subset[index] for index in range(8)], list(range(8)))

    def test_fewer_than_eight_records_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8.*found 7"):
            select_first_eight(FakeDataset(7))  # type: ignore[arg-type]

    def test_token_accuracy_ignores_negative_100(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 9.0, 0.0],
                    [0.0, 0.0, 9.0],
                    [9.0, 0.0, 0.0],
                ]
            ]
        )
        labels = torch.tensor([[1, -100, 2]])
        self.assertEqual(token_accuracy_counts(logits, labels), (1, 2))

    def test_token_overlap_f1_uses_multiset_overlap(self) -> None:
        self.assertAlmostEqual(token_overlap_f1("A a b", "a b b"), 2 / 3)

    def test_token_overlap_f1_handles_empty_predictions(self) -> None:
        self.assertEqual(token_overlap_f1("", "nonempty reference"), 0.0)
        self.assertEqual(token_overlap_f1("   ", ""), 1.0)

    def test_pass_criteria_use_half_initial_loss_and_accuracy_improvement(self) -> None:
        initial = metrics(10.0, 0.2)
        self.assertTrue(passes_overfit_criteria(initial, metrics(5.0, 0.21)))
        self.assertFalse(passes_overfit_criteria(initial, metrics(5.01, 0.9)))
        self.assertFalse(passes_overfit_criteria(initial, metrics(4.0, 0.2)))

    def test_output_directory_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "checkpoint"
            output_dir.mkdir()
            (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite-output-dir"):
                prepare_output_directory(output_dir, overwrite=False)
            self.assertTrue((output_dir / "existing.txt").exists())
            prepare_output_directory(output_dir, overwrite=True)
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_results_json_contains_every_required_field(self) -> None:
        values = {field: 0 for field in REQUIRED_RESULT_FIELDS}
        values.update(
            {
                "model_name": "fake-led",
                "precision": "fp32",
                "device": "cpu",
                "optimizer_settings": {},
                "early_stopping": False,
                "passed": True,
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "overfit_results.json"
            write_results_json(path, values)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(REQUIRED_RESULT_FIELDS <= loaded.keys())


if __name__ == "__main__":
    unittest.main()
