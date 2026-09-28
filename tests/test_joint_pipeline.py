from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from transformers import LEDConfig, LEDForConditionalGeneration

from src.data.triplet_dataset import LEDTripletCollator, LEDTripletDataset
from src.models.led_triplet import (
    JointForwardOutput,
    LEDTripletModel,
    load_led_triplet,
    masked_triplet_margin_loss,
)
from src.models.pooling import masked_mean_pool
from src.training.train_joint import (
    JointTrainingConfig,
    VALIDATION_GENERATION_SUBSET_SEED,
    VALIDATION_GENERATION_SUBSET_SIZE,
    _fixed_subset_indices,
    calculate_joint_steps,
    evaluate_teacher_forced,
    initialize_wandb,
    train_joint_epoch,
)


class FakeTokenizer:
    sep_token = "<sep>"
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2

    @staticmethod
    def token_id(token: str) -> int:
        return 3 + sum((index + 1) * ord(char) for index, char in enumerate(token)) % 1000

    def __call__(
        self,
        text: str | None = None,
        *,
        text_target: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        **_: Any,
    ) -> dict[str, list[int]]:
        del padding
        selected = text_target if text_target is not None else text
        if selected is None:
            raise TypeError("text is required")
        content = [self.token_id(token) for token in selected.split()]
        ids = [0, *content, 2]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            ids[-1] = 2
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def baseline(event: str, ideology: str, body: str | None = None) -> dict[str, str]:
    suffix = "1" if ideology == "left" else "2"
    return {
        "example_id": f"{event}:{ideology}-to-center",
        "split": "train",
        "event_id": event,
        "input_article_id": f"{event}:{suffix}",
        "input_ideology": ideology,
        "input_title": f"{ideology} title {event}",
        "input_text": body or f"{ideology} body {event}",
        "target_article_id": f"{event}:neutral",
        "target_title": "",
        "target_text": f"authoritative neutral target {event}",
    }


def triplet(
    anchor: dict[str, str], positive: dict[str, str], negative: dict[str, str]
) -> dict[str, str]:
    return {
        "triplet_id": f"{anchor['event_id']}:{anchor['input_ideology']}-anchor",
        "split": "train",
        "anchor_event_id": anchor["event_id"],
        "anchor_article_id": anchor["input_article_id"],
        "anchor_ideology": anchor["input_ideology"],
        "anchor_title": anchor["input_title"],
        "anchor_text": anchor["input_text"],
        "positive_event_id": positive["event_id"],
        "positive_article_id": positive["input_article_id"],
        "positive_ideology": positive["input_ideology"],
        "positive_title": positive["input_title"],
        "positive_text": positive["input_text"],
        "negative_event_id": negative["event_id"],
        "negative_article_id": negative["input_article_id"],
        "negative_ideology": negative["input_ideology"],
        "negative_title": negative["input_title"],
        "negative_text": negative["input_text"],
        "target_article_id": "stale-target",
        "target_title": "stale title",
        "target_text": "stale copied target",
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_fixture(root: Path, *, identical_event: str | None = None) -> tuple[Path, Path]:
    baselines: list[dict[str, str]] = []
    by_event: dict[str, dict[str, dict[str, str]]] = {}
    for event in ("1", "2", "3"):
        left = baseline(event, "left")
        right_body = left["input_text"] if event == identical_event else None
        right = baseline(event, "right", right_body)
        baselines.extend((left, right))
        by_event[event] = {"left": left, "right": right}
    triplets = []
    events = ("1", "2", "3")
    for index, event in enumerate(events):
        negative_event = events[(index + 1) % len(events)]
        triplets.append(
            triplet(by_event[event]["left"], by_event[event]["right"], by_event[negative_event]["left"])
        )
        triplets.append(
            triplet(by_event[event]["right"], by_event[event]["left"], by_event[negative_event]["right"])
        )
    baseline_path = root / "train" / "baseline.jsonl"
    triplet_path = root / "train" / "triplets.jsonl"
    write_jsonl(baseline_path, baselines)
    write_jsonl(triplet_path, triplets)
    return baseline_path, triplet_path


def tiny_led() -> LEDTripletModel:
    config = LEDConfig(
        vocab_size=64,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_encoder_position_embeddings=64,
        max_decoder_position_embeddings=64,
        attention_window=[4],
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    return LEDTripletModel(LEDForConditionalGeneration(config))


def tiny_batch(valid: bool = True) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for role, ids in (
        ("anchor", [0, 4, 5, 2]),
        ("positive", [0, 6, 7, 2]),
        ("negative", [0, 8, 9, 2]),
    ):
        result[f"{role}_input_ids"] = torch.tensor([ids])
        result[f"{role}_attention_mask"] = torch.ones(1, 4, dtype=torch.long)
        result[f"{role}_global_attention_mask"] = torch.tensor([[1, 0, 0, 0]])
    result["labels"] = torch.tensor([[0, 10, 11, 2]])
    result["triplet_valid_mask"] = torch.tensor([valid])
    return result


class PoolingAndModelTests(unittest.TestCase):
    def test_masked_mean_correct_padding_invariant_normalized_and_finite(self) -> None:
        hidden = torch.tensor([[[1.0, 0.0], [3.0, 4.0]]])
        normalized, raw = masked_mean_pool(hidden, torch.tensor([[1, 1]]))
        self.assertTrue(torch.allclose(raw, torch.tensor([[2.0, 2.0]])))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(normalized, dim=-1), torch.ones(1)))
        padded = torch.cat((hidden, torch.tensor([[[999.0, 999.0]]])), dim=1)
        normalized_padded, raw_padded = masked_mean_pool(padded, torch.tensor([[1, 1]]))
        self.assertTrue(torch.allclose(raw, raw_padded))
        self.assertTrue(torch.allclose(normalized, normalized_padded))
        self.assertTrue(torch.isfinite(normalized).all())

    def test_all_padding_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "all-padding"):
            masked_mean_pool(torch.ones(1, 2, 3), torch.zeros(1, 2))

    def test_triplet_loss_satisfied_and_violated(self) -> None:
        anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        negative = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        batch_mean, valid_mean, per_example, _, _ = masked_triplet_margin_loss(
            anchor, positive, negative, torch.tensor([True, True]), 0.2
        )
        self.assertAlmostEqual(float(per_example[0]), 0.0, places=5)
        self.assertGreater(float(per_example[1]), 2.0)
        self.assertAlmostEqual(float(batch_mean), float(valid_mean))

    def test_joint_forward_gradients_and_decoder_exclusion(self) -> None:
        combined_model = tiny_led()
        combined_model.eval()
        combined = combined_model(
            **tiny_batch(), triplet_lambda=0.1, triplet_margin=5.0
        )
        self.assertTrue(torch.isfinite(combined.generation_loss))
        self.assertTrue(torch.isfinite(combined.triplet_loss))
        self.assertTrue(torch.isfinite(combined.total_loss))
        self.assertTrue(
            torch.allclose(
                combined.total_loss,
                combined.generation_loss + 0.1 * combined.triplet_loss,
            )
        )
        combined.total_loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in combined_model.parameters()
            )
        )

        model = tiny_led()
        model.eval()
        output = model(**tiny_batch(), triplet_lambda=0.1, triplet_margin=5.0)
        self.assertTrue(torch.isfinite(output.total_loss))
        assert output.anchor_embedding is not None
        assert output.positive_embedding is not None
        assert output.negative_embedding is not None
        for embedding in (
            output.anchor_embedding,
            output.positive_embedding,
            output.negative_embedding,
        ):
            embedding.retain_grad()
        output.triplet_loss.backward()
        self.assertIsNotNone(output.anchor_embedding.grad)
        self.assertIsNotNone(output.positive_embedding.grad)
        self.assertIsNotNone(output.negative_embedding.grad)
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if "encoder.layers" in name and parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)
        decoder_only_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if "decoder.layers" in name
        ]
        self.assertTrue(all(gradient is None for gradient in decoder_only_grads))

    def test_all_invalid_keeps_generation_and_lambda_zero_skips_encoders(self) -> None:
        model = tiny_led()
        model.eval()
        invalid = model(**tiny_batch(False), triplet_lambda=0.1, triplet_margin=0.2)
        self.assertEqual(float(invalid.triplet_loss), 0.0)
        self.assertTrue(torch.equal(invalid.total_loss, invalid.generation_loss))
        direct = model.led(
            input_ids=tiny_batch()["anchor_input_ids"],
            attention_mask=tiny_batch()["anchor_attention_mask"],
            global_attention_mask=tiny_batch()["anchor_global_attention_mask"],
            labels=tiny_batch()["labels"],
            return_dict=True,
        )
        original = model._encode
        model._encode = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("encoded"))  # type: ignore[method-assign]
        control = model(**tiny_batch(), triplet_lambda=0.0, triplet_margin=0.2)
        model._encode = original  # type: ignore[method-assign]
        self.assertTrue(torch.allclose(control.generation_loss, direct.loss))
        self.assertTrue(torch.allclose(control.generation_logits, direct.logits))

    def test_checkpoint_reload_and_anchor_only_generation(self) -> None:
        model = tiny_led()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            model.save_pretrained(path)
            reloaded = load_led_triplet(path, attention_window=4)
            generated = reloaded.led.generate(
                input_ids=tiny_batch()["anchor_input_ids"],
                attention_mask=tiny_batch()["anchor_attention_mask"],
                global_attention_mask=tiny_batch()["anchor_global_attention_mask"],
                max_new_tokens=2,
                do_sample=False,
                num_beams=1,
            )
            self.assertEqual(generated.shape[0], 1)


class TripletDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tokenizer = FakeTokenizer()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authoritative_target_masks_and_attention(self) -> None:
        baseline_path, triplet_path = make_fixture(self.root)
        dataset = LEDTripletDataset(baseline_path, triplet_path, self.tokenizer)
        self.assertEqual(dataset.record(0).target_text, "authoritative neutral target 1")
        self.assertEqual(dataset.resolution_statistics["target_disagreements"], 6)
        examples = [dataset[0], dataset[1]]
        batch = LEDTripletCollator(self.tokenizer)(examples)
        self.assertTrue(torch.all(batch["anchor_global_attention_mask"][:, 0] == 1))
        self.assertTrue(torch.all(batch["positive_global_attention_mask"][:, 0] == 1))
        self.assertTrue(torch.all(batch["negative_global_attention_mask"][:, 0] == 1))
        for role in ("anchor", "positive", "negative"):
            mask = batch[f"{role}_attention_mask"]
            global_mask = batch[f"{role}_global_attention_mask"]
            self.assertTrue(torch.all(global_mask[mask == 0] == 0))
        shorter = min(len(examples[0]["labels"]), len(examples[1]["labels"]))
        self.assertTrue(torch.all(batch["labels"][:, :shorter] != -100))

    def test_invalid_negative_is_replaced_deterministically(self) -> None:
        baseline_path, triplet_path = make_fixture(self.root)
        rows = [json.loads(line) for line in triplet_path.read_text().splitlines()]
        rows[0]["negative_event_id"] = rows[0]["anchor_event_id"]
        rows[0]["negative_article_id"] = rows[0]["anchor_article_id"]
        rows[0]["negative_ideology"] = rows[0]["anchor_ideology"]
        rows[0]["negative_title"] = rows[0]["anchor_title"]
        rows[0]["negative_text"] = rows[0]["anchor_text"]
        write_jsonl(triplet_path, rows)
        first = LEDTripletDataset(baseline_path, triplet_path, self.tokenizer, seed=42)
        second = LEDTripletDataset(baseline_path, triplet_path, self.tokenizer, seed=42)
        a = first.record(0)
        b = second.record(0)
        self.assertTrue(a.negative_replaced)
        self.assertEqual(a.negative.article_id, b.negative.article_id)  # type: ignore[union-attr]
        self.assertNotEqual(a.negative.event_id, a.anchor.event_id)  # type: ignore[union-attr]
        self.assertEqual(a.negative.ideology, a.anchor.ideology)  # type: ignore[union-attr]
        self.assertNotEqual(a.negative.text, a.anchor.text)  # type: ignore[union-attr]

    def test_identical_positive_pair_is_masked_but_generation_remains(self) -> None:
        baseline_path, triplet_path = make_fixture(self.root, identical_event="1")
        dataset = LEDTripletDataset(baseline_path, triplet_path, self.tokenizer)
        self.assertFalse(dataset[0]["triplet_valid_mask"])
        self.assertTrue(dataset[0]["labels"])
        self.assertIn("identical_anchor_positive_body", dataset.record(0).invalid_reasons)

    def test_no_valid_negative_masks_triplet(self) -> None:
        left = baseline("1", "left")
        right = baseline("1", "right")
        baseline_path = self.root / "train" / "baseline.jsonl"
        triplet_path = self.root / "train" / "triplets.jsonl"
        write_jsonl(baseline_path, [left, right])
        write_jsonl(triplet_path, [triplet(left, right, left), triplet(right, left, right)])
        dataset = LEDTripletDataset(baseline_path, triplet_path, self.tokenizer)
        self.assertTrue(all(not dataset[index]["triplet_valid_mask"] for index in range(2)))
        self.assertTrue(all(dataset.record(index).negative is None for index in range(2)))

    def test_512_token_truncation_reporting_and_eos_retention(self) -> None:
        baseline_path, triplet_path = make_fixture(self.root)
        baselines = [json.loads(line) for line in baseline_path.read_text().splitlines()]
        baselines[0]["input_text"] = " ".join(["source"] * 507)
        baselines[0]["target_text"] = " ".join(["target"] * 511)
        baselines[1]["target_text"] = baselines[0]["target_text"]
        write_jsonl(baseline_path, baselines)
        article_lookup = {row["input_article_id"]: row for row in baselines}
        triplets = [json.loads(line) for line in triplet_path.read_text().splitlines()]
        for row in triplets:
            for role in ("anchor", "positive", "negative"):
                article = article_lookup[row[f"{role}_article_id"]]
                row[f"{role}_event_id"] = article["event_id"]
                row[f"{role}_ideology"] = article["input_ideology"]
                row[f"{role}_title"] = article["input_title"]
                row[f"{role}_text"] = article["input_text"]
        write_jsonl(triplet_path, triplets)
        dataset = LEDTripletDataset(
            baseline_path,
            triplet_path,
            self.tokenizer,
            max_input_length=512,
            max_target_length=512,
        )
        statistics = dataset.truncation_statistics()
        self.assertEqual(statistics["anchor"]["truncated"], 1)
        self.assertEqual(statistics["positive"]["truncated"], 1)
        self.assertEqual(statistics["target"]["truncated"], 2)
        for role in ("anchor", "positive", "negative", "target"):
            self.assertEqual(
                statistics[role]["eos_retained"], statistics[role]["records"]
            )


class FakeJointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, **kwargs: Any) -> JointForwardOutput:
        marker = kwargs["anchor_input_ids"][:, 0].float().mean()
        loss = self.weight * marker
        batch = kwargs["anchor_input_ids"].shape[0]
        zeros = loss.new_zeros((batch,))
        target_length = kwargs["labels"].shape[1]
        logits = loss.new_zeros((batch, target_length, 3))
        logits[:, :, 2] = 1.0
        return JointForwardOutput(
            generation_loss=loss,
            triplet_loss=loss.new_zeros(()),
            valid_triplet_mean=loss.new_zeros(()),
            total_loss=loss,
            generation_logits=logits,
            anchor_embedding=None,
            positive_embedding=None,
            negative_embedding=None,
            anchor_raw_embedding=None,
            positive_raw_embedding=None,
            negative_raw_embedding=None,
            anchor_positive_distance=zeros,
            anchor_negative_distance=zeros,
            per_example_triplet_loss=zeros,
            valid_count=0,
        )


class RecordingOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.inner = torch.optim.SGD([parameter], lr=0.0)
        self.gradients: list[float] = []

    @property
    def param_groups(self):  # type: ignore[no-untyped-def]
        return self.inner.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        parameter = self.inner.param_groups[0]["params"][0]
        self.gradients.append(float(parameter.grad))
        self.inner.step()


class Scheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class RecordingWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int | None]] = []

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((values, step))


def training_batch(marker: int, target_length: int = 1) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "labels": torch.tensor([[2] * target_length]),
        "triplet_valid_mask": torch.tensor([False]),
        "example_id": [f"example-{marker}"],
    }
    for role in ("anchor", "positive", "negative"):
        batch[f"{role}_input_ids"] = torch.tensor([[marker]])
        batch[f"{role}_attention_mask"] = torch.tensor([[1]])
        batch[f"{role}_global_attention_mask"] = torch.tensor([[1]])
    return batch


class JointTrainingTests(unittest.TestCase):
    def test_defaults_and_steps(self) -> None:
        config = JointTrainingConfig(Path("data"), Path("output"))
        self.assertEqual(config.train_batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(config.effective_anchor_batch_size, 8)
        self.assertEqual(config.precision, "fp32")
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual(config.triplet_lambda, 0.1)
        self.assertEqual(config.max_input_length, 512)
        self.assertEqual(config.max_target_length, 512)
        self.assertEqual(config.generation_max_new_tokens, 512)
        self.assertEqual(config.attention_window, 512)
        self.assertTrue(config.wandb_enabled)
        self.assertEqual(config.wandb_project, "anlp-project")
        self.assertEqual(config.wandb_entity, "manasagrawal206-iiit-hyderabad")
        self.assertFalse(config.wandb_required)
        steps = calculate_joint_steps(16000, config)
        self.assertEqual(steps.microbatches_per_epoch, 16000)
        self.assertEqual(steps.optimizer_steps_per_epoch, 2000)
        self.assertEqual(steps.maximum_optimizer_steps, 20000)
        self.assertEqual(steps.warmup_steps, 2000)

    def test_validation_generation_subset_is_fixed_unique_and_seeded(self) -> None:
        first = _fixed_subset_indices(
            2000,
            VALIDATION_GENERATION_SUBSET_SEED,
            VALIDATION_GENERATION_SUBSET_SIZE,
            "validation",
        )
        second = _fixed_subset_indices(
            2000,
            VALIDATION_GENERATION_SUBSET_SEED,
            VALIDATION_GENERATION_SUBSET_SIZE,
            "validation",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 256)
        self.assertEqual(len(set(first)), 256)
        self.assertEqual(first, sorted(first))

    def test_wandb_failure_does_not_abort_training_by_default(self) -> None:
        config = JointTrainingConfig(Path("data"), Path("output"))
        with patch("wandb.init", side_effect=RuntimeError("permission denied")):
            with patch("wandb.finish"):
                self.assertIsNone(initialize_wandb(config, {"configuration": {}}))

    def test_wandb_failure_can_be_required_explicitly(self) -> None:
        config = JointTrainingConfig(
            Path("data"), Path("output"), wandb_required=True
        )
        with patch("wandb.init", side_effect=RuntimeError("permission denied")):
            with self.assertRaisesRegex(RuntimeError, "W&B initialization failed"):
                initialize_wandb(config, {"configuration": {}})

    def test_teacher_forced_validation_covers_full_loader_without_generation(self) -> None:
        class FakeLED(torch.nn.Module):
            def forward(self, **kwargs: Any) -> Any:
                labels = kwargs["labels"]
                logits = torch.zeros((*labels.shape, 8))
                for row in range(labels.shape[0]):
                    for column in range(labels.shape[1]):
                        label = int(labels[row, column])
                        if label != -100:
                            logits[row, column, label] = 10.0
                return SimpleNamespace(loss=torch.tensor(2.0), logits=logits)

            def generate(self, **_kwargs: Any) -> torch.Tensor:
                raise AssertionError("teacher-forced validation must not generate")

        class Wrapper(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.led = FakeLED()

        def evaluation_batch(labels: list[list[int]]) -> dict[str, torch.Tensor]:
            batch_size = len(labels)
            return {
                "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
                "global_attention_mask": torch.tensor([[1, 0, 0]] * batch_size),
                "labels": torch.tensor(labels),
            }

        batches = [
            evaluation_batch([[2, 3, -100], [2, -100, -100]]),
            evaluation_batch([[4, 5, 2]]),
        ]
        model = Wrapper()
        model.train()
        result = evaluate_teacher_forced(  # type: ignore[arg-type]
            model, batches, torch.device("cpu")  # type: ignore[arg-type]
        )
        self.assertEqual(result.examples, 3)
        self.assertEqual(result.target_tokens, 6)
        self.assertEqual(result.token_accuracy, 1.0)
        self.assertEqual(result.generation_loss, 2.0)
        self.assertTrue(model.training)

    def test_final_incomplete_accumulation_group_is_not_underweighted(self) -> None:
        model = FakeJointModel()
        optimizer = RecordingOptimizer(model.weight)
        scheduler = Scheduler()
        wandb_run = RecordingWandbRun()
        config = JointTrainingConfig(
            Path("data"),
            Path("output"),
            gradient_accumulation_steps=8,
            max_grad_norm=100.0,
        )
        batches = [training_batch(1) for _ in range(8)] + [training_batch(2) for _ in range(2)]
        result = train_joint_epoch(
            model,  # type: ignore[arg-type]
            batches,  # type: ignore[arg-type]
            optimizer,  # type: ignore[arg-type]
            scheduler,
            torch.device("cpu"),
            config,
            epoch=1,
            global_step=0,
            wandb_run=wandb_run,
        )
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.token_accuracy, 1.0)
        self.assertEqual([step for _, step in wandb_run.logged], [1, 2])
        self.assertIn("train/step_generation_loss", wandb_run.logged[0][0])
        self.assertIn("train/step_token_accuracy", wandb_run.logged[0][0])
        self.assertEqual(scheduler.steps, 2)
        self.assertAlmostEqual(optimizer.gradients[0], 1.0)
        self.assertAlmostEqual(optimizer.gradients[1], 2.0)

    def test_epoch_objective_components_use_one_microbatch_denominator(self) -> None:
        class ObjectiveModel(FakeJointModel):
            def forward(self, **kwargs: Any) -> JointForwardOutput:
                output = super().forward(**kwargs)
                output.triplet_loss = 2 * output.generation_loss
                output.total_loss = (
                    output.generation_loss
                    + kwargs["triplet_lambda"] * output.triplet_loss
                )
                return output

        model = ObjectiveModel()
        config = JointTrainingConfig(
            Path("data"),
            Path("output"),
            gradient_accumulation_steps=2,
            triplet_lambda=0.35,
            max_grad_norm=100.0,
        )
        result = train_joint_epoch(
            model,  # type: ignore[arg-type]
            [training_batch(1, 1), training_batch(3, 3)],  # type: ignore[arg-type]
            RecordingOptimizer(model.weight),  # type: ignore[arg-type]
            Scheduler(),
            torch.device("cpu"),
            config,
            epoch=1,
            global_step=0,
        )
        self.assertAlmostEqual(result.generation_loss, 2.0)
        self.assertAlmostEqual(result.triplet_loss, 4.0)
        self.assertAlmostEqual(result.weighted_triplet_loss, 1.4)
        self.assertAlmostEqual(result.total_loss, 3.4)
        self.assertAlmostEqual(
            result.total_loss,
            result.generation_loss + result.weighted_triplet_loss,
        )

    def test_mismatched_microbatch_objective_is_rejected_before_scaling(self) -> None:
        class MismatchedObjectiveModel(FakeJointModel):
            def forward(self, **kwargs: Any) -> JointForwardOutput:
                output = super().forward(**kwargs)
                output.triplet_loss = output.generation_loss
                return output

        model = MismatchedObjectiveModel()
        config = JointTrainingConfig(
            Path("data"), Path("output"), triplet_lambda=0.35
        )
        with self.assertRaisesRegex(ValueError, "objective mismatch"):
            train_joint_epoch(
                model,  # type: ignore[arg-type]
                [training_batch(1)],  # type: ignore[arg-type]
                RecordingOptimizer(model.weight),  # type: ignore[arg-type]
                Scheduler(),
                torch.device("cpu"),
                config,
                epoch=1,
                global_step=0,
            )


if __name__ == "__main__":
    unittest.main()
