from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import LEDConfig, LEDForConditionalGeneration

from src.models.led_triplet import (
    JointForwardOutput,
    LEDTripletModel,
    load_led_triplet,
    masked_positive_attraction_loss,
    masked_triplet_margin_loss,
)
from src.scripts.train_joint import config_from_args, parse_args
from src.training.train_joint import (
    JointEarlyStopping,
    JointTrainingConfig,
    save_joint_checkpoint,
    train_joint_epoch,
    validate_joint_config,
)


def tiny_model() -> LEDTripletModel:
    config = LEDConfig(
        vocab_size=64, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32, attention_window=[8],
        max_encoder_position_embeddings=64, max_decoder_position_embeddings=32,
        bos_token_id=0, pad_token_id=1, eos_token_id=2,
        decoder_start_token_id=2, dropout=0.0, attention_dropout=0.0,
    )
    return LEDTripletModel(LEDForConditionalGeneration(config))


def batch() -> dict[str, torch.Tensor]:
    result = {
        "labels": torch.tensor([[0, 5, 2], [0, 6, 2]]),
        "triplet_valid_mask": torch.tensor([True, True]),
    }
    for role, offset in (("anchor", 0), ("positive", 1), ("negative", 2)):
        result[f"{role}_input_ids"] = torch.tensor([[0, 4 + offset, 2], [0, 7 + offset, 2]])
        result[f"{role}_attention_mask"] = torch.ones(2, 3, dtype=torch.long)
        result[f"{role}_global_attention_mask"] = torch.tensor([[1, 0, 0], [1, 0, 0]])
    return result


class PositiveAttractionTests(unittest.TestCase):
    def test_hand_calculated_active_and_inactive_cases(self) -> None:
        anchor = torch.tensor([[1., 0.], [1., 0.]])
        positive = torch.tensor([[0., 1.], [1., 0.]])
        negative = torch.tensor([[0.8, 0.6], [-1., 0.]])
        mask = torch.tensor([True, True])
        ranking, _, _, d_ap, d_an = masked_triplet_margin_loss(anchor, positive, negative, mask, 0.2)
        attraction, per_positive = masked_positive_attraction_loss(d_ap, mask, 0.5)
        self.assertAlmostEqual(float(ranking), (math.sqrt(2) - math.sqrt(0.4) + 0.2) / 2, places=4)
        self.assertAlmostEqual(float(attraction), (math.sqrt(2) - 0.5) / 2, places=4)
        self.assertAlmostEqual(float(per_positive[1]), 0.0, places=4)
        self.assertGreater(float(d_an[1]), float(d_ap[1]) + 0.2)

    def test_attraction_when_ranking_satisfied_and_no_negative_gradient(self) -> None:
        anchor = torch.tensor([[1., 0.]], requires_grad=True)
        positive = torch.tensor([[0.6, 0.8]], requires_grad=True)
        negative = torch.tensor([[-1., 0.]], requires_grad=True)
        mask = torch.tensor([True])
        ranking, _, _, d_ap, _ = masked_triplet_margin_loss(anchor, positive, negative, mask, 0.2)
        attraction, _ = masked_positive_attraction_loss(d_ap, mask, 0.5)
        self.assertEqual(float(ranking), 0.0)
        self.assertGreater(float(attraction), 0.0)
        (ranking + 0.3 * attraction).backward()
        self.assertGreater(float(anchor.grad.norm()), 0.0)
        self.assertGreater(float(positive.grad.norm()), 0.0)
        self.assertEqual(float(negative.grad.norm()), 0.0)

    def test_invalid_and_all_invalid_denominators(self) -> None:
        distances = torch.tensor([0.9, 1.2], requires_grad=True)
        loss, per = masked_positive_attraction_loss(distances, torch.tensor([True, False]), 0.5)
        self.assertAlmostEqual(float(loss), 0.2)
        self.assertEqual(per.tolist(), [per[0].item(), 0.0])
        loss.backward()
        self.assertEqual(distances.grad.tolist(), [0.5, 0.0])
        zero, _ = masked_positive_attraction_loss(distances, torch.tensor([False, False]), 0.5)
        self.assertEqual(float(zero), 0.0)

    def test_beta_zero_matches_old_objective_and_gradients(self) -> None:
        model = tiny_model().eval()
        inputs = batch()
        old = model(**inputs, triplet_lambda=0.35, triplet_margin=0.2, triplet_objective="standard")
        old.total_loss.backward()
        encoder_parameter = next(p for p in model.led.get_encoder().parameters() if p.grad is not None)
        old_grad = encoder_parameter.grad.clone()
        model.zero_grad(set_to_none=True)
        new = model(**inputs, triplet_lambda=0.35, triplet_margin=0.2,
                    triplet_objective="attraction", positive_beta=0.0)
        new.total_loss.backward()
        new_grad = encoder_parameter.grad
        torch.testing.assert_close(new.total_loss, old.total_loss)
        torch.testing.assert_close(new.triplet_loss, old.triplet_loss)
        torch.testing.assert_close(new_grad, old_grad)

    def test_combined_forward_backward_and_lambda_zero(self) -> None:
        model = tiny_model().eval()
        inputs = batch()
        output = model(**inputs, triplet_lambda=0.1, triplet_margin=0.2,
                       triplet_objective="attraction", positive_beta=0.3,
                       positive_distance_threshold=0.5)
        torch.testing.assert_close(output.triplet_loss, output.ranking_loss + output.weighted_positive_loss)
        torch.testing.assert_close(output.total_loss, output.generation_loss + 0.1 * output.triplet_loss)
        output.total_loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        zero = model(**inputs, triplet_lambda=0.0, triplet_margin=0.2,
                     triplet_objective="attraction")
        self.assertIsNone(zero.positive_embedding)
        self.assertIsNone(zero.negative_embedding)
        torch.testing.assert_close(zero.total_loss, zero.generation_loss)

    def test_cli_validation_and_checkpoint_round_trip(self) -> None:
        args = parse_args(["--data-dir", "data", "--output-dir", "output",
                           "--triplet-objective", "attraction",
                           "--positive-distance-threshold", "0.5", "--positive-beta", "0.3"])
        config = config_from_args(args)
        validate_joint_config(config)
        self.assertEqual((config.triplet_objective, config.positive_distance_threshold, config.positive_beta),
                         ("attraction", 0.5, 0.3))
        with self.assertRaisesRegex(ValueError, "positive beta"):
            validate_joint_config(JointTrainingConfig(Path("data"), Path("output"), positive_beta=-0.1))
        with self.assertRaisesRegex(ValueError, "require-wandb"):
            validate_joint_config(JointTrainingConfig(Path("data"), Path("output"),
                                                     wandb_enabled=False, wandb_required=True))
        with tempfile.TemporaryDirectory() as directory:
            model = tiny_model()
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            class Tokenizer:
                def save_pretrained(self, path):
                    Path(path, "tokenizer_config.json").write_text("{}")
            save_joint_checkpoint(Path(directory), model, Tokenizer(), optimizer, scheduler, config,
                                  JointEarlyStopping(3, 0.0), torch.Generator(), epoch=2, global_step=7)
            state = json.loads(Path(directory, "joint_state.json").read_text())
            self.assertEqual((state["triplet_objective"], state["positive_distance_threshold"],
                              state["positive_beta"], state["global_optimizer_step"]),
                             ("attraction", 0.5, 0.3, 7))
            reloaded = load_led_triplet(directory, attention_window=8)
            self.assertEqual(reloaded.config.d_model, model.config.d_model)

    def test_epoch_logged_components_reconstruct_objective(self) -> None:
        class SyntheticModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = type("Config", (), {"use_cache": False})()

            def forward(self, **kwargs):
                generation = self.weight * kwargs["anchor_input_ids"][0, 0].float()
                ranking = 2 * generation
                positive = 3 * generation
                weighted = 0.3 * positive
                auxiliary = ranking + weighted
                return JointForwardOutput(
                    generation_loss=generation, triplet_loss=auxiliary,
                    valid_triplet_mean=auxiliary,
                    total_loss=generation + kwargs["triplet_lambda"] * auxiliary,
                    generation_logits=torch.tensor([[[0., 0., 1.]]]),
                    anchor_embedding=None, positive_embedding=None, negative_embedding=None,
                    anchor_raw_embedding=None, positive_raw_embedding=None, negative_raw_embedding=None,
                    anchor_positive_distance=torch.tensor([0.9]),
                    anchor_negative_distance=torch.tensor([1.2]),
                    per_example_triplet_loss=torch.tensor([1.0]), valid_count=1,
                    ranking_loss=ranking, positive_loss=positive,
                    weighted_positive_loss=weighted,
                    per_example_positive_loss=torch.tensor([0.4]),
                )

        class Scheduler:
            def step(self):
                pass

        class Logger:
            def __init__(self):
                self.logged = []

            def log(self, values, step=None):
                self.logged.append((values, step))

        def example(marker):
            values = {"labels": torch.tensor([[2]]), "triplet_valid_mask": torch.tensor([True]),
                      "example_id": [f"synthetic-{marker}"]}
            for role in ("anchor", "positive", "negative"):
                values[f"{role}_input_ids"] = torch.tensor([[marker]])
                values[f"{role}_attention_mask"] = torch.tensor([[1]])
                values[f"{role}_global_attention_mask"] = torch.tensor([[1]])
            return values

        model = SyntheticModel()
        logger = Logger()
        config = JointTrainingConfig(Path("data"), Path("output"), triplet_lambda=0.1,
                                     triplet_objective="attraction", gradient_accumulation_steps=2)
        result = train_joint_epoch(model, [example(1), example(2)],
                                   torch.optim.SGD(model.parameters(), lr=0.0), Scheduler(),
                                   torch.device("cpu"), config, epoch=1, global_step=0,
                                   wandb_run=logger)
        self.assertAlmostEqual(result.triplet_loss, result.ranking_loss + result.weighted_positive_loss, places=6)
        self.assertAlmostEqual(result.weighted_positive_loss, 0.3 * result.positive_loss, places=6)
        self.assertAlmostEqual(result.total_loss, result.generation_loss + result.weighted_triplet_loss, places=6)
        self.assertEqual(result.attraction_active_fraction, 1.0)
        self.assertAlmostEqual(result.mean_anchor_positive_distance, 0.9)
        self.assertEqual(logger.logged[0][1], 1)
        metrics = logger.logged[0][0]
        self.assertAlmostEqual(metrics["train/step_auxiliary_loss"],
                               metrics["train/step_ranking_loss"] + metrics["train/step_weighted_positive_loss"], places=6)


if __name__ == "__main__":
    unittest.main()
