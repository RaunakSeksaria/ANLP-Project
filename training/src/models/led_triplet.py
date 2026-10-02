"""One shared LED model for anchor generation and triplet supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import LEDConfig, LEDForConditionalGeneration

from src.models.pooling import masked_mean_pool


@dataclass
class JointForwardOutput:
    generation_loss: torch.Tensor
    triplet_loss: torch.Tensor
    valid_triplet_mean: torch.Tensor
    total_loss: torch.Tensor
    generation_logits: torch.Tensor
    anchor_embedding: torch.Tensor | None
    positive_embedding: torch.Tensor | None
    negative_embedding: torch.Tensor | None
    anchor_raw_embedding: torch.Tensor | None
    positive_raw_embedding: torch.Tensor | None
    negative_raw_embedding: torch.Tensor | None
    anchor_positive_distance: torch.Tensor
    anchor_negative_distance: torch.Tensor
    per_example_triplet_loss: torch.Tensor
    valid_count: int


def masked_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the requested batch-denominated and valid-only triplet losses."""
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError("anchor, positive, and negative embeddings must have equal shapes")
    if anchor.ndim != 2 or valid_mask.shape != (anchor.shape[0],):
        raise ValueError("invalid embedding or triplet-valid-mask shape")
    criterion = nn.TripletMarginLoss(
        margin=margin, p=2, eps=1e-6, swap=False, reduction="none"
    )
    per_example = criterion(anchor, positive, negative)
    valid = valid_mask.to(device=per_example.device, dtype=torch.bool)
    masked = per_example * valid.to(per_example.dtype)
    batch_mean = masked.sum() / anchor.shape[0]
    valid_count = int(valid.sum().item())
    valid_mean = (
        masked.sum() / valid_count if valid_count else per_example.new_zeros(())
    )
    d_ap = torch.linalg.vector_norm(anchor - positive, ord=2, dim=-1)
    d_an = torch.linalg.vector_norm(anchor - negative, ord=2, dim=-1)
    return batch_mean, valid_mean, per_example, d_ap, d_an


class LEDTripletModel(nn.Module):
    """Wrap one LED instance without adding a projection or a second encoder."""

    def __init__(self, led: LEDForConditionalGeneration) -> None:
        super().__init__()
        self.led = led

    @property
    def config(self) -> LEDConfig:
        return self.led.config

    def save_pretrained(self, path: str | Path) -> None:
        self.led.save_pretrained(path, safe_serialization=True)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.led.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        normalized, raw = masked_mean_pool(encoded.last_hidden_state, attention_mask)
        if normalized.shape[-1] != self.config.d_model:
            raise ValueError(
                f"pooled width {normalized.shape[-1]} != model d_model {self.config.d_model}"
            )
        return normalized, raw

    def forward(
        self,
        *,
        anchor_input_ids: torch.Tensor,
        anchor_attention_mask: torch.Tensor,
        anchor_global_attention_mask: torch.Tensor,
        labels: torch.Tensor,
        triplet_valid_mask: torch.Tensor,
        triplet_lambda: float,
        triplet_margin: float,
        positive_input_ids: torch.Tensor | None = None,
        positive_attention_mask: torch.Tensor | None = None,
        positive_global_attention_mask: torch.Tensor | None = None,
        negative_input_ids: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
        negative_global_attention_mask: torch.Tensor | None = None,
    ) -> JointForwardOutput:
        generation = self.led(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            global_attention_mask=anchor_global_attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )
        if generation.loss is None or generation.encoder_last_hidden_state is None:
            raise ValueError("LED generation forward omitted loss or encoder states")
        generation_loss = generation.loss
        batch_size = anchor_input_ids.shape[0]
        zeros = generation_loss.new_zeros((batch_size,))
        scalar_zero = generation_loss.new_zeros(())
        valid_count = int(triplet_valid_mask.to(torch.bool).sum().item())
        if triplet_lambda == 0.0 or valid_count == 0:
            return JointForwardOutput(
                generation_loss=generation_loss,
                triplet_loss=scalar_zero,
                valid_triplet_mean=scalar_zero,
                total_loss=generation_loss,
                generation_logits=generation.logits,
                anchor_embedding=None,
                positive_embedding=None,
                negative_embedding=None,
                anchor_raw_embedding=None,
                positive_raw_embedding=None,
                negative_raw_embedding=None,
                anchor_positive_distance=zeros,
                anchor_negative_distance=zeros,
                per_example_triplet_loss=zeros,
                valid_count=valid_count,
            )

        required = (
            positive_input_ids,
            positive_attention_mask,
            positive_global_attention_mask,
            negative_input_ids,
            negative_attention_mask,
            negative_global_attention_mask,
        )
        if any(value is None for value in required):
            raise ValueError("positive and negative tensors are required when triplet loss is active")

        anchor_z, anchor_raw = masked_mean_pool(
            generation.encoder_last_hidden_state, anchor_attention_mask
        )
        if anchor_z.shape[-1] != self.config.d_model:
            raise ValueError(
                f"pooled width {anchor_z.shape[-1]} != model d_model {self.config.d_model}"
            )
        positive_z, positive_raw = self._encode(
            positive_input_ids,  # type: ignore[arg-type]
            positive_attention_mask,  # type: ignore[arg-type]
            positive_global_attention_mask,  # type: ignore[arg-type]
        )
        negative_z, negative_raw = self._encode(
            negative_input_ids,  # type: ignore[arg-type]
            negative_attention_mask,  # type: ignore[arg-type]
            negative_global_attention_mask,  # type: ignore[arg-type]
        )
        triplet_loss, valid_mean, per_example, d_ap, d_an = masked_triplet_margin_loss(
            anchor_z,
            positive_z,
            negative_z,
            triplet_valid_mask,
            triplet_margin,
        )
        total = generation_loss + triplet_lambda * triplet_loss
        return JointForwardOutput(
            generation_loss=generation_loss,
            triplet_loss=triplet_loss,
            valid_triplet_mean=valid_mean,
            total_loss=total,
            generation_logits=generation.logits,
            anchor_embedding=anchor_z,
            positive_embedding=positive_z,
            negative_embedding=negative_z,
            anchor_raw_embedding=anchor_raw,
            positive_raw_embedding=positive_raw,
            negative_raw_embedding=negative_raw,
            anchor_positive_distance=d_ap,
            anchor_negative_distance=d_an,
            per_example_triplet_loss=per_example,
            valid_count=valid_count,
        )


def load_led_triplet(
    model_name_or_path: str | Path,
    *,
    attention_window: int = 512,
) -> LEDTripletModel:
    """Configure the required attention window before constructing one LED model."""
    config = LEDConfig.from_pretrained(str(model_name_or_path))
    config.attention_window = [attention_window] * config.encoder_layers
    led = LEDForConditionalGeneration.from_pretrained(
        str(model_name_or_path), config=config, use_safetensors=True
    )
    return LEDTripletModel(led)
