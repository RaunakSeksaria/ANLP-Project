"""Thin pretrained LED baseline model helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import LEDConfig, LEDForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput


def load_led_baseline(
    model_name_or_path: str | Path,
) -> LEDForConditionalGeneration:
    """Load pretrained LED with a conservative local-attention window."""
    config = LEDConfig.from_pretrained(str(model_name_or_path))
    config.attention_window = [512] * config.encoder_layers
    model = LEDForConditionalGeneration.from_pretrained(
        str(model_name_or_path), config=config, use_safetensors=True
    )
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Every pretrained LED parameter must remain trainable")
    return model


def validate_forward_output(
    output: Any,
    *,
    batch_size: int | None = None,
    target_length: int | None = None,
) -> None:
    """Validate loss and logits returned by a sequence-to-sequence forward pass."""
    loss = getattr(output, "loss", None)
    if loss is None:
        raise ValueError("LED forward output is missing loss")
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ValueError("LED forward loss must be a scalar tensor")
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("LED forward loss is not finite")
    logits = getattr(output, "logits", None)
    if logits is None or not isinstance(logits, torch.Tensor):
        raise ValueError("LED forward output is missing logits")
    if logits.ndim != 3:
        raise ValueError("LED logits must have shape [batch_size, target_length, vocabulary_size]")
    if batch_size is not None and logits.shape[0] != batch_size:
        raise ValueError(f"LED logits batch dimension {logits.shape[0]} != {batch_size}")
    if target_length is not None and logits.shape[1] != target_length:
        raise ValueError(f"LED logits target dimension {logits.shape[1]} != {target_length}")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("LED logits contain NaN or infinity")


def forward_led_baseline(
    model: LEDForConditionalGeneration,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    global_attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> Seq2SeqLMOutput:
    """Forward the four baseline tensors through LED and validate its output."""
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        global_attention_mask=global_attention_mask,
        labels=labels,
    )
    validate_forward_output(
        output,
        batch_size=input_ids.shape[0],
        target_length=labels.shape[1],
    )
    return output
