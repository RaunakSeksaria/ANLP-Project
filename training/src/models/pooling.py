"""Pooling operations shared by representation-learning experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def align_pooling_mask(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Align an input attention mask with LED's possibly internally padded output."""
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, length, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, length]")
    if hidden_states.shape[0] != attention_mask.shape[0]:
        raise ValueError("hidden-state and attention-mask batch dimensions differ")
    hidden_length = hidden_states.shape[1]
    mask_length = attention_mask.shape[1]
    if hidden_length < mask_length:
        raise ValueError(
            f"hidden-state length {hidden_length} is shorter than mask length {mask_length}"
        )
    mask = attention_mask
    if hidden_length > mask_length:
        extra = hidden_length - mask_length
        mask = F.pad(mask, (0, extra), value=0)
    return mask


def masked_mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L2-normalized and raw float32 masked-mean article vectors."""
    mask = align_pooling_mask(hidden_states, attention_mask).to(
        device=hidden_states.device, dtype=torch.float32
    )
    counts = mask.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        bad_rows = (counts.squeeze(-1) == 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"cannot pool all-padding inputs at batch rows {bad_rows}")
    raw = (hidden_states.float() * mask.unsqueeze(-1)).sum(dim=1) / counts
    normalized = F.normalize(raw, p=2, dim=-1, eps=1e-12)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("masked-mean pooling produced non-finite vectors")
    return normalized, raw
