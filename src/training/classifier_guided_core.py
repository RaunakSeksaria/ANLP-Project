"""Source-only data, frozen reward model, and self-critical LED objective."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.triplet_dataset import ArticleRef, format_article
from src.models.led_triplet import LEDTripletModel, masked_positive_attraction_loss, masked_triplet_margin_loss


LABELS = ("Left", "Centre", "Right")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceRow:
    article_id: str
    event_id: str
    ideology: str
    title: str
    body: str
    positive: ArticleRef | None
    negative: ArticleRef | None
    valid_triplet: bool


class SourceTripletDataset(Dataset[dict[str, Any]]):
    """Read only source columns from the prepared triplets, never target columns."""

    def __init__(self, path: Path, tokenizer: Any, max_input_length: int = 512) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.rows: list[SourceRow] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                raw = json.loads(line)
                # The prepared file has neutral target columns, but none are
                # accessed or retained by this source-only projection.
                def article(prefix: str) -> ArticleRef:
                    return ArticleRef(
                        event_id=str(raw[f"{prefix}_event_id"]),
                        article_id=str(raw[f"{prefix}_article_id"]),
                        ideology=str(raw[f"{prefix}_ideology"]),
                        split=path.parent.name,
                        title=str(raw[f"{prefix}_title"]),
                        text=str(raw[f"{prefix}_text"]),
                    )

                anchor = article("anchor")
                positive = article("positive")
                negative = article("negative")
                if anchor.ideology not in {"left", "right"} or not anchor.text.strip():
                    raise ValueError(f"{path}:{line_number}: invalid anchor")
                valid = (
                    raw.get("positive_status") == "matched_event"
                    and positive.event_id == anchor.event_id
                    and positive.ideology != anchor.ideology
                    and negative.event_id != anchor.event_id
                    and negative.ideology == anchor.ideology
                    and positive.text.strip() != anchor.text.strip()
                )
                self.rows.append(SourceRow(
                    anchor.article_id, anchor.event_id, anchor.ideology,
                    anchor.title, anchor.text, positive if valid else None,
                    negative if valid else None, valid,
                ))
        if not self.rows:
            raise ValueError(f"{path}: no source articles")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        anchor = ArticleRef(row.event_id, row.article_id, row.ideology,
                            self.path.parent.name, row.title, row.body)

        def encode(article: ArticleRef) -> list[int]:
            text = format_article(article, self.tokenizer)
            return self.tokenizer(text, truncation=True, max_length=self.max_input_length)["input_ids"]

        item = {"input_ids": encode(anchor), "article_id": row.article_id,
                "event_id": row.event_id, "ideology": row.ideology,
                "valid_triplet": row.valid_triplet}
        if row.valid_triplet:
            item["positive_input_ids"] = encode(row.positive)
            item["negative_input_ids"] = encode(row.negative)
        return item


def collate_sources(items: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if not items:
        raise ValueError("empty source batch")
    result: dict[str, Any] = {
        "article_id": [item["article_id"] for item in items],
        "event_id": [item["event_id"] for item in items],
        "ideology": [item["ideology"] for item in items],
        "valid_triplet": torch.tensor([item["valid_triplet"] for item in items], dtype=torch.bool),
    }
    for field in ("input_ids", "positive_input_ids", "negative_input_ids"):
        # Invalid triplets use the anchor as a placeholder. Their auxiliary
        # terms are masked; lambda=0 never invokes positive/negative encoders.
        sequences = [item.get(field, item["input_ids"]) for item in items]
        length = max(map(len, sequences))
        ids = torch.full((len(items), length), pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence)
            mask[i, :len(sequence)] = 1
        prefix = "anchor" if field == "input_ids" else field.removesuffix("_input_ids")
        result[f"{prefix}_input_ids"] = ids
        result[f"{prefix}_attention_mask"] = mask
        global_mask = torch.zeros_like(ids)
        global_mask[:, 0] = 1
        result[f"{prefix}_global_attention_mask"] = global_mask
    return result


class PoliticsClassifier(nn.Module):
    """Architecture documented with Kaggle's politics_media.pt checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = AutoModel.from_pretrained("launch/POLITICS")
        hidden = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(hidden, 768), nn.Tanh(),
            nn.Dropout(0.1), nn.Linear(768, 3),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls)


class FrozenPoliticsClassifier:
    """Documented Kaggle architecture and tokenizer, with frozen local weights."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing politics_media.pt at {checkpoint}; obtain the documented Kaggle classifier first"
            )
        self.model = PoliticsClassifier()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not all(isinstance(v, torch.Tensor) for v in payload.values()):
            raise ValueError("classifier checkpoint is not a tensor state dict")
        self.model.load_state_dict(payload, strict=True)
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
        self.device = device
        self.checkpoint_sha256 = file_sha256(checkpoint)
        self.label_order = LABELS
        self.scoring_policy = "body-only, truncate to 512 launch/POLITICS tokens, Centre=softmax(logits)[:,1]"

    def parameter_sha256(self) -> str:
        """Fingerprint loaded parameters for a bounded frozen-weight check."""
        digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def score(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = self.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
        truncated = torch.tensor([len(ids) > 512 for ids in lengths], dtype=torch.float32, device=self.device)
        encoded = self.tokenizer(texts, truncation=True, max_length=512,
                                 padding=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).float()
        if logits.shape != (len(texts), 3):
            raise ValueError("classifier did not return [batch, 3] logits")
        log_prob = logits.log_softmax(dim=-1)[:, 1]
        return log_prob.exp(), -log_prob, truncated


def valid_generated_tokens(
    sequences: torch.Tensor, eos_id: int, pad_id: int, decoder_start_id: int,
) -> torch.Tensor:
    """Positions predicted by decoder logits: ignore start, pad, and post-EOS."""
    if sequences.ndim != 2 or sequences.shape[1] < 2:
        raise ValueError("generated sequence must include decoder start and a prediction")
    targets = sequences[:, 1:]
    eos_count_before = torch.cumsum((targets == eos_id).int(), dim=1) - (targets == eos_id).int()
    # LED/BART may use EOS itself as decoder_start_token_id. The initial start
    # is removed by [:, 1:], while a generated EOS must still be scored once.
    return ((eos_count_before == 0) & (targets != pad_id)
            & ((targets != decoder_start_id) | (targets == eos_id)))


def policy_gradient_loss(
    logits: torch.Tensor, sequences: torch.Tensor, advantage: torch.Tensor,
    eos_id: int, pad_id: int, decoder_start_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = sequences[:, 1:]
    if logits.shape[:2] != targets.shape or advantage.shape != (targets.shape[0],):
        raise ValueError("sampled tokens and prediction logits are misaligned")
    mask = valid_generated_tokens(sequences, eos_id, pad_id, decoder_start_id)
    log_probs = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    sum_logp = (log_probs * mask).sum(dim=1)
    return -(advantage.detach() * sum_logp).mean(), mask.sum(dim=1)


class ClassifierGuidedLED(nn.Module):
    def __init__(self, led_triplet: LEDTripletModel) -> None:
        super().__init__()
        self.model = led_triplet

    def forward(self, batch: dict[str, torch.Tensor], sampled: torch.Tensor,
                advantage: torch.Tensor, triplet_lambda: float, margin: float = .2,
                beta: float = .3, tau: float = .5) -> dict[str, torch.Tensor]:
        led = self.model.led
        output = led(
            input_ids=batch["anchor_input_ids"],
            attention_mask=batch["anchor_attention_mask"],
            global_attention_mask=batch["anchor_global_attention_mask"],
            decoder_input_ids=sampled[:, :-1], use_cache=False, return_dict=True,
        )
        pg, lengths = policy_gradient_loss(
            output.logits, sampled, advantage, led.config.eos_token_id,
            led.config.pad_token_id, led.config.decoder_start_token_id,
        )
        zero = pg.new_zeros(())
        rank, attraction, d_ap, d_an = zero, zero, zero, zero
        if triplet_lambda and bool(batch["valid_triplet"].any()):
            anchor, _ = self.model._pool(
                output.encoder_last_hidden_state,
                batch["anchor_attention_mask"], batch["anchor_global_attention_mask"],
            )
            positive, _ = self.model._encode(
                batch["positive_input_ids"], batch["positive_attention_mask"],
                batch["positive_global_attention_mask"],
            )
            negative, _ = self.model._encode(
                batch["negative_input_ids"], batch["negative_attention_mask"],
                batch["negative_global_attention_mask"],
            )
            rank, _, _, per_ap, per_an = masked_triplet_margin_loss(
                anchor, positive, negative, batch["valid_triplet"], margin
            )
            attraction, _ = masked_positive_attraction_loss(per_ap, batch["valid_triplet"], tau)
            valid = batch["valid_triplet"].float()
            d_ap = (per_ap * valid).sum() / valid.sum()
            d_an = (per_an * valid).sum() / valid.sum()
        auxiliary = rank + beta * attraction
        return {"total": pg + triplet_lambda * auxiliary, "pg": pg,
                "ranking": rank, "attraction": attraction,
                "auxiliary": auxiliary, "positive_distance": d_ap,
                "negative_distance": d_an, "lengths": lengths.float().mean()}
