"""Authoritative baseline joins and deterministic triplet resolution for LED."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset

from src.data.baseline_dataset import LEDBaselineDataset, TokenizerLike


TRIPLET_FIELDS = frozenset(
    {
        "triplet_id",
        "split",
        "anchor_event_id",
        "anchor_article_id",
        "anchor_ideology",
        "anchor_title",
        "anchor_text",
        "positive_event_id",
        "positive_article_id",
        "positive_ideology",
        "positive_title",
        "positive_text",
        "negative_event_id",
        "negative_article_id",
        "negative_ideology",
        "negative_title",
        "negative_text",
        "target_article_id",
        "target_title",
        "target_text",
    }
)
TENSOR_FIELDS = (
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


@dataclass(frozen=True)
class ArticleRef:
    event_id: str
    article_id: str
    ideology: str
    split: str
    title: str
    text: str
    event_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedTriplet:
    triplet_id: str
    split: str
    anchor: ArticleRef
    positive: ArticleRef
    negative: ArticleRef | None
    target_article_id: str
    target_title: str
    target_text: str
    original_negative_article_id: str
    negative_replaced: bool
    triplet_valid: bool
    invalid_reasons: tuple[str, ...]
    supplied_negative_reasons: tuple[str, ...]
    target_disagreement_fields: tuple[str, ...]

    def assignment_record(self) -> dict[str, Any]:
        return {
            "triplet_id": self.triplet_id,
            "split": self.split,
            "anchor_event_id": self.anchor.event_id,
            "anchor_article_id": self.anchor.article_id,
            "anchor_ideology": self.anchor.ideology,
            "positive_event_id": self.positive.event_id,
            "positive_article_id": self.positive.article_id,
            "positive_ideology": self.positive.ideology,
            "original_negative_article_id": self.original_negative_article_id,
            "resolved_negative_event_id": self.negative.event_id if self.negative else None,
            "resolved_negative_article_id": self.negative.article_id if self.negative else None,
            "resolved_negative_ideology": self.negative.ideology if self.negative else None,
            "negative_replaced": self.negative_replaced,
            "triplet_valid": self.triplet_valid,
            "invalid_reasons": list(self.invalid_reasons),
            "supplied_negative_reasons": list(self.supplied_negative_reasons),
            "target_article_id": self.target_article_id,
            "target_disagreement_fields": list(self.target_disagreement_fields),
        }


class TripletExample(TypedDict):
    anchor_input_ids: list[int]
    anchor_attention_mask: list[int]
    anchor_global_attention_mask: list[int]
    positive_input_ids: list[int]
    positive_attention_mask: list[int]
    positive_global_attention_mask: list[int]
    negative_input_ids: list[int]
    negative_attention_mask: list[int]
    negative_global_attention_mask: list[int]
    labels: list[int]
    triplet_valid_mask: bool
    triplet_id: str
    example_id: str
    anchor_article_id: str
    positive_article_id: str
    negative_article_id: str
    anchor_event_id: str
    negative_event_id: str


class TripletBatch(TypedDict):
    anchor_input_ids: torch.Tensor
    anchor_attention_mask: torch.Tensor
    anchor_global_attention_mask: torch.Tensor
    positive_input_ids: torch.Tensor
    positive_attention_mask: torch.Tensor
    positive_global_attention_mask: torch.Tensor
    negative_input_ids: torch.Tensor
    negative_attention_mask: torch.Tensor
    negative_global_attention_mask: torch.Tensor
    labels: torch.Tensor
    triplet_valid_mask: torch.Tensor
    triplet_id: list[str]
    example_id: list[str]
    anchor_article_id: list[str]
    positive_article_id: list[str]
    negative_article_id: list[str]
    anchor_event_id: list[str]
    negative_event_id: list[str]


def stable_id(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value.split(":", 1)[0]))
    except ValueError:
        return (1, value)


def format_article(article: ArticleRef, tokenizer: TokenizerLike) -> str:
    title = article.title.strip()
    body = article.text.strip()
    return body if not title else f"{title} {tokenizer.sep_token} {body}"


class LEDTripletDataset(Dataset[TripletExample]):
    """Triplets joined to authoritative baseline records and resolved once."""

    def __init__(
        self,
        baseline_path: str | Path,
        triplets_path: str | Path,
        tokenizer: TokenizerLike,
        max_input_length: int = 256,
        max_target_length: int = 256,
        seed: int = 42,
    ) -> None:
        self.baseline_path = Path(baseline_path)
        self.triplets_path = Path(triplets_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.seed = seed
        self.baseline = LEDBaselineDataset(
            self.baseline_path,
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )
        self.split = self.baseline_path.parent.name
        if self.triplets_path.parent.name != self.split:
            raise ValueError("baseline and triplet paths must belong to the same split")
        self._articles, self._targets = self._index_baselines()
        self._encoding_cache: dict[str, tuple[int, ...]] = {}
        self._records = self._load_and_resolve()
        self.resolution_statistics = self._resolution_statistics()

    def _index_baselines(
        self,
    ) -> tuple[dict[str, ArticleRef], dict[tuple[str, str], Any]]:
        articles: dict[str, ArticleRef] = {}
        targets: dict[tuple[str, str], Any] = {}
        event_roles: set[tuple[str, str]] = set()
        for index in range(len(self.baseline)):
            record = self.baseline.record(index)
            if record.input_article_id in articles:
                raise ValueError(
                    f"{self.baseline_path}: ambiguous input article ID "
                    f"{record.input_article_id!r}"
                )
            role_key = (record.event_id, record.input_ideology)
            if role_key in event_roles and not record.event_ids:
                raise ValueError(f"{self.baseline_path}: ambiguous event/ideology {role_key}")
            event_roles.add(role_key)
            articles[record.input_article_id] = ArticleRef(
                event_id=record.event_id,
                article_id=record.input_article_id,
                ideology=record.input_ideology,
                split=record.split,
                title=record.input_title.strip(),
                text=record.input_text.strip(),
                event_ids=record.event_ids,
            )
            targets[(record.event_id, record.input_article_id)] = record
        return articles, targets

    def _token_ids(self, article: ArticleRef) -> tuple[int, ...]:
        cached = self._encoding_cache.get(article.article_id)
        if cached is None:
            encoded = self.tokenizer(
                format_article(article, self.tokenizer),
                truncation=True,
                max_length=self.max_input_length,
                padding=False,
            )
            cached = tuple(int(value) for value in encoded["input_ids"])
            if not cached:
                raise ValueError(f"tokenizer returned no tokens for {article.article_id}")
            self._encoding_cache[article.article_id] = cached
        return cached

    def _require_join(
        self,
        raw: Mapping[str, Any],
        prefix: str,
        line_number: int,
    ) -> ArticleRef:
        article_id = raw.get(f"{prefix}_article_id")
        article = self._articles.get(article_id)
        location = f"{self.triplets_path}:{line_number}"
        if article is None:
            raise ValueError(f"{location}: {prefix} article {article_id!r} is unresolved")
        expected = {
            f"{prefix}_event_id": article.event_id,
            f"{prefix}_article_id": article.article_id,
            f"{prefix}_ideology": article.ideology,
            f"{prefix}_title": article.title,
            f"{prefix}_text": article.text,
        }
        disagreements = [
            name
            for name, value in expected.items()
            if not isinstance(raw.get(name), str) or str(raw.get(name)).strip() != value
        ]
        if raw.get("split") != article.split:
            disagreements.append("split")
        if disagreements:
            raise ValueError(
                f"{location}: {prefix} disagrees with baseline lookup in: "
                f"{', '.join(disagreements)}"
            )
        return article

    def _negative_reasons(
        self,
        candidate: ArticleRef | None,
        anchor: ArticleRef,
        positive: ArticleRef,
        raw: Mapping[str, Any] | None = None,
    ) -> list[str]:
        if candidate is None:
            return ["unresolved_negative_article"]
        reasons: list[str] = []
        if candidate.split != anchor.split:
            reasons.append("different_split")
        if candidate.ideology != anchor.ideology:
            reasons.append("different_ideology")
        if candidate.event_id == anchor.event_id:
            reasons.append("same_event")
        if candidate.text.strip() == anchor.text.strip():
            reasons.append("body_equals_anchor")
        if candidate.text.strip() == positive.text.strip():
            reasons.append("body_equals_positive")
        candidate_tokens = self._token_ids(candidate)
        if candidate_tokens == self._token_ids(anchor):
            reasons.append("tokens_equal_anchor")
        if candidate_tokens == self._token_ids(positive):
            reasons.append("tokens_equal_positive")
        if raw is not None:
            comparisons = {
                "negative_event_id": candidate.event_id,
                "negative_ideology": candidate.ideology,
                "negative_title": candidate.title,
                "negative_text": candidate.text,
            }
            if any(
                not isinstance(raw.get(name), str)
                or str(raw.get(name)).strip() != value
                for name, value in comparisons.items()
            ):
                reasons.append("negative_metadata_disagrees_with_baseline")
        return reasons

    def _positive_reasons(self, anchor: ArticleRef, positive: ArticleRef) -> list[str]:
        reasons: list[str] = []
        if anchor.event_id != positive.event_id:
            reasons.append("anchor_positive_different_event")
        if anchor.event_ids and not set(anchor.event_ids).intersection(positive.event_ids):
            reasons.append("anchor_positive_no_shared_original_event")
        if anchor.ideology == positive.ideology:
            reasons.append("anchor_positive_same_ideology")
        if anchor.text.strip() == positive.text.strip():
            reasons.append("identical_anchor_positive_body")
        if self._token_ids(anchor) == self._token_ids(positive):
            reasons.append("identical_anchor_positive_tokens")
        return reasons

    def _replacement(
        self,
        anchor: ArticleRef,
        positive: ArticleRef,
        rng: random.Random,
    ) -> ArticleRef | None:
        candidates = sorted(
            (
                article
                for article in self._articles.values()
                if article.ideology == anchor.ideology
                and article.event_id != anchor.event_id
            ),
            key=lambda article: stable_id(article.article_id),
        )
        valid = [
            article
            for article in candidates
            if not self._negative_reasons(article, anchor, positive)
        ]
        return rng.choice(valid) if valid else None

    def _load_and_resolve(self) -> list[ResolvedTriplet]:
        if not self.triplets_path.is_file():
            raise FileNotFoundError(f"Triplet JSONL file not found: {self.triplets_path}")
        resolved: list[ResolvedTriplet] = []
        seen_ids: set[str] = set()
        rng = random.Random(self.seed)
        with self.triplets_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                location = f"{self.triplets_path}:{line_number}"
                if not line.strip():
                    raise ValueError(f"{location}: blank lines are not allowed")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{location}: malformed JSON: {exc.msg}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"{location}: record must be a JSON object")
                missing = sorted(TRIPLET_FIELDS - raw.keys())
                if missing:
                    raise ValueError(f"{location}: missing fields: {', '.join(missing)}")
                triplet_id = raw.get("triplet_id")
                if not isinstance(triplet_id, str) or not triplet_id.strip():
                    raise ValueError(f"{location}: triplet_id must be a non-empty string")
                if triplet_id in seen_ids:
                    raise ValueError(f"{location}: duplicate triplet_id {triplet_id!r}")
                seen_ids.add(triplet_id)
                anchor = self._require_join(raw, "anchor", line_number)
                positive = self._require_join(raw, "positive", line_number)
                authoritative = self._targets.get((anchor.event_id, anchor.article_id))
                if authoritative is None:
                    raise ValueError(f"{location}: anchor has no authoritative target")
                target_fields = {
                    "target_article_id": authoritative.target_article_id,
                    "target_title": authoritative.target_title,
                    "target_text": authoritative.target_text,
                }
                target_disagreements = tuple(
                    name
                    for name, value in target_fields.items()
                    if not isinstance(raw.get(name), str)
                    or str(raw.get(name)).strip() != value.strip()
                )
                supplied = self._articles.get(raw.get("negative_article_id"))
                supplied_reasons = tuple(
                    self._negative_reasons(supplied, anchor, positive, raw)
                )
                negative = supplied
                replaced = False
                if supplied_reasons:
                    negative = self._replacement(anchor, positive, rng)
                    replaced = True
                positive_reasons = self._positive_reasons(anchor, positive)
                invalid_reasons = list(positive_reasons)
                if negative is None:
                    invalid_reasons.append("no_valid_negative")
                elif self._negative_reasons(negative, anchor, positive):
                    raise AssertionError("replacement negative failed validation")
                resolved.append(
                    ResolvedTriplet(
                        triplet_id=triplet_id.strip(),
                        split=self.split,
                        anchor=anchor,
                        positive=positive,
                        negative=negative,
                        target_article_id=authoritative.target_article_id,
                        target_title=authoritative.target_title.strip(),
                        target_text=authoritative.target_text.strip(),
                        original_negative_article_id=str(raw.get("negative_article_id", "")),
                        negative_replaced=replaced,
                        triplet_valid=not invalid_reasons,
                        invalid_reasons=tuple(invalid_reasons),
                        supplied_negative_reasons=supplied_reasons,
                        target_disagreement_fields=target_disagreements,
                    )
                )
        baseline_keys = {
            (self.baseline.record(index).event_id, self.baseline.record(index).input_article_id)
            for index in range(len(self.baseline))
        }
        triplet_keys = {(record.anchor.event_id, record.anchor.article_id) for record in resolved}
        if baseline_keys != triplet_keys:
            missing = len(baseline_keys - triplet_keys)
            extra = len(triplet_keys - baseline_keys)
            raise ValueError(
                f"{self.triplets_path}: anchors do not match baseline examples "
                f"(missing={missing}, extra={extra})"
            )
        return resolved

    def _resolution_statistics(self) -> dict[str, Any]:
        reasons = Counter(
            reason for record in self._records for reason in record.invalid_reasons
        )
        supplied = Counter(
            reason
            for record in self._records
            for reason in record.supplied_negative_reasons
        )
        return {
            "anchors": len(self._records),
            "original_valid_negatives": sum(
                not record.supplied_negative_reasons for record in self._records
            ),
            "negative_replacements": sum(record.negative_replaced for record in self._records),
            "identical_positive_pairs": sum(
                any(reason.startswith("identical_anchor_positive") for reason in record.invalid_reasons)
                for record in self._records
            ),
            "triplet_invalid_anchors": sum(not record.triplet_valid for record in self._records),
            "target_disagreements": sum(bool(record.target_disagreement_fields) for record in self._records),
            "invalid_reason_counts": dict(sorted(reasons.items())),
            "supplied_negative_reason_counts": dict(sorted(supplied.items())),
        }

    def __len__(self) -> int:
        return len(self._records)

    def record(self, index: int) -> ResolvedTriplet:
        return self._records[index]

    def assignment_records(self) -> list[dict[str, Any]]:
        return [record.assignment_record() for record in self._records]

    def _encode_article(self, article: ArticleRef) -> tuple[list[int], list[int], list[int]]:
        encoded = self.tokenizer(
            format_article(article, self.tokenizer),
            truncation=True,
            max_length=self.max_input_length,
            padding=False,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        mask = [int(value) for value in encoded["attention_mask"]]
        if not ids or len(ids) != len(mask):
            raise ValueError(f"invalid tokenization for article {article.article_id}")
        global_mask = [0] * len(ids)
        global_mask[0] = 1
        return ids, mask, global_mask

    def _encode_target(self, text: str) -> list[int]:
        options = {
            "truncation": True,
            "max_length": self.max_target_length,
            "padding": False,
        }
        try:
            encoded = self.tokenizer(text_target=text, **options)
        except TypeError as exc:
            if "text_target" not in str(exc):
                raise
            encoded = self.tokenizer(text, **options)
        labels = [int(value) for value in encoded["input_ids"]]
        if not labels:
            raise ValueError("tokenizer returned empty target labels")
        return labels

    def __getitem__(self, index: int) -> TripletExample:
        record = self._records[index]
        negative = record.negative or record.anchor
        anchor_ids, anchor_mask, anchor_global = self._encode_article(record.anchor)
        positive_ids, positive_mask, positive_global = self._encode_article(record.positive)
        negative_ids, negative_mask, negative_global = self._encode_article(negative)
        return {
            "anchor_input_ids": anchor_ids,
            "anchor_attention_mask": anchor_mask,
            "anchor_global_attention_mask": anchor_global,
            "positive_input_ids": positive_ids,
            "positive_attention_mask": positive_mask,
            "positive_global_attention_mask": positive_global,
            "negative_input_ids": negative_ids,
            "negative_attention_mask": negative_mask,
            "negative_global_attention_mask": negative_global,
            "labels": self._encode_target(record.target_text),
            "triplet_valid_mask": record.triplet_valid,
            "triplet_id": record.triplet_id,
            "example_id": self._targets[
                (record.anchor.event_id, record.anchor.article_id)
            ].example_id,
            "anchor_article_id": record.anchor.article_id,
            "positive_article_id": record.positive.article_id,
            "negative_article_id": negative.article_id if record.negative else "",
            "anchor_event_id": record.anchor.event_id,
            "negative_event_id": negative.event_id if record.negative else "",
        }

    def truncation_statistics(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        for role in ("anchor", "positive", "negative"):
            total = truncated = eos_retained = 0
            for record in self._records:
                article = getattr(record, role)
                if article is None:
                    continue
                text = format_article(article, self.tokenizer)
                full = self.tokenizer(text, truncation=False, padding=False)["input_ids"]
                actual = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_input_length,
                    padding=False,
                )["input_ids"]
                total += 1
                truncated += len(full) > self.max_input_length
                eos_retained += eos_id is None or (bool(actual) and actual[-1] == eos_id)
            result[role] = {
                "records": total,
                "truncated": truncated,
                "truncation_rate": truncated / total if total else 0.0,
                "eos_retained": eos_retained,
            }
        total = truncated = eos_retained = 0
        for record in self._records:
            try:
                full = self.tokenizer(
                    text_target=record.target_text, truncation=False, padding=False
                )["input_ids"]
            except TypeError:
                full = self.tokenizer(record.target_text, truncation=False, padding=False)[
                    "input_ids"
                ]
            actual = self._encode_target(record.target_text)
            total += 1
            truncated += len(full) > self.max_target_length
            eos_retained += eos_id is None or (bool(actual) and actual[-1] == eos_id)
        result["target"] = {
            "records": total,
            "truncated": truncated,
            "truncation_rate": truncated / total if total else 0.0,
            "eos_retained": eos_retained,
        }
        return result


class LEDTripletCollator:
    """Independently dynamically pad the three encoder streams and labels."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        self.pad_token_id = int(tokenizer.pad_token_id)

    @staticmethod
    def _pad(
        examples: Sequence[TripletExample],
        ids_name: str,
        mask_name: str,
        global_name: str,
        pad_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum = max(len(example[ids_name]) for example in examples)  # type: ignore[literal-required]
        ids_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        global_rows: list[list[int]] = []
        for example in examples:
            ids = list(example[ids_name])  # type: ignore[literal-required]
            mask = list(example[mask_name])  # type: ignore[literal-required]
            global_mask = list(example[global_name])  # type: ignore[literal-required]
            if not (len(ids) == len(mask) == len(global_mask)):
                raise ValueError(f"unaligned fields for {ids_name}")
            padding = maximum - len(ids)
            ids_rows.append(ids + [pad_id] * padding)
            mask_rows.append(mask + [0] * padding)
            global_rows.append(global_mask + [0] * padding)
        ids_tensor = torch.tensor(ids_rows, dtype=torch.long)
        mask_tensor = torch.tensor(mask_rows, dtype=torch.long)
        global_tensor = torch.tensor(global_rows, dtype=torch.long)
        if not bool((global_tensor.sum(dim=1) == 1).all()) or not bool(
            (global_tensor[:, 0] == 1).all()
        ):
            raise ValueError("every article must have global attention only at position zero")
        if not bool((global_tensor[mask_tensor == 0] == 0).all()):
            raise ValueError("global attention is nonzero on padding")
        return ids_tensor, mask_tensor, global_tensor

    def __call__(self, examples: Sequence[TripletExample]) -> TripletBatch:
        if not examples:
            raise ValueError("cannot collate an empty triplet batch")
        batch: dict[str, Any] = {}
        for role in ("anchor", "positive", "negative"):
            ids, mask, global_mask = self._pad(
                examples,
                f"{role}_input_ids",
                f"{role}_attention_mask",
                f"{role}_global_attention_mask",
                self.pad_token_id,
            )
            batch[f"{role}_input_ids"] = ids
            batch[f"{role}_attention_mask"] = mask
            batch[f"{role}_global_attention_mask"] = global_mask
        max_target = max(len(example["labels"]) for example in examples)
        batch["labels"] = torch.tensor(
            [
                example["labels"] + [-100] * (max_target - len(example["labels"]))
                for example in examples
            ],
            dtype=torch.long,
        )
        batch["triplet_valid_mask"] = torch.tensor(
            [example["triplet_valid_mask"] for example in examples], dtype=torch.bool
        )
        for name in (
            "triplet_id",
            "example_id",
            "anchor_article_id",
            "positive_article_id",
            "negative_article_id",
            "anchor_event_id",
            "negative_event_id",
        ):
            batch[name] = [str(example[name]) for example in examples]  # type: ignore[literal-required]
        return batch  # type: ignore[return-value]


@dataclass(frozen=True)
class TripletDataBundle:
    datasets: dict[str, LEDTripletDataset]
    dataloaders: dict[str, DataLoader[TripletBatch]]
    train_generator: torch.Generator


def create_triplet_dataloaders(
    data_dir: str | Path,
    tokenizer: TokenizerLike,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    max_input_length: int = 256,
    max_target_length: int = 256,
    seed: int = 42,
    num_workers: int = 0,
) -> TripletDataBundle:
    root = Path(data_dir)
    datasets = {
        split: LEDTripletDataset(
            root / split / "baseline.jsonl",
            root / split / "triplets.jsonl",
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
            seed=seed,
        )
        for split in ("train", "validation", "test")
    }
    collator = LEDTripletCollator(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=train_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=num_workers,
            drop_last=False,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            drop_last=False,
        ),
    }
    return TripletDataBundle(
        datasets=datasets,
        dataloaders=dataloaders,
        train_generator=generator,
    )
