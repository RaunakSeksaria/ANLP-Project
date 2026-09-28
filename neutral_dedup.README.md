# left_neutral_dedup.jsonl / right_neutral_dedup.jsonl

Single-article neutral article generation data: **one news article in, the same
article rewritten without framing out.** There is one file per stance, and every
input article appears **exactly once**.

| file | records | train | val | test |
| --- | --- | --- | --- | --- |
| `left_neutral_dedup.jsonl` | 5,695 | 3,942 | 885 | 868 |
| `right_neutral_dedup.jsonl` | 6,160 | 4,360 | 893 | 907 |

Both files are complete: every unique left and right article in the 10k-event
corpus has exactly one neutralised rewrite.

## Why deduplicated

The source corpus (BigNewsAlign) is organised by *event*, and one article often
belongs to several events, up to 32 times. In the 10k-event files
(`left_neutral_10k.jsonl`) that meant 43% of records repeated an input the model
had already seen, each time with a different generated target. That's an
overfitting risk, so these files keep **one record per unique article**.

Every copy of an article was always in the same split, so deduplicating loses no
split information. `event_ids` lists all the events that shared the article.

## Format

One JSON object per line. Every line break inside the text is flattened to a
space.

```json
{
  "story_id": "1000122",
  "split": "train",
  "left_original": {"article_id": "1000122:1", "publisher": "cnn", "title": "...", "body": "..."},
  "left_neutralised": "...",
  "source": "TITLE=> ... . ARTICLE=> ...",
  "edits": ["Removed 'controversial' as subjective evaluation ..."],
  "generator": {"model": "nvidia/nemotron-3-ultra-550b-a55b", "prompt_version": "n2", "temperature": 1.0},
  "event_ids": ["1000122", "1000196"]
}
```

The right file has the same shape, with `right_original` and `right_neutralised`.

| field | use |
| --- | --- |
| `source` | **model input**, in the same `TITLE=> … . ARTICLE=> …` format as `neutral_articles_10k.jsonl` |
| `<stance>_neutralised` | **model target** |
| `split` | use as given; don't re-split |
| `edits` | the generator's own change log, for diagnostics only |

```python
import json
rows = [json.loads(l) for l in open("left_neutral_dedup.jsonl", encoding="utf-8")]
train = [(r["source"], r["left_neutralised"]) for r in rows if r["split"] == "train"]
```

The left and right files share no article text. Val and test come from the same
held-out events as `neutral_articles_10k.jsonl`.

## Quality (left; reproduce with `scripts/eda_neutralised.py`)

| check | result |
| --- | --- |
| parse errors, stray line breaks, duplicate ids, repeated articles, split leakage | 0 |
| loaded vocabulary per 1,000 words, input → output | intensifiers 0.27 → 0.05, subjective adjectives 0.41 → 0.16, loaded attribution verbs 0.83 → 0.50 |
| quotations reproduced verbatim or elided | 99.1% (reworded 0.9%, hallucinated 2 spans) |
| named entities kept | 93.2% |
| length, output/input | median 0.88 |

Right audits the same or slightly better. It keeps 93.6% of entities; 91.5% of
quotations are verbatim; loaded attribution verbs fall from 0.97 to 0.45 per 1,000
words; only 5.1% of outputs contain "the article …" (left: 12.3%). It rewrites
less aggressively than left (median similarity to the input 0.79 against 0.73),
and 2.8% of outputs are near-unchanged copies. A handful of right records (2
truncated, 2 with JSON leaking into the text) are listed in
`right_neutral_dedup.eda.json`.

**Known issues:** the same as in the 10k release, documented in detail in
`left_neutral_10k.README.md`.
- About 12% of outputs contain "The article reports / notes …".
- About 20% add a title or first name the input doesn't give ("President", "Donald").
- About 1% add a date or year from world knowledge.

The record ids behind every flag are in `<file>.eda.json`.

## Build

```bash
uv run python scripts/build_single_release.py --stance left  --dedupe
uv run python scripts/build_single_release.py --stance right --dedupe
uv run python scripts/eda_neutralised.py left_neutral_dedup.jsonl
```
