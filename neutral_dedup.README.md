# left_neutral_dedup.jsonl / right_neutral_dedup.jsonl

Single-article neutral article generation data: **one news article in, the same
article rewritten without framing out.** There is one file per stance, and every
input article appears **exactly once**.

| file | records | train | val | test |
| --- | --- | --- | --- | --- |
| `left_neutral_dedup.jsonl` | 9,437 | 7,684 | 885 | 868 |
| `right_neutral_dedup.jsonl` | 10,000 | 8,200 | 893 | 907 |

**Left holds every distinct left article BigNewsAlign has** (9,441 in the whole
export, minus 4 that were near-copies of articles already in the dataset).
"Unique" is judged ignoring case and punctuation, because some outlets ran the
same article twice with only straight vs curly quotes changed. Those count as
one article, and the kept record's `event_ids` lists both copies' events.

### Where the articles come from

- **From the 10k-event corpus** (5,695 left, 6,158 right): every unique article in
  the events behind `neutral_articles_10k.jsonl`, in their original splits.
- **Added from the rest of the BigNewsAlign export** (3,742 left, 3,842 right),
  chosen with `scripts/make_extra_articles.py`, shortest first. **All added articles
  are in train**, so val and test are exactly those of the 10k-event corpus.
  Candidates were screened against every article already in the dataset (both
  stances, all splits) and against each other. Anything with word 5-gram
  Jaccard >= 0.8 was dropped, so no near-copy of a val/test article entered train.
  Added records have `event_ids` with a single id: the event they were taken from.

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

## Quality (reproduce with `scripts/eda_neutralised.py <file>`)

| check | left | right |
| --- | --- | --- |
| parse errors, stray line breaks, duplicate ids, repeated articles, split leakage | 0 | 0 |
| loaded attribution verbs per 1,000 words, input → output | 0.82 → 0.48 | 0.95 → 0.46 |
| intensifiers per 1,000 words | 0.26 → 0.05 | 0.23 → 0.07 |
| subjective adjectives per 1,000 words | 0.41 → 0.16 | 0.42 → 0.16 |
| quotations verbatim / elided / reworded / hallucinated | 89.4 / 9.5 / 1.0 / 0.1% | 91.1 / 7.8 / 1.0 / 0.0% |
| named entities kept | 93.2% | 93.5% |
| length, output/input (median) | 0.87 | 0.92 |
| median similarity to the input (1.0 = unchanged) | 0.74 | 0.80 |
| near-unchanged copies (≥0.98) | 1.2% | 2.8% |
| "The article reports / notes …" | 12.4% | 5.5% |
| adds a title/first name not in the input | 22.7% | 21.8% |
| adds a date or year not in the input | 1.1% | 0.9% |

Right rewrites less aggressively than left, and left leans more on "the article
…" attribution. A handful of records per file are truncated or carry JSON
fragments (left 2, right 9). Their ids, and those behind every other flag, are in
`<file>.eda.json`.

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
