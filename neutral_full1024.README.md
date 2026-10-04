# left_neutral_full1024.jsonl / right_neutral_full1024.jsonl

Single-article framing-removal data (E2): **one news article in, the same
article rewritten without framing out**, drawn at random from the **whole**
BigNewsAlign export, with every input and output at most 1,024 LED tokens.
One event gives one left and one right article, so every article has an
opposite-side partner for the triplet positive.

| file | records | train | validation | test | outputs < 512 tokens |
| --- | --- | --- | --- | --- | --- |
| `left_neutral_full1024.jsonl` | 9,961 | 7,989 | 964 | 1,008 | 5,880 |
| `right_neutral_full1024.jsonl` | 9,961 | 7,989 | 964 | 1,008 | 6,249 |

The two files list the same events in the same order (`story_id` = event id).

## How it was built

1. **Pool.** Every event in `bignewsalign_full_LCR.jsonl` (1.4M articles) was
   visited in a seeded random order (seed 42). An event was accepted when both
   its left and right article:
   - fit in 1,024 LED tokens, title and body together (`allenai/led-base-16384`; dropped, never truncated);
   - have text not seen before (case- and punctuation-insensitive);
   - are not the same wire copy as each other;
   - are not a near-copy (word 5-gram Jaccard ≥ 0.8, MinHash LSH) of any
     accepted article or of any E2 (`side_neutral_full`) validation/test article.

   That leaves **72,370 events (144,740 articles)**; 63,929 of the articles are
   under 512 tokens.
2. **Sample.** The first 10,000 events of that order. Because the order is
   random, the sample keeps the pool's length distribution (left median 583 vs 578
   tokens, right 531 vs 536). The size was chosen so that both sides would end up
   with at least 5,000 outputs under 512 tokens.
3. **Rewrites.** `nvidia/nemotron-3-ultra-550b-a55b`, prompt `n2` (framing
   removal, add nothing), temperature 1.0. These are the same prompt and settings
   as `left/right_neutral_dedup.jsonl`.
4. **Release** (`scripts/build_full_release.py`):
   - 37 events were dropped because a rewrite exceeded 1,024 tokens.
   - 2 events were dropped as near-copies. An exact 5-gram check caught them
     after MinHash missed them.
   - Both sides of an event are always dropped together.
5. **Splits.** By event, from a hash of the event id: about 80 / 10 / 10. No
   input text appears in two splits.

## Format

The same fields as `left_neutral_dedup.jsonl`: `story_id`, `split`,
`<stance>_original` (`article_id`, `publisher`, `title`, `body`),
`<stance>_neutralised` (the target), `source` (the model input, `TITLE=> … .
ARTICLE=> …`), `edits`, `generator`, `event_ids`.

`edits` is the generator's own change log. It **does not reliably reflect how
much changed**: its entry count correlates only 0.12 with the actual
word-level distance. Use it for diagnostics only.

## Quality (`scripts/eda_neutralised.py`)

| check | left | right |
| --- | --- | --- |
| parse errors, duplicate ids, repeated inputs, split leakage | 0 | 0 |
| loaded attribution verbs per 1,000 words, input → output | 0.79 → 0.55 | 0.99 → 0.59 |
| intensifiers per 1,000 words | 0.22 → 0.07 | 0.23 → 0.07 |
| subjective adjectives per 1,000 words | 0.41 → 0.20 | 0.44 → 0.20 |
| quotes verbatim / elided / reworded / hallucinated | 89.0 / 9.6 / 1.3 / 0.2% | 90.1 / 8.8 / 1.1 / 0.1% |
| input entities kept | 93.7% | 91.8% |
| output / input words (median) | 0.89 | 0.91 |
| similarity to input (median; 1 = unchanged) | 0.81 | 0.83 |
| "The article reports …" | 12.9% | 7.4% |
| adds a date or year not in the input | 2.4% | 2.2% |

**Known issues:** the same as the ≤275-word sets, slightly stronger on long inputs.
- **Entities:** "drops 3+ entities" fires on 24–32% of records, because long
  articles contain more entities; the *share* kept stays around 92–94%.
- **World knowledge:** about 28% add a title or first name the input doesn't give.
- **Malformed outputs:** a few dozen per side are truncated or carry JSON
  fragments.

The ids behind every flag are in `<file>.eda.json`.

## Rebuild

```bash
uv run python scripts/make_full_pairs.py --raw context-transfer/bignewsalign_full_LCR.jsonl \
    --tag full1024b --ratio 0.85 --min-short-per-side 5250 --max-events 10000 --prefix-only
# + the next 112 events of data/final/full1024_pool.jsonl -> single_*_full1024c.jsonl
uv run python scripts/build_full_release.py
```
