# side_neutral_full

The exact split the Kaggle side-neutral runs were trained and tested on
(λ = 0 and λ = 0.35; checkpoints in the HF repo `RaunakSeksaria/anlp-neutral-led`
under `side-neutral-full/`). One news article in, the same article rewritten
neutrally out.

| split | articles | left | right | event groups |
| --- | --- | --- | --- | --- |
| train | 12,484 | 6,034 | 6,450 | 4,217 |
| validation | 1,778 | 885 | 893 | 831 |
| test | 1,775 | 868 | 907 | 834 |

## Where it comes from

`left_neutral_dedup.jsonl` + `right_neutral_dedup.jsonl` on the `data` branch
(commit `2b70898`), keeping only articles that share at least one event with an
article from the other side. The other 3,963 articles (all in train) cannot
have a triplet positive and were dropped. The splits are the release's own;
nothing was re-split. Triplets come from `src/scripts/prepare_side_neutral.py`,
seed 42.

## Files

- `prepared/<split>/baseline.jsonl`: one record per article: `input_title`,
  `input_text`, `target_text` (its neutral rewrite), `input_ideology`,
  `input_article_id`, `event_ids`, `event_id` (connected event group).
- `prepared/<split>/triplets.jsonl`: one triplet per article (it is the anchor).
  The positive is the opposite-side article about the same event; the negative
  is a same-side article from a different event group.
- `prepared/manifest.json`, `validation_report.json`, `eda.json`: counts,
  policies, token-length statistics.

Pass `--data-dir dataset/side_neutral_full/prepared` to `src/scripts/train_joint.py`
or `src/scripts/train_baseline.py`.

## Publisher labels

The prepared files have no publisher field. Join on `input_article_id`:

```python
import json
publisher = {}
for side in ("left", "right"):
    for line in open(f"{side}_neutral_dedup.jsonl", encoding="utf-8"):  # from the data branch
        r = json.loads(line)
        publisher[r[f"{side}_original"]["article_id"]] = r[f"{side}_original"]["publisher"]
```

There are 8 publishers, each on one side only: left `hpo`, `nyt`, `dailykos`,
`wpo`, `cnn`; right `breitbart`, `fox`, `wat`.

## Getting it again

It is a copy of the private Kaggle dataset `raunakseksaria/anlp-side-neutral-full`:

```bash
KAGGLE_API_TOKEN=... uvx kaggle datasets download raunakseksaria/anlp-side-neutral-full -p dataset/side_neutral_full --unzip
```
