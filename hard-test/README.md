# hard-test

A **test set of the most strongly sided articles**: the validation and test articles that the
POLITICS v2 classifier assigns to their own side most confidently. Each record keeps its input and
its existing generated target. Neutralising these articles has the furthest to go, so a model's
output on them shows whether it really removes framing.

| file | records | dedup val / test | full1024 val / test |
| --- | --- | --- | --- |
| `hard_test_left_neutral.jsonl` | 711 | 177 / 174 | 175 / 185 |
| `hard_test_right_neutral.jsonl` | 725 | 179 / 181 | 177 / 188 |

**This is a subset of existing data.** Every record is copied unchanged from one of the two
single-article releases. The `release` field says which one, and the classifier scores are added.

| `release` | source file | HF config | inputs |
| --- | --- | --- | --- |
| `dedup` | `{left,right}_neutral_dedup.jsonl` | `reframing_{left,right}` | ≤ ~275 words, < 512 LED tokens |
| `full1024` | `{left,right}_neutral_full1024.jsonl` | `reframing_{left,right}_long` | ≤ 1,024 LED tokens |

Input is the original BigNewsAlign article. Target is the Nemotron `n2` rewrite (left → left_neutral,
right → right_neutral).

## How it was selected (`scripts/make_hard_subset.py`)

1. **Score every original article** (38,360 unique bodies, both releases, both stances) with the
   POLITICS v2 classifier (Harshit's `politics-ideology-classifier/v2`):
   - all 4 seeds (`seed4{2,3,4,5}_best.pt`);
   - body only, outlet names masked as in `compare_variants.py`;
   - 512-token windows, 64 overlap, mean window logits.
   - Run on Kaggle (`scripts/kaggle_hard_subset/`).
2. **Exclude anything in a train split of either release.** 174 left and 147 right articles are in
   validation or test in `full1024` but in train in `dedup`, so they are excluded. No article here
   appears in any train split.
3. **Rank by logit margin:** own-side logit minus the largest other logit, divided by the ensemble
   temperature (3.079), averaged over seeds. The probabilities saturate (every kept original is
   0.994–0.9986), so they can't rank.
4. **Keep the top 20% within each release × split.** That way long articles don't crowd out
   short ones, and validation and test keep their shares.

An article in both releases is kept once, from `dedup`.

## Format

The fields of the source release (`story_id`, `split`, `<stance>_original`, `<stance>_neutralised`,
`source`, `edits`, `generator`, `event_ids`), plus:

| field | content |
| --- | --- |
| `release` | `dedup` or `full1024` |
| `split` | `validation` or `test` (`dedup`'s `val` renamed) |
| `politics` | classifier on the original body: `p` = [P(Left), P(Centre), P(Right)] (calibrated 4-seed ensemble), `p_own`, `margin`, `windows`, `rank_in_group` |
| `politics_target` | the same for the generated target, `<stance>_neutralised` |

## How the targets score (`p_own.png`)

| | P(own side) original → target, median | own side still the top label | P(Centre) original → target, mean |
| --- | --- | --- | --- |
| left | 0.998 → 0.800 | 100% → 67% | 0.001 → 0.176 |
| right | 0.998 → 0.894 | 100% → 82% | 0.002 → 0.151 |

The targets split into two groups: many are pulled well away from their side (99 left targets are
under 0.1), but 318 left and 359 right stay above 0.9. Hard targets still score a little higher for
their own side than the *other* validation/test articles do before any rewrite. So matching the
targets isn't enough to look neutral on this set; compare a model's outputs against the targets'
scores, not against zero.

## Caveats

- **Left-side outlet skew.** NYT has 222 of the 711 left articles, Daily Kos 185, CNN 154, WaPo 85
  and HuffPost 65. Across both releases HuffPost is about 28% of left articles; here it's 9%. On the
  left, "hard" partly means NYT/Daily Kos/CNN house style, even with outlet names masked. The right
  side keeps its mix (Breitbart 249, Fox 247, Washington Times 229).
- **Classifier contamination.** POLITICS was pretrained on BigNewsAlign, which these articles come
  from (`scratch/politics_classifier_notes_for_harshit.md`). Read probabilities as ranks, not
  rates.
- **Same classifier for selection and evaluation.** If you evaluate with this classifier, its
  scores on the originals are at the ceiling by construction.

## Scores and rebuilding

`scores/originals.jsonl` (38,360 bodies) and `scores/targets.jsonl` (39,918 targets) hold the raw
per-seed logits for **every** article and target in both releases, train included, keyed by
`md5(text)`. Re-selecting from them needs no GPU:

```bash
cp hard-test/scores/originals.jsonl runs/hard/scores.jsonl
cp hard-test/scores/targets.jsonl   runs/hard/scores_targets.jsonl
uv run python scripts/make_hard_subset.py select --test-only \
    --target-scores runs/hard/scores_targets.jsonl --out "hard-test/hard_test_{st}_neutral.jsonl"
```

`select` needs `{left,right}_neutral_full1024.jsonl` at the repo root. Rebuild them with
`scripts/build_full_release.py`; the output is byte-identical to the HF release.
