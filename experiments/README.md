# Generation-setting experiments (E2 single-article neutralisation)

Which Nemotron settings and prompt give the best neutral rewrites? Every variant rewrites the
**same 150 left + 150 right articles** (the first 150 events of `eval500/`), so variants are
compared pairwise on identical inputs, with bootstrap 95% CIs.

- `eval500/{left,right}.jsonl`: the fixed evaluation sample. 500 events from
  `data/final/full1024_pool.jsonl` (index 10,000 on), checked against every article generated
  before (exact text and word 5-gram Jaccard ≥ 0.8). Left and right are paired by event. The
  screening set is the first 150 (field `order`), so extending a variant to 500 adds events
  without redoing any.
- `variants.json`: every setting run, append-only. A new setting always gets a new name.
- `reports/all_variants_150.md`: the full metric table for all 14 runs, with paired differences
  against `think_on_t1.0`. `reports/all_variants_150.per_record.jsonl` holds per-article metrics
  for further analysis.
- Raw outputs live in `runs/experiments/<variant>/` (not in git).

Model: `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM. Unless stated, thinking on,
t = 1.0, top_p 0.95, prompt `n2`.

## Reproduce

```bash
scripts/run_variant.sh <name> 150          # one variant, resumable; settings from variants.json
scripts/run_queue.sh 150 <name> <name> ... # several, one after another
cd scripts && ../.venv/bin/python3 compare_variants.py think_on_t1.0 <name> ...
```

**Outputs are not reproducible.** The endpoint accepts `seed` but ignores it, and even t = 0
gives different text on a rerun. So each variant's outputs are its only copy. The noise floor is
measured instead: `think_on_t1.0_rep` is an identical rerun of `think_on_t1.0`. A 150-per-side
screen moves defect rates by about ±5 points and classifier probabilities by about ±0.05
between identical runs, so smaller differences are noise.

## Metrics

- **Edit volume:** word-sequence similarity to the input, and share of input words changed.
- **Faithfulness:** input entities kept; quotes copied verbatim; outputs that add an entity,
  number or date absent from the input; "The article …" meta-commentary.
  - "Adds entity" is noisy. It mostly catches expanded abbreviations ("Sen." → "Senator") and
    apostrophe tokenisation, so read it relatively.
- **Loaded words per 1,000 words**, input → output (lexicon match).
- **POLITICS classifier** (Harshit's `politics_media.pt`; labels Left / Centre / Right):
  - Outlet names are masked. Articles are scored over 512-token sliding windows (64 overlap).
  - Reported: P(true side), how often the true side is the top label, and P(Centre), each
    original → rewrite.
  - Used as a relative signal only. The encoder was pretrained on BigNewsAlign, which our
    articles come from (see `scratch/politics_classifier_notes_for_harshit.md`).

## Findings (150 per side, ✱ = 95% CI excludes 0 vs `think_on_t1.0`)

Values are left / right. The originals score P(true side) 0.71 / 0.72 and P(Centre) 0.17 / 0.21,
and carry 1.69 / 1.27 loaded words per 1k.

| Variant | What changed | Words changed % | Loaded /1k after | P(true side) | P(Centre) | Adds entity % | "The article" % | Output tokens |
|---|---|---|---|---|---|---|---|---|
| `think_on_t1.0` | reference | 26 / 25 | 0.77 / 0.57 | .37 / .45 | .47 / .44 | 27 / 23 | 13 / 7 | ~2,400 |
| `think_on_t1.0_rep` | identical rerun (noise floor) | 26 / 25 | 0.82 / 0.67 | .37 / .46 | .47 / .45 | 31 / 28 | 11 / 6 | ~2,250–2,400 |
| `think_off_t1.0` | thinking off | 19 / 19 ✱ | 1.20 / 0.89 | .41 / .49 | .43 / .40 | 24 / 20 | 14 / 7 | **~700** |
| `think_on_t0.0` … `t1.3` | temperature 0, 0.3, 0.6, 1.3 | 24–28 / 24–26 | 0.86–1.05 / 0.53–0.80 | .36–.39 / .44–.48 | .46–.48 / .40–.47 | 25–31 / 25–29 | 13–17 / 5–8 | ~2,400–2,700 |
| `think_on_t1.0_p0.8`, `_p1.0` | top_p 0.8, 1.0 | 26 / 25 | 0.85–0.91 / 0.50–0.57 | .38–.40 / .45–.46 | .42–.46 / .43–.45 | 25–28 / 24–26 | 11–13 / 5–7 | ~2,300–2,500 |
| `para_t1.0` | **control:** reword, keep the framing | 33 / 32 ✱ | 2.22 / 1.99 | .52 / .66 ✱ | .23 / .20 ✱ | 56 / 59 ✱ | 2 / 1 ✱ | ~3,700 |
| `n3_t1.0` | n2 + no "The article…", keep abbreviations | 27 / 26 | 0.81 / 0.60 | .39 / .47 | .46 / .44 | 24 / 27 | 9 / 7 | ~2,600 |
| `n2light_t1.0` | word-level edits only | **12 / 11** ✱ | 0.83 / 0.62 | .48 / .52 ✱ | .39 / .37 ✱ | **18 / 13** ✱ | **6 / 3** ✱ | ~2,900 |
| `n2heavy_t1.0` | rebuild as wire copy | 37 / 34 ✱ | 0.74 / 0.61 | **.31 / .43** ✱ | **.55 / .50** ✱ | 40 / 34 ✱ | 12 / 8 | ~2,350 |
| `n2min_t1.0` | one-paragraph prompt, no taxonomy | 25 / 19 ✱ | 0.73 / 0.58 | .49 / .53 ✱ | .39 / .34 ✱ | 35 / 27 ✱ | 3 / 1 ✱ | ~4,250–4,950 |

✱ marks a variant with at least one significant difference in that cell's direction, left or
right. The full CIs are in `reports/all_variants_150.md`.

1. **Decoding settings don't matter with thinking on.** Temperatures 0–1.3 and top_p 0.8–1.0
   all sit inside the rerun noise. Two runs of any two settings give outputs about
   0.81–0.84 word-similar to each other, the same as a run against its own rerun. A top_k
   sweep isn't worth running. **Keep t = 1.0, top_p 0.95, thinking on.**
2. **Thinking off is 3.5× cheaper but edits less.** It changes fewer words and removes about
   half as much loaded language. It also rewords quotes in 7% of left quote spans, concentrated
   in two articles. The classifier doesn't separate it from thinking on. We chose to keep
   thinking on.
3. **The classifier measures neutralisation, not just rewording (paraphrase control).** `para`
   rewords more than `n2` (33% vs 26% of words) while keeping the framing; loaded words even
   rise. Yet P(Centre) barely moves (0.17 → 0.23 on the left), against 0.47 for `n2`.
   - Rewording alone explains about half of `n2`'s P(true side) drop on the left and about a
     quarter on the right.
   - The shift toward Centre is almost entirely neutralisation.
4. **Aggression is the one real dial: light → `n2` → heavy.** Words changed rise 12 → 26 → 37%,
   and P(Centre) rises .39 → .47 → .55 (left).
   - Loaded-word removal is already maxed out at the light level, so the extra aggression buys
     only structural change (reordering, rebalancing). That is what moves the classifier.
   - It costs faithfulness: added entities rise 18 → 27 → 40%, and entities kept fall
     96.8 → 94.5 → 92.6%.
   - **This is a choice between faithfulness and classifier neutrality, still open.**
5. **The bias taxonomy in `n2` earns its tokens.** `n2min` edits as much as `n2` on the left,
   but the classifier moves only as far as with `light`. It also uses about twice the output
   tokens, because the model thinks longer with less guidance. Its wall time in the report
   (172 min per 100 articles) is inflated by a power cut mid-run.
6. **`n3` halves "The article…" meta-commentary but nothing else.** The leftovers come from
   opinion columns ("The author contends…"): when the whole piece is opinion, "cut the
   judgement" leaves nothing, so the model attributes it. That's an input-selection problem
   (detect op-eds), and a candidate criterion for a "hard" subset.

## Next

- Extend the finalists (`n2light`, `n3`, `n2heavy`) to 500 per side before choosing an
  aggression level.
- Later: the few-shot family and a neutralise-specific revision loop.
