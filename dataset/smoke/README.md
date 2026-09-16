# Smoke-test bundle — event-aware triplet learning

A 320-story slice of the project's data, self-contained (full article text inline, no 1.4 GB
crawl needed). Purpose: **measure what a training run costs on your hardware**, and give both
triplet variants from §4.2 of the proposal a concrete, runnable form.

Nothing here is a final training set. 320 stories is ~1/8 of the usable NeuS corpus.

---

## 1. What's in it

| file | rows | what |
| --- | --- | --- |
| `articles.jsonl` | 960 | NeuS/AllSides articles, 320 stories × {left, center, right}, full body text |
| `negatives.jsonl` | 2,198 | Article-Bias-Prediction articles used as triplet negatives, full text |
| `gen/{train,val,test}.source` | 256/32/32 | left + right articles, one story per line |
| `gen/{train,val,test}.target` | 256/32/32 | **the centre article** — our stand-in neutral target |
| `gen/{train,val,test}.target.roundup` | 256/32/32 | the human AllSides roundup, same lines — the alternative target |
| `triplets_publisher.jsonl` | 1,268 | **variant (a)**, publication-invariant |
| `triplets_ideology.jsonl` | 1,280 | **variant (b)**, ideology-invariant |
| `eval_unseen_publisher.jsonl` | 300 | ABP articles from 9 outlets that appear nowhere else in the bundle — for the §5 publisher-leakage probe |
| `manifest.json` | — | counts, length stats, coverage rates, caveats |

### The two variants

Both share the same positive — the same event reported by a different outlet. They differ **only
in the negative**, which is the point of comparing them:

```
(a) publisher   anchor: Fox News, event A    positive: HuffPost, event A    negative: Fox News, event B
(b) ideology    anchor: right,    event A    positive: left,     event A    negative: right (a DIFFERENT outlet), event B
```

### Why negatives come from a second corpus

In NeuS, **248 of 250 publishers only ever appear under one stance**. Draw negatives from NeuS
alone and variant (a)'s negative pool becomes a strict *subset* of variant (b)'s — the two stop
being distinguishable and the comparison is vacuous. Article-Bias-Prediction breaks this: it has
~77–79 distinct outlets per ideology, so a same-ideology negative can be a genuinely different
publisher. Outlet names were matched across the two corpora by normalisation
(`Fox News (Online News)` → `Fox Online News`); **97.8 %** of variant-(a) anchors got a real ABP
negative, the remaining 2.2 % fall back to a NeuS article and are flagged `negative_from: "neus"`.

**This is also the main compute finding.** ABP articles are median ~768 words (~1,000 BPE), p90
~1,669 (~2,169 BPE), max 10,219. So a triplet batch pairs a ~90-BPE anchor with a ~1,000-BPE
negative. That is a completely different memory profile from the 3-excerpt NeuS setup, and it is
why §4.3 specifies LED rather than BART.

---

## 2. How to load it

`gen/*.source` / `*.target` are line-aligned plain text, one story per line, in NeuS's format
(`TITLE=> …. ARTICLE=> …`, multiple articles joined by ` [SEP] `). Feed them to any seq2seq loop.

Triplets are **ids**, needing two lookups:

```python
import json

arts = {a["article_id"]: a for a in map(json.loads, open("articles.jsonl"))}
negs = {n["neg_id"]:     n for n in map(json.loads, open("negatives.jsonl"))}

for line in open("triplets_publisher.jsonl"):      # or triplets_ideology.jsonl
    t = json.loads(line)
    anchor   = arts[t["anchor"]]["body"]
    positive = arts[t["positive"]]["body"]
    negative = (negs[t["negative"]]["content"] if t["negative_from"] == "abp"
                else arts[t["negative"]]["body"])
```

A row looks like:

```json
{"anchor": "4003:right", "positive": "4003:left", "negative": "sJfVGcfW6vWo04q7",
 "negative_from": "abp", "negative_short": "2566:right", "split": "val", "variant": "publisher"}
```

`negative_short` is a NeuS article satisfying the *same* constraint but only ~90 BPE long. Swap
`negative` → `negative_short` and rerun to separate **"the triplet stream costs X"** from
**"long negatives cost Y"**. Everything else stays identical.

The triplet stream is independent of the generation stream: one triplet batch per generation step,
cycled, per `L = L_LED + λ · L_triplet` (§4.3). Filter rows by `split` if you want a clean
val/test; `train` is what matters for timing.

---

## 3. Record your hardware first

So numbers are comparable across four different machines. Paste this with your results:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_bf16_supported())"
```

| | |
| --- | --- |
| GPU / VRAM | |
| bf16 supported | |
| torch / CUDA | |
| precision used | bf16 if supported, else fp16 |

Precision is **reported, not prescribed** — use the best your card supports.

---

## 4. What to run, and what to send back

§4.3 specifies LED-base-16384. LED's cost is dominated by input length, and input length is exactly
what the ABP negatives push on — so work **up this ladder until something OOMs**, and report the
highest rung that fit:

| rung | model | max_source | answers |
| --- | --- | --- | --- |
| 0 | `facebook/bart-base` | 1024 | cheap floor — does the pipeline run at all |
| 1 | `allenai/led-base-16384` | 1024 | LED overhead vs BART at matched context |
| 2 | `allenai/led-base-16384` | 4096 | covers ~p99 of ABP negatives untruncated |
| 3 | `allenai/led-base-16384` | 16384 | the proposal's stated config |
| 4 | `allenai/led-large-16384` | 4096 | the scale-up §4.3 leaves open |

At each rung run three configs, **one epoch each**: baseline (λ=0, no triplet stream), then
`triplets_publisher.jsonl`, then `triplets_ideology.jsonl`. Then repeat your best rung once with
`negative_short`.

| rung | config | peak VRAM | s / epoch | s / opt-step | OOM? |
| --- | --- | --- | --- | --- | --- |
| | baseline | | | | |
| | + publisher triplets | | | | |
| | + ideology triplets | | | | |
| | + publisher, `negative_short` | | | | |

Peak VRAM: `nvidia-smi --query-gpu=memory.used --format=csv -l 5` alongside the run.

**Extrapolating:** full NeuS train is 2,129 stories, so `full run ≈ measured × 8.3 × epochs`.
Triplet configs cost ~2–3× baseline per step (anchor, positive and negative each need an encoder
pass), more when the negative is a full-length ABP article.

Two caveats on the extrapolation, both in `manifest.json`:
- Seed 42 drew NeuS bodies ~13 words shorter than the corpus median (69 vs 82). Generation-side
  cost extrapolates ~10–15 % low. The ABP negatives dominate anyway and are drawn from the full pool.
- Truncation is not neutral for this task — framing concentrates in the article lead (`eda/REPORT.md`).
  The rung that fits decides how much of a negative the model actually sees, which is a
  *methodological* result, not just a budget one.

---

## 5. If you're on Ada

`docs/ada-compute-requirements.md` has the cluster-specific traps: the 2080 Ti is 10.6 GiB with no
bf16 unit; `--constraint=2080ti` is required because featureless nodes are Pascal 1080 Ti, which
current torch wheels don't compile for; never `--mem`, always `--mem-per-cpu`; `uv sync` must run
inside a job. On any other hardware ignore all of that and just fill in §3.

---

## 6. Provenance

- **NeuS / AllSides** — Lee et al. 2022 (NAACL), `data/framing-bias-metric-main/`, flattened by
  `scripts/prepare_data.py` into `data/prepared/articles.jsonl`.
- **Article-Bias-Prediction** — Baly et al. 2020 (EMNLP), `data/Article-Bias-Prediction/`.
- Seed 42 throughout. Sampling: stories with a released NeuS split, exactly 3 articles
  (left/center/right), excluding stories whose left/right slot is an AllSides staff byline rather
  than an outlet. 256/32/32 drawn without replacement, story-disjoint across splits.
- Negatives: drawn from all of ABP **except** outlets reserved for `eval_unseen_publisher.jsonl`
  (ABP's media-split test outlets that don't also occur in NeuS), so the eval outlets are unseen by
  construction. Negatives are also filtered so none shares a normalised headline with any story in
  the bundle.
- Centre articles appear **only** as generation targets — never as a triplet anchor, positive or
  negative, and never in a `.source` line. Verified.
