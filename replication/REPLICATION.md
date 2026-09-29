# ChunkyBERT replication — progress log

**Status: complete.** Table 8 partly reproduced (GRU+attention rows yes, TCN no); Table 10 six of
seven reproduced, with the seventh (`No Transformers`) explained. Best non-ChunkyBERT variant
(`No Classifier`) retrained with weights and evaluated in `../evaluation/best_variant_eval.ipynb`.

Files: `variants.py` (all models as one class) · `run_variant.py` (one-epoch trainer) ·
`sweeps/` (local sweep scripts, run from anywhere) · `logs/` (their output) · `results/` (CSVs) ·
`kaggle/table10/`, `kaggle/best_variant/` (Kaggle kernels + downloaded outputs).

Replicating every model in Loiya et al., *ChunkyBERT*, Sci. Rep. 16, 15323 (2026),
on the same training data (Article-Bias-Prediction, 37,554 articles, 80/20 stratified split, seed 42).


---

## What the authors released, and what they did not

The official repo (`github.com/dloiya/chunkybert`) contains exactly one notebook across its
entire history, and that notebook defines only `Full Model`. **None** of the comparison models
in Table 8, and none of the ablations in Table 10, exist in released code. The paper's
supplementary material is not reachable (Springer returns 503/403). So every model below the
full ChunkyBERT has to be rebuilt from the paper's prose.

**Citation direction.** ChunkyBERT cites the GRU work (ref 43, Tanuku 2022) and the TCN work
(ref 44, Shikalgar & Arage 2023) — not the reverse. The GRU/TCN baselines are therefore borrowed
from those prior papers, not contributions of ChunkyBERT, and the paper gives no architecture
details for them beyond the four columns of Table 8.

## The ambiguity, and how we're resolving it

The paper never states what feeds the GRU/TCN baselines. Two readings:

1. **Standalone** — trainable embedding layer → GRU/TCN. The literal "traditional DL baseline".
2. **BERT-fed** — the ChunkyBERT pipeline with the Transformer encoder swapped for a GRU/TCN.

Evidence for standalone: a GRU over 5,000 timesteps is *sequential*, so ~600 s/epoch is entirely
plausible for a from-scratch recurrent model — the reported times do not require BERT. The
reported accuracies also sit well below ChunkyBERT (36-75%), and two of them (36.4%, 39.2%) are
barely above the 33% chance floor, which is what an under-trained from-scratch model looks like
at `lr=2e-5` for one epoch.

**Plan: start with the cheapest reading (standalone) and escalate only if measured accuracy
fails to land near the paper's numbers.**

## Protocol

Matching the paper's stated setup (Table 4) for every run: `bert-base-uncased` tokenizer,
max 5,000 tokens/document, 512-token chunks, batch 16, AdamW `lr=2e-5` weight decay 0.01,
dropout 0.3, linear warmup 10%, mixed precision, **1 epoch** (Tables 8 and 10 are both
one-epoch results), 80/20 stratified split, seed 42.

## Target numbers from the paper

### Table 8 — architecture comparison (1 epoch)

All 8 baselines run twice — once at the paper's stated `lr=2e-5`, once at `lr=1e-3` (a normal
from-scratch learning rate). Δ = measured − paper.

| Model | Paper | @2e-5 | Δ | @1e-3 | Δ |
|---|---|---|---|---|---|
| GRU, no chunk, no attn | 39.16 | **37.68** ✅ | −1.5 | 71.88 ❌ | +32.7 |
| GRU, no chunk, attn | 73.29 | 39.25 ❌ | −34.0 | **77.42** ✅ | +4.1 |
| GRU, chunk, no attn | 36.44 | **37.32** ✅ | +0.9 | 69.26 ❌ | +32.8 |
| GRU, chunk, attn | 74.82 | 38.82 ❌ | −36.0 | **75.68** ✅ | +0.9 |
| TCN, no chunk, no attn | 71.11 | 36.57 ❌ | −34.5 | 58.85 ❌ | −12.3 |
| TCN, no chunk, attn | 68.02 | 36.57 ❌ | −31.4 | 53.72 ❌ | −14.3 |
| TCN, chunk, no attn | 72.28 | 36.57 ❌ | −35.7 | 54.48 ❌ | −17.8 |
| TCN, chunk, attn | 69.62 | 36.57 ❌ | −33.1 | 54.88 ❌ | −14.7 |
| **ChunkyBERT** | **83.91** | — | — | **82.93** ✅ (authors' own code) | −1.0 |

Measured training times were 139-232 s per run on a local RTX 3050 — far under the paper's
597-1511 s, though that comparison is confounded by different hardware and by `emb_dim=256`.

### What this shows

**No single learning rate reproduces Table 8.** The GRU rows are reproducible, but only by using
*different* learning rates for rows that in the paper differ solely by an attention flag:

- `lr=1e-3` nails both GRU **+attention** rows (77.42 vs 73.29; 75.68 vs 74.82).
- `lr=2e-5` nails both GRU **−attention** rows (37.68 vs 39.16; 37.32 vs 36.44) — but only because
  both our model and theirs sat at chance. `36.57%` is exactly the majority-class share
  (13,734 Right / 37,554), i.e. a model emitting one class.

At a learning rate where GRU trains at all, removing attention costs ~6 points (77.4 → 71.9), not
the ~34 points Table 8 implies. The paper's two near-chance GRU rows look like **training failures
rather than a measured effect of removing attention** — which matters, because those rows are what
make attention appear indispensable.

**TCN never reproduces** under this reading, sitting 12-18 points low at every setting.

### TCN depth escalation (did not help)

Suspecting the 4-level TCN's 31-token receptive field was too local, all four TCN rows were re-run
with 8 levels (dilations 1…128, RF 511) at `lr=1e-3`:

| Model | Paper | 4 levels | 8 levels |
|---|---|---|---|
| TCN, no chunk, no attn | 71.11 | 58.85 | 50.25 |
| TCN, no chunk, attn | 68.02 | 53.72 | 59.77 |
| TCN, chunk, no attn | 72.28 | 54.48 | 36.57 |
| TCN, chunk, attn | 69.62 | 54.88 | 36.57 |

Depth made things *worse and unstable* — two of the four collapsed to majority-class. A deeper stack
is simply harder to fit in one epoch. **Tuning stopped here deliberately**: continuing to search
configurations until one happens to hit 70% would be fitting the hyperparameters to the target, not
replicating. The honest statement is that the TCN baselines are not reproducible from the
information the paper provides.

### Table 10 — ablation study (1 epoch)

All 7 run on Kaggle (2×T4), 1 epoch each, ~2.8 h total.

| Variant | What changes | Paper acc | Measured acc | Δ | Paper min | Measured min |
|---|---|---|---|---|---|---|
| Full Model | — | 84 | 80.99 | −3.0 | 27 | 27.4 |
| No Attention | mean pooling over tokens | 82 | **81.43** ✅ | −0.6 | 27 | 27.4 |
| No Classifier | simplified classification head | 82 | **82.36** ✅ | +0.4 | 28 | 27.4 |
| Max Pool | max instead of mean chunk aggregation | 81 | **81.84** ✅ | +0.8 | 28 | 27.4 |
| No Chunking | first 512 tokens only | 79 | **77.81** ✅ | −1.2 | 12 | 12.0 |
| Freeze BERT | BERT encoder frozen | 58 | **58.15** ✅ | +0.1 | 13 | 12.7 |
| No Transformers | 2-layer Transformer encoder removed | 41 | **82.31** ❌ | **+41.3** | 5 | 24.1 |

Six of seven reproduce, several strikingly well — `Freeze BERT` lands at 58.15 against a reported 58,
and every training time matches to within a minute. That agreement is what makes the last row stand out.

### The `No Transformers` row does not hold up

The paper reports that removing the 2-layer Transformer encoder collapses accuracy to **41%**, and
uses this as its strongest evidence: *"The most severe drop in validation accuracy is observed when
all transformer layers are removed (NoTransformers), with accuracy falling to 41%, confirming that
transformer-based contextual representations are the dominant contributor to performance."*

Measured, that ablation scores **82.31%** — statistically tied with the full model (80.99%) and
second-best of all seven variants. Removing those two layers costs essentially nothing.

**The reported time gives the discrepancy away.** The paper lists 5 minutes for this variant against
27 for the full model. But deleting a 2-layer encoder from a 117.6M-parameter model still leaves
BERT's 109.5M parameters to fine-tune — it cannot run 5× faster. Our measurement takes 24.1 min,
which is the physically sensible number. A 5-minute epoch is what you get when **BERT itself is not
being trained**. So the paper's "NoTransformers" figure looks like it came from a model with *no
BERT at all* — embedding → pooling → classifier — not from the ablation as described.

**Tested, and it fits.** Running exactly that model (embedding → attention pooling → classifier, no
BERT, no encoder, paper's `lr=2e-5`):

| Reading of "No Transformers" | Accuracy | Epoch time | vs paper (41%, 5 min) |
|---|---|---|---|
| 2-layer encoder removed, BERT kept (as described) | 82.31% | 24.1 min | ❌ +41.3 pts, 5× too slow |
| **BERT removed as well** | **37.16%** | **1.9 min** | ✅ within ~4 pts, right order of time |

So the paper's headline ablation result almost certainly measures *"remove BERT"*, while the text
describes it as *"remove the transformer layers"* and draws a conclusion about the 2-layer encoder.

This matters for the paper's architecture argument. Its own numbers, read correctly, say:

- **BERT carries the model.** Freezing it costs 24 points (82 → 58); removing it costs ~45.
- **The 2-layer Transformer encoder contributes nothing measurable.** 82.31% without it vs 80.99%
  with it — and that encoder, plus the attention pooling, is what the paper claims as its novel
  contribution over plain chunked BERT.

Ranked by measured accuracy, the full ChunkyBERT is the *worst* of the five BERT variants that keep
chunking. Every simplification tested — dropping the encoder, dropping attention pooling, dropping
dropout, max-pooling chunks — scored at or above it.

## Already measured

The full ChunkyBERT is done — trained twice on Kaggle (2×T4), paper hyperparameters, and both
runs agree to within noise.

| Run | Epoch 1 val acc | Best val acc | Best val AUC |
|---|---|---|---|
| Run 1 (5 epochs) | 82.93% | 86.58% (ep 3) | 0.9636 |
| Run 2 (5 epochs, weights saved) | 83.01% | 86.87% (ep 3) | 0.9651 |
| **Paper** | **83.91% (Table 8)** / 84% (Table 10) | **86.22%** | **0.962** |

Epoch-1 accuracy lands ~1 point under the paper's; the 5-epoch best lands ~0.5 point over it.
Both well inside run-to-run variance.

## Implementation notes

- `variants.py` — one `VariantClassifier` covering all 15 configurations (14 paper models + the no-BERT test). Chunking routine copied from the
  authors' `model.py`. ChunkyBERT itself is just the configuration
  `backbone=bert, encoder=transformer, pooling=attention, chunk_agg=mean, chunking=True`.
- `run_variant.py` — one-epoch trainer under the paper's protocol, appends to a results CSV.
- Standalone baselines use `emb_dim=256` (a from-scratch embedding table at BERT's 768 would be
  needlessly heavy and slow); GRU is bidirectional with hidden 128 so its output stays 256-d; TCN is
  4 dilated causal conv levels (dilations 1/2/4/8, kernel 3), per the cited TCN work.
- **Unchunked means different things per backbone.** BERT physically cannot exceed 512 tokens, so
  "No Chunking" in Table 10 = first 512 tokens only (as the paper describes). GRU/TCN have no such
  limit, so unchunked Table 8 baselines read the full 5,000 tokens — which is also why a *sequential*
  GRU there is slow enough to explain the paper's ~600 s.
- Attention/mean pooling here is **padding-masked**. The authors' released code softmaxes over
  padding too. This matters little for 512-token chunks but matters a lot for a 5,000-token padded
  sequence, so the masked version is used throughout and `Full Model` is re-run inside this harness
  to keep Table 10 internally like-for-like.

## Run log

- **2026-09-26** — Confirmed no baseline/ablation code exists upstream. Supplementary unreachable.
  Chose standalone-first strategy for Table 8. Wrote the parameterised model class; all 14 configs
  smoke-tested end to end.
- **2026-09-26 17:31** — Timing probe: standalone unchunked models run ~3.7 min/epoch on the local
  RTX 3050, so the whole Table 8 sweep fits locally in ~1 h and costs no Kaggle quota. Sweep launched,
  cheapest variant first.
- **2026-09-26 17:38** — **Standalone at the paper's `lr=2e-5` does not train at all.** Both TCN
  variants returned exactly **36.57%**, which is precisely the majority-class share
  (13,734 Right / 37,554). The models are predicting a single class. One epoch of AdamW at 2e-5 —
  a learning rate meant for *fine-tuning* a pretrained BERT — is far too small to move randomly
  initialised embeddings. So this configuration cannot be what produced the paper's 68-75% numbers.
  Escalating to `lr=1e-3`, the normal range for a from-scratch recurrent/convolutional model, before
  spending anything on the much more expensive BERT-fed reading.
- **2026-09-26 18:03** — All 8 standalone runs at `lr=2e-5` finished (~2.5-4 min each, so the whole
  sweep cost under 25 min locally). **2 of 8 reproduce the paper.** The two that match are
  `GRU no chunk, no attn` (37.68 vs 39.16) and `GRU chunk, no attn` (37.32 vs 36.44) — precisely the
  two rows where the paper itself reports chance-level accuracy. Every row where the paper reports a
  *working* model (68-75%) fails to reproduce, landing at 36-39% instead.

  Worth stating plainly: matching a model that predicts one class is not evidence the setup is right.
  It only shows both our run and theirs failed to train there. The open question is the other six rows.
  Now sweeping all 8 again at `lr=1e-3`.
- **2026-09-26 18:31** — `lr=1e-3` sweep done. Both GRU **+attention** rows reproduce (+4.1, +0.9).
  Both GRU **−attention** rows now overshoot the paper by ~33 points, because at a sane learning rate
  a GRU without attention trains fine (69-72%) where the paper reports 36-39%. So Table 8's GRU block
  is only reproducible by switching learning rate between rows that supposedly share one setup.
- **2026-09-26 18:33** — TCN is uniformly 12-18 points low. Likely cause: 4 dilated levels give a
  receptive field of just 31 tokens (`1 + 2·(1+2+4+8)`), which cannot "analyse complete sequences"
  as the paper describes for TCNs. Re-running the 4 TCN rows with 8 levels (dilations 1…128,
  RF 511) at `lr=1e-3`.
- **2026-09-26 18:38** — Receptive field was **not** the problem: 8 levels scored **50.25%** on
  `TCN no chunk, no attn`, *worse* than the 4-level 58.85%. The deeper stack is simply harder to fit
  in a single epoch. TCN remains unreproduced.
- **2026-09-26 18:42** — Table 10 kernel pushed to Kaggle (7 BERT variants, ~3 h, cheapest first so
  a truncated run still yields results). Running in parallel with the local TCN work — different
  hardware, no contention.
- **2026-09-26 18:51** — 8-level TCN sweep finished and made things worse (two rows collapsed to
  majority-class). Stopped tuning TCN. Table 8 is now closed with the finding above.
  Total local cost for all of Table 8: 20 training runs, ~75 min, zero Kaggle quota.

- **2026-09-27** — Table 10 sweep finished on Kaggle (7 variants, ~2.8 h). Six of seven reproduce.
  `No Transformers` measured 82.31% against a reported 41%; the no-BERT reinterpretation reproduces
  the paper's figure (37.16%, 1.9 min) and is almost certainly what was measured.
- **2026-09-27** — Retraining the best non-ChunkyBERT variant (`No Classifier`, 82.36%) with weight
  saving, so it can be evaluated on `stories_10k.jsonl`. The ablation sweep saved metrics only.

### Known issue

`results/results_table8_lr1e-3.csv` has a ragged schema — the `tcn_levels` column was added partway through,
so the 8-level rows carry one extra field the header does not declare. Read it positionally
(`variant, tag, val_acc` are the first three columns in both schemas) rather than with a plain
`pd.read_csv`. The Kaggle Table 10 script writes a consistent schema and is unaffected.
