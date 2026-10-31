# Fluency, hallucination and POLITICS on E2 long (default evaluation set)

E2 long = Raunak's `{left,right}_neutral_full1024.jsonl` (HF `RaunakSeksaria/anlp-project-datasets`, configs
`reframing_{left,right}_long`): one left or right article (≤ 1,024 LED tokens) and its Nemotron `n2` rewrite.
From 2026-10-30 it is the default set for every evaluation.

**Evaluation set:** the test split (1,008 left + 1,008 right), minus 152 articles that are also in the LED
models' training data (E2 short train) = **1,864 articles** (`inputs/e2long_test.jsonl`).

**Texts compared, per article:** the original body, the Nemotron rewrite (the target), and rewrites by the LED
neutralisers `side-neutral-full/{lambda0, lambda035, first_token_pooling}` (`RaunakSeksaria/anlp-neutral-led`),
generated with `qa_eval.py --stage gen`: input "title </s> body" up to 1,024 tokens (0.2% truncated; the models
were trained at 512), greedy, up to 1,024 new tokens.

## Results

**Fluency** (higher CoLA = better; lower perplexity / repetition / errors = better):

| Text | Words | Repeated word 3-grams % | Texts with > 10% repeated 3-grams | CoLA acceptable sentences % | GPT-2-large perplexity (median) | LanguageTool grammar / 100 words | LanguageTool errors / 100 words |
|---|---|---|---|---|---|---|---|
| Original | 451 | 1.9 | 2.3% | 96.5 | 12.4 | 0.03 | 0.22 |
| Nemotron rewrite | 390 | 1.5 | 0.3% | 97.7 | 11.4 | 0.02 | 0.17 |
| LED λ0 | 447 | 11.0 | 40.1% | 96.6 | 9.6 | 0.03 | 0.24 |
| LED λ0.35 | 462 | 12.4 | 46.2% | 96.4 | 9.2 | 0.03 | 0.25 |
| LED first-token | 421 | 8.9 | 33.2% | 96.5 | 10.2 | 0.03 | 0.23 |

**Hallucination** (lower = better; source = original title + body):

| Text | Unsupported sentences % (MiniCheck) | Texts with ≥ 1 unsupported sentence | New named entities % | New numbers / dates % | Texts with a new named entity |
|---|---|---|---|---|---|
| Nemotron rewrite | 4.3 | 41.7% | 2.1 | 3.8 | 37.7% |
| LED λ0 | 7.7 | 53.8% | 0.9 | 0.9 | 22.4% |
| LED λ0.35 | 8.1 | 56.7% | 0.9 | 0.9 | 22.4% |
| LED first-token | 7.0 | 52.3% | 0.9 | 1.0 | 22.5% |

Unsupported sentences, split by whether the text loops (> 10% repeated 3-grams):

| Text | Not looping | Looping |
|---|---|---|
| LED λ0 | 4.6% (1,116 texts) | 9.7% (748) |
| LED λ0.35 | 4.7% (1,003) | 9.9% (861) |
| LED first-token | 4.5% (1,245) | 9.3% (619) |
| Nemotron | 4.2% (1,859) | (5 texts) |

**POLITICS v2** (calibrated 4-seed ensemble) on the same 1,864 articles; full tables in
`results/politics/e2long_politics.md`:

| Text | Own side top label % | Recall L / R % | Mean P(own side) | Moved to Centre | Moved to other side |
|---|---|---|---|---|---|
| Original | 84.2 | 81.8 / 86.6 | 0.806 | — | — |
| Nemotron rewrite | 62.1 | 52.6 / 71.5 | 0.577 | 14.7% | 12.7% |
| LED λ0 | 63.1 | 58.0 / 68.1 | 0.592 | 18.6% | 7.0% |
| LED λ0.35 | 63.1 | 56.5 / 69.7 | 0.586 | 18.1% | 7.7% |
| LED first-token | 64.7 | 57.4 / 71.9 | 0.595 | 17.3% | 7.0% |

On all 19,783 E2 long articles (every split, articles also in Baly's corpus excluded): originals 85.3% own side,
Nemotron rewrites 63.4% (15.7% to Centre, 11.7% to the other side).

## Reading

- **The LED rewrites loop.** 33-46% of them repeat more than 10% of their 3-grams (originals 2%, Nemotron 0.3%);
  the worst degenerate into repeated phrases ("the 'donor' is not the same as the 'donor' …"). Their low
  perplexity is a symptom of this, not fluency: GPT-2 finds repeated text easy. Sentence-level grammar
  (CoLA, LanguageTool) does not catch it. Likely causes: greedy decoding with no repetition penalty, and outputs
  allowed up to 1,024 tokens while training targets were ≤ 512. A decoding rerun with `no_repeat_ngram_size=3` or
  `repetition_penalty` would test this cheaply.
- **Hallucination comes from the loops.** Where an LED rewrite does not loop, MiniCheck flags 4.5% of its sentences,
  the same as the Nemotron target (4.2%); looping rewrites double it. The LED models add fewer new entities and
  numbers than Nemotron (0.9% vs 2-4%): they copy more of the source.
- **Neutrality:** the LED rewrites sit at the Nemotron target's level on POLITICS (63-65% own side vs 62%), with
  more moves to Centre and fewer flips to the other side. λ0, λ0.35 and first-token differ by ≤ 1.6 points.
- Caveats: one seed per LED model; POLITICS was pretrained on BigNewsAlign, the source of E2 long, so originals are
  likely easier for it than rewrites. "New entity" counts are noisy (expanded abbreviations, possessives).

## Metrics

| Metric | Definition |
|---|---|
| Repeated word 3-grams | 1 − distinct / total word 3-grams in the text |
| CoLA acceptable | share of sentences (NLTK) with P(acceptable) > 0.5 under `textattack/roberta-base-CoLA` |
| Perplexity | GPT-2-large on the first 1,024 tokens of the text |
| LanguageTool | `language_tool_python` en-US matches per 100 words: GRAMMAR category; all categories except style, typography, casing, redundancy, plain English and spelling (spelling is mostly names; stored as `lt_spelling`) |
| MiniCheck | `lytang/MiniCheck-Flan-T5-Large` (MiniCheck @ `b58b9fa`): P(source supports sentence) per rewrite sentence, unsupported if < 0.5 |
| New entities | spaCy `en_core_web_sm` entities of the rewrite whose normalised text is absent from the source; named = person, org, place, group, facility, event, law, product, work; numbers = date, time, money, percent, quantity, cardinal |

## Files

| Path | What |
|---|---|
| `quality_eval.py` | `build` (inputs), `fluency`, `hallucination` (GPU), `languagetool` (CPU, needs Java), `report` |
| `inputs/e2long_test.jsonl` | The 1,864 test articles (qa_eval record format, `target_text` = Nemotron rewrite) |
| `out/gen_own_<model>.jsonl` | LED rewrites |
| `out/{fluency,halluc,languagetool}_<text>.jsonl` | Per-article metrics (MiniCheck probabilities per sentence in `support`) |
| `out/led_politics.csv` | POLITICS v2 on the LED rewrites (per-seed logits, ensemble P) |
| `results/quality_summary.json` | The fluency and hallucination tables |
| `results/politics/` | POLITICS v2 on all E2 long texts (`e2long_politics.csv`), tables (`.md`, `.json`), kernel log |
| `kaggle/{politics-e2long,quality-e2long}/` | The two Kaggle kernels as run |

Run: Kaggle 2×T4 on 2026-10-30/31. POLITICS on 39,844 texts took 13 min. The quality kernel took 10.1 h, of which
MiniCheck was ~9.5 h (≈ 2.3 sentences/s per T4 in fp32): for reruns use a faster MiniCheck variant or a sample.
Kaggle inputs: private dataset `harshitlalwani/e2long-eval` (these inputs, `evaluation/long10k/texts.jsonl`, code)
and `harshitlalwani/politics-ideology-classifier` (weights).
