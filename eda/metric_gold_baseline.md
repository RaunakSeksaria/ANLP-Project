# The NeuS framing-bias metric, reimplemented and baselined on gold data

Reproduce with `uv run python eda/framing_bias.py` (CPU, no network); raw numbers
in `metric_gold_baseline.json`. `--self-test` checks the lexicon load, the
calibration identity, the ordering property and the lemmatiser.

## Why it had to be reimplemented

`data/framing-bias-metric-main/metric_code.ipynb` is a sketch, not runnable code:

- `other_bias_lexicon_analysis(...)` is called but never defined — the auxiliary
  lexicon half of the metric is simply absent.
- `pred_path` and `test_srcs` are referenced but never assigned.
- `blue_score` is printed while its assignment sits commented out one line above.
- `re` and `wordnet_lemmatizer` are used but never imported.
- Four of ten cells are empty; cell 7 is the literal placeholder
  `gens = # load your generations here`, which is a syntax error.
- The VAD loader splits on `"\n"` and unpacks four fields per line, so it raises
  on the file's final line, which has no trailing newline.

`eda/framing_bias.py` implements the definition the notebook describes.

**Definition.** For a (generation, reference) pair, take
`calibrated = set(tokens(generation)) - set(tokens(reference))`, then over every
calibrated token present in NRC-VAD (19,974 entries) accumulate summed valence,
positive/negative valence mass around 0.5, and arousal for tokens with valence
> 0.65 (positive) or < 0.35 (negative). Report the per-instance mean. Subtracting
the reference's tokens is the "calibration" — a generation is charged only for
affect-laden words the neutral reference did not itself use.

**Two documented deviations.** The notebook strips punctuation *before* calling
`nltk.word_tokenize`, so on its own input that call degenerates to a whitespace
split; `re.findall(r"\w+")` is equivalent and removes the dependency.
Lemmatisation resolves against the lexicon being queried — WordNet's `_morphy`
noun rules are applied and a stripped form kept only when it lands in that
lexicon. WordNet's lemmatiser exists here only to make lookups hit; this does
the same job deterministically and needs no corpus download.

## Gold baseline

Every gold article scored as if it were a generation, against the human roundup.
n = 3,095 stories.

| | p_arousal | n_arousal | v_positive | v_negative | calibrated tokens | per-VAD-token p_arousal |
| --- | --- | --- | --- | --- | --- | --- |
| gold **left** article | 3.178 | 1.708 | 2.219 | 0.943 | 33.99 | 0.179 |
| gold **centre** article | 2.916 | 1.478 | 2.029 | 0.825 | 31.32 | 0.178 |
| gold **right** article | 3.079 | 1.584 | 2.155 | 0.880 | 32.99 | 0.180 |
| concat L+C+R | 8.083 | 4.299 | 5.624 | 2.384 | 83.97 | 0.179 |
| copy-centre baseline | 2.916 | 1.478 | 2.029 | 0.825 | 31.32 | 0.178 |
| identity (roundup vs itself) | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

The identity row is the floor and confirms the calibration is implemented
correctly. `concat L+C+R` is the naive extractive baseline any model must beat;
`copy-centre` is the cheap one — and note it already scores *below the left and
right articles*, so "always copy the centrist source" is a non-trivial baseline
on this metric that costs no training at all.

## The metric is largely a length proxy

Across all 9,285 gold articles:

```
pearson  p_arousal        vs n_tokens        0.7555
pearson  p_arousal        vs n_vad_tokens    0.8195
pearson  n_arousal        vs n_vad_tokens    0.5435
pearson  p_arousal+n_arousal vs n_vad_tokens 0.8811
```

And once you divide by the number of calibrated VAD tokens, the stances become
indistinguishable:

```
left    raw p_arousal 3.178    per VAD token 0.1767  (sd 0.0812)
centre  raw p_arousal 2.916    per VAD token 0.1763  (sd 0.0810)
right   raw p_arousal 3.079    per VAD token 0.1792  (sd 0.0790)
```

The left–centre gap is **0.0004 per token, against a standard deviation of
0.081** — about half a percent of one SD. The 9 % gap in the raw scores is
almost entirely explained by left articles contributing 34.0 calibrated tokens
against the centre's 31.3.

So the metric ranks partisan above centrist text, but not because partisan text
is more affect-dense. It ranks it higher because more of its vocabulary diverges
from the roundup. A plain count of calibrated tokens would produce the same
ordering.

**This matters directly for the proposal.** The interim plan names *neutrality by
deletion* — appearing neutral by dropping contested content rather than
reconciling it — as a failure mode to avoid. This metric rewards exactly that: a
model can lower its framing-bias score by generating fewer novel tokens,
regardless of tone. Optimising against it, or reporting it as the headline
neutrality number, would make the deletion failure mode invisible.

Three usable responses, in increasing order of ambition:

1. Report the per-VAD-token rate alongside the raw score. Cheap, and it makes
   the confound visible rather than load-bearing.
2. Compare only at matched generation length, or regress the score on length and
   report the residual.
3. Treat it as a contribution: the metric is from a NAACL 2022 paper and is
   still what this literature reports, so *"the standard framing-bias metric is
   confounded by output length, here is a normalised variant and what changes
   under it"* is a defensible result on its own — and it costs no GPU.

## Cross-corpus check: WCEP-10

Same metric, WCEP-10 test split, each document scored against its 25-word summary.

| | p_arousal | n_arousal | calibrated tokens | per-VAD-token p_arousal |
| --- | --- | --- | --- | --- |
| WCEP-10 test document | 17.623 | 10.135 | 187.26 | 0.177 |
| identity (summary vs itself) | 0.000 | 0.000 | 0.00 | 0.000 |

The raw score is ~5.5× the NeuS gold articles — entirely because WCEP documents
are ~5.6× longer after calibration (187 vs 33 tokens). The per-token rate is
0.177, statistically the same 0.178 seen everywhere else in this table,
including AllSides articles from a different decade and a different crawl.

That constant is the clearest evidence that the affect weighting carries no
corpus-level signal: **English news prose contains roughly 0.18 units of
positive arousal per affect-bearing token regardless of outlet, stance, corpus
or era.** On WCEP the metric is measuring nothing but the length gap between a
337-word article and a 25-word blurb.

## Auxiliary lexicons

Mean hits per instance among calibrated tokens:

| | bias lexicon | subj strong | subj weak | report verbs | hedges | assertives | implicatives | factives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gold left | 5.686 | 2.151 | 3.819 | 0.862 | 0.615 | 0.397 | 0.205 | 0.078 |
| gold centre | 5.346 | 1.746 | 3.385 | 0.837 | 0.552 | 0.394 | 0.174 | 0.059 |
| gold right | 5.583 | 2.083 | 3.583 | 0.878 | 0.564 | 0.414 | 0.221 | 0.074 |
| WCEP-10 doc | 30.831 | 10.452 | 20.983 | 4.489 | 2.933 | 1.893 | 1.241 | 0.387 |

The MPQA strong-subjectivity count is the one signal here that is *not* flat
under length: left 2.151 and right 2.083 against centre 1.746 — a 20–23 % gap
on a 8.5 % token-count gap. Normalised per calibrated token that is 0.0633 /
0.0631 / 0.0557, so the centre really is less subjective per word, unlike the
VAD arousal rate. If a lexical neutrality measure is wanted, MPQA
strong-subjectivity density looks like a better candidate than the paper's own
VAD arousal, and it is worth reporting alongside.

Factives and implicatives are too sparse (0.06–0.22 hits per instance) to
support any per-instance claim; use them only in corpus-level aggregates.
