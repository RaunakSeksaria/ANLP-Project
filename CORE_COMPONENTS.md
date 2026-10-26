# Core code components of the project

The project's most important code across all branches, ranked from most to least central. Each entry
gives where it lives, why it matters, a trimmed excerpt, and an explanation.

**Scope.** Branches surveyed (remote state as of 2026-10-12): `main`, `data`, `data-finding-out`,
`manas`, `mid-eval`, `embedding-evals`, `harshit`.

**Ranking principle.** Code that defines the method (what the model is and what it learns) comes
first, then the data that shapes it, then how it is evaluated, then infrastructure.

---

## Branch map

| Branch | Role | Status |
|---|---|---|
| **`mid-eval`** | Integration branch: data generation, splits, training, embedding evaluation and generation evaluation, merged | **Reference copy; paths below are relative to it unless noted** |
| `manas` | LED training code (as `src/`) | Core training files are byte-identical to `mid-eval`'s `training/src/`. Newer additions are launch scripts (Ada, LED-large, token filter) |
| `data` | Dataset generation (as `neutralize/`, `scripts/`) | Same pipeline as `mid-eval/data_generation`, plus later releases (long-article "full1024" set, publisher-negative variant) |
| `embedding-evals` | Embedding-level evaluation and its results | Code merged into `mid-eval/embedding_eval`; results compiled in `results/RESULTS_COMPILATION.md` |
| `harshit` | Political-bias classifiers and output evaluation | Merged into `mid-eval/generation_eval`, which is a slightly older snapshot of this branch |
| `data-finding-out` | Early exploration: dataset profiling, NeuS baseline, framing-bias metric | Historical |
| `main` | Skeleton (`main.py`, `pyproject.toml`) | Essentially empty |

---

## The project in two sentences

An LED encoder-decoder rewrites a partisan news article into a neutral one. During training, a
triplet loss on its encoder pulls together reports of the *same event* from opposite sides and pushes
apart *different events* from the same side or outlet. The hope is that the model learns what
happened rather than how a given outlet tells it.

Two experiments (from the `mid-eval` README):

| | Input → target | Train / val / test |
|---|---|---|
| **E1** | left or right article → the event's multi-source neutral article | 16,000 / 2,000 / 2,000 |
| **E2** | one article → the same article rewritten without framing | 12,484 / 1,778 / 1,775 |

**Shared settings.**
- *Model and optimiser:* LED-base, AdamW at 3×10⁻⁵, effective batch 8.
- *Decoding:* greedy, up to 512 tokens.
- *Triplet loss:* margin 0.2, weight λ ∈ {0, 0.1, 0.2, 0.35}.

---

## 1. The joint model: generation loss + λ · triplet loss

**Where:** `training/src/models/led_triplet.py` (213 lines) and `training/src/models/pooling.py`
(47 lines).
**Why it ranks first:** this is the method. Every other component exists to feed it or to measure it.

```python
class LEDTripletModel(nn.Module):
    """Wrap one LED instance without adding a projection or a second encoder."""

    def forward(self, *, anchor_input_ids, anchor_attention_mask, anchor_global_attention_mask,
                labels, triplet_valid_mask, triplet_lambda, triplet_margin,
                positive_input_ids=None, ..., negative_input_ids=None, ...):
        # 1. Ordinary seq2seq: anchor article in, neutral target out.
        generation = self.led(input_ids=anchor_input_ids, attention_mask=anchor_attention_mask,
                              global_attention_mask=anchor_global_attention_mask,
                              labels=labels, return_dict=True)
        generation_loss = generation.loss
        if triplet_lambda == 0.0 or valid_count == 0:
            return JointForwardOutput(..., total_loss=generation_loss, ...)

        # 2. Article vectors from the SAME encoder: anchor reuses the generation pass.
        anchor_z, _   = masked_mean_pool(generation.encoder_last_hidden_state, anchor_attention_mask)
        positive_z, _ = self._encode(positive_input_ids, positive_attention_mask, ...)
        negative_z, _ = self._encode(negative_input_ids, negative_attention_mask, ...)

        # 3. Triplet margin loss, then the joint objective.
        triplet_loss, ... = masked_triplet_margin_loss(anchor_z, positive_z, negative_z,
                                                       triplet_valid_mask, triplet_margin)
        total = generation_loss + triplet_lambda * triplet_loss
```

```python
def masked_mean_pool(hidden_states, attention_mask):
    mask = align_pooling_mask(hidden_states, attention_mask).to(dtype=torch.float32)
    raw = (hidden_states.float() * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    normalized = F.normalize(raw, p=2, dim=-1)        # unit-length article vector
    return normalized, raw
```

**What it does.**
- **One model, two jobs.** A single `LEDForConditionalGeneration` is used both to generate the
  rewrite and as the encoder for the triplet loss. There is no separate encoder and no projection
  head, so the triplet gradient directly reshapes the representation the decoder reads from.
- **The article vector** is the mean of the encoder's token states over real tokens, normalised to
  unit length. Normalising means distances depend only on direction; squared distances range 0–4.
- **The anchor costs nothing extra:** its vector comes from the encoder pass already made for
  generation. The positive and negative each need one more encoder pass, which is why triplet
  training costs more than the λ = 0 baseline.
- **The triplet loss**, `max(0, ‖a − p‖ − ‖a − n‖ + margin)`, is zero once the negative is at least
  `margin` farther from the anchor than the positive.
- `triplet_valid_mask` lets a batch contain examples with no valid triplet; they contribute
  generation loss only.
- **λ = 0 is the controlled baseline:** the same model, data and code with the triplet term switched
  off.

**Small details that matter.**
- `align_pooling_mask` pads the mask when LED internally pads the sequence to a multiple of its
  attention window. Without it the mask and hidden states would have different lengths.
- Pooling runs in float32 and raises an error on all-padding rows or non-finite vectors, instead of
  silently training on garbage.

---

## 2. Triplet construction: what the loss is taught to treat as "same" and "different"

**Where:** `training/src/scripts/prepare_side_neutral.py` (`make_triplets`, the default E2 triplets)
and `training/src/scripts/make_negative_variants.py` (alternative negative policies). Both are
validated when loaded by `training/src/data/triplet_dataset.py` (722 lines).
**Why it ranks second:** the triplet loss is only as meaningful as its positives and negatives. This
code decides what "same event" and "different" mean.

```python
def make_triplets(records, token_ids, seed):
    for anchor in records:
        # POSITIVE: the opposite side's article about the same event
        candidates = {row["input_article_id"]: row
                      for event in anchor["event_ids"] for row in by_event[event]
                      if row["input_ideology"] != anchor["input_ideology"]
                      and canonical(row["input_text"]) != canonical(anchor["input_text"])}
        positive = rng.choice([candidates[k] for k in sorted(candidates)])

        # NEGATIVE: same side, same split, a DIFFERENT event (not textually identical)
        pool = pools[anchor["split"], anchor["input_ideology"]]
        for offset in range(len(pool)):
            row = pool[(start + offset) % len(pool)]
            if row["event_id"] == anchor["event_id"]:
                continue
            ...
            negative = row; break
```

```python
POLICIES = ("publisher", "ideology-other-publisher")

def eligible(policy, anchor, candidate, publisher):
    same_publisher = publisher[candidate["input_article_id"]] == publisher[anchor["input_article_id"]]
    if policy == "publisher":
        return same_publisher                          # same outlet, different event
    return candidate["input_ideology"] == anchor["input_ideology"] and not same_publisher
```

**What it encodes.**
- **Positive = same event, opposite side.** The loss pulls a left and a right article about the same
  event together, so the representation should keep *what happened* and discard *whose framing*.
- **Negative = different event from the same side** (the default ideology-anchored variant). The
  loss pushes apart articles that share a slant but not an event, so slant alone must not make two
  articles look alike.
- **The publisher variant** uses a negative from the *same outlet*, which specifically penalises
  outlet style. That is the publisher-leakage problem the evaluation side measures.
- **Same split only,** and never textually identical to the anchor or positive, so nothing leaks
  across splits.

`make_negative_variants.check()` verifies that the variant changes *only* the negatives: every other
field must be byte-identical. A run on the variant therefore differs from the base run in one factor.

**Known weakness** (from the `embedding-evals` audit): "same event" comes from source pairing IDs. In
a manual check of 20 sampled triplets, 11 positives were the same story, 5 related developments, 3
different events, and 1 topic-only. Positives are noisy.

---

## 3. The neutral targets: what "neutral" means to the model

**Where:** `data_generation/neutralize/prompt.py` (453 lines), plus the generation driver
`data_generation/scripts/generate_neutral.py`.
**Why it ranks third:** the model learns to imitate these targets. Whatever Nemotron-3-Ultra was
told counts as neutral is, in effect, the model's definition of neutral.

```python
NEUTRALISE_VERSION = "n2"   # recorded in every output record

NEUTRALISE_SYSTEM = """\
You are a copy editor. You are given ONE news article. Rewrite it so that it
reports the same events with the framing removed.
...
THE ONE HARD RULE: ADD NOTHING
...
KEEP EVERY FACT
Removing framing is not removing content. ...
WHAT TO REMOVE
These are AllSides' types of media bias. Work through the article and strip each one ...
1.  SPIN ...   3. SENSATIONALISM ...   6. WORD CHOICE ...   11. MIND READING ...
ATTRIBUTION VERBS
Use `said`, `wrote`, `reported`, `stated`, `told`. Never `admitted`, `conceded`, `claimed` ..."""

NEUTRALISE_SCHEMA = {"properties": {
    "neutral": {"type": "string"},
    "removed": {"type": "array", "items": {"type": "string"},
                "description": "One short PLAIN SENTENCE per edit, naming the bias type ..."}}}
```

**What it encodes.**
- **Two tasks, two prompts.** `fuse` (several outlets → one neutral account, used for E1) and
  `neutralise` (one article → the same article without framing, used for E2).
- **"Neutral" is made operational with AllSides' 16 bias types.** Each one is turned into an
  instruction the model can apply. The evaluation side's Jev bias-type questions use the same
  taxonomy, so training and evaluation share one definition.
- **"Add nothing" and "keep every fact"** defend against the two easy ways to look neutral:
  inventing balancing context, and deleting content.
- **The `removed` field** makes the model list each edit it made, an audit trail checkable per
  example.
- **Every output records the prompt version** (`v3`, `v4-range`, `n2`), so any dataset can be traced
  back to the exact wording that produced it.

---

## 4. The training loop

**Where:** `training/src/training/train_joint.py` (1,703 lines; the core is `train_joint_epoch`).
**Why it matters:** it turns component 1 into checkpoints. Most of the file is evaluation and
logging; the core is short.

```python
for microbatch_index, cpu_batch in enumerate(progress, 1):
    batch = _move_tensors(cpu_batch, device)
    output = model(**_model_inputs(batch),
                   triplet_lambda=config.triplet_lambda, triplet_margin=config.triplet_margin)
    if not bool(torch.isfinite(output.total_loss)):
        raise FloatingPointError(...)
    expected_total = output.generation_loss + config.triplet_lambda * output.triplet_loss
    if not torch.allclose(output.total_loss, expected_total, rtol=1e-6, atol=1e-7):
        raise ValueError("joint objective mismatch ...")
    (output.total_loss / group_size).backward()          # gradient accumulation
    if not (boundary or final):
        continue
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
```

**What to notice.**
- **Gradient accumulation:** each micro-batch's loss is divided by the accumulation group size, so
  several micro-batches add up to one optimiser step at an effective batch of 8.
  `group_size` shrinks for the last, partial group so it isn't under-weighted.
- **Defensive checks:** non-finite losses or gradients stop the run; the code re-checks that
  `total = generation + λ·triplet`.
- **Per-epoch evaluation** (elsewhere in the file):
  - teacher-forced loss on validation;
  - free generation (with a repeated-trigram check for degenerate output);
  - triplet distances;
  - early stopping on validation loss (patience 3).
- **Reproducibility:** every run records hashes of its input data (`data_hashes`).

---

## 5. Leak-proof splits

**Where:** `data_generation/scripts/prepare_bignews.py` (`components`, `split_components`), and
`splits/` at the repo root (per-split example IDs, SHA-256 manifest, `build.sh`, `check.py`).
**Why it matters:** BigNewsAlign events share articles (the same wire story under several event IDs).
A naive random split by event would leak articles between train and test.

```python
def components(events):
    """Union-find over events that share an article text."""
    parent = {e: e for e in events}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    by_art = defaultdict(list)
    for eid, arts in events.items():
        for a in arts:
            by_art[ahash(a["body"])].append(eid)
    for ids in by_art.values():
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb: parent[ra] = rb
    ...

def split_components(events, ratios=(0.8, 0.1, 0.1), seed=SEED):
    """Largest-first bin packing, so no article crosses a split boundary."""
    comps = sorted(components(events), key=len, reverse=True)
    for comp in comps:
        name = max(quota, key=lambda k: quota[k] - have[k])   # split with most room left
        ...
```

**What it does.**
- **Grouping:** events that share any article body are merged into one connected component, using
  union-find with path compression.
- **Assignment:** whole components go to train, validation or test. The largest go first to whichever
  split is furthest below its quota, so the 80/10/10 ratios hold.
- **Inheritance:** every later dataset keeps the `split` field this assigns.
- **Verification:** `splits/build.sh` rebuilds each prepared file from Hugging Face and checks it
  byte-for-byte against the SHA-256 manifest. Anyone can reproduce exactly the data a model saw.

---

## 6. Event selection: why the articles are short

**Where:** `data_generation/scripts/make_clusters.py` (183 lines).

```python
"""Turn the BigNewsAlign L/C/R export into cluster JSONL for the generator.
    uv run python scripts/make_clusters.py \
        --in data/final/bignewsalign_max250_LCR.jsonl \
        --out data/final/clusters_10k_shortest.jsonl --limit 10000
...
--sort shortest (the default) takes the n events with the fewest total words
across their articles, which is both the cheapest to generate and the least
likely to be truncated."""
```

**Why it matters.**
- **The source is an export already capped at 250 words per article** (`max250`), and the 10,000
  events chosen are the *shortest* ones.
  - That is why `stories_10k` articles have a median of 135 words and never exceed 250.
  - It also explains why the long-document classifiers behave differently on this data.
- **Deduplication:** about 30% of the raw export is exact-duplicate events under different IDs.
- **Long-article set:** a later release on the `data` branch ("full1024", 276–862 words,
  `make_long_pairs.py` / `make_full_pairs.py`) exists because of this limitation.
- **Data provenance:** the data comes from **BigNewsAlign**, the aligned corpus released with the
  POLITICS paper. POLITICS was pretrained on it, so any POLITICS-based evaluation of this data is
  contaminated.

---

## 7. Generation (inference)

**Where:** `training/src/scripts/generate_article.py` (173 lines), with `evaluate_joint.py` for test
metrics.

```python
source_text = article if not title else f"{title.strip()} {tokenizer.sep_token} {article}"
encoded = tokenizer(source_text, truncation=True, max_length=max_input_length,
                    padding=False, return_tensors="pt")
global_attention_mask = torch.zeros_like(attention_mask)
global_attention_mask[:, 0] = 1                    # LED: first token attends globally
model = load_led_triplet(checkpoint, attention_window=attention_window).to(device)
settings = {"num_beams": 1, "do_sample": False, "max_new_tokens": max_new_tokens,
            "early_stopping": False}
with torch.inference_mode():
    generated = generate_with_cache(model.led, batch, device, "fp32", settings)
```

**What to notice.**
- **Global attention.** LED uses sliding-window attention; token 0 gets global attention so every
  position can see a summary slot. The same convention is used in the training dataset
  (`baseline_dataset.py`).
- **Input format:** `"{title} </s> {body}"`, truncated at 512 tokens (`format_article` in
  `triplet_dataset.py`). Any evaluation that regenerates rewrites must reproduce this exactly; the
  generation-evaluation code in `generation_eval/led_eval/generate.py` does.
- **Greedy decoding** makes outputs deterministic, so runs are comparable across λ.

---

## 8. Embedding evaluation: did the triplet loss reshape the representation?

**Where:** `embedding_eval/scripts/` on `mid-eval` (also the `embedding-evals` branch):
- `evaluate_alignment_uniformity.py`
- `verify_and_evaluate_triplets.py`
- `evaluate_ideology_clustering.py`

**Why it matters:** this checks the method's direct effect, the geometry of the encoder's article
vectors, before asking anything about generated text.

```python
def metrics(z, selected, pairs, labels, sampling, seed, repetitions):
    distances = pdist(z[selected], metric='sqeuclidean')
    uniformity = float(np.log(np.exp(-2 * distances).mean()))      # lower = more spread out
    positive_distances = np.sum((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2, axis=1)
    alignment = float(positive_distances.mean())                   # lower = positives closer
    # bootstrap: resample ARTICLES (not the dependent pairwise distances) within ideology strata
```

```python
kmeans = KMeans(n_clusters=2, random_state=seed, n_init=20)
ari = adjusted_rand_score(labels, predicted)          # do 2 clusters recover left vs right?
silhouette = silhouette_score(z, labels, metric="cosine")
```

**The three measurements.**
- **Alignment and uniformity** (Wang & Isola's representation metrics):
  - *alignment* asks whether same-event, cross-ideology pairs are close;
  - *uniformity* asks whether vectors spread over the sphere rather than collapsing.
- **Triplet separation:** for held-out triplets, the gap `d(anchor, negative) − d(anchor, positive)`.
- **Ideology clustering:** whether the vectors split by ideology on their own. For a
  framing-invariant encoder, they shouldn't.

**Results** (`embedding-evals:results/RESULTS_COMPILATION.md`, 11,855 original articles):

| | λ = 0 | λ = 0.35 |
|---|---|---|
| Triplet ordering success (negative farther than positive) | 73.0% | **98.9%** |
| Mean triplet gap (squared distance) | 0.087 | **0.654** |
| Alignment (lower = positives closer) | **0.309** | 0.376 |
| Uniformity (lower = more spread) | −0.76 | **−1.97** |
| Ideology clustering (adjusted Rand index) | 0.005 | 0.0004 |

- **The triplet loss works as designed:** negatives are pushed far away and the space spreads out.
  Note that positives also drift slightly *apart* (alignment worsens). The gain comes from pushing
  negatives away, not from pulling positives together.
- **Neither model's vectors cluster by ideology** (adjusted Rand index ≈ 0).

---

## 9. Generation evaluation: is the generated text neutral?

**Where:** `generation_eval/` on `mid-eval`. The current version is on `harshit`, at the top level.
Documented in `EVALUATION_WALKTHROUGH.md` and `CODE_GUIDE.md`; summarised here.
**Why it ranks below the embedding evaluation:** it measures the downstream outcome, but its
instruments turned out to be weak (they largely recognise outlets). Its main contribution is
knowing how far those instruments can be trusted.

The single most reusable piece is the adversarial gradient-reversal layer in `baly/baly.py`. It is
also the most direct alternative to the project's triplet approach for removing outlet information:

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)              # identity on the way forward

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None        # flip the gradient on the way back

# in forward():  return self.head(self.drop(h)), self.media_head(GradReverse.apply(h, lam))
```

**Components.**
- **Ideology classifiers:**
  - ChunkyBERT, from its authors' code;
  - the rebuilt comparison models (`replication/variants.py`);
  - Baly et al.'s BERT models, including adversarial adaptation (above) and a 73-way outlet
    classifier.
- **Jev zero-shot judgments** (`jev/`):
  - ideology;
  - AllSides bias types, the same taxonomy as the training prompt in §3.
- **VAD** emotional-tone lexicon scores.
- **QA-based content preservation** (`qa_eval/`; written, not yet run in full).

**Key finding.** The trained ideology classifiers mostly identify the outlet rather than the slant.
They score 82 → 37 macro-F1 when test outlets are unseen. Rewrites therefore have to be judged with
paired comparisons and with bias-type questions, not with one classifier's verdict.

---

## 10. Infrastructure

| Where | What |
|---|---|
| `scripts/kaggle/runner.py` (`mid-eval`; `kaggle/runner.py` on `manas`) | Runs joint training on Kaggle GPUs |
| `scripts/kaggle/hf_upload/upload_to_hf.py` | Uploads checkpoints to `RaunakSeksaria/anlp-neutral-led`; refuses to overwrite a run and rebuilds the model card from all runs |
| `scripts/ada/ada_train_side_neutral.sh` (`manas`: also LED-large via `MODEL`) | Slurm launcher for the IIIT Ada cluster |
| `data_generation/scripts/*watchdog*.sh`, `supervise_neutralise.sh` | Restart long API generation jobs when the endpoint drops |
| `training/tests/`, `embedding_eval/tests/` | Unit tests for datasets, joint pipeline and metrics |

---

## 11. Historical: `data-finding-out`

| Where | What |
|---|---|
| `eda/profile_datasets.py`, `eda/REPORT.md` | Profiled candidate corpora (NeuS, AllSides/Baly, WCEP-10, WNC). Concluded that only NeuS had events, publishers and neutral targets together, but it was tiny and made of excerpts. This led to building BigNewsAlign-based data instead |
| `eda/framing_bias.py`, `docs/eda-evaluation.md` | Reimplemented the NeuS framing-bias metric and showed it correlates 0.88 with output length. It rewards writing less, which is why it isn't used |
| `scripts/train_neus.py`, `scripts/probe.py` | First baseline (multi-source summarisation) and a publisher probe. They produced the early "triplet loss cuts publisher recoverability" result that motivated the current design |

---

## Reading path

For someone new to the code, in this order:

1. §1 `led_triplet.py`: the method in about 100 lines of logic.
2. §2 `make_triplets`: what the method is asked to learn.
3. §3 the neutralise prompt: what the target text looks like.
4. §8 embedding evaluation and its results table: what the method achieved.
5. §5 splits, §4 training loop, §7 generation: the rigour around it.
6. §9 and `EVALUATION_WALKTHROUGH.md`: why judging neutrality of output text is still open.
