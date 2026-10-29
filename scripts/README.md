# scripts/

Data pipeline for the neutral article generation project. Everything here is a
CLI; run it with `uv run python scripts/<name>.py --help`.

The library these depend on is `neutralize/` (prompts, API clients, config,
evaluation). These scripts are the entry points.

---

## Pipeline

```
  data/final/bignewsalign_max250_LCR.jsonl        raw export, 1 article per line
        |
        |  make_clusters.py           dedupe, group into events, shortest-first
        v
  data/final/clusters_pool.jsonl                  1 event per line, L + C + R
        |
        +--> prepare_bignews.py       L+R in, CENTRE article out  -> data/prepared_bna/
        |
        +--> generate_neutral.py      generate a neutral article  -> runs/nemotron/*.jsonl
        |         |
        |         +--> prepare_llm_targets.py     L+R in, GENERATED article out
        |         |                                            -> data/prepared_llm/
        |         +--> build_release_jsonl.py     one self-contained file
        |                                            -> neutral_articles_10k.jsonl
        |
        +--> make_single_article_inputs.py        one article per record, per stance
                  |                               -> data/final/single_{left,right}_10k.jsonl
                  +--> generate_neutral.py --task neutralise

  data/prepared/articles.jsonl (NeuS release)
        |
        |  prepare_neus_test.py       L+R in, HUMAN roundup out
        v
  data/prepared_neus_test/                        the only human gold, 274 test items
```

Run `validate_corpus.py` after building any corpus. Every check in it exists
because that failure actually happened here.

---

## The scripts

### Corpus building

| script | what it does |
| --- | --- |
| `make_clusters.py` | Raw export → event clusters. Deduplicates (30% of the raw export is exact-duplicate events under different ids), filters copy-tasks, takes the shortest N. |
| `prepare_bignews.py` | Builds `data/prepared_bna/` — L+R in, **centre article** out. The interim stand-in used before generated targets existed. Also emits `triplets.jsonl` and the probe splits. |
| `prepare_llm_targets.py` | Builds `data/prepared_llm/` — L+R in, **generated neutral article** out. Reuses `prepared_bna`'s splits verbatim so the two differ only in the target. |
| `prepare_neus_test.py` | Builds `data/prepared_neus_test/` from the NeuS release — L+R in, **human roundup** out. 274 test items, 250 publishers. |
| `make_single_article_inputs.py` | Expands clusters to one article per record, one file per stance, for the neutralise task. |

### Generation

| script | what it does |
| --- | --- |
| `generate_neutral.py` | The driver. Resumable, checkpointed per record, three separate retry budgets. `--task fuse` (many articles → one) or `--task neutralise` (one article → same article, framing removed). `--provider gemini\|nvidia`. |

### Packaging and checking

| script | what it does |
| --- | --- |
| `validate_corpus.py` | Exits non-zero on misalignment, stray line breaks, duplicate ids, target-in-source leakage, or articles crossing a split. **Run after every rebuild.** |
| `export_bna.py` | Writes the per-split JSONL for `data/prepared_bna/` and regenerates its README with live numbers. |
| `build_release_jsonl.py` | Joins inputs + generated articles + splits into one root-level file for handoff. |
| `audit_summaries.py` | Quality audit of generated text: structural defects, copy-tasks, template collapse, stance coverage. |
| `make_hard_subset.py` | Scores every original (and generated target) with the POLITICS v2 ensemble and keeps the articles most confidently on their own side → `hard-test/`. Scoring runs on Kaggle (`kaggle_hard_subset/`) or Ada (`ada_hard_subset.sh`); `select` runs on CPU in seconds. |
| `check_consistency.py` | Re-runs a random sample and compares against a previous run. `--similarity module:func` takes your own scorer. |
| `ada_sync.sh` | rsync the repo to the cluster. |

---

## Running a generation job

Fusion (many articles → one neutral article):

```bash
uv run python scripts/generate_neutral.py \
  --provider nvidia --api-key-env NEMOTRON_3_ULTRA_KEY \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --in data/final/clusters_corpus10k.jsonl \
  --out runs/nemotron/out.jsonl \
  --workers 14 --rpm 40 --thinking-level HIGH \
  --temperature 1.0 --top-p 0.95 --stop-after 5
```

Neutralisation (one article in, same article out). The two stances are fully
independent — separate input, output and API key — so run them as two processes:

```bash
uv run python scripts/generate_neutral.py --task neutralise \
  --provider nvidia --api-key-env NEMOTRON_3_ULTRA_KEY \
  --in data/final/single_left_10k.jsonl \
  --out runs/nemotron/neutralised_left.jsonl  ...

uv run python scripts/generate_neutral.py --task neutralise \
  --provider nvidia --api-key-env NEMOTRON_3_ULTRA_KEY_2 \
  --in data/final/single_right_10k.jsonl \
  --out runs/nemotron/neutralised_right.jsonl ...
```

Always `--dry-run` first: it renders the prompt for the first record and exits
without an API call.

---

## Things that will bite you

**Splits are by connected component, not random.** Most events share an article
with another event — 11,576 duplicate article occurrences across the 10k corpus.
A random re-split puts the same text in train and test. Use the `split` field as
provided. As built, **zero** article bodies cross a split.

**Duplicates still exist inside train.** 16,000 training slots from 8,300
distinct articles; one article appears 32 times, each with a different target.
That is not leakage but it is a strong overfitting driver, and it is unfixed.

**Line-break characters are not just `\n`.** Article bodies contain `\r` and
` `. `Path.read_text()` converts `\r` to `\n` and `str.splitlines()` breaks
on ` `, either of which silently shifts every later target against its
source — training proceeds on mismatched pairs with no error, and `wc -l` shows
nothing wrong. All writers here strip
`"\n\r\v\f\x1c\x1d\x1e\x85  "`. If you write your own, do the same.

**The generated articles are longer than their inputs.** Median 215 words out
against a 142-word mean input article — this is fusion, not compression. ROUGE
will not behave like a summarisation benchmark.

**`--stop-after` aborts the run, and unattempted clusters are dropped, not
recorded.** An earlier version wrote a failure row for every remaining cluster
and the watchdog restarted into it: 188,000 junk records for 740 real ones. If
you see a run stop early, the endpoint is unhealthy — fix that, then resume.

**Publisher determines stance completely** in this corpus (11 outlets, each with
one stance). "Publisher leakage" and "stance leakage" are the same measurement
here. NeuS, with 250 publishers, is the only out-of-distribution check.

---

## Not in this folder

The runner and watchdog shell scripts live in `scratch/`, which is
git-excluded — so `run_neutralise.sh`, `watchdog_neutralise.sh` and friends are
**not committed**. They are thin wrappers that pin the flags for a production
run. Move them here if you want them under version control.

`dead_code/` holds the earlier NeuS pipeline (`train_neus.py`, `probe.py`,
`prepare_data.py`, `eval_generations.py`). Also git-excluded. `prepare_data.py`
is still the thing that builds `data/prepared/articles.jsonl`, which
`prepare_neus_test.py` reads.
