# Neutral multi-source summaries with Gemini

Takes clusters of articles covering the same event — one per outlet, left /
centre / right or any other set of leanings — and generates a single neutral
account that carries the union of the facts and none of the
[16 AllSides bias types](https://www.allsides.com/media-bias/how-to-spot-types-of-media-bias).

## Setup

```bash
uv sync
cp .env.example .env        # then fill in GEMINI_API_KEY
```

## Input

One JSON object per line. Any number of articles per cluster.

```json
{"story_id": "1027",
 "articles": [{"stance": "left",   "publisher": "...", "title": "...", "body": "..."},
              {"stance": "center", "publisher": "...", "title": "...", "body": "..."},
              {"stance": "right",  "publisher": "...", "title": "...", "body": "..."}]}
```

Different key names are a config change, not a code change:
`--set id_field=cluster_id --set body_field=text`.

## Running

```bash
# see the exact prompt for the first cluster, no API call
uv run python scripts/generate_neutral.py --in clusters.jsonl --out out.jsonl --dry-run

# generate
uv run python scripts/generate_neutral.py \
    --in clusters.jsonl --out runs/gemini/summaries.jsonl \
    --model <model-id> --temperature 0.0 --workers 4
```

Output is one record per cluster, appended as each finishes. Rerunning the same
command skips every `story_id` already in the file, so a killed run resumes and a
finished one costs nothing.

`--dry-run`, `--limit N` and `--ids a,b,c` are the cheap ways to check the prompt
before spending on a full run.

## Consistency

Temperature and every other sampling parameter is a flag with a recorded value,
defaulting to `temperature=0.0`, `top_k=1`, `seed=42`, thinking disabled. Each
output record carries the `params` and `prompt_version` that produced it.

There is **no response cache**: the same config run twice really calls the API
twice, and the two runs will differ in wording because Gemini is not
bit-deterministic even at temperature 0. To find out how much that matters:

```bash
uv run python scripts/check_consistency.py \
    --run runs/gemini/summaries.jsonl --in clusters.jsonl \
    --regenerate --sample 30 --seed 7 --similarity mymodule:score
```

It re-runs a random sample under the *recorded* config of the first run and
reports similarity, a different-story control, and the mean absolute change in
every evaluation metric. Read the control: two summaries of one event share its
proper nouns whatever the model does, so a same-story score is meaningless
without it. Stable metrics under reworded text is the result you want.

`--similarity` takes `module:function` or `path/to/file.py:function`, either
`score(a, b) -> float` or `score(pairs) -> [float]` (detected from the
signature). Omit it and you get token Jaccard, labelled as the placeholder it is.

## Retries

Three separate loops, three separate budgets:

| Loop | Triggers | Default | Behaviour |
| --- | --- | --- | --- |
| transient | timeout, 429, 5xx | 3 | jittered exponential backoff, honours the server's retry delay. A 429 pauses every worker, not just the one that saw it. |
| malformed | empty, unparseable, truncated, absurd length | 3 | regenerates with a repair note. A safety block is recorded, not retried. |
| revision | an evaluation metric fails | 0 (off) | re-asks with the previous draft and the failing measurements in the prompt. |

A bad key or an unknown model fails immediately and aborts the run — retrying
those only wastes time.

## The evaluation loop

`--max-revisions 2` turns it on. `neutralize/evaluate.py` scores each draft and
returns human-readable failures, which go back into the prompt:

> The summary introduces 11 emotionally charged words that appear in none of the
> source articles (limit 8). Rewrite them in neutral terms. They include:
> slammed, outrage, shocking…

The metrics reuse `eda/framing_bias.py` and cover introduced affect vocabulary,
bias-lexicon hits, per-source coverage (bias by omission), stance parity, shared
numbers retained, and length. Thresholds live in `GenConfig` and are settable
with `--set max_stance_parity_gap=0.05`.

To use your own scorer instead: `--evaluator mymodule:my_evaluate`, returning a
`{name: number}` dict. Thresholds whose metric your dict omits are skipped.

**Calibrate the thresholds before turning revisions on.** The committed defaults
are guesses. Run over the NeuS test split first, where a human neutral roundup
exists to score against.

## Prompt

`neutralize/prompt.py` is the only place bias rules are written down. Sources are
anonymised by default (`Source A / B / C`, shuffled per story, no publisher, no
stance) — telling the model which article is the left one invites it to *balance*
rather than neutralise. `--reveal-stance` and `--reveal-publisher` turn the labels
on; that comparison is a one-flag ablation.

`PROMPT_VERSION` is recorded in every record. **Bump it when you edit the
prompt**, or a set of summaries becomes impossible to attribute to the wording
that produced it.
