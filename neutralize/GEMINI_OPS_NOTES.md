# Gemini ops notes — measured numbers from a completed bulk run

Added 2026-09-20. Operational findings carried over from a separate project
(`Geopolitical_Research`) that ran **477 Gemini calls / 2.36M tokens** to completion with a
full JSONL archive of every call. **Measured** = from that archive. **Inferred** = reasoning
from it, not a measurement.

The point of this file is the short list in §2. `client.py` already does almost everything the
generic advice would tell you to do — shared client, jittered exponential backoff, a run-wide
429 pause, fatal-vs-transient status classification, `finish_reason` handling, structured
output with a schema, safety-block detection, pinned `thinking_budget`, and per-record
provenance. §1 just gives you numbers to cite for choices already made.

---

## 1. Corroboration for what's already here

| choice in this repo | measured elsewhere |
|---|---|
| `thinking_budget = 0`, commented "the main source of drift and cost" | Correct on cost: thinking ran at default and consumed **432,480 tokens = 36.5% of all output tokens**, mean 918/call, median 378, **max 3,265**. Highly variable and invisible unless requested. |
| README: "Gemini is not bit-deterministic even at temperature 0" | Correct, and worse than implied: five byte-identical calls, 40 items, `temperature=0.0` → **0/40 identical across all five runs**, mean spread 0.085, max 0.20. |
| Sources shuffled per story (`shuffle_seed`) so the model can't learn a position convention | Order effects are real and measurable: within a 100-item call, scores drifted **+0.0237** from the first 20 to the last 20 items (paired over 460 batches, t = 5.90). Generation order biases output on its own. See §2.4. |
| 400-on-thinking → disable and continue; 404/401/403 → `FatalAPIError` | Well judged. In that project a backfill lost **117 of 117 calls** to `404 NOT_FOUND` — retrying a 404 three times is pure waste. See §2.5. |

## 2. The actual deltas

### 2.1 `temperature=0.0` + `top_k=1` is right for *this* pipeline and wrong for generating synthetic data

This is the one that matters if the plan is to use Gemini to produce artificial training data.

The current defaults are tuned for a **measurement instrument**: you want two runs of the same
config to differ as little as possible so `check_consistency.py` is interpretable. Entirely
correct for that.

Generating a synthetic *dataset* inverts the objective. `temperature=0.0, top_p=1.0, top_k=1,
seed=42` is greedy decoding with a fixed seed — it maximises the chance that many clusters come
back in the same register, the same sentence shapes, the same transitions. And per §1 you don't
even get determinism in exchange. Diversity is the thing you're paying for; greedy decoding is
the setting that suppresses it.

Suggested, all **inferred**, none measured:

- A separate config preset for data generation rather than reusing the eval defaults —
  `temperature` ~0.9, `top_p` ~0.95, `top_k` unset/high, and **`seed` varied per draw** rather
  than fixed at 42. `GenConfig` already makes all four flags, so this is a preset, not a change.
- **Vary `shuffle_seed` per draw too.** It currently defaults to 42, so every story gets one
  fixed source ordering. For an eval run that's correct (reproducible). For dataset generation
  it throws away a free axis of variation, and §1 says ordering genuinely moves output.
- **Deduplicate, and log the dedup rate per run as a quality metric.** Exact hash first, then
  near-duplicate detection. Near-duplicates are the characteristic failure of bulk generation
  and they are invisible unless measured.
- `reveal_stance` / `reveal_publisher` are already one-flag ablations — as *generation* axes they
  are also a cheap source of legitimate variation across the dataset.

### 2.2 `workers = 4` is the binding constraint, and the `rpm = 60` gate will never engage

**Measured** at `workers=10`, no special quota tier, plain API key:

| | |
|---|---|
| sustained | **51.8 req/min** (0.86 req/s) |
| output tokens | **128,662 tok/min** |
| total tokens | **256,258 tok/min** |
| mean latency @ 10 workers | 11.6 s/call (~2,500 output tokens) |
| **429 / RESOURCE_EXHAUSTED / quota / 503 errors** | **zero, out of 123 real errors across every archive** |

Your `RateGate` is set to `rpm=60` = 1.0 req/s. With `workers=4` and this repo's longer
generations (`max_output_tokens=4096` vs. 2,500 there, so call it 15–20 s), achievable
throughput is roughly `4 / 17s` ≈ **0.24 req/s ≈ 14 rpm** — about a quarter of the gate.
The token bucket is therefore dead code at the current settings; **`workers` is what limits you.**

So: raising `workers` is free throughput up to the point where the gate actually starts
engaging, and 10 concurrent workers produced zero throttling in the measured run. Raise
`workers` first, keep the gate as the safety net it already is, and let an actual 429 tell you
where the ceiling is rather than guessing. (`RateGate` is better than what that project had —
it has no global pause at all — so you're well set up to find the ceiling safely.)

### 2.3 Batch Mode for the bulk synthetic run

**Inferred; confirm current pricing and SDK surface before relying on it.** Generating a dataset
offline is precisely the case Gemini's batch endpoint exists for — advertised at 50% of
interactive pricing, ~24h turnaround. It changes nothing about the prompt, the schema or the
output, so unlike every other knob here it costs no comparability. If the generation job isn't
interactive, this is the single largest cost lever available and it should probably be the
default path for dataset builds, with the current synchronous client kept for eval and dev.

### 2.4 `_fatal_transport` — make sure an SSL EOF isn't treated as a certificate failure

**Measured:** of 6 transport failures in that run, 2 were
`ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol`.
All 6 were **consecutive** and were connection churn — transient, and they succeeded on retry.

`_fatal_transport` keys on `"ssl" in type(e).__name__.lower()`. For `httpx.ConnectError` that's
`"connecterror"`, so today this case is correctly retried. But if any SDK or httpx version
surfaces the same mid-connection truncation as something named `SSLError`, the current heuristic
**aborts the entire run** on what is a retryable blip. `UNEXPECTED_EOF_WHILE_READING` is a
connection dropped mid-stream, not a bad certificate. Worth matching on the message
(`certificate`, `CERTIFICATE_VERIFY_FAILED`, `hostname mismatch`) rather than on the type name.

### 2.5 `model = "gemini-flash-latest"` is an alias — pin it for a dataset build

Records carry `"model": self.cfg.model`, which for the default stores the string
`gemini-flash-latest`. That's the alias, not the version that served the request, and the alias
moves. Two consequences for a dataset:

- Two halves of the same dataset generated a month apart can come from different underlying
  model versions, with nothing in the records to show it.
- The opposite failure is also real and measured: that project pinned `gemini-3-pro-preview`,
  Google retired the ID, and a backfill lost **117/117 calls to 404**. So neither "latest" nor a
  `-preview` pin is safe. A dated stable ID is.

Two cheap fixes: pin a dated model ID for any dataset build, and record the resolved version the
API reports back — `response.model_version` is exposed on `GenerateContentResponse` in recent
`google-genai` (**check it exists in your pinned version**), which would slot straight into
`_usage`-style extraction and into the record next to `model`. `scripts/probe.py` presumably
already smoke-tests the ID; that habit is what would have caught the 404 above in one call
instead of 117.

---

## What isn't a problem here

For the record, so nobody re-litigates these: rate limiting (zero 429s at 0.86 req/s in the
measured run — the gate is a sensible precaution, not a response to an observed problem);
per-call client construction (this repo builds one client in `__init__`, which is right, and
was the actual cause of all 6 transport failures in the other project, where it was constructed
per call); and retry storms (`_backoff` is already jittered, and the run-wide `pause()` is a
better design than per-thread backoff).
