# What needs Ada, and what does not

## Quickstart (three commands)

On the Ada **login** node, from the repo root:

```bash
bash scripts/ada_setup.sh          # venv, torch, transformers, prefetch BART, build data (~10 min)

source .venv-ada/bin/activate      # smoke test on CPU first -- catches bugs in 2 min,
python scripts/train_neus.py --run-name smoke --epochs 1 --limit-train 8 \
    --batch-size 2 --grad-accum 1 --precision fp32 --num-beams 1

sbatch scripts/ada_run.sh          # both runs + probes + metrics (~2-3 h)
```

`scripts/ada_run.sh` does the whole experiment in one job: baseline (generation
loss only), triplet (generation + event-aware loss), then the publisher-leakage
probes on both plus the lexical floor, then ROUGE and framing bias on the
generations. Check its `#SBATCH` lines against your own allocation before
submitting -- partition and account names differ per Ada project.

Results land in `runs/{baseline,triplet}/`, `data/prepared/probe_results.json`
and `runs/generation_metrics.json`. The comparison that matters is
publisher-probe accuracy on the disjoint split, baseline vs triplet, at matched
ROUGE.

This machine has **no GPU** — 16 CPU cores, 15 GB RAM, 106 GB free disk. Every
number below is derived from measured corpus sizes (`eda/REPORT.md`) and NeuS's
own published hyperparameters (`data/framing-bias-metric-main/finetune_neus_script.sh`:
BART-large, micro-batch 2 × grad-accum 8, lr 3e-5, 10 epochs, 1 GPU, source 512
/ target 250).

## Runs fine on the laptop — no cluster account needed

| task | cost | status |
| --- | --- | --- |
| Corpus profiling (all four) | ~3 min, 1 core | done — `eda/profile_datasets.py` |
| Framing-bias metric + gold baseline | ~4 min, 1 core | done — `eda/framing_bias.py` |
| Building the publisher-labelled NeuS splits | seconds | join implemented, 100 % coverage |
| Constructing a media-disjoint split | seconds | not yet written |
| Triplet sampling index (event ↔ publisher) | seconds | not yet written |
| Probe classifiers (logistic regression) on **cached** encoder vectors | ~1 min | needs vectors from a GPU pass first |
| All lexical / statistical analysis and plotting | minutes | — |

The metric work in particular needs nothing: it is stdlib Python over a 20 k-entry
lexicon. Do not spend GPU allocation on it.

## Needs Ada

Ordered by how early it blocks the project.

### 1. BART-large fine-tuning — the core experiments

**Blocks everything downstream.** No baseline, no triplet model, no ablations
without it.

- Memory: 406 M params → 1.5 GB weights + 1.5 GB grads + 3.0 GB Adam states =
  **6.0 GB optimiser state** in fp32, plus ~1–2 GB activations at batch 2 × 512.
  Fits a 16 GB card comfortably; bf16 halves it. **Request ≥16 GB, prefer 24 GB+**
  so the triplet variant can hold three encoder passes per step.
- Throughput: 1,226 micro-steps/epoch, 12,260 for the 10-epoch schedule.
  At 2–5 micro-it/s that is **40–100 min per run**; budget **1.5 GPU-hours**
  including per-epoch validation and test generation.
- Run count: 1 baseline reproduction + ~3 triplet weight/margin settings +
  2 ablations (generation-loss-only, publisher-shuffled control) + reruns on the
  media-disjoint split ≈ **10–15 runs**.

> **Ask: 20–25 GPU-hours, single GPU, ≥16 GB.**

The triplet variant is ~2–3× the baseline per step, since anchor, positive and
negative each need an encoder pass. Budget accordingly if the sweep grows.

### 2. Frozen-encoder representations for the leakage probe

One forward pass over 9,285 NeuS articles (and optionally 37,554 AllSides
articles) to cache encoder vectors, after which the probe itself is a CPU
logistic regression.

- NeuS alone: minutes on any GPU. Feasible on CPU (~1–2 h) if the queue is long.
- AllSides at median 916 words is heavier and genuinely wants a GPU.

> **Ask: <1 GPU-hour.** Bundle it into the tail of a training job rather than
> queueing separately.

### 3. WCEP-10 pretraining — only if extension 1 is in scope

Cross-source latent prediction (the I-JEPA-style extension) is the one part of
the plan that needs real compute, because WCEP clusters are ~9 documents ×
~337 words ≈ 4,400 tokens. That exceeds BART's 1,024 window, so it requires a
long-input model (LED / PRIMERA) and a different memory profile.

- LED-base at 4,096 tokens: **≥24 GB**, realistically an A100-40 GB.
- 8,158 clusters/epoch; at ~1.5 it/s a 3-epoch pass is ~4.5 h.

> **Ask: 15–20 GPU-hours on a 40 GB card — and only after the core result exists.**

This is the item to cut first if allocation is tight. It is an extension, not the
contribution.

### 4. QA-based event-retention evaluation

The DiverseSumm-style evaluation needs an LLM to generate and answer questions
over 307 test summaries.

- Hosted API: no cluster time, but check whether the course permits external API
  calls and who pays.
- Local 7–8 B instruct model: **1 GPU, 16–24 GB, ~2–3 GPU-hours** for the full
  test split.

> Decide the route before requesting the allocation — they have very different
> asks. Note the caveat in `docs/eda-evaluation.md`: the source articles are
> 85-word excerpts, so there is little event content for questions to target.

## Storage and environment

- **Scratch: ~100 GB.** BART-large checkpoints are ~1.6 GB each; 15 runs keeping
  best-only is ~25 GB, plus HF cache, plus the 1.4 GB of corpora, plus
  generations and logs.
- Compute nodes are frequently offline. Pre-fetch weights on the login node and
  pin the caches into the project:

  ```bash
  export HF_HOME="$PWD/.cache/huggingface" TORCH_HOME="$PWD/.cache/torch"
  export TOKENIZERS_PARALLELISM=false
  # after pre-fetching on the login node:
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  ```

- A SLURM script previously in this repo (removed in `174507b`, recoverable with
  `git show 50e4bbe^:scripts/slurm_toxicity.sh`) used
  `--account=research --qos=medium --partition=gpu --gres=gpu:1`. **Confirm those
  are correct for your Ada allocation before relying on them** — they were
  written for an earlier, abandoned experiment and were never run against this
  project's account.

## The environment decision

NeuS's own code cannot run in this repo. It pins **torch 1.7.1, transformers
4.3.3, pytorch-lightning 1.0.8, Python 3.6** (`dataclasses==0.8` gives the Python
version away) against this project's Python 3.14, and it vendors a 14 MB
transformers fork.

That fork is worth understanding before deciding, because it is almost entirely
stock. Its only substantive modification is an auxiliary head:
`src/transformers/models/bart/modeling_bart.py:1377-1533` adds a 2-way
`classification_head` over the encoder's pooled EOS representation, exposed via
`extra_task` / `extra_task_input_ids` / `extra_task_label`, and
`src/finetune.py:203-224` folds its loss in as
`loss = lm_loss + task_loss_ratio * extra_task_loss`.

**That is exactly where a triplet loss attaches** — same pooled representation,
same weighted-sum plumbing, same `task_loss_ratio` hyperparameter.

> **Recommendation: reimplement on the modern stack, copying that design.** It is
> roughly 60 lines against current `transformers` — a `BartForConditionalGeneration`
> wrapper that pools the encoder EOS state and adds
> `nn.TripletMarginLoss` — and it avoids maintaining a Python 3.6 environment on
> Ada for the sake of one model file. The cost is that reproduction numbers will
> not be bit-identical to the paper; state that in the report and cite the
> published table for comparison.

**Python 3.14 is not a problem — checked.** `uv pip install --dry-run torch`
resolves to **torch 2.13.0** on this interpreter, so `requires-python = ">=3.14"`
can stay as it is. Note that the resolution pulls the CUDA 13 `nvidia-*` wheels
(~3 GB), so install once on the Ada login node into a project-local cache before
any compute node needs them.
