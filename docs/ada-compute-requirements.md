# What needs Ada, and what does not

## Quickstart

The hardware and plumbing facts below were established on a previous assignment
that ran end to end on this cluster, and are not guesses.

**From the laptop**, push the repo, then on Ada:

```bash
cd ~/ANLP-Project
python3 scripts/prepare_data.py        # pure stdlib, safe on the login node
mkdir -p logs runs

sbatch scripts/ada_sync.sh             # build venv + cache model (~30 min first time)
squeue -u $USER                        # wait for it to finish
tail -f logs/sync-*.out

sbatch scripts/ada_run.sh              # the whole experiment (~1-2 h on bart-base)
tail -f logs/neus-*.out
```

`prepare_data.py` is the one script that runs on the login node — it imports
nothing outside the standard library. Everything else must run inside a job.

### Five rules the login node enforces

1. **The venv only imports on compute nodes.** The login node is CentOS 7 with
   glibc 2.17 and no `CXXABI_1.3.9`; anything touching torch, numpy or
   transformers dies there. Login node is for `sbatch`, `squeue`, `git` and file
   transfer only.
2. **`uv lock` / `uv sync` must run inside a job** — the head node's `ulimit -v`
   kills the resolver with a bogus `memory allocation of ~700 KB failed`. That
   is what `scripts/ada_sync.sh` is for; run jobs pass `--no-sync` and never
   touch the resolver.
3. **Never reuse a `.venv` built elsewhere.** Mismatched CUDA libraries surface
   as `RuntimeError: CUDA unknown error` from `torch._C._cuda_init()` while
   `nvidia-smi` succeeds. `rm -rf .venv && uv sync` is the only fix.
4. **Never use `--mem`.** `MaxMemPerCPU` is 3000 MB, so `--mem=32G` with `-c 8`
   is rejected outright. Always `--mem-per-cpu`.
5. **A first sync takes ~30 min** (~3.5 GB of CUDA wheels onto NFS). Jobs that
   run short cannot be extended — `scontrol update TimeLimit` is operator-only.

### What the hardware forces

`--constraint=2080ti` gives an **RTX 2080 Ti with 10.6 GiB usable**, and that
drives two choices:

- **bart-base, not bart-large.** BART-large is 6.0 GB of optimiser state before
  activations; adding a triplet stream (three encoder passes per step, with
  gradients) will not fit in 10.6 GiB. bart-base is 139 M parameters, ~2.2 GB of
  optimiser state, and leaves comfortable headroom. Override with
  `MODEL=facebook/bart-large sbatch scripts/ada_run.sh` once the pipeline is
  proven, and expect to drop `--batch-size` and `--triplet-batch-size`.
- **fp16, never bf16.** Turing (sm_75) has no bf16 unit. `train_neus.py`
  defaults to `--precision auto`, which checks `torch.cuda.is_bf16_supported()`
  and falls back to fp16, so this is handled — but do not hard-code bf16.

`--constraint=2080ti` is also required for a second reason: feature-less nodes
are GTX 1080 Ti (Pascal, sm_61), which current torch wheels no longer compile
for. `gnode091` is excluded because it has 11 GB of host RAM rather than 128 GB.
QOS `medium` caps you at 4 running jobs, 4 GPUs, 40 CPUs and 8 submitted.

### torch must be cu128, not cu130

`pyproject.toml` pins torch to the `pytorch-cu128` index. Ada's u22 nodes run a
spread of drivers — 570, 580 and 595 all observed on nodes advertising the same
`2080ti` feature — and cu130 wheels require driver >= 580, aborting with
`The NVIDIA driver on your system is too old` on the 570 nodes. SLURM exposes no
feature for driver version, so node selection cannot solve it; the build has to
match the oldest driver in the pool. cu128 tops out at torch 2.11.0, which is
why the dependency floor is `torch>=2.5`.

**`uv.lock` regenerated on Ada is the authoritative copy** — make sure it comes
back to the laptop repo, or the two diverge.

### What comes out

`runs/{baseline,triplet}/test_generations.txt`, `data/prepared/probe_results.json`
and `runs/generation_metrics.json`. The comparison that matters is
publisher-probe accuracy on the outlet-disjoint split, baseline vs triplet, at
matched ROUGE.

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
  That does **not** fit Ada's 10.6 GiB RTX 2080 Ti once the triplet stream adds
  three encoder passes per step. Use bart-base (139 M, ~2.2 GB optimiser state)
  for the core experiments here; bart-large needs a larger card than this QOS
  reaches, or aggressive gradient checkpointing at micro-batch 1.
- Throughput: 1,226 micro-steps/epoch, 12,260 for the 10-epoch schedule.
  At 2–5 micro-it/s that is **40–100 min per run**; budget **1.5 GPU-hours**
  including per-epoch validation and test generation.
- Run count: 1 baseline reproduction + ~3 triplet weight/margin settings +
  2 ablations (generation-loss-only, publisher-shuffled control) + reruns on the
  media-disjoint split ≈ **10–15 runs**.

> **Ask: 20–25 GPU-hours on the 2080 Ti at bart-base.** QOS `medium` allows 4
> concurrent GPUs, so the sweep parallelises four ways.

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

- LED-base at 4,096 tokens needs well beyond 10.6 GiB. **This does not fit the
  2080 Ti at all**, so the extension needs either a truncated context, a smaller
  long-input model, or a card this account cannot currently reach.

> **Verdict: out of scope on the hardware available.** Reframe the extension to
> fit 10.6 GiB, or drop it. This is the item to cut first regardless.

This is the item to cut first if allocation is tight. It is an extension, not the
contribution.

### 4. QA-based event-retention evaluation

The DiverseSumm-style evaluation needs an LLM to generate and answer questions
over 307 test summaries.

- Hosted API: no cluster time, but check whether the course permits external API
  calls and who pays.
- A local 7–8 B instruct model will not fit 10.6 GiB at usable precision, so on
  this hardware the QA route is effectively API-only.

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

- Storage note: `/home` is a 25 GB NFS quota. BART-base checkpoints are ~0.5 GB
  and the CUDA venv alone is ~3.5 GB, so keep the HF cache and `runs/` inside the
  project and prune between sweeps, or use `/scratch` (purged after 10 days).

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

**Python 3.14 is fine** — the cu128 index ships a cp314 wheel for torch 2.11.0,
so `requires-python = ">=3.14"` stays. Note that a stock `uv sync` on modern
hardware will *not* reproduce this lockfile, because of the cu128 pin above;
that is worth a line in the README and a sentence in the final report.
