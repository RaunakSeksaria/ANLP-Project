"""Kaggle CPU job: push best checkpoints from a training kernel to one HF repo.

Layout is <experiment>/<run>/; the model card is rebuilt from every run already
in the repo, so each upload adds rows instead of replacing the card.
Set EXPERIMENT and RUNS below, and kernel_sources in kernel-metadata.json to the
training kernel. The HF token comes from the private dataset anlp-hf-token.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ID = "RaunakSeksaria/anlp-neutral-led"
EXPERIMENT = "side-neutral-full"
RUNS = {"lambda0": 0.0, "lambda035": 0.35}
EXPERIMENTS = {
    "side-neutral-full": (
        "ideology negatives",
        "Negative = same-side article from a different event group (proposal variant b; "
        "it shares the anchor's publisher ~30% of the time, since each publisher is on one side)."),
    "side-neutral-full-publisher": (
        "publisher negatives",
        "Negative = article from the anchor's own publisher, different event group "
        "(proposal variant a). Anchors, positives and targets are identical to side-neutral-full."),
}
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors",
               "tokenizer.json", "tokenizer_config.json", "joint_state.json")
METRIC_FILES = ("train/training_history.json", "train/training_config.json",
                "train/run_summary.json", "train/validation_metrics_best.json",
                "test_eval/test_metrics.json")
INPUT, STAGE = Path("/kaggle/input"), Path("/kaggle/tmp/hf_stage")


def find(relative):
    hits = sorted(INPUT.rglob(relative))
    if not hits:
        raise FileNotFoundError(relative)
    return hits[0]


def card_rows(api, token):
    from huggingface_hub import hf_hub_download
    rows = []
    for path in sorted(api.list_repo_files(REPO_ID)):
        parts = path.split("/")
        if len(parts) != 4 or parts[2:] != ["metrics", "test_metrics.json"]:
            continue
        exp, run = parts[:2]
        load = lambda name: json.loads(Path(hf_hub_download(REPO_ID, f"{exp}/{run}/{name}", token=token)).read_text())
        test, summary, state = load("metrics/test_metrics.json"), load("metrics/run_summary.json"), load("joint_state.json")
        rows.append(f"| `{exp}/{run}` | {EXPERIMENTS.get(exp, ('?',))[0]} | {state['triplet_lambda']:g} | "
                    f"{summary['best_epoch']} | {test['token_overlap_f1']:.3f} | {test['generation_loss']:.3f} | "
                    f"{test['token_accuracy']:.3f} | {test['mean_word_length_ratio']:.2f} |")
    return rows


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=0.34"], check=True)
    from huggingface_hub import HfApi

    token = find("hf_token.txt").read_text().strip()
    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True)
    existing = [f for f in api.list_repo_files(REPO_ID)
                if any(f.startswith(f"{EXPERIMENT}/{run}/") for run in RUNS)]
    if existing:
        raise SystemExit(f"refusing to overwrite: {len(existing)} files already under "
                         f"{EXPERIMENT}/{{{','.join(RUNS)}}}, e.g. {existing[0]}")

    for run in RUNS:
        root = find(f"runs/{run}/train/best/model.safetensors").parents[2]
        stage = STAGE / EXPERIMENT / run
        (stage / "metrics").mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES:
            shutil.copy2(root / "train" / "best" / name, stage / name)
        for rel in METRIC_FILES:
            shutil.copy2(root / rel, stage / "metrics" / Path(rel).name)
        print(f"staged {run} from {root}", flush=True)
    api.upload_folder(repo_id=REPO_ID, folder_path=str(STAGE),
                      commit_message=f"add {EXPERIMENT}: " + ", ".join(RUNS))

    rows = card_rows(api, token)
    sections = "\n\n".join(f"### {exp}\n\n{desc}" for exp, (_, desc) in EXPERIMENTS.items()
                           if any(f"`{exp}/" in row for row in rows))
    card = README.format(table="\n".join(rows), sections=sections, repo=REPO_ID)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO_ID,
                    commit_message="update model card")
    print("card rows:\n" + "\n".join(rows))
    print("repo files:\n  " + "\n  ".join(sorted(api.list_repo_files(REPO_ID))), flush=True)


README = """---
library_name: transformers
base_model: allenai/led-base-16384
tags: [led, neutral-news, framing-bias, triplet-loss]
---

# anlp-neutral-led

LED-base checkpoints for **neutral article generation** (IIIT Hyderabad ANLP
project: *Event-Aware Representation Learning for Neutral Multi-Source News*).
The model gets one partisan news article and writes the same article
without framing. Each checkpoint lives in `<experiment>/<run>/`.

## Runs

| subfolder | negatives | triplet λ | best epoch | test F1 | test gen loss | test token acc | length ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
{table}

Test set: 1,775 held-out articles. F1 is whitespace-token overlap with the
neutral reference after greedy decoding (max 512 new tokens); token accuracy is
teacher-forced. λ = 0 never uses negatives, so it is the control for every
experiment.

## Experiments

All runs share the data and training setup:

- **Data:** `left/right_neutral_dedup` (BigNewsAlign articles, neutral targets
  generated by Nemotron-3-Ultra), restricted to articles that share an event with
  an opposite-side article: 12,484 train / 1,778 val / 1,775 test, no event,
  article or body shared across splits (`dataset/side_neutral_full` on the
  `data` branch).
- **Objective:** generation loss + λ × triplet margin loss (margin 0.2) on
  L2-normalised masked-mean encoder states. Anchor = input article, positive =
  opposite-side article about the same event; the negative depends on the
  experiment.
- **Training:** fp32, lr 3e-5, effective batch 8, 4 epochs (best by validation
  loss), 10% warmup, max 512 input / 512 target tokens, attention window 512,
  seed 42, one T4 per run.

{sections}

## Loading

```python
from transformers import AutoTokenizer, LEDForConditionalGeneration

sub = "side-neutral-full/lambda035"
tok = AutoTokenizer.from_pretrained("{repo}", subfolder=sub)
model = LEDForConditionalGeneration.from_pretrained("{repo}", subfolder=sub)

source = f"{{title}} {{tok.sep_token}} {{body}}"   # training input format
enc = tok(source, return_tensors="pt", truncation=True, max_length=512)
glob = enc["attention_mask"].new_zeros(enc["attention_mask"].shape)
glob[:, 0] = 1                                   # global attention on the first token
out = model.generate(**enc, global_attention_mask=glob, num_beams=1,
                     do_sample=False, max_new_tokens=512)
print(tok.decode(out[0], skip_special_tokens=True))
```

Article embeddings (what the triplet loss shapes): mean of
`model.get_encoder()(...).last_hidden_state` over non-padding tokens, then L2
normalisation. `metrics/` in each subfolder holds the training history, config
and validation/test metrics; the repo is private.
"""

if __name__ == "__main__":
    main()
