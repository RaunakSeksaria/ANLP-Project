#!/usr/bin/env python3
"""Neutral rewrites of the 2,160 cc2025 articles (dataset/cc2025/cc2025_eval.jsonl, made by select_cc2025.py) with one
LED checkpoint.

Same path as the training-time validation decoding (src/training/train_joint.py):
- source = "title </s> body" with title/body .strip()-ed (prepare_side_neutral.py + LEDBaselineDataset),
  at most 512 tokens (all 2,160 fit), global attention on token 0;
- model = src.models.led_triplet.load_led_triplet (attention window 512), fp32;
- decoding = src.training.overfit_baseline.generate_with_cache with the checkpoint's generation settings
  (greedy, num_beams 1, max_new_tokens 512), one article at a time (training used eval batch size 1).
Resumes: articles already in <out>/<model>.jsonl are skipped.

    python chaitanya/cc2025_eval/generate_cc2025.py --model publisher_standard_m02
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.led_triplet import load_led_triplet  # noqa: E402
from src.training.overfit_baseline import generate_with_cache  # noqa: E402

MODELS = {  # name: (HF repo, subfolder)
    "baseline_lambda0": ("RaunakSeksaria/anlp-neutral-led", "side-neutral-full/lambda0"),
    "ideology_standard_m04": ("tchaitanya0109/anlp-neutral-led-ideology-first-token", "standard-margin04-lambda035"),
    "ideology_attraction_m02": ("tchaitanya0109/anlp-neutral-led-ideology-first-token", "attraction-margin02-lambda035"),
    "ideology_attraction_m05": ("tchaitanya0109/anlp-neutral-led-ideology-first-token", "attraction-margin05-lambda035"),
    "publisher_standard_m02": ("tchaitanya0109/anlp-neutral-led-publisher-first-token", "standard-lambda035"),
    "publisher_attraction_m02": ("tchaitanya0109/anlp-neutral-led-publisher-first-token", "attraction-lambda035"),
    "ideology_attraction_m03_b05_t03": ("tchaitanya0109/anlp-neutral-led-ideology-first-token",
                                        "attraction-margin03-beta05-tau03-lambda035"),
    "ideology_attraction_m03_b06_t05": ("tchaitanya0109/anlp-neutral-led-ideology-first-token",
                                        "attraction-margin03-beta06-tau05-lambda035"),
}
SETTINGS = {"do_sample": False, "early_stopping": False, "max_new_tokens": 512, "num_beams": 1}
CC2025 = ROOT / "dataset" / "cc2025" / "cc2025_eval.jsonl"
META = ("id", "side", "publisher", "date")


def cc2025_articles(sep: str) -> list[dict]:
    rows = []
    for line in CC2025.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        title, body = r["title"].strip(), r["body"].strip()
        rows.append({"id": r["id"], "side": r["side"], "publisher": r["source"], "date": r["date"],
                     "source": f"{title} {sep} {body}" if title else body})
    assert len(rows) == 2160 and len({r["id"] for r in rows}) == 2160, len(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "cc2025_generations")
    args = ap.parse_args()

    repo, sub = MODELS[args.model]
    local = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"])) / sub
    tokenizer = AutoTokenizer.from_pretrained(local)
    rows = cc2025_articles(tokenizer.sep_token)
    assert all(len(tokenizer(r["source"], truncation=False)["input_ids"]) <= 512 for r in rows)
    weights_sha = hashlib.sha256((local / "model.safetensors").read_bytes()).hexdigest()
    model = load_led_triplet(local, attention_window=512).to("cuda").eval()
    device = torch.device("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.model}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.model}: {repo}/{sub} sha256 {weights_sha} | {len(done)} done, {len(todo)} to go", flush=True)

    start = time.time()
    eos = tokenizer.eos_token_id
    for i, r in enumerate(todo, 1):
        enc = tokenizer(r["source"], truncation=True, max_length=512, padding=False)
        ids = torch.tensor([enc["input_ids"]], device=device)
        batch = {"input_ids": ids, "attention_mask": torch.tensor([enc["attention_mask"]], device=device),
                 "global_attention_mask": torch.zeros_like(ids)}
        batch["global_attention_mask"][:, 0] = 1
        with torch.inference_mode():
            generated = generate_with_cache(model.led, batch, device, "fp32", SETTINGS)[0].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        full_tokens = len(tokenizer(r["source"], truncation=False)["input_ids"])
        record = {**{k: r[k] for k in META},
                  "model": args.model, "repo": repo, "subfolder": sub, "weights_sha256": weights_sha,
                  "input_tokens": full_tokens, "truncated": full_tokens > 512,
                  "terminated_with_eos": eos in generated[1:], "output": text}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} | {(time.time() - start) / i:.2f} s/article", flush=True)

    print(f"GENERATION OK {args.model} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
