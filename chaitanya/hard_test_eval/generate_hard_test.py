#!/usr/bin/env python3
"""Neutral rewrites of Raunak's hard-test articles (data branch @ 69ebfdf, hard-test/ -> dataset/hard_test/) with one
LED checkpoint.

Articles: the 1,187 hard-test articles whose "title </s> body" fits in the models' 512-token input (711 from the
dedup release, 476 from full1024).

Same path as the training-time validation decoding (src/training/train_joint.py):
- source = "title </s> body" with title/body .strip()-ed (prepare_side_neutral.py + LEDBaselineDataset),
  global attention on token 0;
- model = src.models.led_triplet.load_led_triplet (attention window 512), fp32;
- decoding = src.training.overfit_baseline.generate_with_cache with the checkpoint's generation settings
  (greedy, num_beams 1, max_new_tokens 512), one article at a time (training used eval batch size 1).

Checks: every dedup article must give exactly the source text of the same article in
dataset/side_neutral/prepared (validation/test); with --check PRED.jsonl the articles shared with a training run's
validation_predictions_best.jsonl are generated and compared with it, then the script exits.
Resumes: articles already in <out>/<model>.jsonl are skipped. The rows have the format of the public HF dataset
tchaitanya0109/anlp-hard-test-generations (generations/<model>.jsonl).

    python chaitanya/hard_test_eval/generate_hard_test.py --model publisher_standard_m02
    python chaitanya/hard_test_eval/generate_hard_test.py --model publisher_standard_m02 --check <predictions.jsonl>
"""

import argparse
import difflib
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
HARD = ROOT / "dataset" / "hard_test"
PREPARED = ROOT / "dataset" / "side_neutral" / "prepared"
META = ("id", "story_id", "side", "article_id", "publisher", "split", "release")


def articles(tokenizer) -> list[dict]:
    sep, rows = tokenizer.sep_token, []
    for side in ("left", "right"):
        for line in (HARD / f"hard_test_{side}_neutral.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            original = r[f"{side}_original"]
            title, body = original["title"].strip(), original["body"].strip()
            rows.append({
                "id": f"{r['story_id']}:{side}", "story_id": r["story_id"], "side": side,
                "article_id": original["article_id"], "publisher": original["publisher"],
                "split": r["split"], "release": r["release"],
                "source": f"{title} {sep} {body}" if title else body,
            })
    for r in rows:
        r["input_tokens"] = len(tokenizer(r["source"], truncation=False)["input_ids"])
    rows = [r for r in rows if r["input_tokens"] <= 512]
    assert len(rows) == 1187 and len({r["id"] for r in rows}) == 1187, len(rows)
    return rows


def check_sources(rows: list[dict], sep: str) -> None:
    prepared = {}
    for split in ("validation", "test"):
        for line in (PREPARED / split / "baseline.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            prepared[r["input_article_id"]] = (r["input_title"], r["input_text"])
    # Only the dedup release shares article ids with side_neutral: full1024 numbers its articles
    # independently, so the same id there can be a different article (e.g. 26040:1).
    shared = mismatched = 0
    for r in rows:
        if r["release"] == "dedup" and r["article_id"] in prepared:
            title, text = prepared[r["article_id"]]
            shared += 1
            mismatched += r["source"] != (f"{title} {sep} {text}" if title else text)
    print(f"source check: {shared} dedup hard-test articles are in side_neutral validation/test, {mismatched} differ", flush=True)
    assert shared == sum(r["release"] == "dedup" for r in rows) and mismatched == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "hard_test_generations")
    ap.add_argument("--check", type=Path, help="validation_predictions_best.jsonl to compare against, then exit")
    ap.add_argument("--check-limit", type=int, default=24)
    ap.add_argument("--check-min", type=int, help="identical outputs needed to pass (default: all); "
                    "a different GPU type than training can change a few greedy tokens")
    args = ap.parse_args()

    repo, sub = MODELS[args.model]
    local = Path(snapshot_download(repo, allow_patterns=[f"{sub}/*"])) / sub
    tokenizer = AutoTokenizer.from_pretrained(local)
    rows = articles(tokenizer)
    check_sources(rows, tokenizer.sep_token)
    weights_sha = hashlib.sha256((local / "model.safetensors").read_bytes()).hexdigest()
    model = load_led_triplet(local, attention_window=512).to("cuda").eval()
    device = torch.device("cuda")

    if args.check:
        reference = {}
        for line in args.check.read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            reference[p["source_text"]] = p["generated_text"]
        todo = [r for r in rows if r["source"] in reference][: args.check_limit]
        assert todo, f"no hard-test source matches a source_text in {args.check}"
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        path = args.out / f"{args.model}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        print(f"{args.model}: {repo}/{sub} sha256 {weights_sha} | {len(done)} done, {len(todo)} to go", flush=True)

    same, start = 0, time.time()
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
        if args.check:
            expected = reference[r["source"]]
            same += text == expected
            if text != expected:
                ratio = difflib.SequenceMatcher(None, text.split(), expected.split()).ratio()
                print(f"  differs: {r['id']} word-sequence similarity {ratio:.4f}", flush=True)
            continue
        record = {**{k: r[k] for k in META}, "model": args.model, "repo": repo, "subfolder": sub,
                  "weights_sha256": weights_sha, "input_tokens": r["input_tokens"],
                  "terminated_with_eos": eos in generated[1:], "output": text}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if i % 50 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} | {(time.time() - start) / i:.2f} s/article", flush=True)

    if args.check:
        print(f"CHECK {args.model}: {same}/{len(todo)} outputs identical to {args.check}", flush=True)
        return 0 if same >= (len(todo) if args.check_min is None else args.check_min) else 1
    print(f"GENERATION OK {args.model} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
