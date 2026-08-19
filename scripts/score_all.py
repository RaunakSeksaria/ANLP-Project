"""Score every text in every fallacy dataset with Detoxify 'original-small'.

Optimized vs the EDA scripts:
  * length-sorted batching  (the EDA pads every batch to its longest member;
    sorting first cuts wasted padding compute by ~3-4x)
  * fp16 + large batches on GPU
  * torch.inference_mode()
  * per-dataset JSONL written incrementally, so a killed job keeps its work
    and a rerun skips finished datasets

    uv run python scripts/score_all.py                 # everything, auto device
    uv run python scripts/score_all.py --only jin      # one dataset
    uv run python scripts/score_all.py --device cpu --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path

import torch

from load_fallacy import ROOT, load_records
from toxicity_refs import load_cocolofa, load_jin

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
OUT_DIR = ROOT / "data/fallacy/toxicity"


def sources():
    """name -> callable returning list of (text, meta_dict)."""

    def v2(field):
        def _f():
            return [
                (r[field], {"answer": r.get("Answer"), "label": r.get("Fallacy Deconstruction")})
                for r in load_records()
                if r.get(field)
            ]

        return _f

    return {
        "v2_question": v2("question"),
        "v2_output": v2("output"),
        "cocolofa": lambda: [(t, {"label": l}) for t, l in load_cocolofa()],
        "jin": lambda: [(t, {"label": l}) for t, l in load_jin()],
    }


def score_dataset(model, tokenizer, device, class_names, rows, batch_size, max_length):
    texts = [t for t, _ in rows]
    # sort by length, remember original positions, restore order at the end
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results = [None] * len(texts)
    t0 = time.time()

    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = [texts[i] for i in idx]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length
        ).to(device)
        with torch.inference_mode():
            logits = model(**enc)[0]
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        for j, i in enumerate(idx):
            results[i] = {k: float(probs[j][n]) for n, k in enumerate(class_names)}

        done = start + len(idx)
        if done % (batch_size * 20) == 0 or done == len(order):
            print(f"    {done}/{len(order)}  {done / (time.time() - t0):.1f} txt/s", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of dataset names")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=None, help="default 256 gpu / 64 cpu")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--force", action="store_true", help="rescore datasets already on disk")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    fp16 = device.type == "cuda"
    batch_size = args.batch_size or (256 if fp16 else 64)

    from detoxify import Detoxify

    dtx = Detoxify("original-small", device=device)
    model, tokenizer, class_names = dtx.model, dtx.tokenizer, dtx.class_names
    model.eval()
    if fp16:
        model.half()
    assert list(class_names) == LABELS, class_names

    print(f"device={device} fp16={fp16} batch={batch_size} threads={torch.get_num_threads()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    todo = args.only or list(sources())
    grand = time.time()
    for name in todo:
        out = OUT_DIR / f"{name}.jsonl"
        if out.exists() and not args.force:
            print(f"[{name}] exists, skipping (use --force to redo)")
            continue
        rows = sources()[name]()
        print(f"[{name}] {len(rows)} texts")
        scores = score_dataset(
            model, tokenizer, device, class_names, rows, batch_size, args.max_length
        )
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for (text, meta), s in zip(rows, scores):
                f.write(json.dumps({"text": text, **meta, **s}, ensure_ascii=False) + "\n")
        tmp.rename(out)
        print(f"[{name}] -> {out.relative_to(ROOT)}")

    print(f"\ntotal wall time: {time.time() - grand:.1f}s")


if __name__ == "__main__":
    main()
