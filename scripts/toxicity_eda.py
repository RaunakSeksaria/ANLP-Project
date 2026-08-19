"""EDA: how toxic is the logical-fallacy dataset?

Runs Detoxify 'original-small' (albert-base-v2, 6 labels) on a random sample
of the dataset on CPU. This is a viability check, not a production pass —
use --n -1 to score everything.

    uv run python scripts/toxicity_eda.py --n 2000
"""

import argparse
import json
import random
import time
from pathlib import Path

from load_fallacy import ROOT, load_records

LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
]
OUT_TMPL = "data/fallacy/toxicity_sample_{field}.jsonl"


def score(texts, batch_size=16):
    from detoxify import Detoxify

    model = Detoxify("original-small", device="cpu")
    rows, t0 = [], time.time()
    for i in range(0, len(texts), batch_size):
        batch = [t[:2000] for t in texts[i : i + batch_size]]
        preds = model.predict(batch)
        for j in range(len(batch)):
            rows.append({k: float(preds[k][j]) for k in LABELS})
        done = i + len(batch)
        if done % (batch_size * 10) == 0 or done == len(texts):
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(texts)}  {rate:.1f} txt/s", flush=True)
    return rows


def quantiles(vals, qs=(0.5, 0.9, 0.99, 1.0)):
    s = sorted(vals)
    return {q: s[min(len(s) - 1, int(q * len(s)))] for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="sample size, -1 for all")
    ap.add_argument("--field", default="question", choices=["question", "output"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [r for r in load_records() if r.get(args.field)]
    random.seed(args.seed)
    sample = recs if args.n < 0 else random.sample(recs, min(args.n, len(recs)))
    texts = [r[args.field] for r in sample]
    print(f"scoring {len(texts)} '{args.field}' texts on CPU...")

    scores = score(texts)

    out = ROOT / OUT_TMPL.format(field=args.field)
    with out.open("w", encoding="utf-8") as f:
        for r, s in zip(sample, scores):
            f.write(json.dumps({"text": r[args.field], "field": args.field, **s}) + "\n")

    print(f"\nwrote {out.relative_to(ROOT)}\n")
    print(f"{'label':17s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s} "
          f"{'>0.5':>6s} {'>0.1':>6s}")
    for lab in LABELS:
        v = [s[lab] for s in scores]
        q = quantiles(v)
        print(
            f"{lab:17s} {sum(v) / len(v):8.4f} {q[0.5]:8.4f} {q[0.9]:8.4f} "
            f"{q[0.99]:8.4f} {q[1.0]:8.4f} "
            f"{sum(x > 0.5 for x in v):6d} {sum(x > 0.1 for x in v):6d}"
        )

    ranked = sorted(zip(texts, scores), key=lambda p: -p[1]["toxicity"])
    print("\n--- most toxic ---")
    for t, s in ranked[:5]:
        print(f"[tox={s['toxicity']:.3f} insult={s['insult']:.3f} "
              f"id_attack={s['identity_attack']:.3f}] {t[:260]}\n")


if __name__ == "__main__":
    main()
