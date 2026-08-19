"""Same toxicity EDA, but for the two human-annotated reference datasets.

    uv run python scripts/toxicity_refs.py cocolofa --n 700
    uv run python scripts/toxicity_refs.py jin --n 700
"""

import argparse
import csv
import json
import random
from pathlib import Path

from load_fallacy import ROOT
from toxicity_eda import LABELS, quantiles, score

REFS = ROOT / "data/fallacy/papers-referenced"


def load_cocolofa():
    """Human-written news comments, 8 fallacy types + none."""
    rows = []
    for split in ("train", "dev", "test"):
        for art in json.loads((REFS / f"cocolofa-main/{split}.json").read_text(encoding="utf-8")):
            for c in art.get("comments") or []:
                if c.get("comment"):
                    rows.append((c["comment"], c.get("fallacy", "?")))
    return rows


def load_jin():
    """Jin et al. edu + climate splits (source_article -> label)."""
    rows = []
    for name, label_col in (("edu_all", "updated_label"), ("climate_all", "logical_fallacies")):
        path = REFS / f"logical-fallacy-main/data/{name}.csv"
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                text = (r.get("source_article") or "").strip()
                if text:
                    rows.append((text, r.get(label_col, "?")))
    return rows


LOADERS = {"cocolofa": load_cocolofa, "jin": load_jin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(LOADERS))
    ap.add_argument("--n", type=int, default=700, help="sample size, -1 for all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = LOADERS[args.dataset]()
    random.seed(args.seed)
    sample = rows if args.n < 0 else random.sample(rows, min(args.n, len(rows)))
    texts = [t for t, _ in sample]
    print(f"{args.dataset}: {len(rows)} total, scoring {len(texts)} on CPU...")

    scores = score(texts)

    out = ROOT / f"data/fallacy/toxicity_sample_{args.dataset}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for (t, lab), s in zip(sample, scores):
            f.write(json.dumps({"text": t, "label": lab, **s}, ensure_ascii=False) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}\n")

    print(f"{'label':17s} {'mean':>8s} {'p50':>8s} {'p90':>8s} {'p99':>8s} {'max':>8s} "
          f"{'>0.5':>6s} {'>0.1':>6s}")
    for lab in LABELS:
        v = [s[lab] for s in scores]
        q = quantiles(v)
        print(f"{lab:17s} {sum(v) / len(v):8.4f} {q[0.5]:8.4f} {q[0.9]:8.4f} "
              f"{q[0.99]:8.4f} {q[1.0]:8.4f} "
              f"{sum(x > 0.5 for x in v):6d} {sum(x > 0.1 for x in v):6d}")

    ranked = sorted(zip(sample, scores), key=lambda p: -p[1]["toxicity"])
    print("\n--- most toxic ---")
    for (t, lab), s in ranked[:5]:
        print(f"[tox={s['toxicity']:.3f} insult={s['insult']:.3f} "
              f"id_attack={s['identity_attack']:.3f} label={lab!r}] {t[:240]}\n")


if __name__ == "__main__":
    main()
