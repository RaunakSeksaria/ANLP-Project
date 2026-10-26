"""Fixed evaluation sample for generation-setting experiments.

    uv run python scripts/make_eval_sample.py --n 500
    -> experiments/eval500/{left,right}.jsonl

Takes events from data/final/full1024_pool.jsonl starting at --start (default
10,000: the first 10,000 were used for the full1024 release). The pool is
already in a seeded random order, deduplicated, <= 1,024 LED tokens, and free of
near-copies of the E2 validation/test articles. Each candidate is re-checked here
against EVERY article ever generated from (all single_*.jsonl inputs at the repo
root): exact text (case/punctuation-insensitive) and word 5-gram Jaccard >= 0.8.

Records carry an `order` field (0-based position in the sample). Every variant
screens on the first 150 and finalists extend to the full set, so the screening
set is a random subset of the full one by construction.
"""
import argparse, glob, json
from pathlib import Path

from make_extra_articles import NearDupIndex, fingerprint, shingles

ROOT = Path(__file__).resolve().parents[1]
INDEX = {"left": 1, "right": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="context-transfer/bignewsalign_full_LCR.jsonl")
    ap.add_argument("--pool", default="data/final/full1024_pool.jsonl")
    ap.add_argument("--start", type=int, default=10000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="experiments/eval500")
    ap.add_argument("--jaccard", type=float, default=0.8)
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already has files; the sample is fixed once written -- refusing to overwrite")

    index, seen = NearDupIndex(args.jaccard), set()
    n_existing = 0
    for path in sorted(glob.glob(str(ROOT / "single_*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            body = json.loads(line)["articles"][0]["body"]
            index.add(shingles(body)); seen.add(fingerprint(body)); n_existing += 1
    print(f"indexed {n_existing:,} previously generated articles")

    raw = open(ROOT / args.raw, "rb")
    def read(o):
        raw.seek(o[0]); return json.loads(raw.read(o[1]))

    pool = [json.loads(l) for l in open(ROOT / args.pool)]
    taken, skipped = [], 0
    for a in pool[args.start:]:
        if len(taken) >= args.n:
            break
        recs = {st: read(a[st]) for st in ("left", "right")}
        fps = {st: fingerprint(recs[st]["textString"]) for st in recs}
        shs = {st: shingles(recs[st]["textString"]) for st in recs}
        if any(f in seen for f in fps.values()) or any(index.near(s) for s in shs.values()):
            skipped += 1
            continue
        for st in recs:
            seen.add(fps[st]); index.add(shs[st])
        taken.append((a, recs))
    print(f"taken {len(taken)} events (skipped {skipped} as duplicates of earlier data)")

    out.mkdir(parents=True, exist_ok=True)
    for st in ("left", "right"):
        with (out / f"{st}.jsonl").open("w", encoding="utf-8") as f:
            for i, (a, recs) in enumerate(taken):
                r = recs[st]
                f.write(json.dumps({
                    "story_id": f"{a['group_uid']}:{st}", "source_story_id": a["group_uid"], "stance": st,
                    "order": i, "split": "eval", "article_id": f"{a['group_uid']}:{INDEX[st]}",
                    "words": a[f"{st}_words"], "led_tokens": a[f"{st}_tokens"],
                    "articles": [{"stance": st, "publisher": r["source"], "title": r["title"], "body": r["textString"]}],
                }, ensure_ascii=False) + "\n")
    print(f"wrote {out}/left.jsonl, {out}/right.jsonl")


if __name__ == "__main__":
    main()
