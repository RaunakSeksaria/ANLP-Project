"""Expand event clusters into one-article inputs, one file per stance.

    uv run python scripts/make_single_article_inputs.py

Writes data/final/single_{stance}_10k.jsonl. Each record is a "cluster" holding
exactly ONE article, so scripts/generate_neutral.py --task neutralise consumes
it with no special-casing.

One file per stance on purpose: the left and right runs are meant to go in
parallel, on separate API keys, writing to separate outputs. Nothing is shared,
so either can die without touching the other.

  story_id "42806:left"  -- stance-suffixed so left and right never collide if
                            the outputs are later concatenated.
"""
import argparse, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})
def flat(s): return " ".join((s or "").translate(_FLAT).split())
def jdump(o):
    return (json.dumps(o, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "data/final/clusters_corpus10k.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/final"))
    ap.add_argument("--stances", default="left,right")
    args = ap.parse_args()

    stances = [s.strip() for s in args.stances.split(",") if s.strip()]
    out = {s: (Path(args.out_dir) / f"single_{s}_10k.jsonl").open("w", encoding="utf-8")
           for s in stances}
    counts, words = Counter(), Counter()
    for line in open(args.inp, encoding="utf-8"):
        c = json.loads(line)
        for a in c["articles"]:
            st = a["stance"]
            if st not in out:
                continue
            out[st].write(jdump({
                "story_id": f"{c['story_id']}:{st}",
                "source_story_id": str(c["story_id"]),
                "stance": st,
                "articles": [{"stance": st, "publisher": a["publisher"],
                              "title": flat(a["title"]), "body": flat(a["body"])}],
            }) + "\n")
            counts[st] += 1
            words[st] += len(a["body"].split())
    for f in out.values():
        f.close()
    for s in stances:
        print(f"  single_{s}_10k.jsonl  {counts[s]:,} articles   "
              f"mean {words[s]/max(counts[s],1):.0f} words")


if __name__ == "__main__":
    main()
