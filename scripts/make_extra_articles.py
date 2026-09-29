"""New unique single articles from the raw export, to grow a stance past the 10k-event corpus.

    uv run python scripts/make_extra_articles.py --raw data/final/bignewsalign_max250_LCR.jsonl \
        --stance right --target 10000
    -> single_right_extra.jsonl

The 10k events reuse articles heavily (5,695 unique left, 6,160 unique right),
so reaching N unique articles per stance means drawing on the rest of the
export. An article is taken only if it is not already in <stance>_neutral_dedup.jsonl
and is not a near-copy (word 5-gram Jaccard >= 0.8) of any article already in the
dataset -- in either stance, in any split -- or of another article taken here.
New articles all go to train, so val and test stay exactly as released. Shortest
first, as before.

Output records have the shape generate_neutral.py reads, plus the split and
article_id that build_single_release.py needs for articles outside the 10k release.
"""
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = {"center": 0, "left": 1, "right": 2}      # article_id suffix, as in the 10k release


def fingerprint(t): return " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))
def shingles(t, n=5):
    w = fingerprint(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}
def sketch(sh, k=8):
    """Bottom-k hashes: two near-copies almost always share at least one."""
    return sorted(int(hashlib.md5(s.encode()).hexdigest()[:12], 16) for s in sh)[:k]


class NearDupIndex:
    def __init__(self, threshold):
        self.t, self.docs, self.by_hash = threshold, [], defaultdict(list)

    def add(self, sh):
        i = len(self.docs)
        self.docs.append(sh)
        for h in sketch(sh):
            self.by_hash[h].append(i)

    def near(self, sh):
        cands = {i for h in sketch(sh) for i in self.by_hash.get(h, ())}
        return any(len(sh & self.docs[i]) / max(len(sh | self.docs[i]), 1) >= self.t for i in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw export, one article per line")
    ap.add_argument("--stance", choices=("left", "right"), required=True)
    ap.add_argument("--target", type=int, default=10000, help="unique articles wanted in total")
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--out", default=None, help="default single_<stance>_extra.jsonl")
    args = ap.parse_args()
    st = args.stance
    out = Path(args.out or ROOT / f"single_{st}_extra.jsonl")

    # everything already in the dataset, both stances, all splits
    index, have_fp, have_ids = NearDupIndex(args.jaccard), set(), set()
    for s in ("left", "right"):
        for line in (ROOT / f"{s}_neutral_dedup.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            body = r[f"{s}_original"]["body"]
            index.add(shingles(body))
            if s == st:
                have_fp.add(fingerprint(body))
    for line in (ROOT / "neutral_articles_10k.jsonl").open(encoding="utf-8"):
        have_ids.add(json.loads(line)["story_id"])
    have = len(have_fp)

    cands, seen = [], set()
    for line in open(args.raw, encoding="utf-8"):
        r = json.loads(line)
        if r["political_bias"] != st:
            continue
        f = fingerprint(r["textString"])
        if len(f.split()) < args.min_words or f in have_fp or f in seen:
            continue
        seen.add(f)
        cands.append(r)
    cands.sort(key=lambda r: (len(r["textString"].split()), str(r["group_uid"])))

    need = max(args.target - have, 0)
    taken, near = [], 0
    for r in cands:
        if len(taken) >= need:
            break
        sh = shingles(r["textString"])
        if index.near(sh):
            near += 1
            continue
        index.add(sh)
        taken.append(r)

    with out.open("w", encoding="utf-8") as fo:
        for r in taken:
            gid = str(r["group_uid"])
            assert gid not in have_ids, f"group_uid {gid} is already an event in the 10k release"
            fo.write(json.dumps({
                "story_id": f"{gid}:{st}", "source_story_id": gid, "stance": st,
                "split": "train", "article_id": f"{gid}:{INDEX[st]}",
                "articles": [{"stance": st, "publisher": r["source"],
                              "title": r["title"], "body": r["textString"]}],
            }, ensure_ascii=False) + "\n")
    print(f"{st}: have {have:,}, new candidates {len(cands):,}, near-copies skipped {near:,}, "
          f"taken {len(taken):,} (wanted {need:,}) -> {out.name}; total would be {have + len(taken):,}")


if __name__ == "__main__":
    main()
