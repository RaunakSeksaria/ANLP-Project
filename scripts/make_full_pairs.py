"""Single-article neutralisation set drawn from the WHOLE BigNewsAlign export.

    uv run python scripts/make_full_pairs.py --raw context-transfer/bignewsalign_full_LCR.jsonl
    -> data/final/full1024_pool.jsonl         every accepted event (ids, lengths; no text)
       single_{left,right}_full1024.jsonl     generation inputs: a random prefix of the pool
       scratch/full1024_stats.json            counts for the EDA

Unit = one event's left and right article (the E2 triplet positive is the
opposite-side article about the same event).

1. Events are visited in a seeded random order, so any prefix of the accepted
   list is a random sample of the pool and preserves its length distribution.
2. An event is accepted when both articles
     - fit in --max-tokens LED tokens (title + body, specials included) -- dropped, never truncated,
     - have text never seen before (case/punctuation-insensitive),
     - are not near-copies of each other (outlets running the same wire story),
     - are not near-copies (estimated word 5-gram Jaccard >= --jaccard, MinHash
       LSH) of any accepted article, or of any E2 validation/test article, so an
       old held-out article never reappears.
3. The generation input is the shortest prefix of the accepted list holding at
   least --min-short articles under --short-tokens tokens.
4. Splits: by event, from a hash of the event id (--test/--val percent).
"""
import argparse, hashlib, json, random, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NPERM, BANDS = 64, 8
ROWS = NPERM // BANDS
P = (1 << 61) - 1
_rng = np.random.default_rng(12345)
A = _rng.integers(1, P, NPERM, dtype=np.uint64)
B = _rng.integers(0, P, NPERM, dtype=np.uint64)


def words(t): return re.findall(r"[a-z0-9]+", (t or "").lower())
def fingerprint(t): return hashlib.md5(" ".join(words(t)).encode()).digest()


def signature(text):
    w = words(text)
    sh = {" ".join(w[i:i + 5]) for i in range(max(len(w) - 4, 1))}
    h = np.fromiter((int.from_bytes(hashlib.blake2b(s.encode(), digest_size=7).digest(), "little") for s in sh),
                    dtype=np.uint64, count=len(sh))
    # (a*h + b) mod 2^64 is a fine universal-ish hash for MinHash purposes
    return (np.outer(A, h) + B[:, None]).min(axis=1)


class LSH:
    def __init__(self, cap):
        self.sigs = np.zeros((cap, NPERM), dtype=np.uint64)
        self.n = 0
        self.buckets = [defaultdict(list) for _ in range(BANDS)]

    def _keys(self, sig):
        return [hash(sig[b * ROWS:(b + 1) * ROWS].tobytes()) for b in range(BANDS)]

    def near(self, sig, th):
        cands = {i for b, k in enumerate(self._keys(sig)) for i in self.buckets[b].get(k, ())}
        return any((self.sigs[i] == sig).mean() >= th for i in cands)

    def add(self, sig):
        if self.n == len(self.sigs):
            self.sigs = np.vstack([self.sigs, np.zeros_like(self.sigs)])
        self.sigs[self.n] = sig
        for b, k in enumerate(self._keys(sig)):
            self.buckets[b][k].append(self.n)
        self.n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--short-tokens", type=int, default=512)
    ap.add_argument("--min-short", type=int, default=5500,
                    help="articles under --short-tokens the generation prefix must hold")
    ap.add_argument("--prefilter-words", type=int, default=1100,
                    help="cheap word cap before tokenising (1024 LED tokens is ~790 words)")
    ap.add_argument("--ratio", type=float, default=None,
                    help="estimate output tokens as input tokens x RATIO; with --min-short-per-side the "
                         "prefix target is per stance on estimated OUTPUT length")
    ap.add_argument("--min-short-per-side", type=int, default=None)
    ap.add_argument("--max-events", type=int, default=None, help="cap on the generation prefix")
    ap.add_argument("--prefix-only", action="store_true",
                    help="stop once the generation inputs are written (skip the full-pool count)")
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--test", type=int, default=10)
    ap.add_argument("--val", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="full1024")
    ap.add_argument("--tokenizer", default="allenai/led-base-16384")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # pass 1: byte offsets of every event's left and right article
    ev = defaultdict(dict)
    with open(args.raw, "rb") as f:
        off = 0
        for i, line in enumerate(f):
            if b'"political_bias": "center"' not in line and b'"political_bias":"center"' not in line:
                r = json.loads(line)
                st = r["political_bias"]
                if st in ("left", "right"):
                    nw = len((r["textString"] or "").split(" "))
                    if nw <= args.prefilter_words:
                        ev[str(r["group_uid"])][st] = (off, len(line))
            off += len(line)
            if i % 200000 == 0:
                print(f"  pass 1: {i:,} lines", flush=True)
    gids = sorted(g for g, v in ev.items() if "left" in v and "right" in v)
    print(f"events with left and right both <= {args.prefilter_words} words: {len(gids):,}", flush=True)

    # held-out E2 articles must never reappear
    lsh, seen = LSH(1 << 20), set()
    held = 0
    for split in ("validation", "test"):
        p = ROOT / f"dataset/side_neutral_full/prepared/{split}/baseline.jsonl"
        if p.exists():
            for line in p.open(encoding="utf-8"):
                t = json.loads(line)["input_text"]
                lsh.add(signature(t)); seen.add(fingerprint(t)); held += 1
    held_n = lsh.n
    print(f"held-out E2 val/test articles indexed: {held}", flush=True)

    random.Random(args.seed).shuffle(gids)
    skipped = defaultdict(int)
    accepted = []
    raw = open(args.raw, "rb")

    def read(o):
        raw.seek(o[0])
        return json.loads(raw.read(o[1]))

    INDEX = {"left": 1, "right": 2}

    def write_inputs(prefix):
        # written to .tmp then renamed, so a reader never sees a half-written file
        for st in ("left", "right"):
            out = ROOT / f"single_{st}_{args.tag}.jsonl"
            tmp = out.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fo:
                for a in prefix:
                    r = read(a[st])
                    fo.write(json.dumps({
                        "story_id": f"{a['group_uid']}:{st}", "source_story_id": a["group_uid"], "stance": st,
                        "split": a["split"], "article_id": f"{a['group_uid']}:{INDEX[st]}",
                        "words": a[f"{st}_words"], "led_tokens": a[f"{st}_tokens"],
                        "articles": [{"stance": st, "publisher": r["source"], "title": r["title"], "body": r["textString"]}],
                    }, ensure_ascii=False) + "\n")
            tmp.rename(out)

    prefix = None
    CH = 1000
    for c in range(0, len(gids), CH):
        chunk = gids[c:c + CH]
        recs = [(g, read(ev[g]["left"]), read(ev[g]["right"])) for g in chunk]
        texts = [f"{r['title'] or ''}\n{r['textString'] or ''}" for _, L, R in recs for r in (L, R)]
        ntoks = [len(x) for x in tok(texts)["input_ids"]]
        for j, (g, L, R) in enumerate(recs):
            tl, tr = ntoks[2 * j], ntoks[2 * j + 1]
            if tl > args.max_tokens or tr > args.max_tokens:
                skipped["over token cap"] += 1; continue
            fl, fr = fingerprint(L["textString"]), fingerprint(R["textString"])
            if fl in seen or fr in seen or fl == fr:
                skipped["text already seen"] += 1; continue
            sl, sr = signature(L["textString"]), signature(R["textString"])
            if (sl == sr).mean() >= args.jaccard:
                skipped["left/right same wire copy"] += 1; continue
            if lsh.near(sl, args.jaccard) or lsh.near(sr, args.jaccard):
                skipped["near-copy of accepted or held-out article"] += 1; continue
            seen.update((fl, fr)); lsh.add(sl); lsh.add(sr)
            h = int(hashlib.md5(g.encode()).hexdigest(), 16) % 100
            split = "test" if h < args.test else "validation" if h < args.test + args.val else "train"
            accepted.append({"group_uid": g, "split": split, "left": ev[g]["left"], "right": ev[g]["right"],
                             "left_tokens": tl, "right_tokens": tr,
                             "left_words": len(L["textString"].split()), "right_words": len(R["textString"].split())})
        if prefix is None:
            if args.min_short_per_side:
                lim = args.short_tokens / (args.ratio or 1.0)        # input length whose output fits
                cl = cr = k = 0
                cap = args.max_events or len(accepted)
                while k < min(len(accepted), cap) and (cl < args.min_short_per_side or cr < args.min_short_per_side):
                    cl += accepted[k]["left_tokens"] < lim; cr += accepted[k]["right_tokens"] < lim
                    k += 1
                done = (cl >= args.min_short_per_side and cr >= args.min_short_per_side) or k >= cap
                short = f"est. output <{args.short_tokens}: left {cl:,}, right {cr:,}"
            else:
                short, k = 0, 0
                while k < len(accepted) and short < args.min_short:
                    short += (accepted[k]["left_tokens"] < args.short_tokens) + (accepted[k]["right_tokens"] < args.short_tokens)
                    k += 1
                done = short >= args.min_short
            if done:
                prefix = accepted[:k]
                write_inputs(prefix)
                print(f"  generation inputs written: {k:,} events, {short}", flush=True)
                if args.prefix_only:
                    print("  --prefix-only: stopping"); return
        if (c // CH) % 20 == 0:
            print(f"  pass 2: {c + CH:,}/{len(gids):,} events visited, {len(accepted):,} accepted", flush=True)

    pool_path = ROOT / f"data/final/{args.tag}_pool.jsonl"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    with pool_path.open("w") as fo:
        for a in accepted:
            fo.write(json.dumps(a) + "\n")

    if prefix is None:                      # pool too small to reach --min-short
        prefix = accepted
        write_inputs(prefix)
    k = len(prefix)
    short = sum((a["left_tokens"] < args.short_tokens) + (a["right_tokens"] < args.short_tokens) for a in prefix)

    tok_all = [a[f"{s}_tokens"] for a in accepted for s in ("left", "right")]
    stats = {
        "events_considered": len(gids), "held_out_indexed": held_n, "skipped": dict(skipped),
        "pool_events": len(accepted), "pool_articles": 2 * len(accepted),
        "pool_articles_under_short": int(sum(t < args.short_tokens for t in tok_all)),
        "prefix_events": k, "prefix_articles": 2 * k, "prefix_articles_under_short": short,
        "prefix_splits": {s: sum(a["split"] == s for a in prefix) for s in ("train", "validation", "test")},
    }
    (ROOT / "scratch").mkdir(exist_ok=True)
    json.dump(stats, open(ROOT / f"scratch/{args.tag}_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
