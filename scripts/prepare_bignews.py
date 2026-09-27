"""Build a trainable corpus from the BigNewsAlign L/C/R clusters.

    uv run python scripts/prepare_bignews.py

Interim target: the CENTRE article stands in for the neutral summary until the
LLM-generated targets exist. That forces one design choice --

    source = left + right      target = centre

-- because a centre article that appears in the input is a copy task, not a
summarisation one. So this is 2-source, where the proposal's setup is 3-source.
Swapping in real neutral targets later restores the third source; the file
layout here is identical either way, so only --data-dir changes.

Splits are by CONNECTED COMPONENT over shared articles, not by event. 57.7% of
events share an article with another event, so a random split would put the same
text in train and test.

Emits into data/prepared_bna/, matching what dead_code/train_neus.py and
dead_code/probe.py already read:

  gen/{train,val,test}.{source,target}   NeuS's TITLE=>/ARTICLE=> [SEP] format
  articles.jsonl                         every article, with split membership
  triplets.jsonl                         anchor / positive (same event, other
                                         publisher) / negative (same publisher,
                                         other event)
  probe_random.json, probe_media.json    article-id splits for the leakage probe
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# json.dumps(ensure_ascii=False) passes U+2028/U+2029 through as raw characters.
# They are legal inside a JSON string and safe for line-by-line readers, but
# str.splitlines() breaks on them -- which is how a stray one silently shifted
# this corpus's targets against its sources once already. Escape them so the
# files are safe under any reader.
def jdump(obj):
    return (json.dumps(obj, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))

WORD = re.compile(r"[a-z0-9]+")
SEED = 42


def norm(t):
    return " ".join(WORD.findall(t.lower()))


def ahash(body):
    return hashlib.md5(norm(body).encode()).hexdigest()


# Every character Python's str.splitlines() treats as a line boundary. The
# generation files are line-aligned and train_neus.py reads them with
# .read_text().splitlines(), so ONE stray \r or \u2028 inside an article
# silently shifts every later target against its source. read_text() also
# translates \r to \n, which is why wc -l and Python disagree about these files.
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})


def flatten(s):
    return " ".join((s or "").translate(_FLAT).split())


def neus_format(articles):
    """NeuS's own source encoding, so the existing training script is unchanged."""
    return " [SEP] ".join(
        f"TITLE=> {flatten(a['title'])} . ARTICLE=> {flatten(a['body'])}"
        for a in articles)


STOP = set("a an the and or of to in on at by for with from as is are was were be been it its "
           "this that he she they we you his her their our not no so than then there here which "
           "who what when where why how all any both each more most other some such only own same "
           "too very can will just should now said say says over under after before during about "
           "into out up down off again further once would could has have had do does did".split())


def _content(t):
    return [w for w in re.findall(r"[a-z0-9']+", (t or "").lower())
            if w not in STOP and len(w) > 2]


def target_overlap(articles, target_stance, n=8):
    """Fraction of the target's content 8-grams that already appear in the input.

    Near 0 for a genuine summarisation pair; near 1 when outlets ran the same
    wire story and the 'target' is just one of the inputs again.
    """
    tgt = [a for a in articles if a["stance"] == target_stance]
    src = [a for a in articles if a["stance"] != target_stance]
    if not tgt or not src:
        return 1.0
    # Measured on the formatted strings that actually ship, titles included --
    # measuring raw bodies here let a record through at 0.52 that the validator
    # then caught, because the title contributes content words too.
    c = _content(neus_format(tgt[:1]))
    s = _content(neus_format(src))
    if len(c) < n:
        return 0.0
    g = {tuple(c[i:i + n]) for i in range(len(c) - n + 1)}
    gs = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    return len(g & gs) / len(g) if g else 0.0


def components(events):
    """Union-find over events that share an article text."""
    parent = {e: e for e in events}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_art = defaultdict(list)
    for eid, arts in events.items():
        for a in arts:
            by_art[ahash(a["body"])].append(eid)
    for ids in by_art.values():
        for other in ids[1:]:
            ra, rb = find(ids[0]), find(other)
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for eid in events:
        groups[find(eid)].append(eid)
    return list(groups.values())


def split_components(events, ratios=(0.8, 0.1, 0.1), seed=SEED):
    """Largest-first bin packing, so no article crosses a split boundary."""
    comps = components(events)
    comps.sort(key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(comps[len(comps) // 2:])      # break ties among the small ones
    total = len(events)
    quota = {"train": ratios[0] * total, "val": ratios[1] * total,
             "test": ratios[2] * total}
    have = {k: 0 for k in quota}
    out = {}
    for comp in comps:
        name = max(quota, key=lambda k: quota[k] - have[k])
        have[name] += len(comp)
        for eid in comp:
            out[eid] = name
    return out, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "data/final/clusters_10k_shortest.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/prepared_bna"))
    ap.add_argument("--target-stance", default="center",
                    help="which stance stands in for the neutral summary")
    ap.add_argument("--max-triplets", type=int, default=40000)
    ap.add_argument("--target-events", type=int, default=0,
                    help="after filtering, keep exactly this many events, shortest "
                         "first. Feed a pool larger than N so the filter has room; "
                         "0 keeps everything that survives.")
    ap.add_argument("--max-target-overlap", type=float, default=0.5,
                    help="drop events where this fraction of the target's 8-grams "
                         "already appears in the source. Outlets republish wire "
                         "copy, so a few percent of centre articles are near-copies "
                         "of a left/right article -- those are a copy task, not a "
                         "summarisation one. 0 disables the filter.")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    events = {}
    for line in open(args.inp, encoding="utf-8"):
        c = json.loads(line)
        events[str(c["story_id"])] = c["articles"]
    print(f"read {len(events):,} events from {Path(args.inp).name}")

    if args.max_target_overlap:
        before = len(events)
        events = {e: a for e, a in events.items()
                  if target_overlap(a, args.target_stance) < args.max_target_overlap}
        if before != len(events):
            print(f"dropped {before - len(events):,} event(s) whose "
                  f"{args.target_stance} article is largely reproduced in the "
                  f"other sources (>={args.max_target_overlap:.0%} of 8-grams)")

    if args.target_events:
        if len(events) < args.target_events:
            raise SystemExit(
                f"only {len(events):,} event(s) survive the filters but "
                f"--target-events {args.target_events:,} was asked for. Rebuild the "
                f"pool with a larger --limit in make_clusters.py.")
        # Shortest first, story_id as the tiebreak, matching make_clusters.py so
        # topping up a corpus extends the same ordering rather than reshuffling it.
        keep = sorted(events, key=lambda e: (sum(len(a["body"].split())
                                                 for a in events[e]), e))
        keep = set(keep[:args.target_events])
        print(f"kept the {args.target_events:,} shortest of {len(events):,} "
              f"surviving event(s)")
        events = {e: a for e, a in events.items() if e in keep}

    assign, comps = split_components(events, seed=args.seed)
    print(f"connected components: {len(comps):,}  (largest {len(comps[0])})")
    print(f"split: " + "  ".join(
        f"{k}={sum(1 for v in assign.values() if v == k):,}"
        for k in ("train", "val", "test")))

    out = Path(args.out)
    (out / "gen").mkdir(parents=True, exist_ok=True)

    # ---- generation pairs: source = every non-target stance, target = centre --
    counts = Counter()
    dropped = 0
    # {split}.ids records which event produced each line, in order. Without it
    # anything downstream has to re-derive the order from the cluster file, which
    # breaks the moment the cluster file changes.
    handles = {s: ((out / "gen" / f"{s}.source").open("w", encoding="utf-8"),
                   (out / "gen" / f"{s}.target").open("w", encoding="utf-8"),
                   (out / "gen" / f"{s}.ids").open("w", encoding="utf-8"))
               for s in ("train", "val", "test")}
    rng = random.Random(args.seed)
    for eid, arts in events.items():
        tgt = [a for a in arts if a["stance"] == args.target_stance]
        src = [a for a in arts if a["stance"] != args.target_stance]
        if not tgt or not src:
            dropped += 1
            continue
        # Seeded per event: a fixed left-then-right order would let the model
        # learn a positional convention rather than read both sources.
        src = sorted(src, key=lambda a: a["stance"])
        random.Random(f"{args.seed}:{eid}").shuffle(src)
        s = assign[eid]
        fs, ft, fi = handles[s]
        fi.write(eid + "\n")
        fs.write(neus_format(src) + "\n")
        ft.write(f"TITLE=> {flatten(tgt[0]['title'])} . ARTICLE=> "
                 f"{flatten(tgt[0]['body'])}\n")
        counts[s] += 1
    for hs in handles.values():
        for h in hs:
            h.close()
    if dropped:
        print(f"dropped {dropped} event(s) with no {args.target_stance} article")
    print(f"gen pairs: " + "  ".join(f"{k}={counts[k]:,}" for k in ("train", "val", "test")))

    # ---- articles.jsonl -----------------------------------------------------
    articles, by_event, by_pub = [], defaultdict(list), defaultdict(list)
    for eid, arts in events.items():
        for i, a in enumerate(arts):
            aid = f"{eid}:{i}"
            rec = {"article_id": aid, "story_id": eid, "stance": a["stance"],
                   "publisher": a["publisher"], "title": a["title"],
                   "body": a["body"], "split": assign[eid]}
            articles.append(rec)
            by_event[eid].append(rec)
            by_pub[a["publisher"]].append(rec)
    with (out / "articles.jsonl").open("w", encoding="utf-8") as f:
        for r in articles:
            f.write(jdump(r) + "\n")
    print(f"articles.jsonl: {len(articles):,}")

    # ---- triplets -----------------------------------------------------------
    # positive shares the EVENT, negative shares the PUBLISHER. The encoder is
    # pulled toward event identity and away from source identity.
    rng = random.Random(args.seed)
    triples = []
    for eid, recs in by_event.items():
        for anchor in recs:
            pos = [r for r in recs if r["publisher"] != anchor["publisher"]]
            neg = [r for r in by_pub[anchor["publisher"]] if r["story_id"] != eid
                   and assign[r["story_id"]] == assign[eid]]
            if not pos or not neg:
                continue
            triples.append({"anchor": anchor["article_id"],
                            "positive": rng.choice(pos)["article_id"],
                            "negative": rng.choice(neg)["article_id"],
                            "split": assign[eid]})
    rng.shuffle(triples)
    triples = triples[:args.max_triplets]
    with (out / "triplets.jsonl").open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    print(f"triplets.jsonl: {len(triples):,}  "
          + "  ".join(f"{k}={sum(1 for t in triples if t['split'] == k):,}"
                      for k in ("train", "val", "test")))

    # ---- probe splits -------------------------------------------------------
    ids = [a["article_id"] for a in articles]
    rng.shuffle(ids)
    n = len(ids)
    json.dump({"train": ids[:int(.8 * n)], "val": ids[int(.8 * n):int(.9 * n)],
               "test": ids[int(.9 * n):]},
              (out / "probe_random.json").open("w"))

    # publisher-disjoint: no outlet on both sides. Stances are assigned round
    # robin so a stance cannot vanish from the test side.
    pubs = defaultdict(list)
    for p, recs in by_pub.items():
        pubs[recs[0]["stance"]].append(p)
    media = {"train": [], "val": [], "test": []}
    for stance, ps in pubs.items():
        ps = sorted(ps, key=lambda p: -len(by_pub[p]))
        for i, p in enumerate(ps):
            bucket = "train" if i % 4 < 2 else ("val" if i % 4 == 2 else "test")
            media[bucket] += [r["article_id"] for r in by_pub[p]]
    json.dump(media, (out / "probe_media.json").open("w"))
    print(f"probe_random: {n:,} ids   probe_media: "
          + "  ".join(f"{k}={len(v):,}" for k, v in media.items()))
    print(f"\nwrote {out.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
