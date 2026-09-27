"""Turn the BigNewsAlign L/C/R export into cluster JSONL for the generator.

    uv run python scripts/make_clusters.py \
        --in data/final/bignewsalign_max250_LCR.jsonl \
        --out data/final/clusters_10k_shortest.jsonl --limit 10000

The input is one article per line (group_uid, title, source, political_bias,
textString). The output is one event per line in the shape scripts/
generate_neutral.py reads:

    {"story_id": ..., "articles": [{"stance", "publisher", "title", "body"}, ...]}

--sort shortest (the default) takes the n events with the fewest total words
across their articles, which is both the cheapest to generate and the least
likely to be truncated. Ties are broken by story_id so the selection is
reproducible.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_articles(path, fields):
    gid, title, source, bias, text = fields
    groups = defaultdict(list)
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            body = (r.get(text) or "").strip()
            if not body or gid not in r:
                skipped += 1
                continue
            groups[str(r[gid])].append({
                "stance": (r.get(bias) or "").strip().lower(),
                "publisher": (r.get(source) or "").strip(),
                "title": (r.get(title) or "").strip(),
                "body": body,
            })
    return groups, skipped


_WORD = re.compile(r"[a-z0-9]+")


def _fingerprint(articles):
    """Identity of an event: the multiset of its normalised article texts.

    Whitespace, case and punctuation are ignored, and article order does not
    matter, so the same three articles under a different group_uid collide.
    """
    per = sorted(hashlib.md5(" ".join(_WORD.findall(a["body"].lower())).encode()).hexdigest()
                 for a in articles)
    return hashlib.md5("||".join(per).encode()).hexdigest()


def dedup(groups):
    """Keep one event per fingerprint, choosing the lowest id for reproducibility."""
    best = {}
    for gid, arts in groups.items():
        fp = _fingerprint(arts)
        cur = best.get(fp)
        if cur is None or (len(gid), gid) < (len(cur), cur):
            best[fp] = gid
    keep = set(best.values())
    return {k: v for k, v in groups.items() if k in keep}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=10000,
                   help="how many events to keep (0 = all)")
    p.add_argument("--sort", choices=("shortest", "longest", "random", "none"),
                   default="shortest")
    p.add_argument("--seed", type=int, default=42, help="for --sort random")
    p.add_argument("--require-stances", default="left,center,right",
                   help="comma-separated; drop events missing any of them. "
                        "Empty string keeps every event.")
    p.add_argument("--no-dedup", action="store_true",
                   help="keep events whose article texts duplicate another event's. "
                        "30%% of this corpus is exact duplicates under different ids, "
                        "so leaving them in wastes generation budget and contaminates "
                        "any train/test split.")
    p.add_argument("--fields", default="group_uid,title,source,political_bias,textString",
                   help="input key names: id,title,publisher,stance,body")
    args = p.parse_args(argv)

    fields = [s.strip() for s in args.fields.split(",")]
    if len(fields) != 5:
        sys.exit("--fields needs exactly 5 comma-separated names")

    groups, skipped = read_articles(args.inp, fields)
    if not groups:
        sys.exit(f"no usable articles in {args.inp}")
    print(f"read {sum(len(v) for v in groups.values()):,} articles in "
          f"{len(groups):,} events"
          + (f" ({skipped:,} lines skipped)" if skipped else ""))

    required = {s.strip().lower() for s in args.require_stances.split(",") if s.strip()}
    if required:
        before = len(groups)
        groups = {k: v for k, v in groups.items()
                  if required <= {a["stance"] for a in v}}
        if before != len(groups):
            print(f"dropped {before - len(groups):,} event(s) missing one of "
                  f"{sorted(required)}")

    if not args.no_dedup:
        before = len(groups)
        groups = dedup(groups)
        if before != len(groups):
            print(f"dropped {before - len(groups):,} duplicate event(s) "
                  f"({100*(before-len(groups))/before:.1f}%); kept the lowest id of each set")

    keys = sorted(groups)
    words = {k: sum(len(a["body"].split()) for a in groups[k]) for k in keys}
    if args.sort in ("shortest", "longest"):
        # story_id as the tiebreaker keeps the selection reproducible at the
        # cut point, where many events share a word count.
        keys.sort(key=lambda k: (words[k], k), reverse=args.sort == "longest")
    elif args.sort == "random":
        import random
        random.Random(args.seed).shuffle(keys)

    kept = keys[:args.limit] if args.limit else keys

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for k in kept:
            # Stable article order within an event; the generator shuffles for
            # the prompt anyway, seeded on the story id.
            arts = sorted(groups[k], key=lambda a: (a["stance"], a["publisher"]))
            f.write(json.dumps({"story_id": k, "articles": arts},
                               ensure_ascii=False) + "\n")

    sel = [words[k] for k in kept]
    per_art = [len(a["body"].split()) for k in kept for a in groups[k]]
    print(f"\nwrote {len(kept):,} events -> {out}")
    print(f"  total words/event   min {min(sel)}  median {int(statistics.median(sel))}  "
          f"max {max(sel)}  mean {statistics.fmean(sel):.0f}")
    print(f"  words/article       min {min(per_art)}  median "
          f"{int(statistics.median(per_art))}  max {max(per_art)}")
    print(f"  articles/event      {Counter(len(groups[k]) for k in kept).most_common()}")
    print(f"  stances             {dict(Counter(a['stance'] for k in kept for a in groups[k]))}")
    pubs = Counter(a["publisher"] for k in kept for a in groups[k])
    print(f"  publishers ({len(pubs)})      {pubs.most_common()}")

    # A publisher that only ever appears with one stance makes the leakage probe
    # trivial -- worth knowing before these become training data.
    by_pub = defaultdict(set)
    for k in kept:
        for a in groups[k]:
            by_pub[a["publisher"]].add(a["stance"])
    single = [p for p, s in by_pub.items() if len(s) == 1]
    if single:
        print(f"\n  note: {len(single)}/{len(by_pub)} publishers carry exactly one "
              f"stance, so stance is\n  almost fully determined by publisher in this "
              f"subset. Expected for AllSides-style\n  labels, but it makes the "
              f"publisher probe and the stance label near-redundant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
