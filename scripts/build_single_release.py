"""One self-contained file for the single-article neutralisation task.

    uv run python scripts/build_single_release.py --stance left
    -> left_neutral_10k.jsonl

Each record is one article and its neutralised rewrite. The split assignment is
taken verbatim from neutral_articles_10k.jsonl, so the fusion and neutralise
corpora share splits event-for-event and no article body crosses a split.
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})
def flat(s): return " ".join((s or "").translate(_FLAT).split())
def fingerprint(s):
    """Case- and punctuation-insensitive text. Some outlets ran the same article
    twice, differing only in straight vs curly quotes; exact text misses those."""
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))
def jdump(o):
    return (json.dumps(o, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stance", choices=("left", "right"), required=True)
    ap.add_argument("--run", nargs="+", default=None,
                    help="default runs/nemotron/neutralised_<stance>.jsonl plus any "
                         "runs/nemotron/extra_<stance>_*.jsonl (disjoint slices run on other keys)")
    ap.add_argument("--inputs", nargs="+", default=None,
                    help="default single_<stance>_10k.jsonl plus every single_<stance>_extra*.jsonl")
    ap.add_argument("--out", default=None,
                    help="default <stance>_neutral_10k.jsonl, or <stance>_neutral_dedup.jsonl with --dedupe")
    ap.add_argument("--dedupe", action="store_true",
                    help="one record per unique input article. The 10k are events and one "
                         "article belongs to several of them; every copy sits in the same split, "
                         "so keeping the first loses no split information")
    args = ap.parse_args()
    st = args.stance
    runs = ([Path(r) for r in args.run] if args.run else
            [ROOT / f"runs/nemotron/neutralised_{st}.jsonl",
             *sorted((ROOT / "runs/nemotron").glob(f"extra_{st}_*.jsonl"))])
    inputs = ([Path(i) for i in args.inputs] if args.inputs else
              [ROOT / f"single_{st}_10k.jsonl", *sorted(ROOT.glob(f"single_{st}_extra*.jsonl"))])
    out = Path(args.out or ROOT / (f"{st}_neutral_dedup.jsonl" if args.dedupe else f"{st}_neutral_10k.jsonl"))

    # split + article_id, from the fusion release (same 10k events)
    meta = {}
    for line in (ROOT / "neutral_articles_10k.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        meta[r["story_id"]] = (r["split"], r[st])

    src = {}
    for path in inputs:
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            src[r["source_story_id"]] = r["articles"][0]
            # articles from outside the 10k release carry their own split and id
            if r["source_story_id"] not in meta and "split" in r:
                meta[r["source_story_id"]] = (r["split"], {"article_id": r["article_id"],
                                                          "body": flat(r["articles"][0]["body"])})

    # the run file is append-only: a failed attempt can precede a later success,
    # so keep the last successful record per id
    gen = {}
    for run in runs:
        for line in run.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("summary"):
                gen[r["story_id"].rsplit(":", 1)[0]] = r
    print("runs:", ", ".join(r.name for r in runs))

    missing = sorted(set(src) - set(gen))
    mismatched = 0
    order = sorted(gen, key=lambda s: ({"train": 0, "val": 1, "test": 2}.get(meta.get(s, ("",))[0], 3), s))

    # every event that carries each article, so a deduplicated record still says
    # which events it stands for
    events = {}
    for sid in sorted(src):
        events.setdefault(fingerprint(src[sid]["body"]), []).append(sid)
    if args.dedupe:
        missing = [b for b, ids in events.items() if not any(i in gen for i in ids)]
    seen = set()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for sid in order:
            if sid not in src or sid not in meta:
                continue
            split, rel = meta[sid]
            a, g = src[sid], gen[sid]
            if flat(a["body"]) != rel["body"]:
                mismatched += 1          # the input is not the article the release holds
                continue
            key = fingerprint(a["body"])
            if args.dedupe:
                if key in seen:
                    continue
                seen.add(key)
            orig = {"article_id": rel["article_id"], "publisher": a["publisher"],
                    "title": flat(a["title"]), "body": flat(a["body"])}
            params = g.get("params") or {}
            rec = {
                "story_id": sid,
                "split": split,
                f"{st}_original": orig,
                f"{st}_neutralised": flat(g["summary"]),
                "source": f"TITLE=> {orig['title']} . ARTICLE=> {orig['body']}",
                "edits": [flat(e) for e in g.get("conflicts") or []],
                "generator": {"model": g.get("model"),
                              "prompt_version": params.get("prompt_version"),
                              "temperature": params.get("temperature")},
            }
            if args.dedupe:
                rec["event_ids"] = events[key]       # all events sharing this article
            f.write(jdump(rec) + "\n")
            n += 1
    print(f"wrote {out.name}: {n:,} records")
    print(f"  {'articles' if args.dedupe else 'inputs'} without a generated article: {len(missing):,}")
    print(f"  body mismatch against release (dropped): {mismatched:,}")


if __name__ == "__main__":
    main()
