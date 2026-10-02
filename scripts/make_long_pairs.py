"""Longer-article train set: N events, each giving one left and one right article.

    uv run python scripts/make_long_pairs.py --raw context-transfer/bignewsalign_full_LCR.jsonl --n 1000
    -> single_left_long1k.jsonl, single_right_long1k.jsonl

The existing single-article sets top out at 275 words. This draws longer
articles from the full BigNewsAlign export, under the same event rules: an
event qualifies only if its left AND right article both qualify, so every
article has an opposite-side partner about the same event (what the triplet
objective needs for a positive -- see dataset/side_neutral_full/README.md).

An article qualifies when
  - it is longer than --min-words (default 275: above every existing article)
    and at most --max-words (default 1000),
  - title + body fit in --max-tokens (default 1024) LED tokens, specials
    included; longer articles are dropped, never truncated,
  - its text is new: not already in the dataset, and not a near-copy (word
    5-gram Jaccard >= 0.8) of any existing article in either stance and any
    split, or of an article already taken here.
The event must not already be represented anywhere in the dataset, and its
left and right articles must not be near-copies of each other (outlets often
run the same wire story). Events are
drawn in a seeded random order rather than shortest-first, so the lengths
spread across the whole allowed range. Every record goes to train.
"""
import argparse, json, random
from pathlib import Path

from make_extra_articles import INDEX, NearDupIndex, fingerprint, shingles

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n", type=int, default=1000, help="events, i.e. articles per stance")
    ap.add_argument("--min-words", type=int, default=275)
    ap.add_argument("--max-words", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tokenizer", default="allenai/led-base-16384")
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="long1k")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ntok = lambda r: len(tok(f"{r['title'] or ''}\n{r['textString'] or ''}")["input_ids"])
    nwords = lambda r: len((r["textString"] or "").split(" "))     # as the export counts

    # everything already in the dataset: texts (both stances, all splits) and events
    index, have_fp, have_events = NearDupIndex(args.jaccard), set(), set()
    for s in ("left", "right"):
        for line in (ROOT / f"{s}_neutral_dedup.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            body = r[f"{s}_original"]["body"]
            index.add(shingles(body))
            have_fp.add(fingerprint(body))
            have_events.update(map(str, r.get("event_ids") or [r["story_id"]]))
    for line in (ROOT / "neutral_articles_10k.jsonl").open(encoding="utf-8"):
        have_events.add(json.loads(line)["story_id"])
    for path in ROOT.glob("single_*_*.jsonl"):
        long_set = "_long" in path.name
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            have_events.add(str(r["source_story_id"]))
            if long_set:                 # earlier long batches are text-deduped against too
                body = r["articles"][0]["body"]
                index.add(shingles(body))
                have_fp.add(fingerprint(body))
    side = ROOT / "dataset/side_neutral_full/prepared"
    for path in side.glob("*/baseline.jsonl"):
        for line in path.open(encoding="utf-8"):
            have_events.update(map(str, json.loads(line).get("event_ids") or []))
    print(f"existing: {len(have_fp):,} articles, {len(have_events):,} events")

    # candidate events: left and right both within the word range
    events = {}
    for line in open(args.raw, encoding="utf-8"):
        r = json.loads(line)
        st, gid = r["political_bias"], str(r["group_uid"])
        if st not in ("left", "right") or gid in have_events:
            continue
        if not args.min_words < nwords(r) <= args.max_words:
            continue
        events.setdefault(gid, {})[st] = r
    cands = sorted(g for g, v in events.items() if "left" in v and "right" in v)
    print(f"candidate events (left and right both {args.min_words + 1}-{args.max_words} words): {len(cands):,}")

    random.Random(args.seed).shuffle(cands)
    taken, skipped = [], {"tokens": 0, "seen text": 0, "same wire copy": 0, "near-copy": 0}
    for gid in cands:
        if len(taken) >= args.n:
            break
        pair = events[gid]
        if any(ntok(pair[s]) > args.max_tokens for s in ("left", "right")):
            skipped["tokens"] += 1
            continue
        fps = [fingerprint(pair[s]["textString"]) for s in ("left", "right")]
        if any(f in have_fp for f in fps):
            skipped["seen text"] += 1
            continue
        shs = [shingles(pair[s]["textString"]) for s in ("left", "right")]
        # both outlets running the same wire story makes a near-identical
        # anchor/positive pair; skip the event
        if len(shs[0] & shs[1]) / max(len(shs[0] | shs[1]), 1) >= args.jaccard:
            skipped["same wire copy"] += 1
            continue
        if any(index.near(sh) for sh in shs):
            skipped["near-copy"] += 1
            continue
        for f, sh in zip(fps, shs):
            have_fp.add(f)
            index.add(sh)
        taken.append(gid)
    print(f"taken {len(taken):,} events (wanted {args.n:,}); skipped {skipped}")

    for st in ("left", "right"):
        out = ROOT / f"single_{st}_{args.tag}.jsonl"
        with out.open("w", encoding="utf-8") as fo:
            for gid in taken:
                r = events[gid][st]
                fo.write(json.dumps({
                    "story_id": f"{gid}:{st}", "source_story_id": gid, "stance": st,
                    "split": "train", "article_id": f"{gid}:{INDEX[st]}",
                    "words": nwords(r), "led_tokens": ntok(r),
                    "articles": [{"stance": st, "publisher": r["source"],
                                  "title": r["title"], "body": r["textString"]}],
                }, ensure_ascii=False) + "\n")
        print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
