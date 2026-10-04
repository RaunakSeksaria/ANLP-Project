"""Release files for the full1024 set (one random, deduplicated sample of the whole export).

    uv run python scripts/build_full_release.py
    -> left_neutral_full1024.jsonl, right_neutral_full1024.jsonl

Unlike build_single_release.py, every input record here carries its own split
and article_id, and those always win. An event is kept only when both its left
and right rewrites exist and both fit in --max-tokens LED tokens; the two sides
are dropped together. Texts are re-checked for exact duplicates and near-copies
(word 5-gram Jaccard >= --jaccard) across both sides before writing.
"""
import argparse, glob, json
from pathlib import Path

from build_single_release import flat, jdump
from make_extra_articles import NearDupIndex, fingerprint, shingles

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["single_{st}_full1024b.jsonl", "single_{st}_full1024c.jsonl"])
    ap.add_argument("--runs", default="runs/nemotron/extra_{st}_full*_k*.jsonl")
    ap.add_argument("--out", default="{st}_neutral_full1024.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--tokenizer", default="allenai/led-base-16384")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    src, gen = {}, {}
    for st in ("left", "right"):
        src[st] = {}
        for pat in args.inputs:
            for line in (ROOT / pat.format(st=st)).open(encoding="utf-8"):
                r = json.loads(line)
                src[st][r["source_story_id"]] = r
        gen[st] = {}
        for path in sorted(glob.glob(str(ROOT / args.runs.format(st=st)))):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r.get("summary"):
                    gen[st][r["story_id"].rsplit(":", 1)[0]] = r      # last success wins

    events = sorted(set(src["left"]) & set(src["right"]))
    drop = {"missing a side": set(), "rewrite over token cap": set(), "duplicate / near-copy": set()}
    for g in events:
        if g not in gen["left"] or g not in gen["right"]:
            drop["missing a side"].add(g)
    keep = [g for g in events if g not in drop["missing a side"]]
    for st in ("left", "right"):
        lens = [len(x) for x in tok([gen[st][g]["summary"] for g in keep])["input_ids"]]
        drop["rewrite over token cap"].update(g for g, n in zip(keep, lens) if n > args.max_tokens)
    keep = [g for g in keep if g not in drop["rewrite over token cap"]]

    index, seen = NearDupIndex(args.jaccard), set()
    for g in keep:
        bodies = [src[st][g]["articles"][0]["body"] for st in ("left", "right")]
        fps, shs = [fingerprint(b) for b in bodies], [shingles(b) for b in bodies]
        cross = len(shs[0] & shs[1]) / max(len(shs[0] | shs[1]), 1) >= args.jaccard
        if cross or any(f in seen for f in fps) or fps[0] == fps[1] or any(index.near(s) for s in shs):
            drop["duplicate / near-copy"].add(g)
            continue
        seen.update(fps)
        for s in shs:
            index.add(s)
    keep = [g for g in keep if g not in drop["duplicate / near-copy"]]

    order = sorted(keep, key=lambda g: ({"train": 0, "validation": 1, "test": 2}[src["left"][g]["split"]], g))
    for st in ("left", "right"):
        with (ROOT / args.out.format(st=st)).open("w", encoding="utf-8") as f:
            for g in order:
                s, r = src[st][g], gen[st][g]
                a = s["articles"][0]
                orig = {"article_id": s["article_id"], "publisher": a["publisher"],
                        "title": flat(a["title"]), "body": flat(a["body"])}
                p = r.get("params") or {}
                f.write(jdump({
                    "story_id": g, "split": s["split"],
                    f"{st}_original": orig, f"{st}_neutralised": flat(r["summary"]),
                    "source": f"TITLE=> {orig['title']} . ARTICLE=> {orig['body']}",
                    "edits": [flat(e if isinstance(e, str) else json.dumps(e)) for e in r.get("conflicts") or []],
                    "generator": {"model": r.get("model"), "prompt_version": p.get("prompt_version"),
                                  "temperature": p.get("temperature")},
                    "event_ids": [g],
                }) + "\n")
    print(f"events in inputs: {len(events):,} | kept: {len(keep):,}")
    for k, v in drop.items():
        print(f"  dropped ({k}): {len(v)}  {sorted(v)[:8]}")


if __name__ == "__main__":
    main()
