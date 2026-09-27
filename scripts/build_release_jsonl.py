"""One self-contained file: inputs, the generated neutral article, and the split.

    uv run python scripts/build_release_jsonl.py

Joins the article bodies, the generated neutral articles and the split
assignment into a single JSONL at the repo root, so nothing downstream has to
redo the join (and cannot get the split wrong).
"""
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})
def flat(s): return " ".join((s or "").translate(_FLAT).split())
def jdump(o):
    return (json.dumps(o, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))

# split + article bodies
arts = defaultdict(dict)
split = {}
for line in (ROOT / "data/prepared_bna/articles.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    arts[str(r["story_id"])][r["stance"]] = r
    split[str(r["story_id"])] = r["split"]

# generated neutral articles
neutral = {}
for line in (ROOT / "runs/nemotron/summaries_10k.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    if r.get("summary"):
        neutral[str(r["story_id"])] = r

# stable order: split, then id -- so the file diffs cleanly on regeneration
order = sorted(neutral, key=lambda s: ({"train": 0, "val": 1, "test": 2}.get(split.get(s), 3), s))
out = ROOT / "neutral_articles_10k.jsonl"
n = 0
with out.open("w", encoding="utf-8") as f:
    for sid in order:
        a, g = arts.get(sid), neutral[sid]
        if not a or not {"left", "right"} <= set(a):
            continue
        def art(st):
            x = a.get(st)
            # article_id is what triplets.jsonl and the probe splits key on --
            # without it the triplet objective cannot be wired up at all.
            return None if x is None else {"article_id": x["article_id"],
                                           "publisher": x["publisher"],
                                           "title": flat(x["title"]),
                                           "body": flat(x["body"])}
        L, R, C = art("left"), art("right"), art("center")
        f.write(jdump({
            "story_id": sid,
            "split": split[sid],
            "left": L,
            "right": R,
            "center": C,          # held out of the input; kept for reference
            "neutral": flat(g["summary"]),
            "source": f"TITLE=> {L['title']} . ARTICLE=> {L['body']}"
                      f" [SEP] TITLE=> {R['title']} . ARTICLE=> {R['body']}",
            "generator": {"model": g.get("model"),
                          "prompt_version": (g.get("params") or {}).get("prompt_version"),
                          "temperature": (g.get("params") or {}).get("temperature")},
        }) + "\n")
        n += 1
print(f"wrote {out.name}: {n:,} records")
