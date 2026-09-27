"""NeuS gold test set: LEFT+RIGHT in, human roundup out.

    uv run python scripts/prepare_neus_test.py

Same record shape and source formatting as data/prepared_llm, so a model trained
there can be evaluated here without any adapter. The target is a human-written
AllSides roundup, not a generated one -- this is the only human gold available.

Keeps NeuS's own released splits. `test` (274 stories) is the one to report on.
"""
import json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})
def flatten(s): return " ".join((s or "").translate(_FLAT).split())

STOP = set("a an the and or of to in on at by for with from as is are was were be been it its "
           "this that he she they we you his her their our not no so than then there here which "
           "who what when where why how all any both each more most other some such only own same "
           "too very can will just should now said say says over under after before during about "
           "into out up down off again further once would could has have had do does did".split())


def _content(t):
    return [w for w in re.findall(r"[a-z0-9']+", (t or "").lower())
            if w not in STOP and len(w) > 2]


def copy_overlap(target, source, n=8):
    """Share of the roundup's 8-grams already in the input.

    AllSides editors sometimes used one source's lede verbatim as the roundup,
    which makes that item a copy task and inflates any score computed on it.
    Tagged rather than dropped so the eval can filter without losing data.
    """
    a = _content(target)
    ga = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    if not ga:
        return 0.0
    b = _content(source)
    gb = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return round(len(ga & gb) / len(ga), 4)

def jdump(o):
    return (json.dumps(o, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))

src = ROOT / "data/prepared/articles.jsonl"
out = ROOT / "data/prepared_neus_test"
(out / "gen").mkdir(parents=True, exist_ok=True)

stories = defaultdict(lambda: {"arts": [], "roundup": None, "split": None, "topic": None})
for line in src.open(encoding="utf-8"):
    r = json.loads(line)
    # AllSides' own staff blog is an author, not an outlet -- prepare_data.py
    # flags it; keep it out of a test set that measures publisher generalisation.
    if r.get("is_staff_blog") in (True, "True"):
        continue
    d = stories[r["story_id"]]
    d["arts"].append(r)
    d["split"] = r.get("split") or d["split"]
    d["topic"] = r.get("topic") or d["topic"]
    if r.get("roundup"):
        d["roundup"] = r["roundup"]

counts, skipped = defaultdict(int), 0
handles = {}
for split in ("train", "val", "test"):
    handles[split] = ((out / "gen" / f"{split}.source").open("w", encoding="utf-8"),
                      (out / "gen" / f"{split}.target").open("w", encoding="utf-8"),
                      (out / "gen" / f"{split}.ids").open("w", encoding="utf-8"),
                      (out / f"{split}.jsonl").open("w", encoding="utf-8"))

import random
for sid in sorted(stories):
    d = stories[sid]
    split = d["split"]
    if split not in handles or not d["roundup"]:
        skipped += 1; continue
    lr = [a for a in d["arts"] if a["stance"] in ("left", "right")]
    ctr = next((a for a in d["arts"] if a["stance"] == "center"), None)
    if len({a["stance"] for a in lr}) < 2:
        skipped += 1; continue
    lr = sorted(lr, key=lambda a: a["stance"])
    random.Random(f"42:{sid}").shuffle(lr)
    source = " [SEP] ".join(
        f"TITLE=> {flatten(a['title'])} . ARTICLE=> {flatten(a['body'])}" for a in lr)
    target = flatten(d["roundup"])
    fs, ft, fi, fj = handles[split]
    fs.write(source + "\n"); ft.write(target + "\n"); fi.write(str(sid) + "\n")
    fj.write(jdump({
        "id": str(sid), "split": split, "source": source, "target": target,
        "target_kind": "human_roundup", "topic": d["topic"],
        "copy_overlap": copy_overlap(target, source),
        "inputs": [{"stance": a["stance"], "publisher": a["publisher"],
                    "title": a["title"], "body": a["body"]} for a in lr],
        "held_out_center": ({"publisher": ctr["publisher"], "title": ctr["title"],
                             "body": ctr["body"]} if ctr else None),
    }) + "\n")
    counts[split] += 1

for hs in handles.values():
    for h in hs: h.close()
print("  ".join(f"{k}={counts[k]:,}" for k in ("train", "val", "test"))
      + f"   total={sum(counts.values()):,}   skipped={skipped}")
