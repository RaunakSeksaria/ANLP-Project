"""Package data/prepared_bna as self-contained JSONL for handoff, and verify it.

    uv run python scripts/export_bna.py

Writes {train,val,test}.jsonl next to the plain .source/.target files. Each line
carries both the ready-to-train strings and the structured articles they were
built from, so whoever trains can either use `source`/`target` directly or
re-format without going back to the raw corpus.
"""
import hashlib
import json
import re
import sys
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

P = ROOT / "data/prepared_bna"
CLUSTERS = ROOT / "data/final/clusters_10k_shortest.jsonl"
W = re.compile(r"[a-z0-9]+")
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85  "


def norm(t):
    return " ".join(W.findall((t or "").lower()))


STOP = set("a an the and or of to in on at by for with from as is are was were be been it its "
           "this that he she they we you his her their our not no so than then there here which "
           "who what when where why how all any both each more most other some such only own same "
           "too very can will just should now said say says over under after before during about "
           "into out up down off again further once would could has have had do does did".split())


def ngram_overlap(target, source, n=8):
    def c(x):
        return [w for w in re.findall(r"[a-z0-9\']+", (x or "").lower())
                if w not in STOP and len(w) > 2]
    a, b = c(target), c(source)
    if len(a) < n:
        return 0.0
    g = {tuple(a[i:i + n]) for i in range(len(a) - n + 1)}
    if not g:
        return 0.0
    gs = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    return len(g & gs) / len(g)


def main():
    arts = [json.loads(l) for l in (P / "articles.jsonl").open()]
    by_event = defaultdict(list)
    for a in arts:
        by_event[a["story_id"]].append(a)

    fails = []
    counts = {}
    per_split = {}
    for split in ("train", "val", "test"):
        src = (P / f"gen/{split}.source").read_text().splitlines()
        tgt = (P / f"gen/{split}.target").read_text().splitlines()
        ids = (P / f"gen/{split}.ids").read_text().split()
        counts[split] = (len(src), len(tgt), len(ids))
        per_split[split] = ids
        if not (len(src) == len(tgt) == len(ids)):
            fails.append(f"{split}: src={len(src)} tgt={len(tgt)} ids={len(ids)}")

    total = 0
    for split in ("train", "val", "test"):
        src = (P / f"gen/{split}.source").read_text().splitlines()
        tgt = (P / f"gen/{split}.target").read_text().splitlines()
        ids = per_split[split]
        if not (len(src) == len(tgt) == len(ids)):
            fails.append(f"{split}: src={len(src)} tgt={len(tgt)} ids={len(ids)}")
            continue
        with (P / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for eid, s, t in zip(ids, src, tgt):
                recs = by_event[eid]
                centre = [r for r in recs if r["stance"] == "center"][0]
                inputs = [r for r in recs if r["stance"] != "center"]
                # A verbatim quote shared between outlets is normal; a target
                # wholly reproduced in its own source is a copy task and should
                # have been filtered upstream.
                if norm(centre["body"])[:400] and norm(centre["body"]) in norm(s):
                    fails.append(f"{split}/{eid}: target reproduced in source")
                f.write(jdump({
                    "id": eid,
                    "split": split,
                    "source": s,
                    "target": t,
                    "target_stance": "center",
                    "target_publisher": centre["publisher"],
                    "inputs": [{"stance": r["stance"], "publisher": r["publisher"],
                                "title": r["title"], "body": r["body"]} for r in inputs],
                    "target_article": {"title": centre["title"], "body": centre["body"]},
                }) + "\n")
                total += 1

    # split integrity: no article text on two sides
    seen, leak = {}, 0
    for a in arts:
        k = hashlib.md5(norm(a["body"]).encode()).hexdigest()
        if seen.setdefault(k, a["split"]) != a["split"]:
            leak += 1
    if leak:
        fails.append(f"{leak} article(s) appear in more than one split")

    # no stray line-break characters anywhere in the shipped files
    for fn in list(P.glob("*.jsonl")) + list((P / "gen").glob("*")):
        raw = fn.read_bytes().decode("utf-8")
        body = raw.replace("\n", "")
        bad = [repr(c) for c in BREAKS if c != "\n" and c in body]
        if bad:
            fails.append(f"{fn.name}: contains {bad}")

    print(f"{'split':>6} {'events':>7} {'inputs':>7} {'targets':>8}")
    print("-" * 33)
    for s in ("train", "val", "test"):
        n = counts[s][0]
        print(f"{s:>6} {n:>7,} {n*2:>7,} {n:>8,}")
    print("-" * 33)
    print(f"{'TOTAL':>6} {total:>7,} {total*2:>7,} {total:>8,}")
    print(f"\npublishers: {len({a['publisher'] for a in arts})}   "
          f"articles: {len(arts):,}   stances: {dict(Counter(a['stance'] for a in arts))}")

    write_readme(P, counts, total, arts)

    if fails:
        print("\nFAILED CHECKS:")
        for x in fails[:20]:
            print("  -", x)
        return 1
    print("\nall checks passed: alignment, no target leakage, no cross-split "
          "articles, no stray line breaks")
    return 0




README = """# BigNewsAlign neutral-summarisation corpus (interim)

*Generated by `scripts/export_bna.py` — edit that script, not this file.*

**Task:** given a left-leaning and a right-leaning article about the same event,
generate the centre-leaning article.

**This is a stand-in.** The centre article is *not* a neutral summary — it is one
outlet's coverage, carrying its own framing. It is here so training can start
before the LLM-generated neutral targets exist. Swapping those in changes only the
`target` field; the layout stays identical.

## Files

| file | what |
| --- | --- |
| `train.jsonl` / `val.jsonl` / `test.jsonl` | **start here** — self-contained |
| `gen/{{split}}.source`, `.target` | same pairs as plain text, NeuS format — what `dead_code/train_neus.py --data-dir` reads |
| `gen/{{split}}.ids` | the event id behind each line, in order |
| `articles.jsonl` | all {n_articles:,} articles individually, with split membership |
| `triplets.jsonl` | anchor / positive / negative ids for the triplet loss |
| `probe_random.json`, `probe_media.json` | article-id splits for the leakage probe |

## Counts

```
{table}
```

{n_pub} publishers, {n_articles:,} articles, one left + one centre + one right per event.
Source median {src_w}w, target median {tgt_w}w — compression about {ratio:.1f}:1.

## JSONL schema

```json
{{"id": "42806", "split": "train",
 "source": "TITLE=> ... . ARTICLE=> ... [SEP] TITLE=> ... . ARTICLE=> ...",
 "target": "TITLE=> ... . ARTICLE=> ...",
 "target_stance": "center", "target_publisher": "hill",
 "inputs": [{{"stance": "left", "publisher": "hpo", "title": "...", "body": "..."}}],
 "target_article": {{"title": "...", "body": "..."}}}}
```

`source` and `target` tokenise as-is. `inputs` / `target_article` are the same
content unformatted, for a different encoding.

## Things that will bite you

**The centre article is never in its own input.** Source is left + right only —
including the target would make it a copy task. Hence 2-source where the proposal
says 3-source.

**Splits are by connected component, not random.** Most events share an article
with another event (outlets republish; the aligner reuses articles across
clusters). A random split puts the same text in train and test. Components are
packed largest-first into 80/10/10, so sizes are approximate.
**Do not re-shuffle these splits.**

**Line-break characters are stripped from `gen/*`.** Bodies contained `\r` and
`\u2028`. `read_text()` turns `\r` into `\n` and `splitlines()` breaks on
`\u2028` — either silently shifts every later target against its source, and
training proceeds on mismatched pairs with no error. The JSONL escapes them. If
you reformat from `body` yourself, strip `"\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"`.

**Copy tasks are filtered out.** Outlets run the same wire story, so some centre
articles reproduce a left/right article near-verbatim. Anything with >=50% of its
target 8-grams already in the source is gone, drawn from a larger pool so the
corpus still lands on its target size. Survivors: median 8-gram overlap
**{ov_med:.3f}**, max {ov_max:.3f}.

**Publisher determines stance completely.** All {n_pub} outlets carry exactly one
stance, so "publisher leakage" and "stance leakage" are the same measurement here
— not separable axes as they are on NeuS. `probe_media.json` holds outlets out
entirely, stratified by stance.

**Targets are full articles, not summaries.** ~{ratio:.0f}:1 compression, much gentler
than a real summarisation target. Expect ROUGE to look flattering.

## Reproducing

```bash
uv run python scripts/make_clusters.py \\
  --in data/final/bignewsalign_max250_LCR.jsonl \\
  --out data/final/clusters_pool.jsonl --limit 11000
uv run python scripts/prepare_bignews.py \\
  --in data/final/clusters_pool.jsonl --target-events {total}
uv run python scripts/export_bna.py
uv run python scripts/validate_corpus.py data/prepared_bna --expect {total}
```

The pool is deliberately larger than the corpus: the copy-task filter removes a
couple hundred events, then `--target-events` takes the shortest survivors.
Deterministic at `--seed 42`.

**Validate after every rebuild.** `validate_corpus.py` exits non-zero on
misalignment, stray line breaks, duplicate events, target-in-source leakage, or
articles crossing a split. Every check in it exists because that failure actually
happened here.

## Training

```bash
uv run python dead_code/train_neus.py --run-name bna-baseline \\
  --data-dir data/prepared_bna/gen --triplet-weight 0
```

`dead_code/train_neus.py` reads `triplets.jsonl` and `articles.jsonl` from a hardcoded
`data/prepared`, so `--triplet-weight > 0` needs that path pointed here first.
"""


def write_readme(P, counts, total, arts):
    from statistics import median
    rows = [json.loads(l) for l in (P / "train.jsonl").open(encoding="utf-8")]
    sw = [len(o["source"].split()) for o in rows]
    tw = [len(o["target"].split()) for o in rows]
    ov = sorted(ngram_overlap(o["target"], o["source"]) for o in rows)
    lines = [f"{'split':>7} {'events':>8} {'L+R inputs':>12} {'centre targets':>15}"]
    for s in ("train", "val", "test"):
        n = counts[s][0]
        lines.append(f"{s:>7} {n:>8,} {n*2:>12,} {n:>15,}")
    lines.append(f"{'TOTAL':>7} {total:>8,} {total*2:>12,} {total:>15,}")
    (P / "README.md").write_text(README.format(
        table="\n".join(lines), total=total,
        n_articles=len(arts), n_pub=len({a["publisher"] for a in arts}),
        src_w=int(median(sw)), tgt_w=int(median(tw)),
        ratio=sum(sw) / max(sum(tw), 1),
        ov_med=ov[len(ov)//2], ov_max=ov[-1]), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
