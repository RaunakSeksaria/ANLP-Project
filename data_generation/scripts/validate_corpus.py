"""Validate a prepared summarisation corpus. Exits non-zero if anything fails.

    uv run python scripts/validate_corpus.py data/prepared_bna
    uv run python scripts/validate_corpus.py data/prepared_bna --max-target-overlap 0.5

Every check here exists because the thing it looks for actually happened while
building this corpus. Run it after any rebuild, and before handing a corpus to
anyone.

Checks
  alignment      every {split}.source has as many lines as {split}.target, read
                 the way the training script reads them (str.splitlines(), which
                 breaks on more characters than \\n)
  linebreaks     no U+2028/U+2029/NEL/\\r/etc. surviving inside a record -- one of
                 these shifts every later target against its source, silently
  structure      source lines carry the expected number of articles, targets one
  duplicates     no event appears twice; no article text appears in two splits
  leakage        the target is not reproduced inside its own source
  balance        stance composition, publisher-stance coupling, length ratios
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORD = re.compile(r"[a-z0-9']+")
BREAKS = "\r\v\f\x1c\x1d\x1e\x85  "      # \n excluded: it is the delimiter
STOP = set("a an the and or of to in on at by for with from as is are was were be been it its "
           "this that he she they we you his her their our not no so than then there here which "
           "who what when where why how all any both each more most other some such only own same "
           "too very can will just should now said say says over under after before during about "
           "into out up down off again further once would could has have had do does did".split())
SPLITS = ("train", "val", "test")


def content(t):
    return [w for w in WORD.findall((t or "").lower()) if w not in STOP and len(w) > 2]


def ngram_overlap(target, source, n=8):
    c, s = content(target), content(source)
    if len(c) < n:
        return 0.0
    g = {tuple(c[i:i + n]) for i in range(len(c) - n + 1)}
    if not g:
        return 0.0
    gs = {tuple(s[i:i + n]) for i in range(len(s) - n + 1)}
    return len(g & gs) / len(g)


class Report:
    def __init__(self):
        self.fails, self.warns, self.notes = [], [], []

    def fail(self, msg):
        self.fails.append(msg)

    def warn(self, msg):
        self.warns.append(msg)

    def note(self, msg):
        self.notes.append(msg)


def check_gen(d, r, n_src_articles, n_tgt_articles=1):
    """Line alignment and structure of the plain source/target files."""
    gen = d / "gen"
    if not gen.is_dir():
        r.warn(f"no {gen.name}/ directory -- skipping generation-file checks")
        return {}
    counts = {}
    for split in SPLITS:
        fs, ft = gen / f"{split}.source", gen / f"{split}.target"
        if not (fs.exists() and ft.exists()):
            r.fail(f"gen/{split}.source or .target missing")
            continue
        # read_text() translates \r to \n; splitlines() breaks on U+2028 etc.
        # This is exactly how the NeuS-format trainer reads them.
        src = fs.read_text(encoding="utf-8").splitlines()
        tgt = ft.read_text(encoding="utf-8").splitlines()
        if len(src) != len(tgt):
            r.fail(f"gen/{split}: {len(src):,} source lines vs {len(tgt):,} target "
                   f"lines -- sources and targets are MISALIGNED")
        counts[split] = len(src)

        bad_s = sum(1 for l in src if l.count("TITLE=>") != n_src_articles)
        # A target may be plain prose (an LLM summary) rather than a formatted
        # article; --target-articles 0 says so.
        bad_t = (sum(1 for l in tgt if l.count("TITLE=>") != n_tgt_articles)
                 if n_tgt_articles else 0)
        if bad_s:
            r.fail(f"gen/{split}.source: {bad_s:,} line(s) do not hold exactly "
                   f"{n_src_articles} articles")
        if bad_t:
            r.fail(f"gen/{split}.target: {bad_t:,} line(s) do not hold exactly "
                   f"{n_tgt_articles} article(s)")

        for fn, raw in ((fs.name, fs.read_bytes().decode("utf-8")),
                        (ft.name, ft.read_bytes().decode("utf-8"))):
            found = [repr(c) for c in BREAKS if c in raw]
            if found:
                r.fail(f"gen/{fn}: contains line-break character(s) {found} -- these "
                       f"shift every later line when read with splitlines()")
    return counts


def check_jsonl(d, r, max_overlap):
    rows, ids = [], Counter()
    for split in SPLITS:
        f = d / f"{split}.jsonl"
        if not f.exists():
            r.warn(f"{f.name} missing -- skipping record checks for {split}")
            continue
        raw = f.read_bytes().decode("utf-8")
        found = [repr(c) for c in BREAKS if c in raw]
        if found:
            r.fail(f"{f.name}: raw line-break character(s) {found} inside records")
        for i, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError as e:
                r.fail(f"{f.name}:{i} is not valid JSON ({e})")
                continue
            o["_split"] = split
            rows.append(o)
            ids[str(o.get("id"))] += 1
    if not rows:
        return rows
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        r.fail(f"{len(dup):,} event id(s) appear more than once, e.g. {dup[:5]}")

    missing = [k for k in ("id", "source", "target") if not all(k in o for o in rows)]
    if missing:
        r.fail(f"records missing required field(s): {missing}")
        return rows

    # target reproduced inside its own source
    over = [(o["id"], ngram_overlap(o["target"], o["source"])) for o in rows]
    bad = [(i, v) for i, v in over if v >= max_overlap]
    if bad:
        worst = sorted(bad, key=lambda x: -x[1])[:5]
        r.fail(f"{len(bad):,} record(s) have >={max_overlap:.0%} of their target's "
               f"8-grams already in the source -- these are copy tasks, not "
               f"summarisation. Worst: {[(i, round(v,2)) for i,v in worst]}")
    vals = sorted(v for _, v in over)
    r.note(f"target-in-source 8-gram overlap: median {vals[len(vals)//2]:.3f}  "
           f"p90 {vals[int(.9*len(vals))]:.3f}  max {vals[-1]:.3f}")

    # identical (source, target) pairs across the corpus
    seen = {}
    dupe_pairs = 0
    for o in rows:
        k = hashlib.md5((o["source"] + "||" + o["target"]).encode()).hexdigest()
        if k in seen:
            dupe_pairs += 1
        seen[k] = o["id"]
    if dupe_pairs:
        r.fail(f"{dupe_pairs:,} duplicate (source,target) pair(s)")

    sw = [len(o["source"].split()) for o in rows]
    tw = [len(o["target"].split()) for o in rows]
    r.note(f"length: source median {sorted(sw)[len(sw)//2]}w, target median "
           f"{sorted(tw)[len(tw)//2]}w, compression "
           f"{sum(sw)/max(sum(tw),1):.2f}:1")
    return rows


def check_articles(d, r):
    f = d / "articles.jsonl"
    if not f.exists():
        r.warn("articles.jsonl missing -- skipping split-leakage and probe checks")
        return
    arts = [json.loads(l) for l in f.open(encoding="utf-8") if l.strip()]
    seen, leak = {}, 0
    for a in arts:
        k = hashlib.md5(" ".join(WORD.findall(a["body"].lower())).encode()).hexdigest()
        if seen.setdefault(k, a["split"]) != a["split"]:
            leak += 1
    if leak:
        r.fail(f"{leak:,} article text(s) appear in more than one split -- "
               f"train/test contamination")

    stances = Counter(a["stance"] for a in arts)
    if len(set(stances.values())) != 1:
        r.warn(f"stance counts are unbalanced: {dict(stances)}")
    r.note(f"articles {len(arts):,}   events {len({a['story_id'] for a in arts}):,}   "
           f"stances {dict(stances)}")

    by_pub = defaultdict(set)
    for a in arts:
        by_pub[a["publisher"]].add(a["stance"])
    single = [p for p, s in by_pub.items() if len(s) == 1]
    if len(single) == len(by_pub):
        r.note(f"all {len(by_pub)} publishers carry exactly one stance -- publisher "
               f"and stance are the same variable here, so the leakage probe "
               f"cannot separate them")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="directory, e.g. data/prepared_bna")
    ap.add_argument("--max-target-overlap", type=float, default=0.5)
    ap.add_argument("--source-articles", type=int, default=2,
                    help="how many articles a source line should hold")
    ap.add_argument("--target-articles", type=int, default=1,
                    help="articles per target line; 0 = plain prose (LLM summary)")
    ap.add_argument("--expect", type=int, default=0,
                    help="fail unless the corpus holds exactly this many events")
    args = ap.parse_args()

    d = Path(args.corpus)
    if not d.is_dir():
        sys.exit(f"{d} is not a directory")
    r = Report()

    counts = check_gen(d, r, args.source_articles, args.target_articles)
    rows = check_jsonl(d, r, args.max_target_overlap)
    check_articles(d, r)

    total = sum(counts.values()) if counts else len(rows)
    if counts and rows and total != len(rows):
        r.fail(f"gen/ holds {total:,} pairs but the JSONL holds {len(rows):,}")
    if args.expect and total != args.expect:
        r.fail(f"expected {args.expect:,} events, found {total:,}")

    print(f"corpus: {d}")
    if counts:
        print(f"\n{'split':>7} {'events':>8} {'inputs':>8} {'targets':>8}")
        print("-" * 35)
        for s in SPLITS:
            n = counts.get(s, 0)
            print(f"{s:>7} {n:>8,} {n*args.source_articles:>8,} {n:>8,}")
        print("-" * 35)
        print(f"{'TOTAL':>7} {total:>8,} {total*args.source_articles:>8,} {total:>8,}")
    for n in r.notes:
        print(f"  {n}")
    for w in r.warns:
        print(f"\n  WARN  {w}")
    if r.fails:
        print(f"\n{len(r.fails)} CHECK(S) FAILED:")
        for x in r.fails:
            print(f"  FAIL  {x}")
        return 1
    print("\nOK: alignment, line breaks, structure, duplicates, leakage, balance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
