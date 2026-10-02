"""Quality and correctness audit for a single-article neutralisation release.

    uv run python scripts/eda_neutralised.py left_neutral_10k.jsonl

The same families of check the fusion corpus got, re-aimed at one article in,
one article out:

  file integrity    parse, ids, line breaks, split sizes, split leakage
  structure         truncation, scaffolding, refusals, duplicates, mojibake
  change            did the rewrite change anything, and by how much
  length            output/input word ratio
  additions         entities, numbers and dates the input does not contain --
                    the world-knowledge leak the prompt forbids
  retention         entities, numbers and quotations from the input that the
                    output dropped
  quote fidelity    verbatim / elided / reworded / hallucinated
  bias vocabulary   loaded terms per 100 words, input vs output, outside quotes

Prints a report and writes flagged ids to <file>.eda.json next to the input.
"""
import argparse, difflib, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

WORD = re.compile(r"[A-Za-z0-9']+")
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?(?![\w])")
MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December|Jan\\.|Feb\\.|Aug\\.|Sept?\\.|Oct\\.|Nov\\.|Dec\\.")
DATE = re.compile(rf"\b(?:{MONTHS})\s+\d{{1,2}}\b|\b(?:19|20)\d\d\b")
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85  "
STOP = set("a an the and or of to in on at by for with from as is are was were be been it its this "
           "that he she they we you his her their our not no so than then there here which who what "
           "when where why how all any both each more most other some such only own same too very can "
           "will just should now said say says over under after before during about into out up down "
           "off again further once would could has have had do does did also but i".split())
# sentence-initial and generic capitalised words that are not entities
NOT_ENT = set("the a an in on at he she they it this that but and or as after before when while "
              "if for his her their its we i there here some many one two three mr mrs ms dr "
              "monday tuesday wednesday thursday friday saturday sunday".split())

# AllSides-derived loaded vocabulary the prompt tells the model to remove.
LOADED = {
    "attribution": "admitted conceded claimed claims claiming alleged insisted insists boasted "
                   "slammed slams blasted blasts",
    "intensifier": "massive explosive stunning shocking bombshell erupts erupted outrageous "
                   "devastating disastrous horrific brutal furious scathing",
    "subjective":  "controversial so-called radical hardline extreme extremist disgraced failed "
                   "embattled notorious beleaguered",
}
LOADED_RE = {k: re.compile(r"\b(" + "|".join(re.escape(w) for w in v.split()) + r")\b", re.I)
             for k, v in LOADED.items()}


_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'"})
def toks(t):
    """Lowercase words; curly apostrophes normalised, possessive 's dropped,
    hyphenated words split -- otherwise don’t/don't and Canada's/Canada
    count as different words and every check downstream inflates."""
    t = (t or "").translate(_APOS).replace("-", " ")
    return [re.sub(r"'s$", "", w.lower()) for w in WORD.findall(t)]
def content(t): return [w for w in toks(t) if w not in STOP and len(w) > 2]
def stem(w):
    for suf in ("ing", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w
def entities(t):
    """Capitalised tokens that do not open a sentence, possessive stripped."""
    out = set()
    t = (t or "").translate(_APOS)
    for m in re.finditer(r"[A-Z][a-zA-Z]+(?:'s)?", t):
        before = (t[:m.start()].rstrip() or ".")[-1]
        if before in ".!?\"“:(-–—":          # sentence/quote/clause opener
            continue
        w = re.sub(r"'s$", "", m.group()).lower()
        if w not in NOT_ENT and len(w) > 1:
            out.add(w)
    return out
def nums(t): return {n.replace(",", "").rstrip("%") for n in NUM.findall(t or "")}
def quotes(t, min_len=25):
    """Quoted spans, pairing marks in order: curly quotes open/close, straight
    quotes alternate. A regex over straight quotes pairs a closing mark with
    the next opening one and returns the prose between two quotations."""
    spans, start = [], None
    for i, c in enumerate(t or ""):
        if c == "“" or (c == '"' and start is None):
            start = i + 1
        elif (c == "”" or c == '"') and start is not None:
            if i - start >= min_len:
                spans.append(t[start:i])
            start = None
    return spans
def strip_quotes(t):
    for q in quotes(t, 1):
        t = t.replace(q, " ")
    return t


def is_subseq(needle, hay):
    it = iter(hay)
    return all(w in it for w in needle)


def classify_quote(q, src_toks, src_norm):
    qt = toks(q)
    if not qt:
        return None
    if " ".join(qt) in src_norm:
        return "verbatim"
    if is_subseq(qt, src_toks):
        return "elided/merged"
    present = sum(w in set(src_toks) for w in qt) / len(qt)
    return "reworded" if present >= 0.6 else "hallucinated"


_MON = {m[:3].lower(): m for m in "January February March April May June July August "
        "September October November December".split()}
def norm_date(d):
    m = re.match(r"([A-Za-z]+)\.?\s+(\d+)", d)
    return f"{_MON.get(m.group(1)[:3].lower(), m.group(1))} {int(m.group(2))}" if m else d


def pct(n, d): return f"{100 * n / max(d, 1):5.1f}%"
def q(xs, p): return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else (xs or [0])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--examples", type=int, default=3, help="examples printed per flagged check")
    args = ap.parse_args()
    path = Path(args.path)

    raw = path.read_bytes().decode("utf-8")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rows, bad_json = [], 0
    for ln in lines:
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            bad_json += 1
    ok = [c for c in raw if c in BREAKS and c != "\n"]
    stance = next(k[: -len("_original")] for k in rows[0] if k.endswith("_original"))
    O, N = f"{stance}_original", f"{stance}_neutralised"
    total = len(rows)
    flags = defaultdict(list)

    # ---------------------------------------------------------------- integrity
    print(f"== {path.name}: {total:,} records, stance={stance}\n")
    print("FILE INTEGRITY")
    ids = Counter(r["story_id"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    by_body = defaultdict(set)
    for r in rows:
        by_body[r[O]["body"]].add(r["split"])
    cross = sum(1 for s in by_body.values() if len(s) > 1)
    dup_inputs = sum(1 for r in rows) - len(by_body)
    print(f"  unparseable lines                 {bad_json}")
    print(f"  raw line-break chars (not \\n)     {len(ok)}")
    print(f"  duplicate story_ids               {sum(v - 1 for v in ids.values())}")
    print(f"  splits                            {dict(splits)}")
    print(f"  input bodies crossing a split     {cross}")
    print(f"  records whose input body repeats  {dup_inputs:,}  ({pct(dup_inputs, total).strip()})"
          f"  -- same article in several events")
    print(f"  prompt versions                   {dict(Counter(r['generator']['prompt_version'] for r in rows))}")

    # ------------------------------------------------------------ per record
    ratio, sim, add_ent_share, keep_share, ent_keep = [], [], [], [], []
    quote_cls = Counter()
    quote_examples = defaultdict(list)
    loaded_in = Counter(); loaded_out = Counter()
    words_in = words_out = 0
    edits_n = []
    outputs = defaultdict(list)

    for r in rows:
        sid, o, n = r["story_id"], r[O], r[N]
        src = f"{o['title']} . {o['body']}"
        s_toks, n_toks = toks(src), toks(n)
        s_norm = " ".join(s_toks)
        wi, wo = len(o["body"].split()), len(n.split())
        ratio.append(wo / max(wi, 1))
        words_in += wi; words_out += wo
        outputs[" ".join(n_toks)].append((sid, o["body"]))
        edits_n.append(len(r.get("edits") or []))

        # structure
        if not n.strip():
            flags["empty output"].append(sid); continue
        if not n.rstrip().endswith((".", "!", "?", '"', "”", "'", ")")):
            flags["no terminal punctuation (truncated?)"].append(sid)
        if re.search(r"TITLE=>|ARTICLE=>|^\s*[\[{`]", n):
            flags["scaffolding / JSON in output"].append(sid)
        prose = strip_quotes(n)
        if re.search(r"\b(I cannot|I'm unable|as an AI|I apologi[sz]e|Here is the|Here's the)\b", prose, re.I):
            flags["refusal / assistant voice"].append(sid)
        if re.search(r"\b(the|this) (article|piece|headline|rewrite)\b|\brewritten\b", prose, re.I):
            flags["meta-commentary about the article"].append(sid)
        if re.search(r"â€|Ã©|�", n):
            flags["mojibake"].append(sid)
        sents = [x.strip().lower() for x in re.split(r"(?<=[.!?])\s+", n) if len(x) > 40]
        if len(sents) != len(set(sents)):
            flags["repeated sentence"].append(sid)

        # change
        sm = difflib.SequenceMatcher(None, toks(o["body"]), n_toks, autojunk=False).ratio()
        sim.append(sm)
        if sm >= 0.98:
            flags["essentially unchanged (>=0.98 token similarity)"].append(sid)

        # length
        if wo / max(wi, 1) < 0.6:
            flags["output < 0.6x input words (possible drops)"].append(sid)
        if wo / max(wi, 1) > 1.5:
            flags["output > 1.5x input words (padding/additions)"].append(sid)

        # additions: things in the output the article does not contain
        src_stems = {stem(w) for w in s_toks}
        new_ent = {e for e in entities(n) if e not in set(s_toks) and stem(e) not in src_stems}
        if new_ent:
            flags["adds an entity not in the input"].append(sid)
            if len(quote_examples["_ent"]) < 30:
                quote_examples["_ent"].append((sid, sorted(new_ent)[:6]))
        ne = entities(n)
        add_ent_share.append(len(new_ent) / max(len(ne), 1))
        new_nums = nums(n) - nums(src)
        if new_nums:
            flags["adds a number not in the input"].append(sid)
            if len(quote_examples["_num"]) < 30:
                quote_examples["_num"].append((sid, sorted(new_nums)[:6]))
        src_dates = {norm_date(d) for d in DATE.findall(src)}
        new_dates = {d for d in DATE.findall(n) if norm_date(d) not in src_dates}
        if new_dates:
            flags["adds a calendar date/year not in the input"].append(sid)
            if len(quote_examples["_date"]) < 30:
                quote_examples["_date"].append((sid, sorted(new_dates)[:4]))

        # retention: things in the input the output lost
        n_stems = {stem(w) for w in n_toks}
        body_ent = entities(o["body"])
        lost_ent = {e for e in body_ent if e not in set(n_toks) and stem(e) not in n_stems}
        ent_keep.append(1 - len(lost_ent) / max(len(body_ent), 1))
        if len(lost_ent) >= 3:
            flags["drops 3+ entities from the input body"].append(sid)
        lost_nums = nums(o["body"]) - nums(n)
        if lost_nums:
            flags["drops a number from the input body"].append(sid)
        c_in = set(content(o["body"]))
        keep_share.append(len({stem(w) for w in c_in} & n_stems) / max(len(c_in), 1))
        in_quotes = [x for x in quotes(o["body"])]
        dropped_q = [x for x in in_quotes if " ".join(toks(x)) not in " ".join(n_toks)
                     and not is_subseq(toks(x)[:6], n_toks)]
        if dropped_q:
            flags["drops a quotation from the input"].append(sid)

        # quote fidelity (quotes in the output)
        for qt in quotes(n):
            c = classify_quote(qt, s_toks, s_norm)
            if c:
                quote_cls[c] += 1
                if c in ("reworded", "hallucinated") and len(quote_examples[c]) < 30:
                    quote_examples[c].append((sid, qt[:110]))
                if c == "hallucinated":
                    flags["hallucinated quotation"].append(sid)
                if c == "reworded":
                    flags["reworded quotation"].append(sid)

        # loaded vocabulary, outside quotation marks
        for k, rx in LOADED_RE.items():
            loaded_in[k] += len(rx.findall(strip_quotes(src)))
            loaded_out[k] += len(rx.findall(strip_quotes(n)))

        if not r.get("edits"):
            flags["empty edit log"].append(sid)

    # identical inputs (43% of records repeat an article) should give identical
    # outputs; only a shared output across DIFFERENT inputs is a defect
    flags["same output for different inputs"] = [
        i for v in outputs.values() if len({b for _, b in v}) > 1 for i, _ in v[1:]]

    # ----------------------------------------------------------------- report
    print("\nLENGTH (words, body only)")
    print(f"  output/input ratio   median {statistics.median(ratio):.2f}   "
          f"p5 {q(ratio, 5):.2f}   p95 {q(ratio, 95):.2f}   mean {statistics.fmean(ratio):.2f}")
    print(f"  total words          in {words_in:,}   out {words_out:,}")

    print("\nCHANGE (token similarity of output to input body; 1.0 = identical)")
    print(f"  median {statistics.median(sim):.2f}   p5 {q(sim, 5):.2f}   p95 {q(sim, 95):.2f}")

    print("\nADDITIONS / RETENTION")
    print(f"  share of output entities absent from input   mean {statistics.fmean(add_ent_share):.3f}")
    print(f"  share of input entities kept                 mean {statistics.fmean(ent_keep):.3f}")
    print(f"  share of input content words kept (stemmed)  mean {statistics.fmean(keep_share):.3f}"
          f"   median {statistics.median(keep_share):.3f}")

    qt = sum(quote_cls.values())
    print(f"\nQUOTE FIDELITY (quoted spans >= 25 chars in the output: {qt:,})")
    for c in ("verbatim", "elided/merged", "reworded", "hallucinated"):
        print(f"  {c:<14} {quote_cls[c]:>6,}  {pct(quote_cls[c], qt)}")

    print("\nLOADED VOCABULARY outside quotes (per 1,000 words)")
    print(f"  {'':<12} {'input':>7} {'output':>7}")
    for k in LOADED:
        print(f"  {k:<12} {1000 * loaded_in[k] / words_in:7.2f} {1000 * loaded_out[k] / words_out:7.2f}")

    print(f"\nEDIT LOG  entries per record: median {statistics.median(edits_n):.0f}, "
          f"mean {statistics.fmean(edits_n):.2f}")

    print(f"\nFLAGGED RECORDS  (of {total:,})")
    for k, v in sorted(flags.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<50} {len(v):>6,}  {pct(len(v), total)}")

    for key, title in (("_date", "added dates"), ("_num", "added numbers"), ("_ent", "added entities"),
                       ("hallucinated", "hallucinated quotes"), ("reworded", "reworded quotes")):
        if quote_examples[key]:
            print(f"\nexamples -- {title}:")
            for sid, x in quote_examples[key][: args.examples]:
                print(f"  {sid}: {x}")

    out = path.with_suffix(".eda.json")
    json.dump({k: v for k, v in flags.items()}, out.open("w"), indent=1)
    print(f"\nflagged ids -> {out.name}")


if __name__ == "__main__":
    main()
