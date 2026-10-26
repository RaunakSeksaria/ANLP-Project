"""Compare generation-setting variants on the fixed evaluation sample.

    uv run python scripts/compare_variants.py think_on_t1.0 think_off_t1.0 --ref think_on_t1.0
    -> runs/experiments/report/<variants>/summary.md, per_record.jsonl, review.md

Only events that every listed variant finished are compared, so the comparison is
paired. For each variant and side it reports:
  - speed and cost: output tokens per article, wall time per 100 articles;
  - edit size: word-level similarity to the input and the share of input words
    deleted or replaced;
  - faithfulness: input entities kept, quotations verbatim / reworded / invented,
    and added entities, numbers and calendar dates;
  - framing: loaded vocabulary per 1,000 words (outside quotes) and the
    "The article reports ..." rate;
  - classifier: POLITICS (Harshit's politics_media.pt) on the original and the
    rewrite -- P(true side), whether the true side is still the top label, and
    P(Centre). Outlet names are masked first and long texts are read in
    512-token windows with 64 overlap (logits averaged), as the paper does.
    Treat it as a RELATIVE signal: the encoder was pretrained on BigNewsAlign
    (see scratch/politics_classifier_notes_for_harshit.md);
  - output length under 512 LED tokens.
Differences against --ref come with bootstrap 95% CIs over events.
"""
import argparse, difflib, glob, hashlib, json, random, re, statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from eda_neutralised import (DATE, LOADED_RE, classify_quote, entities, norm_date,
                             nums, quotes, stem, strip_quotes, toks)

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
TRUE = {"left": 0, "right": 2}
OUTLETS = [r"Associated Press", r"\bAP\b", r"The Hill", r"USA Today", r"CNN", r"Fox News", r"\bFox\b",
           r"Breitbart(?: News)?", r"Huff(?:ington )?Post", r"HuffPost", r"New York Times", r"\bNYT\b",
           r"Washington Post", r"Washington Times", r"Daily Kos"]
OUTLET_RE = re.compile("|".join(OUTLETS), re.I)
META_RE = re.compile(r"\b(the|this) (article|piece|headline|rewrite)\b|\brewritten\b", re.I)


def load_variant(name):
    out = {}
    for path in sorted(glob.glob(str(ROOT / f"runs/experiments/{name}/extra_*_k*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("summary"):
                out[r["story_id"]] = r
    return out


def record_metrics(src, out_text):
    body = src["articles"][0]["body"]
    full = f"{src['articles'][0]['title']} . {body}"
    a, b = toks(body), toks(out_text)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changed = sum(i2 - i1 for t, i1, i2, *_ in sm.get_opcodes() if t in ("delete", "replace"))
    s_toks, b_set = toks(full), set(b)
    b_stems = {stem(w) for w in b}
    body_ent = entities(body)
    kept = [e for e in body_ent if e in b_set or stem(e) in b_stems]
    src_set, src_stems = set(s_toks), {stem(w) for w in s_toks}
    added_ent = [e for e in entities(out_text) if e not in src_set and stem(e) not in src_stems]
    src_dates = {norm_date(d) for d in DATE.findall(full)}
    added_dates = [d for d in DATE.findall(out_text) if norm_date(d) not in src_dates]
    qs = [c for c in (classify_quote(q, s_toks, " ".join(s_toks)) for q in quotes(out_text)) if c]
    prose = strip_quotes(out_text)
    words = max(len(out_text.split()), 1)
    return {
        "similarity": sm.ratio(),
        "changed_share": changed / max(len(a), 1),
        "entities_kept": len(kept) / len(body_ent) if body_ent else None,
        "adds_entity": bool(added_ent),
        "adds_number": bool(nums(out_text) - nums(full)),
        "adds_date": bool(added_dates),
        "meta_commentary": bool(META_RE.search(prose)),
        "quotes": qs,
        "loaded_per_1k": 1000 * sum(len(rx.findall(prose)) for rx in LOADED_RE.values()) / words,
        "loaded_per_1k_input": 1000 * sum(len(rx.findall(strip_quotes(body))) for rx in LOADED_RE.values())
                               / max(len(body.split()), 1),
        "out_words": len(out_text.split()),
        "in_words": len(body.split()),
    }


class Politics:
    """politics_media.pt with outlet masking and sliding windows; results cached on disk."""

    def __init__(self, cache_path):
        import torch, torch.nn as nn
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        enc_name = "launch/POLITICS"

        class Clf(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = AutoModel.from_pretrained(enc_name)
                h = self.enc.config.hidden_size
                self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 768), nn.Tanh(),
                                          nn.Dropout(0.1), nn.Linear(768, 3))

            def forward(self, ids, mask):
                return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

        self.tok = AutoTokenizer.from_pretrained(enc_name)
        self.model = Clf()
        state = torch.load(ROOT / "data/classifiers/politics-ideology-classifier/politics_media.pt",
                           map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path)) if cache_path.exists() else {}

    def _windows(self, text, size=510, overlap=64):
        ids = self.tok(OUTLET_RE.sub("<mask>", text), add_special_tokens=False)["input_ids"]
        step = size - overlap
        return [ids[i:i + size] for i in range(0, max(len(ids) - overlap, 1), step)] or [[]]

    def probs(self, text):
        key = hashlib.md5(text.encode()).hexdigest()
        if key not in self.cache:
            wins = self._windows(text)
            batch = [[self.tok.cls_token_id] + w + [self.tok.sep_token_id] for w in wins]
            m = max(len(x) for x in batch)
            ids = self.torch.tensor([x + [self.tok.pad_token_id] * (m - len(x)) for x in batch])
            mask = (ids != self.tok.pad_token_id).long()
            with self.torch.no_grad():
                logits = self.model(ids, mask).mean(0)
            self.cache[key] = self.torch.softmax(logits, -1).tolist()
        return self.cache[key]

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w"))


def boot_ci(diffs, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(d, (n, len(d))).mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def wall_per_100(recs):
    ts = sorted(datetime.fromisoformat(r["generated_at"]) for r in recs)
    if len(ts) < 2:
        return None
    return (ts[-1] - ts[0]).total_seconds() / 60 / len(ts) * 100


def word_diff(a, b):
    aw, bw = a.split(), b.split()
    out = []
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if t == "equal":
            out.append(" ".join(aw[i1:i2]))
        if t in ("delete", "replace"):
            out.append("~~" + " ".join(aw[i1:i2]) + "~~")
        if t in ("insert", "replace"):
            out.append("**" + " ".join(bw[j1:j2]) + "**")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--ref", default=None, help="reference variant for paired differences (default: first)")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--review", type=int, default=20, help="events in the side-by-side review file")
    args = ap.parse_args()
    ref = args.ref or args.variants[0]

    src = {st: {json.loads(l)["story_id"]: json.loads(l) for l in open(ROOT / f"experiments/eval500/{st}.jsonl")}
           for st in SIDES}
    outs = {v: load_variant(v) for v in args.variants}
    report = ROOT / "runs/experiments/report" / "__".join(args.variants)
    report.mkdir(parents=True, exist_ok=True)
    clf = None if args.no_classifier else Politics(ROOT / "runs/experiments/report/politics_cache.json")

    per = defaultdict(dict)          # (variant, story_id) -> metrics
    common = {}
    for st in SIDES:
        common[st] = sorted(set.intersection(*[{k for k in outs[v] if k.endswith(":" + st)} for v in args.variants]),
                            key=lambda k: src[st][k]["order"])
        for sid in common[st]:
            s = src[st][sid]
            p_orig = clf.probs(s["articles"][0]["body"]) if clf else None
            for v in args.variants:
                text = outs[v][sid]["summary"]
                m = record_metrics(s, text)
                m["out_tokens_api"] = (outs[v][sid].get("usage") or {}).get("output_tokens")
                if clf:
                    p = clf.probs(text)
                    m.update(p_true_orig=p_orig[TRUE[st]], p_true_out=p[TRUE[st]],
                             top_true_orig=int(np.argmax(p_orig)) == TRUE[st], top_true_out=int(np.argmax(p)) == TRUE[st],
                             p_centre_orig=p_orig[1], p_centre_out=p[1])
                per[(v, st)][sid] = m
    if clf:
        clf.save()

    from transformers import AutoTokenizer
    led = AutoTokenizer.from_pretrained("allenai/led-base-16384")

    def summarise(v, st):
        ms = list(per[(v, st)].values())
        recs = [outs[v][sid] for sid in per[(v, st)]]
        mean = lambda k: statistics.fmean([m[k] for m in ms if m.get(k) is not None]) if ms else float("nan")
        rate = lambda k: 100 * statistics.fmean([bool(m[k]) for m in ms]) if ms else float("nan")
        qs = [q for m in ms for q in m["quotes"]]
        led_len = [len(x) for x in led([r["summary"] for r in recs])["input_ids"]] if recs else []
        row = {
            "n": len(ms),
            "out tokens/article (API)": statistics.median([m["out_tokens_api"] for m in ms if m["out_tokens_api"]]) if ms else None,
            "wall min / 100 articles": wall_per_100(recs),
            "similarity to input": mean("similarity"),
            "input words changed %": 100 * mean("changed_share"),
            "output/input words": statistics.median([m["out_words"] / m["in_words"] for m in ms]) if ms else None,
            "entities kept %": 100 * mean("entities_kept"),
            "quotes verbatim %": 100 * sum(q in ("verbatim", "elided/merged") for q in qs) / max(len(qs), 1),
            "quotes reworded+invented %": 100 * sum(q in ("reworded", "hallucinated") for q in qs) / max(len(qs), 1),
            "adds entity %": rate("adds_entity"), "adds number %": rate("adds_number"),
            "adds date %": rate("adds_date"), "'the article ...' %": rate("meta_commentary"),
            "loaded words /1k (in -> out)": f"{mean('loaded_per_1k_input'):.2f} -> {mean('loaded_per_1k'):.2f}",
            "output < 512 LED tokens %": 100 * statistics.fmean([n < 512 for n in led_len]) if led_len else None,
        }
        if clf:
            row.update({
                "POLITICS P(true side) orig -> out": f"{mean('p_true_orig'):.3f} -> {mean('p_true_out'):.3f}",
                "POLITICS true side on top % orig -> out": f"{rate('top_true_orig'):.1f} -> {rate('top_true_out'):.1f}",
                "POLITICS P(Centre) orig -> out": f"{mean('p_centre_orig'):.3f} -> {mean('p_centre_out'):.3f}",
            })
        return row

    lines = [f"# Variant comparison: {', '.join(args.variants)}", "",
             f"Paired on events every variant finished. Reference for differences: `{ref}`.", ""]
    for st in SIDES:
        rows = {v: summarise(v, st) for v in args.variants}
        keys = list(next(iter(rows.values())).keys())
        lines += [f"## {st} (n = {len(common[st])} paired events)", "",
                  "| metric | " + " | ".join(f"`{v}`" for v in args.variants) + " |",
                  "| --- |" + " --- |" * len(args.variants)]
        for k in keys:
            cells = []
            for v in args.variants:
                x = rows[v][k]
                cells.append(f"{x:.2f}" if isinstance(x, float) else str(x))
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += ["", f"**Paired differences vs `{ref}`** (mean, bootstrap 95% CI):", "",
                  "| variant | metric | diff | 95% CI |", "| --- | --- | --- | --- |"]
        for v in args.variants:
            if v == ref:
                continue
            for k, lab, scale in (("similarity", "similarity to input", 1), ("changed_share", "input words changed (pts)", 100),
                                  ("entities_kept", "entities kept (pts)", 100), ("meta_commentary", "'the article ...' (pts)", 100),
                                  ("adds_entity", "adds entity (pts)", 100), ("adds_date", "adds date (pts)", 100),
                                  ("p_true_out", "POLITICS P(true side) on output", 1), ("p_centre_out", "POLITICS P(Centre) on output", 1)):
                if clf is None and k.startswith("p_"):
                    continue
                diffs = [None if per[(v, st)][s][k] is None or per[(ref, st)][s][k] is None
                         else scale * (float(per[(v, st)][s][k]) - float(per[(ref, st)][s][k])) for s in common[st]]
                m, lo, hi = boot_ci(diffs)
                lines.append(f"| `{v}` | {lab} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        lines.append("")
    (report / "summary.md").write_text("\n".join(lines))

    with open(report / "per_record.jsonl", "w") as f:
        for (v, st), d in per.items():
            for sid, m in d.items():
                f.write(json.dumps({"variant": v, "story_id": sid, **{k: x for k, x in m.items() if k != "quotes"},
                                    "quote_classes": m["quotes"]}) + "\n")

    random.seed(11)
    rv = ["# Side-by-side review", "",
          "~~struck~~ = removed from the original, **bold** = added. Same events for every variant.", ""]
    for st in SIDES:
        for sid in random.sample(common[st], min(args.review // 2, len(common[st]))):
            s = src[st][sid]["articles"][0]
            rv += [f"---", f"## {st} · `{sid}` · {s['publisher']} · {src[st][sid]['led_tokens']} tokens",
                   f"**Title:** {s['title']}", "", f"**Original:** {s['body']}", ""]
            for v in args.variants:
                m = per[(v, st)][sid]
                cl = f" · POLITICS P(true) {m['p_true_orig']:.2f}→{m['p_true_out']:.2f}" if clf else ""
                rv += [f"**`{v}`** (similarity {m['similarity']:.2f}{cl}):", "",
                       word_diff(s["body"], outs[v][sid]["summary"]), ""]
    (report / "review.md").write_text("\n".join(rv))
    print((report / "summary.md").read_text())
    print(f"\nwrote {report}/summary.md, per_record.jsonl, review.md")


if __name__ == "__main__":
    main()
