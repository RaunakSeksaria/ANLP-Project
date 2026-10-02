"""Audit LLM-generated neutral summaries before they become training targets.

    uv run python scripts/audit_summaries.py runs/nemotron/summaries_10k.jsonl

Checks the failure modes that actually matter for a generation target set:
degenerate/duplicate output, truncation, refusals, leaked scaffolding, copying a
single source instead of summarising, hallucinated numbers, and template collapse.
"""
import argparse, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[a-z0-9']+")
NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?%?\b")
STOP = set("a an the and or of to in on at by for with from as is are was were be been it its this "
           "that he she they we you his her their our not no so than then there here which who what "
           "when where why how all any both each more most other some such only own same too very can "
           "will just should now said say says over under after before during about into out up down "
           "off again further once would could has have had do does did".split())

def toks(t): return WORD.findall((t or "").lower())
def content(t): return [w for w in toks(t) if w not in STOP and len(w) > 2]
def grams(ws, n=8): return {tuple(ws[i:i+n]) for i in range(len(ws)-n+1)}

def overlap(a, b, n=8):
    ga = grams(content(a), n)
    if not ga: return 0.0
    return len(ga & grams(content(b), n)) / len(ga)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--clusters", default=str(ROOT / "data/final/clusters_corpus10k.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.path, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("summary")]
    clusters = {}
    for line in open(args.clusters, encoding="utf-8"):
        c = json.loads(line)
        clusters[str(c["story_id"])] = c["articles"]

    print(f"{len(rows):,} records, {len(ok):,} with a summary\n")
    issues = {}

    # --- structural -------------------------------------------------------
    issues["empty or missing summary"] = [r["story_id"] for r in rows if not r.get("summary")]
    issues["under 30 words"] = [r["story_id"] for r in ok if r["actual_words"] < 30]
    issues["over 600 words"] = [r["story_id"] for r in ok if r["actual_words"] > 600]
    issues["finish_reason != stop"] = [r["story_id"] for r in ok
                                       if (r.get("finish_reason") or "").lower() != "stop"]
    # truncation: ends without terminal punctuation
    issues["no terminal punctuation"] = [r["story_id"] for r in ok
                                         if not r["summary"].rstrip().endswith((".", "!", "?", '"'))]

    # --- scaffolding / meta ----------------------------------------------
    issues["'Source X' in summary"] = [r["story_id"] for r in ok
                                       if re.search(r"\bSource [A-Z]\b", r["summary"])]
    issues["outlet named as reporter"] = [
        r["story_id"] for r in ok if re.search(
            r"(Washington Post|New York Times|CNN|Breitbart|Fox News|HuffPost|USA Today|"
            r"Daily Kos|Associated Press|The Hill|POLITICO)\s+(reported|said|compiled|covered|"
            r"published|ran|noted|wrote)", r["summary"])]
    issues["meta-commentary about articles"] = [
        r["story_id"] for r in ok if re.search(
            r"\b(the (article|report|piece|coverage)|this (article|report)|the sources?)\b",
            r["summary"], re.I)]
    issues["TITLE=>/ARTICLE=> scaffolding"] = [r["story_id"] for r in ok
                                               if "TITLE=>" in r["summary"] or "ARTICLE=>" in r["summary"]]
    issues["refusal / assistant voice"] = [
        r["story_id"] for r in ok if re.search(
            r"\b(I cannot|I'm unable|as an AI|I apologize|Here is|Here's the summary)\b",
            r["summary"], re.I)]
    issues["JSON fragment in summary"] = [r["story_id"] for r in ok
                                          if r["summary"].lstrip().startswith(("{", "[", "```"))]

    # --- degenerate output -------------------------------------------------
    seen = defaultdict(list)
    for r in ok:
        seen[" ".join(toks(r["summary"]))].append(r["story_id"])
    issues["exact duplicate summaries"] = [i for v in seen.values() if len(v) > 1 for i in v[1:]]
    # a sentence repeated inside one summary
    rep = []
    for r in ok:
        sents = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", r["summary"]) if len(s) > 40]
        if len(sents) != len(set(sents)):
            rep.append(r["story_id"])
    issues["repeated sentence within summary"] = rep

    # --- content fidelity --------------------------------------------------
    copies, halluc, per_stance = [], [], defaultdict(list)
    for r in ok:
        arts = clusters.get(r["story_id"])
        if not arts: continue
        src_all = " ".join(a["body"] for a in arts)
        # copying one article rather than summarising
        best = max(overlap(r["summary"], a["body"]) for a in arts)
        if best >= 0.5: copies.append(r["story_id"])
        # numbers invented
        snums, cnums = set(NUM.findall(r["summary"])), set(NUM.findall(src_all))
        made_up = snums - cnums
        if len(made_up) >= 3: halluc.append((r["story_id"], sorted(made_up)[:5]))
        for a in arts:
            per_stance[a["stance"]].append(overlap(r["summary"], a["body"], 5))
    issues["copies one source (>=50% 8-grams)"] = copies
    issues["3+ numbers not in any source"] = [i for i, _ in halluc]

    # --- report ------------------------------------------------------------
    print(f"{'issue':<42} {'count':>7}  {'%':>6}")
    print("-" * 60)
    for k, v in issues.items():
        n = len(v)
        flag = "  <-- " if n > len(ok) * 0.01 else ""
        print(f"{k:<42} {n:>7,}  {100*n/max(len(ok),1):5.2f}%{flag}")

    print(f"\nlexical overlap with each stance (5-gram, mean):")
    for s in sorted(per_stance):
        print(f"  {s:<8} {statistics.fmean(per_stance[s]):.3f}")
    gap = abs(statistics.fmean(per_stance['left']) - statistics.fmean(per_stance['right']))
    print(f"  |left - right| = {gap:.3f}")

    print(f"\ntemplate check -- most common opening 5 words:")
    for phrase, n in Counter(" ".join(toks(r["summary"])[:5]) for r in ok).most_common(8):
        print(f"  {n:>5}  {phrase}")

    if halluc:
        print(f"\nexamples of numbers not in any source:")
        for i, nums in halluc[:5]:
            print(f"  {i}: {nums}")

    (ROOT / "runs").mkdir(exist_ok=True)
    json.dump({k: v[:200] for k, v in issues.items()},
              open(ROOT / "runs/audit_issues.json", "w"), indent=1)
    print(f"\nids written to runs/audit_issues.json")

if __name__ == "__main__":
    main()
