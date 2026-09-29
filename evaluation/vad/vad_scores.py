"""Lexicon VAD (valence/arousal/dominance) per slot on stories_10k.jsonl.

Lexicon: Warriner, Kuperman & Brysbaert (2013) affective norms, 13,915 lemmas,
1-9 scale (5 = neutral). Document score = mean over covered tokens.
"""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "stories_10k.jsonl"
LEX = Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "warriner_vad.csv"))
SLOTS = ["left", "center", "right", "neutral"]

w = pd.read_csv(LEX, usecols=["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"])
lex = {str(r.Word).lower(): (r[1], r[2], r[3]) for r in w.itertuples(index=False)}

TOKEN = re.compile(r"[a-z']+")

def vad(text):
    hits = [lex[t] for t in TOKEN.findall(text.lower()) if t in lex]
    n = len(hits)
    if not n:
        return None
    return (sum(h[0] for h in hits) / n, sum(h[1] for h in hits) / n,
            sum(h[2] for h in hits) / n, n)

rows = []
for line in DATA.open(encoding="utf-8"):
    if not line.strip():
        continue
    s = json.loads(line)
    for slot in SLOTS:
        v = s[slot]
        text = v["body"] if isinstance(v, dict) else v
        r = vad(text)
        if r:
            rows.append({"story_id": s["story_id"], "split": s.get("split"), "slot": slot,
                         "valence": r[0], "arousal": r[1], "dominance": r[2], "n_words": r[3]})

df = pd.DataFrame(rows)
df.to_csv(ROOT / "vad_scores.csv", index=False)

tbl = df.groupby("slot")[["valence", "arousal", "dominance"]].agg(["mean", "std"])
tbl.columns = [f"{a}_{b}" for a, b in tbl.columns]
tbl["n_docs"] = df.groupby("slot").size()
print(tbl.reindex(SLOTS).round(4).to_string())
