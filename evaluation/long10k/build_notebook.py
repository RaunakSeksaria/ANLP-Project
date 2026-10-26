"""Builds evaluation/<dataset>_eval.ipynb (analysis only; scores come from evaluation/long10k/score.py).

    python evaluation/long10k/build_notebook.py long10k
    python evaluation/long10k/build_notebook.py short
"""
import sys
import nbformat as nbf
from pathlib import Path

DATASET = sys.argv[1] if len(sys.argv) > 1 else "long10k"
INTRO = {
    "long10k": """# Evaluation of the long Nemotron rewrite set (`long_10k`)

**Data.** Raunak's single-article framing-removal set, "full1024" release (HF
`RaunakSeksaria/anlp-project-datasets`, `data/long/`). 9,961 events sampled from BigNewsAlign; each
has one **left** and one **right** article (≤ 1,024 LED tokens, median 421 words), and each article
has a **Nemotron-3-Ultra rewrite** that removes framing (prompt `n2`). That is 19,922 originals and
19,922 rewrites. There are no centre articles in this set.""",
    "short": """# Evaluation of the short Nemotron rewrite set (`{left,right}_neutral_dedup`)

**Data.** Raunak's single-article framing-removal set (`origin/data`, `left_neutral_dedup.jsonl` and
`right_neutral_dedup.jsonl`): 10,000 left and 10,000 right articles of at most ~275 words, from
BigNewsAlign, each with a **Nemotron-3-Ultra rewrite** that removes framing (prompt `n2`). These
rewrites are the **training targets of the team's LED neutralisers**. That is 20,000 originals and
20,000 rewrites. There are no centre articles in this set.""",
}[DATASET]

nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s.replace("{DATASET}", DATASET)))

md(INTRO + """

**What is measured** (the instruments in `eval_metrics.md`, all validated earlier on this branch):
- **Ideology classifiers**: ChunkyBERT, its "No Classifier" variant, and Baly's BERT trained on the
  random split, the media split, and with adversarial adaptation. All output P(Left), P(Centre), P(Right).
- **Publisher classifier**: Baly's 73-way outlet classifier (78% accurate on its own test set).
- **VAD**: Warriner lexicon valence / arousal / dominance.

**Two questions:**
1. *On the originals:* do the classifiers recover each article's side? (Confusion matrices, true L/R.)
2. *Original → rewrite, paired:* does the rewrite move toward Centre, and does it hide the
   outlet and side? Each rewrite is compared with its own original, which cancels each
   classifier's fixed bias.

All texts are scored body-only, like-for-like with the rewrites. Originals whose
title also appears in Baly's corpus, the classifiers' training data, are excluded together with
their rewrites (the count is printed below).""")

co("""import json
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path("/root/ANLP/ANLP-Project")
H = ROOT / "evaluation/{DATASET}"
CLASSES = ["Left", "Centre", "Right"]
IDEO = ["chunkybert", "no_classifier", "baly_random", "baly_media", "baly_aa"]
NAMES = {"chunkybert": "ChunkyBERT (5 ep)", "no_classifier": "No Classifier variant (1 ep)",
         "baly_random": "Baly BERT, random split", "baly_media": "Baly BERT, media split",
         "baly_aa": "Baly BERT + adversarial"}

T = pd.DataFrame([json.loads(l) for l in open(H / "texts.jsonl")]).drop(columns="text")
excluded = T[T.in_baly_corpus].story_id + ":" + T[T.in_baly_corpus].side
T["pair"] = T.story_id + ":" + T.side
T = T[~T.pair.isin(set(excluded))].reset_index(drop=True)
print(f"texts after excluding {len(set(excluded))} contaminated originals (+ their rewrites): {len(T)}")

P = {}
for m in IDEO:
    d = pd.DataFrame([json.loads(l) for l in open(H / f"scores/{m}.jsonl")]).drop_duplicates("id").set_index("id")
    P[m] = np.stack(d.loc[T.id, "p"].values)
print("scored:", {m: len(P[m]) for m in P})
T.groupby(["side", "publisher"]).size().unstack(0).fillna(0).astype(int)""")

md("""## 1. Originals: do the classifiers recover the side?

True labels are Left or Right only, so each matrix has two true rows and three predicted columns.
A balanced coin flip between Left and Right would score 50%.""")

co("""orig = (T.kind == "original").values
y = T.label.values
rows = []
fig, ax = plt.subplots(1, len(IDEO), figsize=(4.2 * len(IDEO), 3.6))
for a, m in zip(ax, IDEO):
    pred = P[m][orig].argmax(1); yt = y[orig]
    cm = confusion_matrix(yt, pred, labels=[0, 1, 2])[[0, 2]]
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(3):
            a.text(j, i, f"{cm[i, j]}\\n{cm[i, j] / cm[i].sum():.0%}", ha="center", va="center", fontsize=9)
    a.set(xticks=range(3), xticklabels=CLASSES, yticks=range(2), yticklabels=["Left", "Right"],
          xlabel="Predicted", ylabel="True", title=NAMES[m])
    rows.append({"model": NAMES[m], "accuracy": 100 * (pred == yt).mean(),
                 "recall_Left": (pred[yt == 0] == 0).mean(), "recall_Right": (pred[yt == 2] == 2).mean(),
                 "pred_Centre_share": (pred == 1).mean(),
                 "binary_acc_L_vs_R": 100 * ((P[m][orig][:, 2] > P[m][orig][:, 0]) == (yt == 2)).mean()})
plt.tight_layout(); plt.show()
print("binary_acc_L_vs_R ignores Centre: is P(Right) > P(Left) for right articles, and vice versa.")
pd.DataFrame(rows).set_index("model").round(3)""")

co("""for m in IDEO:
    cm = pd.DataFrame(confusion_matrix(y[orig], P[m][orig].argmax(1), labels=[0, 1, 2])[[0, 2]],
                      index=["true Left", "true Right"], columns=[f"pred {c}" for c in CLASSES])
    print(f"\\n=== {NAMES[m]} ===\\n{cm.to_string()}")""")

md("### Accuracy by publisher (originals)")

co("""pd.DataFrame({NAMES[m]: T[orig].assign(ok=P[m][orig].argmax(1) == y[orig]).groupby(["side", "publisher"]).ok.mean() * 100
              for m in IDEO}).round(1).assign(articles=T[orig].groupby(["side", "publisher"]).size())""")

md("""## 2. Original → rewrite, paired

For every article, the same classifier scores the original and its rewrite. Reported:
- **Δ P(own side)**: mean change in the probability of the article's true side. Negative means
  the rewrite reads less like its side.
- **Δ P(Centre)**: positive means the rewrite reads more centrist.
- **Side recovered**: share whose argmax is still the true side, before → after. Lower after is
  better for a neutraliser.
- **Moved to Centre**: share whose argmax was not Centre before and is Centre after.""")

co("""O = T[T.kind == "original"].set_index("pair"); R = T[T.kind == "rewrite"].set_index("pair")
pairs = O.index.intersection(R.index)
io = pd.Series(np.arange(len(T)), index=T.id)
oi = io[O.loc[pairs, "id"]].values; ri = io[R.loc[pairs, "id"]].values
own = np.where(O.loc[pairs, "label"].values == 0, 0, 2)
k = np.arange(len(pairs))

shift = []
for m in IDEO:
    po, pr = P[m][oi], P[m][ri]
    ao, ar = po.argmax(1), pr.argmax(1)
    shift.append({"model": NAMES[m],
                  "P(own side) orig": po[k, own].mean(), "P(own side) rewrite": pr[k, own].mean(),
                  "Δ P(own side)": (pr[k, own] - po[k, own]).mean(),
                  "Δ P(Centre)": (pr[:, 1] - po[:, 1]).mean(),
                  "side recovered orig %": 100 * (ao == own).mean(), "side recovered rewrite %": 100 * (ar == own).mean(),
                  "moved to Centre %": 100 * ((ao != 1) & (ar == 1)).mean(),
                  "own-side prob fell %": 100 * (pr[k, own] < po[k, own]).mean()})
print(f"{len(pairs)} original/rewrite pairs")
pd.DataFrame(shift).set_index("model").round(3)""")

co("""fig, ax = plt.subplots(1, len(IDEO), figsize=(4.2 * len(IDEO), 3.4), sharey=True)
for a, m in zip(ax, IDEO):
    d = P[m][ri][k, own] - P[m][oi][k, own]
    a.hist(d, bins=60, color="#4C72B0"); a.axvline(0, color="k", lw=0.8)
    a.axvline(d.mean(), color="#C44E52", ls="--", label=f"mean {d.mean():+.3f}")
    a.set(title=NAMES[m], xlabel="Δ P(own side), rewrite − original"); a.legend()
ax[0].set_ylabel("pairs"); plt.tight_layout(); plt.show()""")

md("### Paired shift by side")

co("""by_side = []
for m in IDEO:
    for s, lab in (("left", 0), ("right", 2)):
        sel = O.loc[pairs, "side"].values == s
        po, pr = P[m][oi][sel], P[m][ri][sel]
        by_side.append({"model": NAMES[m], "side": s,
                        "mean P L/C/R orig": " / ".join(f"{x:.2f}" for x in po.mean(0)),
                        "mean P L/C/R rewrite": " / ".join(f"{x:.2f}" for x in pr.mean(0)),
                        "side recovered orig %": 100 * (po.argmax(1) == lab).mean(),
                        "side recovered rewrite %": 100 * (pr.argmax(1) == lab).mean()})
pd.DataFrame(by_side).set_index(["model", "side"]).round(1)""")

md("""## 3. Publisher leakage: can the outlet still be identified?

Baly's 73-way outlet classifier, on originals and rewrites. Top-1 / top-5 accuracy counts only
publishers that are among its 73 outlets (HuffPost is not). A neutraliser that removes outlet style
should push both toward zero. As a second view, each predicted outlet is mapped to its AllSides
side (the majority label of that outlet in Baly's corpus), which gives a side prediction that comes
purely from outlet identity.""")

co("""PUB2DOM = {"cnn": "www.cnn.com", "dailykos": "www.dailykos.com", "hpo": None, "nyt": "www.nytimes.com",
           "wpo": "www.washingtonpost.com", "breitbart": "www.breitbart.com", "fox": "www.foxnews.com",
           "wat": "www.washingtontimes.com"}
od = pd.DataFrame([json.loads(l) for l in open(H / "scores/baly_outlet.jsonl")]).drop_duplicates("id").set_index("id")
arts = pd.read_csv(ROOT / "data/baly_articles.csv", usecols=["source_url", "bias"])
dom_side = arts.groupby("source_url").bias.agg(lambda s: s.mode()[0])
vocab = set(d for t5 in od.top5 for d, _ in t5)

def leak(kind):
    d = T[T.kind == kind].copy()
    d["dom"] = d.publisher.map(PUB2DOM)
    top5 = od.loc[d.id, "top5"].values
    d["top1"] = [t[0][0] for t in top5]
    d["in_top5"] = [dm in {x for x, _ in t} for dm, t in zip(d.dom, top5)]
    d["outlet_side"] = d.top1.map(dom_side)
    return d

L = {k: leak(k) for k in ("original", "rewrite")}
rows = []
for k, d in L.items():
    known = d.dom.notna()
    rows.append({"text": k, "outlet top-1 acc %": 100 * (d.top1 == d.dom)[known].mean(),
                 "outlet top-5 acc %": 100 * d.in_top5[known].mean(),
                 "side via predicted outlet %": 100 * (d.outlet_side == d.label).mean()})
print("publishers covered by the 73-way vocab:", sorted(p for p, dm in PUB2DOM.items() if dm in vocab),
      "| not covered:", sorted(p for p, dm in PUB2DOM.items() if dm not in vocab))
pd.DataFrame(rows).set_index("text").round(2)""")

co("""pd.DataFrame({k: d[d.dom.notna()].assign(ok=lambda x: x.top1 == x.dom).groupby("publisher").ok.mean() * 100
              for k, d in L.items()}).round(1).rename(columns=lambda c: f"top-1 % ({c})")""")

co("""print("Most common predicted outlets for rewrites:")
L["rewrite"].top1.value_counts(normalize=True).head(8).round(3)""")

md("## 4. VAD (valence / arousal / dominance)")

co("""V = pd.read_csv(H / "scores/vad.csv").set_index("id")
vo, vr = V.loc[O.loc[pairs, "id"]].reset_index(drop=True), V.loc[R.loc[pairs, "id"]].reset_index(drop=True)
side = O.loc[pairs, "side"].values
out = []
for s in ("left", "right", "all"):
    m = np.ones(len(pairs), bool) if s == "all" else side == s
    for dim in ("valence", "arousal", "dominance"):
        out.append({"side": s, "dim": dim, "original": vo[dim][m].mean(), "rewrite": vr[dim][m].mean(),
                    "mean Δ": (vr[dim] - vo[dim])[m].mean(), "share decreased %": 100 * (vr[dim] < vo[dim])[m].mean()})
pd.DataFrame(out).set_index(["side", "dim"]).round(4)""")

md("""## Notes

- **No centre class.** Accuracy on the originals is a two-row problem, but the models can still
  predict Centre, and over-predicting it costs accuracy. `binary_acc_L_vs_R` ignores Centre.
- **Labels are outlet-level**, inherited from the publisher.
- **What a shift means.** Earlier on this branch, these classifiers turned out to mostly identify
  the outlet rather than the ideology. A rewrite that removes house style will lower P(own side)
  even if its slant is unchanged. Read section 2 together with section 3: if the outlet becomes
  unrecognisable, the ideology shift is at least partly the same effect.
- **Contamination:** originals whose titles appear in the classifiers' training corpus are
  excluded together with their rewrites.
- Per-text scores: `evaluation/long10k/scores/` (ideology `p` = [Left, Centre, Right]; outlet = top-5).""")

nb.cells = c
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
nbf.write(nb, Path(__file__).resolve().parents[1] / f"{DATASET}_eval.ipynb")
print("written")
