"""How recoverable is the publisher from a representation of the article?

This is the number the whole proposal rests on. If publisher identity is not
recoverable, there is no leakage to remove; if it is recoverable from plain
word counts, then a representation-level fix is aimed at the wrong level.

Two feature sets, deliberately:

  tfidf   word counts. Runs on a laptop. The lexical floor -- how much of the
          publisher signal is just vocabulary and boilerplate?
  vectors a frozen encoder's pooled representation, cached by
          scripts/encode_articles.py. What the triplet loss would actually act on.

Two splits, deliberately:

  random  articles split at random. Every test publisher was seen in training.
  media   publisher-disjoint. No outlet appears on two sides. This is the
          generalisation question, and the released NeuS split cannot ask it
          (92 of its 99 test publishers also occur in train).

    uv run python scripts/probe.py --features tfidf
    uv run python scripts/probe.py --features vectors --vectors data/prepared/vectors_bart-large.npz
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data/prepared"


def load_articles():
    return [json.loads(l) for l in (PREP / "articles.jsonl").open()]


def load_split(name):
    blob = json.loads((PREP / f"probe_{name}.json").read_text())
    return blob["articles"] if "articles" in blob else blob


def subset(articles, ids):
    index = {a["article_id"]: a for a in articles}
    return [index[i] for i in ids if i in index]


def baselines(train_y, test_y):
    """Three reference points, because one is not enough here.

    `train_majority` (always predict the training set's most common class) is
    the standard baseline but it misleads when the test split has a different
    class balance -- which a publisher-disjoint split always does, since stance
    is a property of the outlet. Report `test_majority` and `chance` beside it.
    """
    train_top = Counter(train_y).most_common(1)[0][0]
    test_top = Counter(test_y).most_common(1)[0][0]
    return {
        "train_majority": float(accuracy_score(test_y, [train_top] * len(test_y))),
        "test_majority": float(accuracy_score(test_y, [test_top] * len(test_y))),
        "chance": 1.0 / len(set(train_y)),
    }


def tfidf_featuriser(max_features=50000):
    def f(train_texts, test_texts):
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=2)
        return vec.fit_transform(train_texts), vec.transform(test_texts)
    return f


def vector_featuriser(path):
    blob = np.load(path, allow_pickle=True)
    ids = list(blob["ids"])
    vectors = blob["vectors"]
    index = {a: i for i, a in enumerate(ids)}

    def by_ids(train, test):
        Xtr = vectors[[index[a["article_id"]] for a in train]]
        Xte = vectors[[index[a["article_id"]] for a in test]]
        scaler = StandardScaler().fit(Xtr)
        return scaler.transform(Xtr), scaler.transform(Xte)

    return by_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["tfidf", "vectors"], default="tfidf")
    ap.add_argument("--vectors", help="npz from scripts/encode_articles.py")
    ap.add_argument("--tag", default=None, help="name for this feature set in the output")
    ap.add_argument("--out", default=str(PREP / "probe_results.json"))
    args = ap.parse_args()

    articles = load_articles()
    tag = args.tag or args.features

    if args.features == "vectors":
        if not args.vectors:
            ap.error("--features vectors needs --vectors")
        featurise_pair = vector_featuriser(args.vectors)
    else:
        tf = tfidf_featuriser()
        def featurise_pair(train, test):
            return tf([a["body"] for a in train], [a["body"] for a in test])

    def probe(train, test, label, name, note=""):
        ytr = [a[label] for a in train]
        yte_all = [a[label] for a in test]
        seen = set(ytr)
        keep = [i for i, y in enumerate(yte_all) if y in seen]
        if not keep:
            return {"probe": name, "label": label, "n_train": len(train),
                    "n_test": len(test), "note": "no shared classes -- skipped"}
        Xtr, Xte = featurise_pair(train, test)
        Xte = Xte[keep]
        yte = [yte_all[i] for i in keep]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return {
            "probe": name, "label": label, "note": note,
            "n_train": len(train), "n_test": len(yte), "n_classes": len(seen),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro", zero_division=0)), 4),
            **{k: round(v, 4) for k, v in baselines(ytr, yte).items()},
        }

    results = []

    rnd = load_split("random")
    tr, te = subset(articles, rnd["train"]), subset(articles, rnd["test"])
    results.append(probe(tr, te, "publisher", f"{tag} · publisher · random split",
                         "every test outlet was seen in training"))
    results.append(probe(tr, te, "stance", f"{tag} · stance · random split"))

    med = load_split("media")
    mtr, mte = subset(articles, med["train"]), subset(articles, med["test"])
    results.append(probe(mtr, mte, "stance", f"{tag} · stance · publisher-disjoint",
                         "no outlet appears on both sides"))
    results.append(probe(mtr, mte, "publisher", f"{tag} · publisher · publisher-disjoint",
                         "expected to be impossible -- labels unseen by construction"))

    width = max(len(r["probe"]) for r in results)
    print(f"\n{'probe'.ljust(width)}  {'label':10} {'acc':>7} {'testmaj':>8} "
          f"{'chance':>7} {'macroF1':>8} {'classes':>8} {'n_test':>7}")
    print("-" * (width + 62))
    for r in results:
        if "accuracy" not in r:
            print(f"{r['probe'].ljust(width)}  {r['label']:10} {r['note']}")
            continue
        print(f"{r['probe'].ljust(width)}  {r['label']:10} {r['accuracy']:7.3f} "
              f"{r['test_majority']:8.3f} {r['chance']:7.3f} {r['macro_f1']:8.3f} "
              f"{r['n_classes']:8d} {r['n_test']:7d}")
    for r in results:
        if r.get("note") and "accuracy" in r:
            print(f"  {r['probe']}: {r['note']}")

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[tag] = results
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
