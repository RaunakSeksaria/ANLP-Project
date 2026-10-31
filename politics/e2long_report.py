"""POLITICS v2 on E2 long: tables from the politics-e2long kernel (originals + Nemotron rewrites, all splits) and,
if present, the quality-e2long kernel's scores of the LED rewrites of the test articles.

Labels are the outlet's side (left / right); E2 long has no centre articles. "Own side" = the article's side.

    python e2long_report.py --texts ../evaluation/long10k/texts.jsonl --scores e2long_politics.csv \
                            [--led led_politics.csv --test ../quality_eval/inputs/e2long_test.jsonl] --out e2long/
"""
import argparse, json
from pathlib import Path

import pandas as pd

P = ["p_left", "p_centre", "p_right"]


def summary(d):
    """d: rows of one text kind, with label (0 / 2) and P columns."""
    pred = d[P].values.argmax(1)
    own = d[P].values[range(len(d)), d.label.values]
    lr = d[["p_left", "p_right"]].values.argmax(1) * 2                      # forced Left vs Right
    side = lambda lab: {k: round(100 * float(((pred == k) & (d.label == lab)).sum() / max(1, (d.label == lab).sum())), 1)
                        for k in (0, 1, 2)}
    return {"n": len(d), "acc_own_side": round(100 * float((pred == d.label.values).mean()), 1),
            "acc_left_vs_right": round(100 * float((lr == d.label.values).mean()), 1),
            "recall_left": side(0)[0], "recall_right": side(2)[2],
            "pred_LCR": [round(100 * float((pred == k).mean()), 1) for k in (0, 1, 2)],
            "left_articles_pred_LCR": list(side(0).values()), "right_articles_pred_LCR": list(side(2).values()),
            "mean_P_LCR": [round(float(x), 3) for x in d[P].mean()], "mean_P_own": round(float(own.mean()), 3)}


def paired(o, r):
    """o, r: originals and their rewrites, aligned."""
    po, pr = o[P].values.argmax(1), r[P].values.argmax(1)
    own_o = o[P].values[range(len(o)), o.label.values]
    own_r = r[P].values[range(len(r)), r.label.values]
    return {"moved_to_centre": round(100 * float(((po != 1) & (pr == 1)).mean()), 1),
            "moved_to_other_side": round(100 * float(((po == o.label.values) & (pr == 2 - o.label.values)).mean()), 1),
            "P_own_drop_mean": round(float((own_o - own_r).mean()), 3),
            "P_centre_gain_mean": round(float((r.p_centre.values - o.p_centre.values).mean()), 3)}


def row(name, s, p=None):
    t = (f"| {name} | {s['n']:,} | {s['acc_own_side']} | {s['acc_left_vs_right']} | {s['recall_left']} / {s['recall_right']} | "
         f"{'/'.join(f'{x:.2f}' for x in s['mean_P_LCR'])} | {s['mean_P_own']:.3f} | "
         f"{'/'.join(f'{x:.0f}' for x in s['left_articles_pred_LCR'])}% | {'/'.join(f'{x:.0f}' for x in s['right_articles_pred_LCR'])}% |")
    return t + (f" {p['moved_to_centre']}% | {p['moved_to_other_side']}% |" if p else " — | — |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True); ap.add_argument("--scores", required=True)
    ap.add_argument("--led"); ap.add_argument("--test"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame([{k: v for k, v in json.loads(l).items() if k != "text"} for l in open(a.texts)])
    D = meta.merge(pd.read_csv(a.scores), on="id", validate="1:1")
    assert len(D) == len(meta), "scores do not cover every text"
    D["key"] = D.story_id.astype(str) + ":" + D.side
    res, lines = {}, []
    head = ("| Texts | n | Own side top label % | Left vs Right % | Recall L / R % | Mean P(L/C/R) | Mean P(own side) "
            "| Left articles → L/C/R | Right articles → L/C/R | Moved to Centre | Moved to other side |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    for scope, m in (("all", ~D.in_baly_corpus), ("test", ~D.in_baly_corpus & (D.split == "test"))):
        o = D[m & (D.kind == "original")].set_index("key").sort_index()
        r = D[m & (D.kind == "rewrite")].set_index("key").sort_index()
        assert (o.index == r.index).all()
        so, sr, pp = summary(o), summary(r), paired(o, r)
        res[scope] = {"original": so, "nemotron": sr, "paired": pp}
        lines += [f"\n**E2 long, {scope} splits** (articles also in Baly's corpus excluded)\n" if scope == "all" else
                  "\n**E2 long, test split** (articles also in Baly's corpus excluded)\n", head,
                  row("Original", so), row("Nemotron rewrite", sr, pp)]
        if scope == "all":
            res["all"]["by_split_acc_own_side"] = {s: [summary(o[o.split == s])["acc_own_side"], summary(r[r.split == s])["acc_own_side"]]
                                                   for s in ("train", "validation", "test")}
            res["all"]["by_publisher"] = {pub: [summary(o[o.publisher == pub])["mean_P_own"], summary(r[r.publisher == pub])["mean_P_own"],
                                                int((o.publisher == pub).sum())] for pub in sorted(o.publisher.unique())}
    if a.led and Path(a.led).exists():
        test = [json.loads(l) for l in open(a.test)]
        ids = [t["input_article_id"] for t in test]
        L = pd.read_csv(a.led)
        L["key"] = L.id.str.rsplit(":", n=1).str[0]; L["model"] = L.id.str.rsplit(":", n=1).str[1]
        base = D[D.kind == "original"].set_index("key").loc[ids]
        lines += [f"\n**E2 long test split, LED quality-eval set** ({len(ids):,} articles not in LED training data)\n", head]
        res["led_set"] = {}
        for name, frame in [("Original", base), ("Nemotron rewrite", D[D.kind == "rewrite"].set_index("key").loc[ids])] + \
                [(f"LED {m}", L[L.model == m].set_index("key").loc[ids].assign(label=base.label.values))
                 for m in ("lambda0", "lambda035", "first_token")]:
            s = summary(frame); p = None if name == "Original" else paired(base, frame)
            res["led_set"][name] = {**s, **(p or {})}
            lines.append(row(name, s, p))
    md = "\n".join(lines) + "\n"
    (out / "e2long_politics.md").write_text(md)
    json.dump(res, open(out / "e2long_politics.json", "w"), indent=2)
    print(md)
    print("acc own side by split (original, rewrite):", res["all"]["by_split_acc_own_side"])


if __name__ == "__main__":
    main()
