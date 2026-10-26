"""Score the team's stories_10k (30,000 labelled source articles + 10,000 neutral summaries) with the POLITICS
v2 seed models: sliding-window logits per seed, then the calibrated seed ensemble.

    python score_v2.py --weights runs/seed*/seed*_best.pt --calibration v2_results/calibration.json \
                       --stories stories_10k.jsonl --out v2_results/
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from politics_classifier import (PoliticsClassifier, tokenizer, to_windows, article_logits, softmax, metrics)

SLOT2ID = {"left": 0, "center": 1, "right": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--stories", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    temps = json.load(open(a.calibration))["temperature"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for line in open(a.stories):
        s = json.loads(line)
        for slot in ("left", "center", "right"):
            rows.append({"story_id": s["story_id"], "split": s.get("split"), "slot": slot, "label": SLOT2ID[slot],
                         "publisher": s[slot]["publisher"], "text": s[slot]["body"]})
        rows.append({"story_id": s["story_id"], "split": s.get("split"), "slot": "neutral", "label": -1,
                     "publisher": None, "text": s["neutral"]})
    D = pd.DataFrame(rows)
    tok = tokenizer()
    wins = to_windows(tok, D.text.tolist())
    print(f"{len(D)} texts, {sum(map(len, wins))} windows", flush=True)

    P = []
    for w in a.weights:
        seed = Path(w).stem.split("_")[0]                   # "seed42"
        m = PoliticsClassifier()
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        m = m.float().to(dev).eval()
        lg = article_logits(m, wins, tok.pad_token_id, dev)
        P.append(softmax(lg, temps["ensemble"]))
        D[[f"{seed}_p_left", f"{seed}_p_centre", f"{seed}_p_right"]] = softmax(lg, temps[seed])
        print(f"scored with {seed}", flush=True)
        del m; torch.cuda.empty_cache()
    pe = np.mean(P, 0)
    D[["p_left", "p_centre", "p_right"]] = pe
    D["pred_id"] = pe.argmax(1)
    D.drop(columns="text").to_csv(out / "stories_10k_scores.csv", index=False)

    lab = D[D.slot != "neutral"]; neu = D[D.slot == "neutral"]
    res = {"labelled": metrics(lab.label.values, lab[["p_left", "p_centre", "p_right"]].values),
           "labelled_per_seed_macro_f1": {s: metrics(lab.label.values, lab[[f"{s}_p_left", f"{s}_p_centre", f"{s}_p_right"]].values)["macro_f1"]
                                          for s in [Path(w).stem.split("_")[0] for w in a.weights]},
           "neutral_pred_share_LCR": (np.bincount(neu.pred_id, minlength=3) / len(neu)).round(4).tolist(),
           "neutral_mean_p_LCR": neu[["p_left", "p_centre", "p_right"]].mean().round(4).tolist()}
    json.dump(res, open(out / "stories_10k_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "labelled"} | {"macro_f1": res["labelled"]["macro_f1"],
                      "acc": res["labelled"]["acc"]}), flush=True)


if __name__ == "__main__":
    main()
