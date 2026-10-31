"""Score any JSONL of texts ({"id", "text", ...}) with the POLITICS v2 seed models: sliding-window logits per seed,
then the calibrated seed ensemble. Writes one CSV row per text: id, raw logits per seed, ensemble P(L, C, R).

    python score_texts.py --weights runs/seed*/seed*_best.pt --calibration v2_results/calibration.json \
                          --texts texts.jsonl --out scores.csv
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from politics_classifier import PoliticsClassifier, tokenizer, to_windows, article_logits, softmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    temps = json.load(open(a.calibration))["temperature"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(a.texts)]
    D = pd.DataFrame({"id": [r["id"] for r in rows]})
    tok = tokenizer()
    wins = to_windows(tok, [r["text"] for r in rows])
    print(f"{len(rows)} texts, {sum(map(len, wins))} windows", flush=True)

    P = []
    for w in a.weights:
        seed = Path(w).stem.split("_")[0]                   # "seed42"
        m = PoliticsClassifier()
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        lg = article_logits(m.float().to(dev).eval(), wins, tok.pad_token_id, dev)
        D[[f"{seed}_logit_l", f"{seed}_logit_c", f"{seed}_logit_r"]] = lg
        P.append(softmax(lg, temps["ensemble"]))
        print(f"scored with {seed}", flush=True)
        del m; torch.cuda.empty_cache()
    D[["p_left", "p_centre", "p_right"]] = np.mean(P, 0)
    D["windows"] = [len(w) for w in wins]
    D.to_csv(a.out, index=False)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
