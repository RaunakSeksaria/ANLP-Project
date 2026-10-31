#!/usr/bin/env python3
"""POLITICS-v2 scores of the cc2025 originals and the eight models' rewrites (2,160 articles).

Same scoring as Raunak's cc2025 scores and data-branch scripts (scripts/kaggle_hard_subset/hard_subset_score.py +
scripts/make_hard_subset.py): outlet names masked with his OUTLET_RE plus `Salon`, 512-token windows with 64
overlap (politics_classifier.to_windows), raw mean-window logits per seed for seeds 42-45, then politics(): margin
and p_own averaged over seeds with the ensemble temperature, p = mean calibrated probabilities.

Input: dataset/cc2025/cc2025_eval.jsonl (original = article body) and outputs/cc2025_generations/<model>.jsonl.
Check: the originals must reproduce the `politics_original` scores stored with the articles.
Classifier: Harshit's POLITICS v2 release (seed42-45_best.pt, calibration.json, politics_classifier.py) in
~/anlp-project/politics_v2/ (--classifier).

    python chaitanya/cc2025_eval/score_cc2025_politics.py [--out outputs/cc2025_politics]
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CC2025 = ROOT / "dataset" / "cc2025" / "cc2025_eval.jsonl"
GENERATIONS = ROOT / "outputs" / "cc2025_generations"
SEEDS = ("seed42", "seed43", "seed44", "seed45")
OWN = {"left": 0, "right": 2}
MODELS = ("baseline_lambda0", "ideology_standard_m04", "ideology_attraction_m02", "ideology_attraction_m05",
          "publisher_standard_m02", "publisher_attraction_m02",
          "ideology_attraction_m03_b05_t03", "ideology_attraction_m03_b06_t05")
# copied verbatim from scripts/kaggle_hard_subset/hard_subset_score.py (= scripts/compare_variants.py), plus Salon
OUTLETS = [r"Associated Press", r"\bAP\b", r"The Hill", r"USA Today", r"CNN", r"Fox News", r"\bFox\b",
           r"Breitbart(?: News)?", r"Huff(?:ington )?Post", r"HuffPost", r"New York Times", r"\bNYT\b",
           r"Washington Post", r"Washington Times", r"Daily Kos", r"Salon"]
OUTLET_RE = re.compile("|".join(OUTLETS), re.I)
SYSTEMS = ("original",) + MODELS


def politics(logits, own, T):  # verbatim from scripts/make_hard_subset.py
    z = np.array(list(logits.values()), dtype=np.float64) / T                      # seeds x 3
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    others = np.delete(z, own, axis=1).max(1)
    return {"margin": round(float((z[:, own] - others).mean()), 4),
            "p_own": round(float(p[:, own].mean()), 4), "p": p.mean(0).round(4).tolist()}


def describe(items):
    p_own = np.array([x["p_own"] for x in items])
    p = np.array([x["p"] for x in items])
    own = np.array([x["own"] for x in items])
    top = p.argmax(1)
    return {"n": len(items), "median_p_own": round(float(np.median(p_own)), 4),
            "mean_p_own": round(float(p_own.mean()), 4),
            "own_side_top_pct": round(100 * float((top == own).mean()), 2),
            "centre_top_pct": round(100 * float((top == 1).mean()), 2),
            "mean_p_centre": round(float(p[:, 1].mean()), 4),
            "mean_margin": round(float(np.mean([x["margin"] for x in items])), 4)}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cc2025_rows() -> list[dict]:
    outputs = {m: {r["id"]: r["output"] for r in read(GENERATIONS / f"{m}.jsonl")} for m in MODELS}
    rows = [{"id": e["id"], "side": e["side"], "release": "cc2025", "split": "cc2025", "publisher": e["source"],
             "original": e["body"], "politics_original": e["politics_original"],
             "outputs": {m: outputs[m][e["id"]] for m in MODELS}} for e in read(CC2025)]
    for m in MODELS:
        assert len(outputs[m]) == len(rows) and all(v.strip() for v in outputs[m].values()), m
    return rows


def text_of(row, system):
    return row["original"] if system == "original" else row["outputs"][system]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "cc2025_politics")
    ap.add_argument("--classifier", type=Path, default=Path.home() / "anlp-project" / "politics_v2")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("politics_classifier", args.classifier / "politics_classifier.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    T = json.loads((args.classifier / "calibration.json").read_text())["temperature"]["ensemble"]
    rows = cc2025_rows()
    assert len(rows) == 2160, len(rows)

    texts = {}
    for r in rows:
        for system in SYSTEMS:
            text = text_of(r, system)
            texts[hashlib.md5(text.encode()).hexdigest()] = text
    items = sorted(texts.items(), key=lambda kv: len(kv[1]))
    tok = pc.tokenizer()
    wins = pc.to_windows(tok, [OUTLET_RE.sub("<mask>", t) for _, t in items])
    print(f"{len(rows)} articles, {len(items)} unique texts, {sum(map(len, wins))} windows, T={T}", flush=True)
    models, _ = pc.load([args.classifier / f"{s}_best.pt" for s in SEEDS])
    device = next(models[0].parameters()).device
    L = [pc.article_logits(m, wins, tok.pad_token_id, device) for m in models]
    logits = {k: {s: L[j][i].round(4).tolist() for j, s in enumerate(SEEDS)} for i, (k, _) in enumerate(items)}
    n_windows = {k: len(wins[i]) for i, (k, _) in enumerate(items)}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "merged.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    scored, check = [], []
    with (args.out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            own = OWN[r["side"]]
            row = {k: r[k] for k in ("id", "side", "release", "split", "publisher")}
            for system in SYSTEMS:
                k = hashlib.md5(text_of(r, system).encode()).hexdigest()
                res = politics(logits[k], own, T) | {"windows": n_windows[k]}
                row[system] = res | {"logits": logits[k]}
                scored.append({"system": system, "side": r["side"], "publisher": r["publisher"], "own": own, **res})
                if system == "original":
                    stored = r["politics_original"]
                    check.append((abs(res["margin"] - stored["margin"]), abs(res["p_own"] - stored["p_own"]),
                                  int(np.argmax(res["p"]) == np.argmax(stored["p"]))))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    d = np.array(check)
    report = {"temperature": T, "articles": len(rows), "check_vs_stored_scores": {"original": {
        "max_abs_margin_diff": round(float(d[:, 0].max()), 4), "max_abs_p_own_diff": round(float(d[:, 1].max()), 4),
        "same_top_label_pct": round(100 * float(d[:, 2].mean()), 2)}}}
    groups = [("all", lambda x: True), ("left", lambda x: x["side"] == "left"), ("right", lambda x: x["side"] == "right")]
    groups += [(o, lambda x, o=o: x["publisher"] == o) for o in ("dailykos", "salon", "fox", "breitbart")]
    groups += [("strict", lambda x: x["publisher"] != "salon")]
    for group, keep in groups:
        report[group] = {s: describe([x for x in scored if x["system"] == s and keep(x)]) for s in SYSTEMS}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"check": report["check_vs_stored_scores"], "all": report["all"]}, indent=1), flush=True)
    print(f"SCORING OK -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
