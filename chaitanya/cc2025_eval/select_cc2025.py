#!/usr/bin/env python3
"""Pick the cc2025 evaluation articles (Raunak's 2025-26 Common Crawl set, cc2025_dataset.zip of 31 Oct 2026).

Pool = union of cc2025_strict_LR.jsonl and cc2025_mixed_LR.jsonl (12,500 unique articles; Daily Kos, Salon = left,
Fox, Breitbart = right). Keeps an article when
  - its "title </s> body" fits in the models' 512-token input (LED tokenizer, special tokens included), and
  - Raunak's POLITICS-v2 4-seed scores give P(own side) >= 0.75 (Manas's threshold; own side = the outlet's
    left/right label), with politics() exactly as in score_cc2025_politics.py (ensemble temperature).
Writes cc2025_eval.jsonl: id, source, side, date, url, title, body, input_tokens, politics_original.

usage: python select_cc2025.py <cc2025 data dir> <calibration.json> <LED tokenizer dir> <out.jsonl>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

OWN = {"left": 0, "right": 2}
THRESHOLD = 0.75


def politics(logits, own, T):  # verbatim from score_cc2025_politics.py (= Raunak's scripts/make_hard_subset.py)
    z = np.array(list(logits.values()), dtype=np.float64) / T                      # seeds x 3
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    others = np.delete(z, own, axis=1).max(1)
    return {"margin": round(float((z[:, own] - others).mean()), 4),
            "p_own": round(float(p[:, own].mean()), 4), "p": p.mean(0).round(4).tolist()}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    data, calibration, tok_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    T = json.loads(calibration.read_text())["temperature"]["ensemble"]
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    articles = {}
    for name in ("cc2025_strict_LR.jsonl", "cc2025_mixed_LR.jsonl"):
        for r in read(data / name):
            assert articles.get(r["id"], r) == r, f"{r['id']} differs between files"
            articles[r["id"]] = r
    scores = {r["id"]: r for r in read(data / "cc2025_politics_v2_scores.jsonl")}
    assert len(articles) == 12500 and set(articles) <= set(scores), (len(articles), len(set(articles) - set(scores)))

    kept = []
    for i, r in articles.items():
        title, body = r["title"].strip(), r["textString"].strip()
        source = f"{title} {tokenizer.sep_token} {body}" if title else body
        tokens = len(tokenizer(source, truncation=False)["input_ids"])
        pol = politics(scores[i]["logits"], OWN[r["political_bias"]], T) | {"windows": scores[i]["windows"],
                                                                         "logits": scores[i]["logits"]}
        if tokens <= 512 and pol["p_own"] >= THRESHOLD:
            kept.append({"id": i, "source": r["source"], "side": r["political_bias"], "date": r["date"], "url": r["url"],
                         "title": r["title"], "body": r["textString"], "input_tokens": tokens, "politics_original": pol})
    kept.sort(key=lambda r: (r["side"], r["source"], r["id"]))
    counts = {"articles": len(kept), "side": dict(Counter(r["side"] for r in kept)),
              "source": dict(Counter(r["source"] for r in kept)), "max_input_tokens": max(r["input_tokens"] for r in kept)}
    print(json.dumps(counts), flush=True)
    assert counts["articles"] == 2160, counts
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
