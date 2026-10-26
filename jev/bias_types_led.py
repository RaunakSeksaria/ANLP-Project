"""AllSides bias types (bias_types.QUESTIONS) on the LED neutraliser outputs, paired with their originals.

Same 750 test stories as results/bias_types.jsonl. For each story's left and right article: the original
body (body only, so it is like-for-like with the rewrites) and the outputs of every LED model found in
led_eval/out/gen_<model>.jsonl. Resumable; --budget-usd caps this invocation's spend.

    python jev/bias_types_led.py --budget-usd 0.40
"""
import argparse, asyncio, json
from pathlib import Path

import jev_classify
from jev_classify import OUT, ROOT, Ledger, score_all
from bias_types import QUESTIONS, flatten


def items():
    stories = {json.loads(l)["story_id"] for l in open(OUT / "bias_types.jsonl")}
    src = {s["story_id"]: s for s in map(json.loads, open(ROOT / "data/stories_10k.jsonl")) if s["story_id"] in stories}
    out = []
    for sid in sorted(src):
        for side in ("left", "right"):
            a = src[sid][side]
            out.append({"id": f"orig_body:{sid}:{side}", "story_id": sid, "split": "test", "slot": f"orig_body_{side}",
                        "label": 0 if side == "left" else 2, "publisher": a["publisher"], "title": None, "text": a["body"]})
    for f in sorted((ROOT / "led_eval/out").glob("gen_*.jsonl")):
        model = f.stem[4:]
        for r in map(json.loads, open(f)):
            sid, side = r["id"].split(":")
            if sid in src:
                out.append({"id": f"{model}:{r['id']}", "story_id": sid, "split": "test", "slot": model,
                            "label": 0 if side == "left" else 2, "publisher": src[sid][side]["publisher"],
                            "title": None, "text": r["output"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    jev_classify.QUESTIONS.update(QUESTIONS)
    jev_classify.probs = flatten
    ledger = Ledger(0)
    ledger.cap = (ledger.usd + a.budget_usd) / jev_classify.PRICE_PER_TOKEN
    it = items()
    print(f"{len(it)} texts; spent before ${ledger.usd:.4f}; this run may spend up to ${a.budget_usd:.2f}", flush=True)
    asyncio.run(score_all(it, list(QUESTIONS), False, OUT / "bias_types_led.jsonl", ledger, 2600, a.concurrency))


if __name__ == "__main__":
    main()
