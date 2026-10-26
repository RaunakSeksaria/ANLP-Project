"""Jev `outlet_style` ideology scores for every neutral text: the remaining stories_10k neutral summaries,
the Nemotron rewrites of the short and long sets, and the LED neutraliser outputs once generated.

Same question, client, runner and spend ledger as jev_classify.py. Rewrites are scored body-only
(they have no title). Resumable; spend accumulates in results/spend.json and stops at --cap-usd.

    python jev/score_neutral.py summaries
    python jev/score_neutral.py rewrites --dataset short|long10k
    python jev/score_neutral.py led                   # needs led_eval/out/gen_<model>.jsonl
"""
import argparse, asyncio, json
from pathlib import Path

import jev_classify as J

ROOT = J.ROOT


def summary_items():
    return J.load_items()[1]


def rewrite_items(dataset):
    out = []
    for l in open(ROOT / "evaluation" / dataset / "texts.jsonl"):
        t = json.loads(l)
        if t["kind"] == "rewrite":
            out.append({"id": f"{dataset}:{t['id']}", "story_id": t["story_id"], "split": t["split"],
                        "slot": "rewrite", "label": t["label"], "publisher": t["publisher"],
                        "title": None, "text": t["text"]})
    return out


def led_items():
    out = []
    stories = {s["story_id"]: s for s in map(json.loads, open(ROOT / "data/stories_10k.jsonl"))}
    for f in sorted((ROOT / "led_eval/out").glob("gen_*.jsonl")):
        model = f.stem[4:]
        for r in map(json.loads, open(f)):
            sid, side = r["id"].split(":")
            out.append({"id": f"{model}:{r['id']}", "story_id": sid, "split": stories[sid]["split"],
                        "slot": model, "label": 0 if side == "left" else 2,
                        "publisher": stories[sid][side]["publisher"], "title": None, "text": r["output"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["summaries", "rewrites", "led"])
    ap.add_argument("--dataset", choices=["short", "long10k"])
    ap.add_argument("--cap-usd", type=float, required=True, help="cumulative cap across all Jev runs")
    ap.add_argument("--est-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()

    if a.what == "summaries":   # same file as the original run, so the earlier 8,001 count as done
        items, f = summary_items(), J.OUT / "run_outlet_style_title.jsonl"
    elif a.what == "rewrites":
        items, f = rewrite_items(a.dataset), J.OUT / f"rewrites_{a.dataset}.jsonl"
    else:
        items, f = led_items(), J.OUT / "led_outputs.jsonl"
    print(f"{a.what} {a.dataset or ''}: {len(items)} items -> {f.name}", flush=True)
    asyncio.run(J.score_all(items, ["outlet_style"], False, f, J.Ledger(a.cap_usd),
                            a.est_tokens, a.concurrency))


if __name__ == "__main__":
    main()
