"""Build a corpus with LEFT+RIGHT as input and the generated neutral article as target.

    uv run python scripts/prepare_llm_targets.py

Same events, same connected-component splits and same source formatting as
data/prepared_bna, so the only variable against that corpus is the target:
centre article there, LLM-generated neutral article here.

Writes data/prepared_llm/ and never touches data/prepared_bna/.
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Same trap as before: str.splitlines() breaks on more than \n, and the trainer
# reads the gen/ files that way. One stray char shifts every later target.
BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_FLAT = str.maketrans({c: " " for c in BREAKS})
def flatten(s): return " ".join((s or "").translate(_FLAT).split())

def jdump(o):
    return (json.dumps(o, ensure_ascii=False)
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            .replace("\x85", "\\u0085"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default=str(ROOT / "runs/nemotron/summaries_10k.jsonl"))
    ap.add_argument("--clusters", default=str(ROOT / "data/final/clusters_corpus10k.jsonl"))
    ap.add_argument("--splits-from", default=str(ROOT / "data/prepared_bna/gen"))
    ap.add_argument("--out", default=str(ROOT / "data/prepared_llm"))
    args = ap.parse_args()

    summaries = {}
    for line in open(args.summaries, encoding="utf-8"):
        r = json.loads(line)
        if r.get("summary"):
            summaries[str(r["story_id"])] = r
    clusters = {}
    for line in open(args.clusters, encoding="utf-8"):
        c = json.loads(line)
        clusters[str(c["story_id"])] = c["articles"]

    # split assignment and ORDER carried over verbatim from prepared_bna
    order = {}
    for s in ("train", "val", "test"):
        order[s] = Path(args.splits_from, f"{s}.ids").read_text().split()
    print(f"summaries {len(summaries):,}   clusters {len(clusters):,}")
    print("splits from prepared_bna: " + "  ".join(f"{k}={len(v):,}" for k, v in order.items()))

    out = Path(args.out); (out / "gen").mkdir(parents=True, exist_ok=True)
    missing, counts = [], Counter()
    for split, ids in order.items():
        fs = (out / "gen" / f"{split}.source").open("w", encoding="utf-8")
        ft = (out / "gen" / f"{split}.target").open("w", encoding="utf-8")
        fi = (out / "gen" / f"{split}.ids").open("w", encoding="utf-8")
        fj = (out / f"{split}.jsonl").open("w", encoding="utf-8")
        for eid in ids:
            rec, arts = summaries.get(eid), clusters.get(eid)
            if not rec or not arts:
                missing.append(eid); continue
            src = sorted((a for a in arts if a["stance"] != "center"),
                         key=lambda a: a["stance"])
            # seeded per event, same convention as prepare_bignews.py
            import random as _r
            _r.Random(f"42:{eid}").shuffle(src)
            source = " [SEP] ".join(
                f"TITLE=> {flatten(a['title'])} . ARTICLE=> {flatten(a['body'])}" for a in src)
            target = flatten(rec["summary"])
            fs.write(source + "\n"); ft.write(target + "\n"); fi.write(eid + "\n")
            fj.write(jdump({
                "id": eid, "split": split,
                "source": source, "target": target,
                "target_kind": "llm_neutral_article",
                "inputs": [{"stance": a["stance"], "publisher": a["publisher"],
                            "title": a["title"], "body": a["body"]} for a in src],
                "held_out_center": next(({"publisher": a["publisher"], "title": a["title"],
                                          "body": a["body"]}
                                         for a in arts if a["stance"] == "center"), None),
                "conflicts": rec.get("conflicts") or [],
                "generator": {"model": rec.get("model"),
                              "prompt_version": (rec.get("params") or {}).get("prompt_version"),
                              "temperature": (rec.get("params") or {}).get("temperature")},
            }) + "\n")
            counts[split] += 1
        for f in (fs, ft, fi, fj): f.close()

    if missing:
        print(f"skipped {len(missing)} event(s) with no summary: {missing[:5]}")
    print("\n" + "  ".join(f"{k}={counts[k]:,}" for k in ("train", "val", "test"))
          + f"   total={sum(counts.values()):,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
