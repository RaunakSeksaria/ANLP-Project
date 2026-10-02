"""One table of every text the LED evaluation scores: data/led_eval/texts.jsonl.

Two panels:
  test  the 1,000 test-split stories of stories_10k (held out from LED training). Per story:
        original left/right article, centre article, LLM multi-source summary, Nemotron rewrite
        (the LED training target) of the left/right article, and each LED model's rewrite.
  data  Raunak's single-article release (left/right_neutral_dedup.jsonl, all splits): every
        original article and its Nemotron rewrite.

Every row carries `h`, a hash of its text; scorers score each distinct text once.
Rerun after generation to pick up the LED rewrites (led_eval/out/gen_<model>.jsonl).
"""
import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "out"
LED = ["led_lambda0", "led_lambda035"]
SIDES = ("left", "right")


def norm(s):
    return " ".join(s.split())


def key(s):                       # Raunak's dedup criterion: ignore case and punctuation
    return re.sub(r"[^a-z0-9]", "", s.lower())


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def row(id_, panel, story_id, split, side, kind, publisher, text):
    text = norm(text)
    return {"id": id_, "panel": panel, "story_id": story_id, "split": split, "side": side,
            "kind": kind, "publisher": publisher, "h": h(text), "text": text}


def main():
    rows, nemo = [], {}
    for side in SIDES:
        for l in open(DATA / f"led_eval/{side}_neutral_dedup.jsonl"):
            r = json.loads(l)
            o = r[f"{side}_original"]
            nemo[key(o["body"])] = r[f"{side}_neutralised"]
            for kind, text in (("original", o["body"]), ("nemotron", r[f"{side}_neutralised"])):
                rows.append(row(f"data:{o['article_id']}:{kind}", "data", r["story_id"], r["split"],
                                side, kind, o["publisher"], text))

    gen = {m: {g["id"]: g["output"] for g in map(json.loads, open(OUT / f"gen_{m}.jsonl"))}
           for m in LED if (OUT / f"gen_{m}.jsonl").exists()}
    missing = 0
    for s in map(json.loads, open(DATA / "stories_10k.jsonl")):
        if s["split"] != "test":
            continue
        sid = s["story_id"]
        t = lambda side, kind, text, pub=None: rows.append(
            row(f"{sid}:{side}:{kind}", "test", sid, "test", side, kind, pub, text))
        t("center", "centre", s["center"]["body"], s["center"]["publisher"])
        t(None, "llm_summary", s["neutral"])
        for side in SIDES:
            a = s[side]
            t(side, "original", a["body"], a["publisher"])
            n = nemo.get(key(a["body"]))
            if n is None:
                missing += 1
            else:
                t(side, "nemotron", n, a["publisher"])
            for m, g in gen.items():
                if f"{sid}:{side}" in g:
                    t(side, m, g[f"{sid}:{side}"], a["publisher"])

    with open(DATA / "led_eval/texts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(Counter((r["panel"], r["kind"]) for r in rows))
    print(f"{len(rows)} rows, {len({r['h'] for r in rows})} distinct texts; "
          f"test articles without a Nemotron rewrite: {missing}")


if __name__ == "__main__":
    main()
