"""Summarise the LED bias-type run against originals (body only) and the existing summary/centre rows."""
import json
import numpy as np, pandas as pd
from pathlib import Path

from bias_types import PRESENCE

R = Path(__file__).resolve().parent / "results"
TYPES = list(PRESENCE)
p = lambda rec, q: rec[q]["p"]

rows = [json.loads(l) for l in open(R / "bias_types_led.jsonl")]
ref = [json.loads(l) for l in open(R / "bias_types.jsonl")]
df = pd.DataFrame([{"slot": r["slot"], "story_id": r["story_id"], "side": "left" if r["label"] == 0 else "right",
                    **{t: p(r, t) for t in TYPES}, "slant": r["slant_side"]["p"]} for r in rows])
df["slot"] = df.slot.str.replace("orig_body_left", "original").str.replace("orig_body_right", "original")
rf = pd.DataFrame([{"slot": {"center": "ref: centre article (title+body)", "neutral": "ref: neutral summary"}.get(r["slot"], None),
                    **{t: p(r, t) for t in TYPES}, "slant": r["slant_side"]["p"]} for r in ref])
rf = rf[rf.slot.notna()]
both = pd.concat([df, rf])
both["overall"] = both[TYPES].mean(1)

order = ["original", "led_lambda0", "led_lambda035", "led_first_token", "ref: centre article (title+body)", "ref: neutral summary"]
print("== Mean presence probability (lower = less bias) ==")
print(both.groupby("slot")[["overall"] + TYPES].mean().reindex(order).round(3).T.to_string())

print("\n== Slant direction (slant_side: P left / neither / right) ==")
sl = both.groupby("slot").slant.apply(lambda s: np.mean([[x.get("left", 0), x.get("neither", 0), x.get("right", 0)] for x in s], 0))
for k in order:
    print(f"  {k:34} " + " / ".join(f"{v:.2f}" for v in sl[k]))

print("\n== Paired: rewrite overall bias lower than its own original ==")
o = df[df.slot == "original"].set_index(["story_id", "side"])[TYPES].mean(1)
for m in ("led_lambda0", "led_lambda035", "led_first_token"):
    x = df[df.slot == m].set_index(["story_id", "side"])[TYPES].mean(1).reindex(o.index)
    d = (x - o)
    print(f"  {m:16} win-rate {(d < 0).mean():.1%}   mean Δ overall {d.mean():+.3f}   "
          f"left {(d.xs('left', level='side') < 0).mean():.1%}  right {(d.xs('right', level='side') < 0).mean():.1%}")
