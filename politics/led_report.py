"""POLITICS on the LED outputs vs the same originals (body only). Usage: python led_report.py <results dir> <name>"""
import sys, numpy as np, pandas as pd
d, name = sys.argv[1], sys.argv[2]
S = pd.read_csv(f"{d}/{name}_stories_scores.csv", dtype={"story_id": str})
L = pd.read_csv(f"{d}/{name}_led_scores.csv", dtype={"id": str})
P = ["p_left", "p_centre", "p_right"]
O = S[S.slot.isin(["left", "right"])].assign(id=lambda x: x.story_id + ":" + x.slot).set_index("id")
ids = sorted(set(L.id))
own = np.array([0 if i.endswith(":left") else 2 for i in ids])
po = O.loc[ids, P].values
rows = []
for k in ["original", "led_lambda0", "led_lambda035", "led_first_token"]:
    p = po if k == "original" else L[L.kind == k].set_index("id").loc[ids, P].values
    a = p.argmax(1)
    rows.append({"texts": k, "pred L/C/R %": "/".join(f"{x:.0%}" for x in np.bincount(a, minlength=3) / len(a)),
                 "own side %": round(100 * (a == own).mean(), 1), "P(own)": round(p[np.arange(len(ids)), own].mean(), 3),
                 "→Centre %": None if k == "original" else round(100 * ((po.argmax(1) != 1) & (a == 1)).mean(), 1),
                 "left→L/C/R": "/".join(f"{x:.0%}" for x in np.bincount(a[own == 0], minlength=3) / (own == 0).sum()),
                 "right→L/C/R": "/".join(f"{x:.0%}" for x in np.bincount(a[own == 2], minlength=3) / (own == 2).sum())})
print(f"{name}: {len(ids)} held-out test articles"); print(pd.DataFrame(rows).to_string(index=False))
