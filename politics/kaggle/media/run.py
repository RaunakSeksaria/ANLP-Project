# POLITICS fine-tuned on Article-Bias-Prediction (media split), weights saved, LED outputs scored; generated from politics/politics_baly.py
"""Fine-tune POLITICS (Liu et al. 2022, HF `launch/POLITICS`) as a Left/Centre/Right classifier on the
Article-Bias-Prediction corpus, then score the team's stories_10k (30k labelled articles + 10k neutral
summaries) and the corpus's own test set.

POLITICS ships no fine-tuning code. Head and optimiser follow the paper's Table A7: [CLS] -> 2-layer
FFNN (hidden 768, dropout 0.1), AdamW lr 2e-5, weight decay 0.001, betas (0.9, 0.999), linear decay with
6% warmup, early stopping on validation macro-F1. Epochs are capped at 5 with patience 2 (paper: 10 / 4),
to match the Baly models this is compared against. Input is 512 tokens, body only.

    python politics/politics_baly.py --split random|media [--encoder launch/POLITICS] [--limit N]
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Left", "Centre", "Right"]
SLOT2ID = {"left": 0, "center": 1, "right": 2}


def find(name, roots):
    for root in roots:
        for d, dirs, files in os.walk(root):
            if name in files or name in dirs:
                return Path(d) / name
    return None


def metrics(y, p):
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(p).argmax(1)
    return {"macro_f1": 100 * f1_score(y, pred, average="macro"), "acc": 100 * (y == pred).mean(),
            "mae": float(np.abs(y - pred).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "media"], required=True)
    ap.add_argument("--encoder", default="launch/POLITICS")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--roots", default="/kaggle/input,data,external")
    ap.add_argument("--save-weights", action="store_true")
    a = ap.parse_args()

    import torch, torch.nn as nn
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    roots = a.roots.split(",")
    articles = find("baly_articles.csv", roots)
    splits = find("splits", roots)
    stories = find("stories_10k_eval.jsonl", roots) or find("stories_10k.jsonl", roots)
    print(f"articles={articles} splits={splits} stories={stories}", flush=True)

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = f"politics_{a.split}" if "POLITICS" in a.encoder else f"{a.encoder.split('/')[-1]}_{a.split}"

    df = pd.read_csv(articles).set_index("ID")
    parts = {p: df.loc[pd.read_csv(splits / a.split / f"{p}.tsv", sep="\t").ID].reset_index()
             for p in ("train", "valid", "test")}
    if a.limit:
        parts = {k: v.sample(min(a.limit, len(v)), random_state=a.seed).reset_index(drop=True) for k, v in parts.items()}
    print(f"[{name}] " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()), flush=True)

    tok = AutoTokenizer.from_pretrained(a.encoder)

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(a.encoder)
            h = self.enc.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 768), nn.Tanh(),
                                      nn.Dropout(0.1), nn.Linear(768, 3))

        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    model = Clf().to(dev)
    par = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    def batches(texts, labels=None, shuffle=False, bs=None):
        bs = bs or a.batch_size
        idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
        for s in range(0, len(idx), bs):
            j = idx[s:s + bs]
            e = tok([texts[i] for i in j], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
            yield e["input_ids"].to(dev), e["attention_mask"].to(dev), (
                torch.tensor([labels[i] for i in j], device=dev) if labels is not None else None)

    @torch.no_grad()
    def predict(texts):
        model.eval(); outp = []
        order = np.argsort([len(t) for t in texts])                  # length-sorted for speed
        sorted_texts = [texts[i] for i in order]
        for ids, mask, _ in batches(sorted_texts, bs=64):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                outp.append(torch.softmax(par(ids, mask).float(), 1).cpu())
        p = torch.cat(outp).numpy()
        res = np.empty_like(p); res[order] = p
        return res

    tr_x = parts["train"].content.astype(str).tolist(); tr_y = parts["train"].bias.tolist()
    va_x = parts["valid"].content.astype(str).tolist(); va_y = parts["valid"].bias.values
    steps = a.epochs * ((len(tr_x) + a.batch_size - 1) // a.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.001, betas=(0.9, 0.999))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    ce = nn.CrossEntropyLoss()

    best, best_state, bad, log = -1.0, None, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = n = 0
        for step, (ids, mask, y) in enumerate(batches(tr_x, tr_y, shuffle=True)):
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = ce(par(ids, mask), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if step % 400 == 0:
                print(f"  [{name}] ep{ep} step {step} loss {tot / n:.4f}", flush=True)
        v = metrics(va_y, predict(va_x))
        log.append({"epoch": ep, "train_loss": tot / n, "minutes": (time.time() - t0) / 60,
                    **{f"val_{k}": x for k, x in v.items()}})
        print(f"[{name}] epoch {ep}: {json.dumps(log[-1])}", flush=True)
        if v["macro_f1"] > best:
            best, bad = v["macro_f1"], 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[{name}] early stop after epoch {ep}", flush=True); break
    model.load_state_dict(best_state)
    pd.DataFrame(log).to_csv(out / f"{name}_epochs.csv", index=False)
    if a.save_weights:
        torch.save(best_state, out / f"{name}.pt")

    # corpus test set
    te = parts["test"]
    pt = predict(te.content.astype(str).tolist())
    res = {"model": name, "encoder": a.encoder, "split": a.split, "best_val_macro_f1": best,
           "epochs_run": len(log), **{f"test_{k}": x for k, x in metrics(te.bias.values, pt).items()}}

    # team data: labelled source articles + neutral summaries, body only
    rows = []
    for l in open(stories):
        s = json.loads(l)
        for slot in ("left", "center", "right"):
            rows.append({"story_id": s["story_id"], "split": s.get("split"), "slot": slot, "label": SLOT2ID[slot],
                         "publisher": s[slot]["publisher"], "text": s[slot]["body"]})
        rows.append({"story_id": s["story_id"], "split": s.get("split"), "slot": "neutral", "label": -1,
                     "publisher": None, "text": s["neutral"]})
    if a.limit:
        rows = rows[: 4 * a.limit]
    S = pd.DataFrame(rows)
    ps = predict(S.text.tolist())
    S[["p_left", "p_centre", "p_right"]] = ps
    S["pred_id"] = ps.argmax(1)
    S.drop(columns="text").to_csv(out / f"{name}_stories_scores.csv", index=False)
    lab = S[S.slot != "neutral"]
    res.update({f"stories_{k}": x for k, x in metrics(lab.label.values, lab[["p_left", "p_centre", "p_right"]].values).items()})
    neu = S[S.slot == "neutral"]
    res["neutral_pred_share_LCR"] = (np.bincount(neu.pred_id, minlength=3) / max(1, len(neu))).round(4).tolist()
    res["neutral_mean_p_LCR"] = neu[["p_left", "p_centre", "p_right"]].mean().round(4).tolist()
    # LED / neutraliser outputs, if mounted (gen_<model>.jsonl rows: {id, output})
    gens = []
    for root in roots:
        for d, _, files in os.walk(root):
            gens += [Path(d) / f for f in files if f.startswith("gen_led") and f.endswith(".jsonl")]
    if gens:
        G = pd.DataFrame([{"kind": g.stem[4:], "id": r["id"], "text": " ".join(str(r["output"]).split())}
                          for g in sorted(set(gens)) for r in map(json.loads, open(g))]).drop_duplicates(["kind", "id"])
        pg = predict(G.text.tolist())
        G[["p_left", "p_centre", "p_right"]] = pg
        G.drop(columns="text").to_csv(out / f"{name}_led_scores.csv", index=False)
        res["led_outputs_scored"] = G.kind.value_counts().to_dict()
    print("RESULT " + json.dumps(res), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")



import sys
sys.argv = ["run.py", "--split", "media", "--out", "results", "--save-weights"]
main()
