"""Fine-tune POLITICS as a Left / Centre / Right classifier on Article-Bias-Prediction (Baly et al. 2020),
random split, one seed per run. Version 2 of politics_baly.py, addressing the classifier review notes:

  * sliding windows for training and inference (512 tokens, overlap 64, mean of window logits per article)
  * up to 10 epochs with early-stopping patience 4 on validation macro-F1 (paper, Table A7)
  * validation AND test metrics every epoch, incl. per-class P/R/F1 and confusion matrices
  * first / best / last checkpoints kept (same PoliticsClassifier state_dict format as release 1)
  * best checkpoint's validation and test logits saved for calibration and seed ensembling

Unchanged from release 1: encoder launch/POLITICS, <s> -> 2-layer head (768, dropout 0.1), AdamW lr 2e-5,
weight decay 0.001, betas (0.9, 0.999), 6% warmup + linear decay, clip 1.0, fp16 autocast, 16 articles per
step, article body only, data as released (no extra masking).

    python train_v2.py --articles baly_articles.csv --splits splits/random --seed 42 --out runs/seed42
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from politics_classifier import (PoliticsClassifier, tokenizer, to_windows, pad, article_logits, softmax,
                                 metrics, WINDOW, OVERLAP, MAX_WINDOWS)

T0 = time.time()
log = lambda m: print(f"[{(time.time() - T0) / 60:6.1f} min] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", required=True)
    ap.add_argument("--splits", required=True, help="dir with train.tsv / valid.tsv / test.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--articles-per-step", type=int, default=16)
    ap.add_argument("--micro-windows", type=int, default=16, help="max windows per forward pass (memory)")
    ap.add_argument("--max-windows", type=int, default=MAX_WINDOWS)
    ap.add_argument("--limit", type=int, default=None, help="subsample each split (smoke tests)")
    ap.add_argument("--half-checkpoints", action="store_true", help="store checkpoints in fp16 (half the size)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log(f"device {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'} | seed {a.seed}")

    df = pd.read_csv(a.articles).set_index("ID")
    parts = {p: df.loc[pd.read_csv(Path(a.splits) / f"{p}.tsv", sep="\t").ID].reset_index()
             for p in ("train", "valid", "test")}
    if a.limit:
        parts = {k: v.sample(min(a.limit, len(v)), random_state=0).reset_index(drop=True) for k, v in parts.items()}
    tok = tokenizer()
    W, Y, ID = {}, {}, {}
    for p, d in parts.items():
        W[p] = to_windows(tok, d.content.astype(str).tolist(), WINDOW, OVERLAP, a.max_windows)
        Y[p] = d.bias.values.astype(int); ID[p] = d.ID.tolist()
        nw = np.array([len(w) for w in W[p]])
        log(f"{p}: {len(nw)} articles, {nw.sum()} windows (mean {nw.mean():.2f}, capped {np.mean(nw == a.max_windows):.1%})")

    model = PoliticsClassifier().to(dev)
    n_art = len(W["train"])
    steps_per_epoch = (n_art + a.articles_per_step - 1) // a.articles_per_step
    total = a.epochs * steps_per_epoch
    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.001, betas=(0.9, 0.999))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    ce = nn.CrossEntropyLoss(reduction="sum")

    def save(name):
        sd = {k: (v.half() if a.half_checkpoints and v.is_floating_point() else v).detach().cpu().clone()
              for k, v in model.state_dict().items()}
        torch.save(sd, out / f"seed{a.seed}_{name}.pt")

    def evaluate():
        r = {}
        for p in ("valid", "test"):
            lg = article_logits(model, W[p], tok.pad_token_id, dev)
            r[p] = (lg, metrics(Y[p], softmax(lg)))
        return r

    best, bad, hist = -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); loss_sum = 0.0
        order = np.random.permutation(n_art)
        for s in range(steps_per_epoch):
            arts = order[s * a.articles_per_step:(s + 1) * a.articles_per_step]
            opt.zero_grad(set_to_none=True)
            # micro-batches of whole articles, so each article's window logits are averaged in one graph
            micro, cur = [], []
            for i in arts:
                if cur and sum(len(W["train"][j]) for j in cur) + len(W["train"][i]) > a.micro_windows:
                    micro.append(cur); cur = []
                cur.append(i)
            micro.append(cur)
            for mb in micro:
                wins = [w for i in mb for w in W["train"][i]]
                owner = torch.tensor([k for k, i in enumerate(mb) for _ in W["train"][i]], device=dev)
                ids, mask = pad(wins, tok.pad_token_id, dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    wl = model(ids, mask).float()
                art = torch.zeros(len(mb), 3, device=dev).index_add_(0, owner, wl)
                art = art / torch.bincount(owner, minlength=len(mb)).unsqueeze(1)
                loss = ce(art, torch.tensor(Y["train"][mb], device=dev)) / len(arts)
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if s % 200 == 0:
                log(f"ep{ep} step {s}/{steps_per_epoch} loss {loss_sum / (s + 1):.4f}")
        train_min = (time.time() - t0) / 60
        r = evaluate()
        rec = {"epoch": ep, "train_loss": round(loss_sum / steps_per_epoch, 4), "train_minutes": round(train_min, 1),
               "valid": r["valid"][1], "test": r["test"][1]}
        hist.append(rec)
        with open(out / f"seed{a.seed}_epochs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"epoch {ep}: val F1 {r['valid'][1]['macro_f1']:.2f} acc {r['valid'][1]['acc']:.2f} | "
            f"test F1 {r['test'][1]['macro_f1']:.2f} acc {r['test'][1]['acc']:.2f} | train {train_min:.1f} min")
        if ep == 1:
            save("first")
        if r["valid"][1]["macro_f1"] > best:
            best, bad = r["valid"][1]["macro_f1"], 0
            save("best")
            np.savez(out / f"seed{a.seed}_best_logits.npz", epoch=ep,
                     valid=r["valid"][0], valid_y=Y["valid"], valid_id=np.array(ID["valid"]),
                     test=r["test"][0], test_y=Y["test"], test_id=np.array(ID["test"]))
        else:
            bad += 1
        if bad >= a.patience or ep == a.epochs:
            save("last")
            log(f"stopping after epoch {ep} ({'early stop' if bad >= a.patience else 'max epochs'}); best val F1 {best:.2f}")
            break

    best_ep = max(hist, key=lambda h: h["valid"]["macro_f1"])["epoch"]
    json.dump({"seed": a.seed, "epochs_run": len(hist), "best_epoch": best_ep, "last_epoch": hist[-1]["epoch"],
               "window": WINDOW, "overlap": OVERLAP, "max_windows": a.max_windows, "minutes": round((time.time() - T0) / 60, 1),
               "best_valid": hist[best_ep - 1]["valid"], "best_test": hist[best_ep - 1]["test"]},
              open(out / f"seed{a.seed}_summary.json", "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
