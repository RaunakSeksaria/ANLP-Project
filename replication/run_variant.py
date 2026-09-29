"""Train one variant for one epoch under the paper's protocol (Table 4) and log the result.

The paper's Tables 8 and 10 are both one-epoch results, so this trains a single epoch and
reports validation accuracy and AUC. Results append to a CSV so a long sweep survives a crash.
"""
import argparse, csv, json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VariantClassifier, TABLE8, TABLE10

ALL = {**{k: dict(v) for k, v in TABLE8.items()}, **{k: dict(v) for k, v in TABLE10.items()}}


class Docs(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels, self.tok, self.max_length = texts, labels, tokenizer, max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        e = self.tok(self.texts[i], return_tensors="pt", max_length=self.max_length,
                     truncation=True, padding="max_length")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(ALL))
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--results-csv", default="results.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--tcn-levels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(a.data_csv).reset_index(drop=True)
    if a.limit:
        df = df.groupby("bias", group_keys=False).head(max(1, a.limit // 3)).reset_index(drop=True)
    texts, labels = df["content"].tolist(), df["bias"].tolist()
    tr_x, va_x, tr_y, va_y = train_test_split(texts, labels, test_size=0.2,
                                              random_state=42, stratify=labels)

    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh,
                                   num_workers=2, pin_memory=True, drop_last=sh)
    tr = dl(Docs(tr_x, tr_y, tok, a.max_length), True)
    va = dl(Docs(va_x, va_y, tok, a.max_length), False)

    cfg = dict(ALL[a.variant]); cfg["emb_dim"] = a.emb_dim; cfg["tcn_levels"] = a.tcn_levels
    model = VariantClassifier(num_classes=3, **cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{a.variant}] {json.dumps(model.cfg)}  trainable={n_par/1e6:.1f}M  "
          f"train={len(tr_x)} val={len(va_x)}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.learning_rate, weight_decay=0.01)
    steps = len(tr) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        loss_sum = 0.0
        for b in tqdm(tr, desc=f"{a.variant} ep{ep+1}", file=sys.stdout, mininterval=30):
            ids, msk, y = (b["input_ids"].to(device), b["attention_mask"].to(device),
                           b["label"].to(device))
            opt.zero_grad()
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = crit(model(ids, msk, tok), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item()
    train_s = time.time() - t0

    model.eval()
    P, Y, vloss = [], [], 0.0
    with torch.no_grad():
        for b in tqdm(va, desc=f"{a.variant} val", file=sys.stdout, mininterval=30):
            ids, msk, y = (b["input_ids"].to(device), b["attention_mask"].to(device),
                           b["label"].to(device))
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                lg = model(ids, msk, tok)
                vloss += crit(lg, y).item()
            P.append(torch.softmax(lg.float(), 1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    acc = float((P.argmax(1) == Y).mean())
    try:
        auc = float(roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr", average="weighted"))
    except Exception:
        auc = float("nan")

    row = {"variant": a.variant, "tag": a.tag, "val_acc": round(acc, 4), "val_auc": round(auc, 4),
           "train_loss": round(loss_sum / max(1, len(tr)), 4), "val_loss": round(vloss / max(1, len(va)), 4),
           "train_time_s": round(train_s, 1), "epochs": a.epochs, "lr": a.learning_rate,
           "emb_dim": a.emb_dim, "tcn_levels": a.tcn_levels, "n_train": len(tr_x), "trainable_params_M": round(n_par / 1e6, 2),
           **model.cfg}
    print(f"RESULT {json.dumps(row)}", flush=True)

    new = not os.path.exists(a.results_csv)
    with open(a.results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
