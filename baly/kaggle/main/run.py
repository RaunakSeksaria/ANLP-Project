# Baly et al. 2020 replication on Kaggle — generated from baly/baly.py, do not edit by hand.
"""Baly et al. 2020, "We Can Detect Your Bias" (arXiv 2010.05338) — BERT models.

No code was released for the models, so they are rebuilt from Section 4-5 of the paper.
Hyperparameters are the paper's reported best (Sec 5.1): bert-base, 512 tokens, lr 2e-5,
gradient clipping 1.0, early stopping on validation macro-F1, AA lambda 0.7, 35,017 triplets.

Modes
  prep      build baly_articles.csv (ID, topic, source_url, bias, content) from the released jsons
  majority  Table 3 majority-class baseline
  bert      Table 3 / Table 5 "None"      --split media|random
  medium    Table 4 predict the outlet    (own stratified 80/10/10 split, as the paper did)
  aa        Table 5 adversarial media adaptation (gradient reversal), media split
  tlp       Table 5 triplet-loss pre-training, then fine-tuning, media split
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Left", "Centre", "Right"]


# ----------------------------------------------------------------------------- data
def find(name, roots=("/kaggle/input", ".")):
    """Locate a file or directory by name (Kaggle mounts datasets under unpredictable paths)."""
    for root in roots:
        for d, dirs, files in os.walk(root):
            if name in files or name in dirs:
                return Path(d) / name
    return None


def prep(jsons, out):
    rows = []
    for f in sorted(Path(jsons).glob("*.json")):
        r = json.load(open(f, encoding="utf-8"))
        rows.append({k: r[k] for k in ("ID", "topic", "source_url", "bias", "content")})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {len(rows)} articles -> {out}")


def load(articles, splits, split):
    df = pd.read_csv(articles).set_index("ID")
    parts = {p: pd.read_csv(Path(splits) / split / f"{p}.tsv", sep="\t").ID for p in ("train", "valid", "test")}
    return {p: df.loc[ids].reset_index() for p, ids in parts.items()}


def medium_split(articles, seed=42, n_media=73):
    """Table 4: stratified random split so every outlet appears in train, valid and test.
    Outlet = domain (`source_url`), the unit Baly's media split is disjoint on. The released corpus
    has 467 domains with a long tail; the paper's task was 73-way, so keep the 73 largest."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(articles)
    counts = df.source_url.value_counts()
    df = df[df.source_url.isin(counts.index[:n_media])].reset_index(drop=True)
    tr, rest = train_test_split(df, test_size=0.2, random_state=seed, stratify=df.source_url)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest.source_url)
    return {"train": tr.reset_index(drop=True), "valid": va.reset_index(drop=True),
            "test": te.reset_index(drop=True)}, sorted(df.source_url.unique()), int(counts.iloc[n_media:].sum())


def sample_triplets(train, n, seed=42):
    """Anchor; positive = same ideology, different outlet; negative = different ideology, same
    outlet. All three share a topic, 'to ensure that the change in topic has minimal impact'."""
    rng = random.Random(seed)
    by_mt, by_bt = {}, {}
    for i, r in enumerate(train.itertuples()):
        by_mt.setdefault((r.source_url, r.topic), []).append(i)
        by_bt.setdefault((r.bias, r.topic), []).append(i)
    anchors = []
    for i, r in enumerate(train.itertuples()):
        pos = [j for j in by_bt[(r.bias, r.topic)] if train.source_url.iat[j] != r.source_url]
        neg = [j for j in by_mt[(r.source_url, r.topic)] if train.bias.iat[j] != r.bias]
        if pos and neg:
            anchors.append((i, pos, neg))
    if not anchors:
        raise ValueError("no anchor has both a same-topic positive and negative")
    trip = []
    for _ in range(n):
        i, pos, neg = rng.choice(anchors)
        trip.append((i, rng.choice(pos), rng.choice(neg)))
    return trip, len(anchors)


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(p).argmax(1)
    m = {"macro_f1": 100 * f1_score(y, pred, average="macro"), "acc": 100 * (y == pred).mean()}
    if np.asarray(p).shape[1] == 3:           # MAE is only meaningful on the ordinal L<C<R scale
        m["mae"] = float(np.abs(y - pred).mean())
    return m


# ----------------------------------------------------------------------------- model
def build(n_classes, n_media=0, lam=0.0):
    import torch, torch.nn as nn
    from transformers import BertModel

    class GradReverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    class BertClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.drop = nn.Dropout(0.1)
            self.head = nn.Linear(768, n_classes)
            self.media_head = nn.Linear(768, n_media) if n_media else None

        def encode(self, ids, mask):
            return self.bert(ids, attention_mask=mask).pooler_output

        def forward(self, ids, mask):
            h = self.encode(ids, mask)
            y = self.head(self.drop(h))
            if self.media_head is None:
                return y
            return y, self.media_head(GradReverse.apply(h, lam))

    return BertClf()


# ----------------------------------------------------------------------------- training
def run(args, parts, label_col, n_classes, name, media_vocab=None, init_state=None):
    import torch, torch.nn as nn
    from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

    n_media = len(media_vocab) if media_vocab else 0
    model = build(n_classes, n_media, args.lam)
    if init_state is not None:
        missing, _ = model.load_state_dict(init_state, strict=False)
        print(f"initialised from pre-trained encoder ({len(missing)} head params fresh)", flush=True)
    model.to(dev)
    par = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    def batches(df, shuffle):
        idx = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
        for s in range(0, len(idx), args.batch_size):
            b = df.iloc[idx[s:s + args.batch_size]]
            enc = tok(b.content.astype(str).tolist(), truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt")
            yield (enc["input_ids"].to(dev), enc["attention_mask"].to(dev),
                   torch.tensor(b[label_col].values, device=dev),
                   torch.tensor([media_vocab.get(m, 0) for m in b.source_url], device=dev) if n_media else None)

    @torch.no_grad()
    def predict(df):
        model.eval(); out = []
        for ids, mask, _, _ in batches(df, False):
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = par(ids, mask)
                o = o[0] if isinstance(o, tuple) else o
            out.append(torch.softmax(o.float(), 1).cpu())
        return torch.cat(out).numpy()

    train = parts["train"]
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    ce = nn.CrossEntropyLoss()

    best, best_state, bad, log = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; n = 0
        for step, (ids, mask, y, m) in enumerate(batches(train, True)):
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = par(ids, mask)
                loss = ce(o[0], y) + ce(o[1], m) if n_media else ce(o, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
            if step % 400 == 0:
                print(f"  [{name}] ep{ep} step {step} loss {tot/n:.4f}", flush=True)
        v = metrics(parts["valid"][label_col], predict(parts["valid"]))
        log.append({"epoch": ep, "train_loss": tot / n, "minutes": (time.time() - t0) / 60,
                    **{f"val_{k}": x for k, x in v.items()}})
        print(f"[{name}] epoch {ep}: {json.dumps(log[-1])}", flush=True)
        if v["macro_f1"] > best:
            best, bad = v["macro_f1"], 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}] early stop after epoch {ep}", flush=True)
                break

    model.load_state_dict(best_state)
    probs = predict(parts["test"])
    res = {"model": name, **{f"test_{k}": x for k, x in metrics(parts["test"][label_col], probs).items()},
           "best_val_macro_f1": best, "epochs_run": len(log), "n_train": len(train),
           "n_test": len(parts["test"]), "n_classes": n_classes}
    return res, log, probs, best_state


def pretrain_triplets(args, train):
    import torch, torch.nn as nn
    from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    trip, n_anchor = sample_triplets(train, args.n_triplets, args.seed)
    print(f"[tlp] {len(trip)} triplets from {n_anchor} eligible anchors", flush=True)

    model = build(3).to(dev)

    class Enc(nn.Module):
        def __init__(self, bert):
            super().__init__(); self.bert = bert
        def forward(self, ids, mask):
            return self.bert(ids, attention_mask=mask).pooler_output

    enc_fn = Enc(model.bert)
    enc_fn = nn.DataParallel(enc_fn) if torch.cuda.device_count() > 1 else enc_fn
    loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)
    B = args.triplet_batch
    steps = args.triplet_epochs * ((len(trip) + B - 1) // B)
    opt = torch.optim.AdamW(model.bert.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    text = train.content.astype(str).tolist()

    for ep in range(args.triplet_epochs):
        random.Random(args.seed + ep).shuffle(trip)
        model.train(); t0 = time.time(); tot = 0.0
        for s in range(0, len(trip), B):
            chunk = trip[s:s + B]
            flat = [text[i] for t in chunk for i in t]          # a, p, n, a, p, n, ...
            e = tok(flat, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            opt.zero_grad()
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                h = enc_fn(e["input_ids"].to(dev), e["attention_mask"].to(dev))
            h = h.float().view(len(chunk), 3, -1)
            loss = loss_fn(h[:, 0], h[:, 1], h[:, 2])
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.bert.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            if (s // B) % 400 == 0:
                print(f"  [tlp-pretrain] ep{ep+1} {s}/{len(trip)} loss {tot/((s//B)+1):.4f}", flush=True)
        print(f"[tlp-pretrain] epoch {ep+1} loss {tot/((len(trip)+B-1)//B):.4f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith("bert.")}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "majority", "bert", "medium", "aa", "tlp"])
    ap.add_argument("--split", default="media", choices=["media", "random"])
    ap.add_argument("--articles", default=None, help="baly_articles.csv (auto-located if omitted)")
    ap.add_argument("--splits", default=None, help="splits dir with media/ and random/ (auto-located)")
    ap.add_argument("--jsons", default="external/Article-Bias-Prediction/data/jsons")
    ap.add_argument("--out", default="baly/results")
    ap.add_argument("--limit", type=int, default=None, help="subsample each part (smoke tests)")
    ap.add_argument("--save-weights", action="store_true")
    # paper-reported (Sec 5.1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--n-triplets", type=int, default=35017)
    # not reported by the paper — assumptions, logged with every result
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--triplet-epochs", type=int, default=1)
    ap.add_argument("--triplet-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.mode == "prep":
        return prep(a.jsons, a.articles or "data/baly_articles.csv")

    articles = a.articles or find("baly_articles.csv", ("/kaggle/input", "data"))
    splits = a.splits or find("splits", ("/kaggle/input",)) or "external/Article-Bias-Prediction/data/splits"
    print(f"articles={articles} splits={splits}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    adv_vocab, extra = None, {}          # set only for AA: outlets the adversary must not recover
    if a.mode == "medium":
        parts, vocab, dropped = medium_split(articles, a.seed)
        ids = {m: i for i, m in enumerate(vocab)}
        for p in parts.values():
            p["medium_id"] = p.source_url.map(ids)
        label_col, n_classes, name = "medium_id", len(vocab), "bert_medium"
        extra = {"n_media": len(vocab), "articles_outside_top_media": dropped}
    else:
        split = "media" if a.mode in ("aa", "tlp") else a.split
        parts = load(articles, splits, split)
        label_col, n_classes = "bias", 3
        name = {"majority": f"majority_{split}", "bert": f"bert_{split}",
                "aa": "bert_aa_media", "tlp": "bert_tlp_media"}[a.mode]
        if a.mode == "aa":
            adv_vocab = {m: i for i, m in enumerate(sorted(parts["train"].source_url.unique()))}
            extra = {"n_train_media": len(adv_vocab), "lambda": a.lam}

    full_train = parts["train"]           # triplets are always drawn from the full training set
    if a.limit:
        parts = {k: v.sample(min(a.limit, len(v)), random_state=a.seed).reset_index(drop=True)
                 for k, v in parts.items()}
    print(f"[{name}] sizes: " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()), flush=True)

    if a.mode == "majority":
        maj = parts["train"][label_col].mode()[0]
        probs = np.eye(n_classes)[[maj] * len(parts["test"])]
        res, log, state = {"model": name, **{f"test_{k}": x for k, x in
                           metrics(parts["test"][label_col], probs).items()},
                           "majority_class": CLASSES[maj], "n_test": len(parts["test"])}, [], None
    else:
        init = None
        if a.mode == "tlp":
            init = pretrain_triplets(a, full_train)
            extra = {"n_triplets": a.n_triplets, "margin": a.margin, "triplet_epochs": a.triplet_epochs}
        res, log, probs, state = run(a, parts, label_col, n_classes, name, adv_vocab, init)

    res.update(extra)
    res["assumptions"] = {k: getattr(a, k) for k in ("batch_size", "epochs", "patience", "seed")}
    print("RESULT " + json.dumps(res), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    if log:
        pd.DataFrame(log).to_csv(out / f"{name}_epochs.csv", index=False)
    if n_classes == 3:
        t = parts["test"][["ID", "source_url", "topic", label_col]].copy()
        t[["p_left", "p_centre", "p_right"]] = probs
        t.to_csv(out / f"{name}_test_probs.csv", index=False)
    if a.save_weights and state is not None:
        import torch
        torch.save(state, out / f"{name}.pt")



RUNS = [["bert", "--split", "media"], ["aa"], ["bert", "--split", "random"], ["medium"]]

# ------------------------------------------------------------------ Kaggle driver
import sys, traceback
for r in RUNS:
    sys.argv = ["baly.py", *r, "--out", "results", "--save-weights"]
    print("\n" + "=" * 80 + f"\nRUN {' '.join(r)}\n" + "=" * 80, flush=True)
    try:
        main()
    except Exception:
        traceback.print_exc()
