# ChunkyBERT Table 10 ablation sweep (Kaggle). Paper protocol: 1 epoch per variant.

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

BERT_DIM = 768


class Chomp(nn.Module):
    """Trim right-hand padding so a dilated conv stays causal."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, :-self.n] if self.n > 0 else x


class TCN(nn.Module):
    """Dilated causal convolutions, as described for the TCN baseline (ref 44)."""
    def __init__(self, d, levels=4, k=3, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(levels):
            pad = (k - 1) * 2**i
            layers += [nn.Conv1d(d, d, k, padding=pad, dilation=2**i),
                       Chomp(pad), nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class VariantClassifier(nn.Module):
    def __init__(self, num_classes=3, backbone="bert", encoder="transformer",
                 pooling="attention", chunk_agg="mean", chunking=True,
                 freeze_backbone=False, simple_head=False,
                 hidden_size=256, nhead=8, dropout=0.3, max_chunk_length=512,
                 emb_dim=256, vocab_size=30522, bert_model_name="bert-base-uncased",
                 tcn_levels=4):
        super().__init__()
        self.cfg = dict(backbone=backbone, encoder=encoder, pooling=pooling,
                        chunk_agg=chunk_agg, chunking=chunking,
                        freeze_backbone=freeze_backbone, simple_head=simple_head)
        self.max_chunk_length = max_chunk_length
        self.chunking = chunking
        self.pooling = pooling
        self.chunk_agg = chunk_agg
        self.backbone_kind = backbone

        if backbone == "bert":
            cfg = BertConfig.from_pretrained(bert_model_name)
            cfg.output_hidden_states = False
            cfg.return_dict = True
            self.bert = BertModel.from_pretrained(bert_model_name, config=cfg)
            d = cfg.hidden_size
            if freeze_backbone:
                for p in self.bert.parameters():
                    p.requires_grad = False
        else:
            d = emb_dim
            self.embed = nn.Embedding(vocab_size, d, padding_idx=0)

        if encoder == "transformer":
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead,
                                               dim_feedforward=hidden_size * 4,
                                               dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        elif encoder == "gru":
            self.encoder = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        elif encoder == "tcn":
            self.encoder = TCN(d, levels=tcn_levels, dropout=dropout)
        elif encoder == "none":
            self.encoder = None
        else:
            raise ValueError(encoder)
        self.encoder_kind = encoder

        self.attention = nn.Sequential(nn.Linear(d, hidden_size), nn.GELU(),
                                       nn.Linear(hidden_size, 1, bias=False))
        self.classifier = (nn.Linear(d, num_classes) if simple_head
                           else nn.Sequential(nn.Dropout(dropout), nn.Linear(d, num_classes)))
        self.out_features = num_classes

    # --- chunking: from the authors' released model.py ---
    def chunk_text_batched(self, input_ids, attention_mask, tokenizer):
        batch_size, _ = input_ids.shape
        device = input_ids.device
        cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
        content_size = self.max_chunk_length - 2
        ids, masks, owner = [], [], []

        for i in range(batch_size):
            n = attention_mask[i].sum().item()
            seq = input_ids[i, :n]
            content = seq[1:-1] if (n >= 2 and seq[0] == cls_id and seq[-1] == sep_id) else seq
            for j in range((len(content) + content_size - 1) // content_size):
                piece = content[j * content_size:(j + 1) * content_size]
                chunk = torch.cat([torch.tensor([cls_id], device=device), piece,
                                   torch.tensor([sep_id], device=device)])
                pad = self.max_chunk_length - len(chunk)
                if pad > 0:
                    chunk = torch.cat([chunk, torch.full((pad,), pad_id, device=device)])
                masks.append(torch.cat([torch.ones(len(chunk) - pad, device=device),
                                        torch.zeros(pad, device=device)]))
                ids.append(chunk)
                owner.append(i)

        if not ids:
            return None, None, None
        return torch.stack(ids), torch.stack(masks), torch.tensor(owner, device=device)

    def encode(self, ids, mask):
        if self.backbone_kind == "bert":
            h = self.bert(ids, attention_mask=mask).last_hidden_state
        else:
            h = self.embed(ids)
        if self.encoder_kind == "gru":
            h, _ = self.encoder(h)
        elif self.encoder is not None:
            h = self.encoder(h)
        return h

    def pool(self, h, mask):
        m = mask.unsqueeze(-1).to(h.dtype)
        if self.pooling == "attention":
            scores = self.attention(h)
            scores = scores.masked_fill(m == 0, torch.finfo(scores.dtype).min)
            return (torch.softmax(scores, dim=1).to(h.dtype) * h).sum(1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)          # masked mean

    def forward(self, input_ids, attention_mask, tokenizer):
        B = input_ids.size(0)
        device = input_ids.device

        if not self.chunking:
            # BERT cannot exceed its 512-token limit; GRU/TCN read the whole document.
            if self.backbone_kind == "bert":
                ids, mask = input_ids[:, :self.max_chunk_length], attention_mask[:, :self.max_chunk_length]
            else:
                ids, mask = input_ids, attention_mask
            return self.classifier(self.pool(self.encode(ids, mask), mask))

        ids, mask, owner = self.chunk_text_batched(input_ids, attention_mask, tokenizer)
        if ids is None:
            return torch.zeros(B, self.out_features, device=device)

        pooled = self.pool(self.encode(ids, mask), mask)

        docs = []
        for i in range(B):
            sel = owner == i
            if sel.any():
                v = pooled[sel]
                docs.append(v.max(0).values if self.chunk_agg == "max" else v.mean(0))
            else:
                docs.append(torch.zeros(pooled.size(1), device=device))
        return self.classifier(torch.stack(docs))


# Every model in the paper, as configuration.
TABLE8 = {
    "GRU no chunk, no attn":  dict(backbone="embedding", encoder="gru", pooling="mean",      chunking=False),
    "GRU no chunk, attn":     dict(backbone="embedding", encoder="gru", pooling="attention", chunking=False),
    "GRU with chunk, no attn":dict(backbone="embedding", encoder="gru", pooling="mean",      chunking=True),
    "GRU with chunk, attn":   dict(backbone="embedding", encoder="gru", pooling="attention", chunking=True),
    "TCN no chunk, no attn":  dict(backbone="embedding", encoder="tcn", pooling="mean",      chunking=False),
    "TCN no chunk, attn":     dict(backbone="embedding", encoder="tcn", pooling="attention", chunking=False),
    "TCN with chunk, no attn":dict(backbone="embedding", encoder="tcn", pooling="mean",      chunking=True),
    "TCN with chunk, attn":   dict(backbone="embedding", encoder="tcn", pooling="attention", chunking=True),
}

TABLE10 = {
    "Full Model":      dict(),
    "No Attention":    dict(pooling="mean"),
    "No Classifier":   dict(simple_head=True),
    "Max Pool":        dict(chunk_agg="max"),
    "No Chunking":     dict(chunking=False),
    "Freeze BERT":     dict(freeze_backbone=True),
    "No Transformers": dict(encoder="none"),
}

# ---------------------------------------------------------------- driver
import argparse, csv, json, os, sys, time, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CSV = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == "out.csv":
            CSV = os.path.join(root, f)
print("data:", CSV, flush=True)

RESULTS = "best_variant_results.csv"
BATCH, LR, MAX_LEN, SEED = 16, 2e-5, 5000, 42


class Docs(Dataset):
    def __init__(self, texts, labels, tok, max_length):
        self.texts, self.labels, self.tok, self.max_length = texts, labels, tok, max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        e = self.tok(self.texts[i], return_tensors="pt", max_length=self.max_length,
                     truncation=True, padding="max_length")
        return {"input_ids": e["input_ids"].squeeze(0),
                "attention_mask": e["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[i], dtype=torch.long)}


def run_one(name, cfg, tr, va, tok, device, n_train, n_val):
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); np.random.seed(SEED)
    model = VariantClassifier(num_classes=3, **cfg)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    inner = model.module if isinstance(model, nn.DataParallel) else model
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name} {json.dumps(inner.cfg)} trainable={n_par/1e6:.1f}M ===", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = len(tr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time(); model.train(); loss_sum = 0.0
    for b in tqdm(tr, desc=name, file=sys.stdout, mininterval=60):
        ids, msk, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
        opt.zero_grad()
        with torch.autocast("cuda"):
            loss = crit(model(ids, msk, tok), y)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        loss_sum += loss.item()
    train_s = time.time() - t0

    model.eval(); P, Y, vl = [], [], 0.0
    with torch.no_grad():
        for b in tqdm(va, desc=name + " val", file=sys.stdout, mininterval=60):
            ids, msk, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["label"].to(device)
            with torch.autocast("cuda"):
                lg = model(ids, msk, tok); vl += crit(lg, y).item()
            P.append(torch.softmax(lg.float(), 1).cpu()); Y.append(y.cpu())
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    acc = float((P.argmax(1) == Y).mean())
    try:
        auc = float(roc_auc_score(Y, P / P.sum(1, keepdims=True), multi_class="ovr", average="weighted"))
    except Exception:
        auc = float("nan")

    row = {"variant": name, "val_acc": round(acc, 4), "val_auc": round(auc, 4),
           "train_loss": round(loss_sum / len(tr), 4), "val_loss": round(vl / len(va), 4),
           "train_time_s": round(train_s, 1), "train_time_min": round(train_s / 60, 1),
           "trainable_params_M": round(n_par / 1e6, 2), "n_train": n_train, **inner.cfg}
    print("RESULT " + json.dumps(row), flush=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)
    sd = (model.module if isinstance(model, nn.DataParallel) else model).state_dict()
    torch.save(sd, "best_variant_no_classifier.pt")
    print("saved weights", flush=True)
    del model; torch.cuda.empty_cache()


def main():
    device = torch.device("cuda")
    print("GPUs:", torch.cuda.device_count(), flush=True)
    df = pd.read_csv(CSV).reset_index(drop=True)
    texts, labels = df["content"].tolist(), df["bias"].tolist()
    tr_x, va_x, tr_y, va_y = train_test_split(texts, labels, test_size=0.2,
                                              random_state=42, stratify=labels)
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    tr = DataLoader(Docs(tr_x, tr_y, tok, MAX_LEN), batch_size=BATCH, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    va = DataLoader(Docs(va_x, va_y, tok, MAX_LEN), batch_size=BATCH, shuffle=False,
                    num_workers=2, pin_memory=True)

    # cheapest first, so partial results are still useful if the kernel is cut short
    order = ["No Classifier"]   # top-ranked non-ChunkyBERT variant (82.36%)
    for name in order:
        try:
            run_one(name, dict(TABLE10[name]), tr, va, tok, device, len(tr_x), len(va_x))
        except Exception as e:
            print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True)

    print(pd.read_csv(RESULTS).to_string(index=False), flush=True)


main()
