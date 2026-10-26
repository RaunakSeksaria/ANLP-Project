"""POLITICS Left / Centre / Right classifier: model class, sliding-window inference, metrics, calibration.

Shared by training (`train_v2.py`), evaluation (`aggregate_v2.py`) and anyone scoring text with the released
weights. The model class is unchanged from the first release, so old and new `.pt` files load the same way.

Sliding windows (POLITICS paper, Table A7: window 512, overlap 64): an article is tokenised in full, cut into
512-token windows (510 body tokens + <s> ... </s>) that overlap by 64 tokens, each window is classified, and
the article's logits are the mean of its windows' logits. Training uses the same scheme, so training and
inference match. At most `max_windows` windows per article are used (default 8, about 3,600 tokens).

    from politics_classifier import load, predict
    models, temps = load(["seed42_best.pt", "seed43_best.pt", "seed44_best.pt"], "calibration.json")
    probs = predict(models, texts, temperature=temps["ensemble"])     # (n, 3): P(Left), P(Centre), P(Right)
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ENCODER = "launch/POLITICS"
LABELS = ["Left", "Centre", "Right"]
WINDOW, OVERLAP, MAX_WINDOWS = 512, 64, 8


class PoliticsClassifier(nn.Module):
    def __init__(self, encoder=ENCODER):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(encoder)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 768), nn.Tanh(),
                                  nn.Dropout(0.1), nn.Linear(768, 3))

    def forward(self, input_ids, attention_mask):
        cls = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(cls)


# ---------------------------------------------------------------- windowing
def tokenizer(encoder=ENCODER):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder)
    tok.model_max_length = 10 ** 9            # full-length tokenisation is intended; silence the >512 warning
    return tok


def to_windows(tok, texts, window=WINDOW, overlap=OVERLAP, max_windows=MAX_WINDOWS, batch=512):
    """Return one list of window token-id arrays per text. Every text yields at least one window."""
    body, stride = window - 2, window - 2 - overlap
    cls, sep = tok.cls_token_id, tok.sep_token_id
    out = []
    for s in range(0, len(texts), batch):
        enc = tok([str(t) for t in texts[s:s + batch]], add_special_tokens=False)["input_ids"]
        for ids in enc:
            wins, start = [], 0
            while True:
                wins.append(np.asarray([cls] + ids[start:start + body] + [sep], dtype=np.int32))
                if start + body >= len(ids) or len(wins) == max_windows:
                    break
                start += stride
            out.append(wins)
    return out


def pad(wins, pad_id, device):
    L = max(len(w) for w in wins)
    ids = torch.full((len(wins), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(wins), L), dtype=torch.long)
    for i, w in enumerate(wins):
        ids[i, :len(w)] = torch.from_numpy(w.astype(np.int64)); mask[i, :len(w)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def article_logits(model, windows, pad_id, device, batch=64):
    """Mean window logits per article. `windows` is the output of to_windows."""
    model.eval()
    flat = [(a, w) for a, ws in enumerate(windows) for w in ws]
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][1]))      # length-sorted for less padding
    win_logits = np.zeros((len(flat), 3), dtype=np.float32)
    for s in range(0, len(order), batch):
        j = order[s:s + batch]
        ids, mask = pad([flat[i][1] for i in j], pad_id, device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            win_logits[j] = model(ids, mask).float().cpu().numpy()
    owner = np.array([a for a, _ in flat])
    sums = np.zeros((len(windows), 3), dtype=np.float64)
    np.add.at(sums, owner, win_logits)
    return (sums / np.bincount(owner, minlength=len(windows))[:, None]).astype(np.float32)


# ---------------------------------------------------------------- loading / prediction for users
def load(weights, calibration=None, device=None):
    """Load one or more state_dicts (paths). Returns (models, temperatures dict or {})."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for w in ([weights] if isinstance(weights, (str, Path)) else weights):
        m = PoliticsClassifier()
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))   # fp16 or fp32 both load
        models.append(m.float().to(device).eval())
    temps = json.load(open(calibration))["temperature"] if calibration else {}
    return models, temps


def predict(models, texts, temperature=1.0, window=WINDOW, overlap=OVERLAP, max_windows=MAX_WINDOWS):
    """P(Left, Centre, Right) per text. With several models, their temperature-scaled probabilities are averaged."""
    models = models if isinstance(models, list) else [models]
    tok = tokenizer()
    wins = to_windows(tok, texts, window, overlap, max_windows)
    device = next(models[0].parameters()).device
    P = [softmax(article_logits(m, wins, tok.pad_token_id, device), temperature) for m in models]
    return np.mean(P, axis=0)


# ---------------------------------------------------------------- metrics / calibration
def softmax(logits, t=1.0):
    z = np.asarray(logits, dtype=np.float64) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def metrics(y, probs):
    """Macro-F1 / accuracy / MAE (ordinal L<C<R), per-class P/R/F1, confusion matrix, NLL, ECE."""
    y = np.asarray(y); probs = np.asarray(probs); pred = probs.argmax(1)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    per = {}
    for c, name in enumerate(LABELS):
        tp = cm[c, c]; p = tp / max(1, cm[:, c].sum()); r = tp / max(1, cm[c].sum())
        per[name] = {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
                     "f1": round(100 * (2 * p * r / (p + r) if p + r else 0.0), 2), "support": int(cm[c].sum())}
    return {"macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 3),
            "acc": round(100 * float((pred == y).mean()), 3),
            "mae": round(float(np.abs(pred - y).mean()), 4),
            "nll": round(float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()), 4),
            "ece": round(ece(y, probs), 4),
            "per_class": per, "confusion": cm.tolist()}


def ece(y, probs, bins=15):
    conf = probs.max(1); correct = probs.argmax(1) == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(y, probs, bins=15):
    conf = probs.max(1); correct = probs.argmax(1) == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "confidence": float(conf[m].mean()) if m.any() else None,
                     "accuracy": float(correct[m].mean()) if m.any() else None})
    return rows


def fit_temperature(logits, y):
    """Scalar temperature minimising validation NLL (grid on log T, then a fine grid)."""
    y = np.asarray(y)
    nll = lambda t: -np.log(np.clip(softmax(logits, t)[np.arange(len(y)), y], 1e-12, 1)).mean()
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 400))
    t0 = grid[np.argmin([nll(t) for t in grid])]
    fine = np.linspace(t0 * 0.9, t0 * 1.1, 201)
    return float(fine[np.argmin([nll(t) for t in fine])])
