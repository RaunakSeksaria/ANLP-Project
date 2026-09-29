"""Metrics for Jev runs: accuracy, macro-F1, per-class recall, confusion matrix."""
import json, sys
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def load(path):
    return [json.loads(l) for l in open(path)]

def metrics(y, P):
    y, pred = np.asarray(y), np.asarray(P).argmax(1)
    rec = [(pred[y == c] == c).mean() for c in range(3)]
    return {"acc": 100 * (pred == y).mean(), "macro_f1": 100 * f1_score(y, pred, average="macro"),
            "rec_L": rec[0], "rec_C": rec[1], "rec_R": rec[2]}

def fit_prior(y, P, grid=np.linspace(-3, 3, 61)):
    """Per-class log-offsets (centre fixed at 0) that maximise macro-F1. Corrects a model that
    systematically over- or under-predicts a class; learned on dev only."""
    y, L = np.asarray(y), np.log(np.asarray(P) + 1e-9)
    best = (-1, (0, 0))
    for a in grid:
        for b in grid:
            f = f1_score(y, (L + [a, 0, b]).argmax(1), average="macro")
            if f > best[0]: best = (f, (a, b))
    return np.array([best[1][0], 0, best[1][1]])

def apply_prior(P, off):
    z = np.log(np.asarray(P) + 1e-9) + off
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)
