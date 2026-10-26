"""Score Raunak's Nemotron single-article rewrite sets with every evaluation instrument on this branch.

Datasets (--dataset):
  long10k  data/long_10k/        full1024 release, HF RaunakSeksaria/anlp-project-datasets data/long/
                                 9,961 events x {left, right}, <= 1,024 LED tokens
  short    data/short_dedup/     left/right_neutral_dedup.jsonl (origin/data), <= ~275 words,
                                 10,000 per side; the LED neutralisers' training targets

Texts: each original article body and its Nemotron rewrite. Body only, like-for-like with the
rewrites and with the stories_10k evaluations. Outputs go to evaluation/<dataset>/.

    python evaluation/long10k/score.py build --dataset D
    python evaluation/long10k/score.py time  --dataset D --model X   # time 64 texts
    python evaluation/long10k/score.py run   --dataset D --model X   # resumable, appends scores/X.jsonl
    python evaluation/long10k/score.py vad   --dataset D

Models: chunkybert, no_classifier (ChunkyBERT family, 5,000-token chunked), baly_random, baly_media,
baly_aa (3-way ideology, 512 tokens), baly_outlet (73-way outlet classifier, 512 tokens).
"""
import argparse, gzip, json, re, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = TEXTS = SCORES = None


def use(dataset):
    global HERE, TEXTS, SCORES
    HERE = ROOT / "evaluation" / dataset
    HERE.mkdir(parents=True, exist_ok=True)
    TEXTS, SCORES = HERE / "texts.jsonl", HERE / "scores"


def records(dataset):
    """Yield (side, record) for every record of a rewrite set."""
    if dataset == "long10k":
        for side in ("left", "right"):
            for part in ("train", "validation", "test"):
                for line in gzip.open(ROOT / "data/long_10k" / side / f"{part}.jsonl.gz", "rt"):
                    yield side, json.loads(line)
    else:
        for side in ("left", "right"):
            for line in open(ROOT / "data/short_dedup" / f"{side}_neutral_dedup.jsonl"):
                yield side, json.loads(line)


def build(dataset):
    norm = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())
    baly_titles = set()
    for f in (ROOT / "external/Article-Bias-Prediction/data/jsons").glob("*.json"):
        t = norm(json.load(open(f))["title"])
        if len(t) > 20:
            baly_titles.add(t)
    n = 0
    with open(TEXTS, "w") as out:
        for side, r in records(dataset):
            o = r[f"{side}_original"]
            common = {"story_id": r["story_id"], "split": r["split"], "side": side,
                      "label": 0 if side == "left" else 2, "publisher": o["publisher"],
                      "in_baly_corpus": norm(o["title"]) in baly_titles}
            for kind, text in (("original", o["body"]), ("rewrite", r[f"{side}_neutralised"])):
                out.write(json.dumps({"id": f"{r['story_id']}:{side}:{kind}", "kind": kind,
                                      **common, "text": " ".join(str(text).split())}) + "\n")
                n += 1
    print(f"wrote {n} texts -> {TEXTS}")


def load_texts():
    return [json.loads(l) for l in open(TEXTS)]


# ------------------------------------------------------------------ models
def load_model(name, device):
    import torch
    if name in ("chunkybert", "no_classifier"):
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        if name == "chunkybert":
            sys.path.insert(0, str(ROOT / "chunkybert"))
            from model import ChunkedBERTClassifier
            m = ChunkedBERTClassifier(num_classes=3)
            w = ROOT / "models/chunkybert_5ep.pt"
        else:
            sys.path.insert(0, str(ROOT / "replication"))
            from variants import VariantClassifier, TABLE10
            m = VariantClassifier(num_classes=3, **TABLE10["No Classifier"])
            w = ROOT / "models/no_classifier_1ep.pt"
        m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True))
        return m.eval().to(device), tok, 5000, True, None

    from transformers import BertTokenizerFast
    sys.path.insert(0, str(ROOT / "baly"))
    from baly import build as baly_build, medium_split
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    if name == "baly_outlet":
        _, vocab, _ = medium_split(ROOT / "data/baly_articles.csv")      # same 73 domains, same order
        m = baly_build(len(vocab))
        m.load_state_dict(torch.load(ROOT / "models/baly/bert_medium.pt", map_location="cpu", weights_only=True))
        return m.eval().to(device), tok, 512, False, vocab
    w = {"baly_random": "bert_random", "baly_media": "bert_media", "baly_aa": "bert_aa_media"}[name]
    m = baly_build(3)
    m.load_state_dict(torch.load(ROOT / f"models/baly/{w}.pt", map_location="cpu", weights_only=True),
                      strict=False)                                      # AA also holds its outlet head
    return m.eval().to(device), tok, 512, False, None


def score(name, limit=None, batch=16):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SCORES.mkdir(exist_ok=True)
    out_path = SCORES / f"{name}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() and not limit else set()
    items = [t for t in load_texts() if t["id"] not in done]
    items.sort(key=lambda t: (t["split"] != "test", len(t["text"])))   # test split first; length-sorted
    if limit:
        items = items[len(items) // 2: len(items) // 2 + limit]           # mid-length sample for timing
    if not items:
        print(f"{name}: nothing to do"); return
    model, tok, max_len, needs_tok, vocab = load_model(name, device)
    if needs_tok:
        batch = 8                                                         # up to 10 chunks per text
    t0 = time.time()
    sink = open(out_path, "a") if not limit else None
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = items[i:i + batch]
            enc = tok([t["text"] for t in b], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(ids, mask, tok) if needs_tok else model(ids, mask)
            p = torch.softmax(logits.float(), 1).cpu().numpy()
            if sink:
                for t, row in zip(b, p):
                    rec = {"id": t["id"]}
                    if vocab:
                        top = row.argsort()[::-1][:5]
                        rec["top5"] = [[vocab[j], round(float(row[j]), 5)] for j in top]
                    else:
                        rec["p"] = [round(float(x), 6) for x in row]
                    sink.write(json.dumps(rec) + "\n")
                sink.flush()
            if (i // batch) % 250 == 0:
                print(f"  {name} {i + len(b)}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    print(f"{name}: {len(items)} texts in {dt:.0f}s ({dt / len(items) * 1000:.0f} ms/text)", flush=True)


def vad():
    """Same lexicon and averaging as evaluation/vad/vad_scores.py (Warriner et al. 2013)."""
    w = pd.read_csv(ROOT / "evaluation/vad/warriner_vad.csv", usecols=["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"])
    lex = {str(r.Word).lower(): (r[1], r[2], r[3]) for r in w.itertuples(index=False)}
    tok = re.compile(r"[a-z']+")
    rows = []
    for t in load_texts():
        hits = [lex[x] for x in tok.findall(t["text"].lower()) if x in lex]
        if hits:
            n = len(hits)
            rows.append({"id": t["id"], "valence": sum(h[0] for h in hits) / n,
                         "arousal": sum(h[1] for h in hits) / n, "dominance": sum(h[2] for h in hits) / n,
                         "n_words": n})
    SCORES.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(SCORES / "vad.csv", index=False)
    print(f"vad: {len(rows)} texts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "time", "run", "vad"])
    ap.add_argument("--dataset", choices=["long10k", "short"], default="long10k")
    ap.add_argument("--model", choices=["chunkybert", "no_classifier", "baly_random", "baly_media",
                                        "baly_aa", "baly_outlet"])
    a = ap.parse_args()
    use(a.dataset)
    if a.mode == "build":
        build(a.dataset)
    elif a.mode == "vad":
        vad()
    else:
        score(a.model, limit=64 if a.mode == "time" else None)
