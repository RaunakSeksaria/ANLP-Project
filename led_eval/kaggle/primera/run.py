# Kaggle kernel: zero-shot PRIMERA (allenai/PRIMERA, allenai/PRIMERA-multinews) on the 1,000 stories_10k
# test stories, then every evaluation instrument on its summaries and on the Nemotron neutral summaries.
#   input  : left and right article ("title. body") joined by <doc-sep>, the same pair Nemotron summarised
#   decode : the checkpoints' own settings: beam 5, no_repeat_ngram 3; max 256 new tokens (PRIMERA's Multi-News length)
#   global attention on <s> and every <doc-sep>, as in the PRIMERA code
#   scores : ChunkyBERT, No-Classifier, Baly random/media/AA, Baly outlet (top-5), POLITICS media/random, VAD,
#            ROUGE-1/2/L against the Nemotron summary
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOTS = [r for r in os.environ.get("EVAL_ROOTS", "/kaggle/input").split(",") if r]
OUT = Path(os.environ.get("EVAL_OUT", "/kaggle/working/out")); OUT.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None
MODELS = {"primera": "allenai/PRIMERA", "primera_multinews": "allenai/PRIMERA-multinews"}


def find(*names):
    for root in ROOTS:
        for d, dirs, files in os.walk(root):
            for n in names:
                if n in files or n in dirs:
                    return Path(d) / n
    raise FileNotFoundError(names)


norm = lambda s: " ".join(str(s).split())


def stories():
    """1,000 test stories: id, two documents, and the Nemotron neutral summary."""
    art = {r["id"]: r for r in map(json.loads, open(find("test_inputs.jsonl")))}
    neu = {s["story_id"]: s["neutral"] for s in map(json.loads, open(find("stories_10k_eval.jsonl"))) if s["split"] == "test"}
    out = []
    for sid in sorted({k.split(":")[0] for k in art}):
        docs = [f"{norm(art[f'{sid}:{side}']['title'])}. {norm(art[f'{sid}:{side}']['body'])}" for side in ("left", "right")]
        out.append({"id": sid, "docs": docs, "neutral": norm(neu[sid])})
    return out[:LIMIT] if LIMIT else out


WORKER = r'''
import json, sys, time, torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
inp, out, shard, nshard, gpu, repo = sys.argv[1:7]
shard, nshard = int(shard), int(nshard)
rows = [json.loads(l) for l in open(inp)][shard::nshard]
tok = AutoTokenizer.from_pretrained(repo)
sep = tok.convert_tokens_to_ids("<doc-sep>")
assert sep != tok.unk_token_id, "<doc-sep> missing from tokenizer"
dev = f"cuda:{gpu}"
model = LEDForConditionalGeneration.from_pretrained(repo).to(dev).half().eval()
texts = [" <doc-sep> ".join(r["docs"]) for r in rows]
t0 = time.time()
with open(out, "w") as fh:
    for s in range(0, len(rows), 8):
        b = rows[s:s + 8]
        enc = tok(texts[s:s + 8], return_tensors="pt", padding=True, truncation=True, max_length=4096).to(dev)
        g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1; g[enc["input_ids"] == sep] = 1
        with torch.no_grad():
            ids = model.generate(**enc, global_attention_mask=g, num_beams=5, no_repeat_ngram_size=3,
                                 max_new_tokens=256, early_stopping=True)
        for r, o, n, ni in zip(b, tok.batch_decode(ids, skip_special_tokens=True), (ids != tok.pad_token_id).sum(1),
                               enc["attention_mask"].sum(1)):
            fh.write(json.dumps({"id": r["id"], "output": o.strip(), "input_tokens": int(ni),
                                 "output_tokens": int(n), "hit_cap": int(n) >= 256}) + "\n")
        fh.flush()
        print(f"  shard{shard} {min(s + 8, len(rows))}/{len(rows)} {(time.time() - t0) / 60:.1f} min", flush=True)
'''


def generate(rows):
    import torch
    n = max(1, torch.cuda.device_count())
    inp = OUT / "inputs.jsonl"
    with open(inp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    w = OUT / "gen_worker.py"; w.write_text(WORKER)
    for name, repo in MODELS.items():
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, str(w), str(inp), str(OUT / f"{name}.shard{i}.jsonl"), str(i), str(n), str(i), repo])
                 for i in range(n)]
        codes = [p.wait() for p in procs]
        print(f"{name}: exit codes {codes}, {(time.time() - t0) / 60:.1f} min", flush=True)
        if any(codes):
            raise SystemExit(f"generation failed for {name}")
        with open(OUT / f"gen_{name}.jsonl", "w") as f:
            for i in range(n):
                f.write(open(OUT / f"{name}.shard{i}.jsonl").read())


def texts(rows):
    out = [{"id": f"nemotron:{r['id']}", "kind": "nemotron", "text": r["neutral"]} for r in rows]
    for name in MODELS:
        for g in map(json.loads, open(OUT / f"gen_{name}.jsonl")):
            out.append({"id": f"{name}:{g['id']}", "kind": name, "text": norm(g["output"])})
    return out


def vad(rows):
    import pandas as pd
    w = pd.read_csv(find("warriner_vad.csv"), usecols=["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"])
    lex = {str(r.Word).lower(): (r[1], r[2], r[3]) for r in w.itertuples(index=False)}
    tokre = re.compile(r"[a-z']+")
    out = []
    for t in rows:
        h = [lex[x] for x in tokre.findall(t["text"].lower()) if x in lex]
        if h:
            out.append({"id": t["id"], "valence": sum(x[0] for x in h) / len(h), "arousal": sum(x[1] for x in h) / len(h),
                        "dominance": sum(x[2] for x in h) / len(h), "n_words": len(h)})
    pd.DataFrame(out).to_csv(OUT / "vad.csv", index=False)
    print(f"vad: {len(out)} texts", flush=True)


def rouge(rows, texts_):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rouge_score"], check=False)
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    ref = {r["id"]: r["neutral"] for r in rows}
    with open(OUT / "rouge_vs_nemotron.jsonl", "w") as f:
        for t in texts_:
            if t["kind"] == "nemotron":
                continue
            s = sc.score(ref[t["id"].split(":", 1)[1]], t["text"])
            f.write(json.dumps({"id": t["id"], **{k: round(v.fmeasure, 4) for k, v in s.items()}}) + "\n")
    print("rouge done", flush=True)


def score(rows):
    import torch, torch.nn as nn
    sys.path.insert(0, str(find("variants.py").parent))
    from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertTokenizerFast
    from model import ChunkedBERTClassifier
    from variants import VariantClassifier, TABLE10
    from baly import build
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = json.load(open(find("outlet_vocab.json")))

    class PoliticsClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained("launch/POLITICS")
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 768), nn.Tanh(),
                                      nn.Dropout(0.1), nn.Linear(768, 3))

        def forward(self, ids, mask):
            return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])

    bert = lambda: BertTokenizerFast.from_pretrained("bert-base-uncased")
    specs = [  # name, constructor, weight file names, tokenizer, max_len, chunked-forward, strict
        ("chunkybert", lambda: ChunkedBERTClassifier(num_classes=3), ("chunkybert_final_epoch.pt",),
         lambda: BertTokenizer.from_pretrained("bert-base-uncased"), 5000, True, True),
        ("no_classifier", lambda: VariantClassifier(num_classes=3, **TABLE10["No Classifier"]), ("best_variant_no_classifier.pt",),
         lambda: BertTokenizer.from_pretrained("bert-base-uncased"), 5000, True, True),
        ("baly_random", lambda: build(3), ("bert_random.pt",), bert, 512, False, False),
        ("baly_media", lambda: build(3), ("bert_media.pt",), bert, 512, False, False),
        ("baly_aa", lambda: build(3), ("bert_aa_media.pt",), bert, 512, False, False),
        ("baly_outlet", lambda: build(len(vocab)), ("bert_medium.pt",), bert, 512, False, True),
        ("politics_media", PoliticsClassifier, ("politics_media.pt",),
         lambda: AutoTokenizer.from_pretrained("launch/POLITICS"), 512, False, True),
        ("politics_random", PoliticsClassifier, ("politics_random.pt",),
         lambda: AutoTokenizer.from_pretrained("launch/POLITICS"), 512, False, True),
    ]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    for name, mk, wnames, mktok, max_len, chunked, strict in specs:
        t0 = time.time()
        m = mk(); m.load_state_dict(torch.load(find(*wnames), map_location="cpu", weights_only=True), strict=strict)
        m.eval().to(dev); tok = mktok()
        bs = 16 if chunked else 64
        with open(OUT / f"scores_{name}.jsonl", "w") as f, torch.no_grad():
            for s in range(0, len(order), bs):
                b = [rows[i] for i in order[s:s + bs]]
                e = tok([t["text"] for t in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
                ids, mask = e["input_ids"].to(dev), e["attention_mask"].to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    lg = m(ids, mask, tok) if chunked else m(ids, mask)
                p = torch.softmax(lg.float(), 1).cpu().numpy()
                for t, row in zip(b, p):
                    rec = {"id": t["id"]}
                    if name == "baly_outlet":
                        top = row.argsort()[::-1][:5]
                        rec["top5"] = [[vocab[j], round(float(row[j]), 5)] for j in top]
                    else:
                        rec["p"] = [round(float(x), 6) for x in row]
                    f.write(json.dumps(rec) + "\n")
        print(f"{name}: {len(rows)} texts in {time.time() - t0:.0f}s", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    rows = stories()
    print(f"{len(rows)} stories", flush=True)
    generate(rows)
    T = texts(rows)
    print(f"{len(T)} texts to score", flush=True)
    vad(T)
    rouge(rows, T)
    score(T)
    print("ALL DONE", flush=True)
