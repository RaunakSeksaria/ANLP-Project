# Kaggle kernel: evaluate the LED neutraliser `side-neutral-full/first_token_pooling`
# (RaunakSeksaria/anlp-neutral-led, uploaded 2026-10-15).
#   1. generate rewrites of the 2,000 held-out stories_10k test articles (both GPUs, sharded)
#   2. VAD on every text
#   3. ChunkyBERT, No-Classifier, Baly random / media / AA, Baly outlet classifier on
#      new outputs + existing LED lambda0 / lambda035 outputs + the originals (body only)
# Generation settings = the checkpoint's training config: "title </s> body" truncated to 512 tokens,
# global attention on token 0, greedy, max 512 new tokens.
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOTS = [r for r in os.environ.get("EVAL_ROOTS", "/kaggle/input").split(",") if r]
OUT = Path(os.environ.get("EVAL_OUT", "/kaggle/working/out")); OUT.mkdir(parents=True, exist_ok=True)
LIMIT = int(os.environ.get("EVAL_LIMIT", "0")) or None
NEW = "led_first_token"
HF_REPO, SUB = "RaunakSeksaria/anlp-neutral-led", "side-neutral-full/first_token_pooling"


def find(*names):
    for root in ROOTS:
        for d, dirs, files in os.walk(root):
            for n in names:
                if n in files or n in dirs:
                    return Path(d) / n
    raise FileNotFoundError(names)


norm = lambda s: " ".join(str(s).split())

WORKER = r'''
import json, sys, time, torch
from transformers import AutoTokenizer, LEDForConditionalGeneration
inp, out, shard, nshard, gpu, repo, sub, limit = sys.argv[1:9]
shard, nshard, limit = int(shard), int(nshard), int(limit)
rows = [json.loads(l) for l in open(inp)]
if limit: rows = rows[:limit]
rows = rows[shard::nshard]
try:
    tok = AutoTokenizer.from_pretrained(repo, subfolder=sub)
except Exception as e:
    print("tokenizer fallback:", type(e).__name__, flush=True)
    tok = AutoTokenizer.from_pretrained("allenai/led-base-16384")
dev = f"cuda:{gpu}"
model = LEDForConditionalGeneration.from_pretrained(repo, subfolder=sub).to(dev).half().eval()
norm = lambda s: " ".join(str(s).split())
texts = [f"{norm(r['title'])} {tok.sep_token} {norm(r['body'])}" for r in rows]
lens = [len(tok(x, truncation=True, max_length=512)["input_ids"]) for x in texts]
order = sorted(range(len(rows)), key=lambda i: -lens[i])
t0, i = time.time(), 0
with open(out, "w") as fh:
    while i < len(order):
        bs = max(1, 12288 // (lens[order[i]] + 512))
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
        g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
        with torch.no_grad():
            ids = model.generate(**enc, global_attention_mask=g, num_beams=1, do_sample=False, max_new_tokens=512)
        for j, o, n in zip(idx, tok.batch_decode(ids, skip_special_tokens=True), (ids != tok.pad_token_id).sum(1)):
            fh.write(json.dumps({"id": rows[j]["id"], "output": o.strip(), "input_tokens": lens[j],
                                 "truncated": lens[j] >= 512, "output_tokens": int(n), "hit_cap": int(n) >= 512}) + "\n")
        fh.flush(); i += bs
        print(f"  shard{shard} {i}/{len(order)} {(time.time()-t0)/60:.1f} min", flush=True)
'''


def generate():
    import torch
    n = max(1, torch.cuda.device_count())
    Path("/kaggle/working" if Path("/kaggle/working").exists() else OUT).mkdir(exist_ok=True)
    w = OUT / "gen_worker.py"; w.write_text(WORKER)
    inp = find("test_inputs.jsonl")
    t0 = time.time()
    procs = [subprocess.Popen([sys.executable, str(w), str(inp), str(OUT / f"gen_shard{i}.jsonl"), str(i), str(n),
                               str(i), HF_REPO, SUB, str(LIMIT or 0)]) for i in range(n)]
    codes = [p.wait() for p in procs]
    print(f"generation exit codes {codes}, {(time.time() - t0) / 60:.1f} min", flush=True)
    if any(codes):
        raise SystemExit("generation failed")
    with open(OUT / f"gen_{NEW}.jsonl", "w") as f:
        for i in range(n):
            f.write(open(OUT / f"gen_shard{i}.jsonl").read())


def texts():
    rows = []
    for r in map(json.loads, open(find("test_inputs.jsonl"))):
        if LIMIT and len(rows) >= LIMIT: break
        rows.append({"id": f"original:{r['id']}", "kind": "original", "text": norm(r["body"])})
    for kind, f in ((NEW, OUT / f"gen_{NEW}.jsonl"), ("led_lambda0", find("gen_led_lambda0.jsonl")),
                    ("led_lambda035", find("gen_led_lambda035.jsonl"))):
        for i, r in enumerate(map(json.loads, open(f))):
            if LIMIT and i >= LIMIT: break
            rows.append({"id": f"{kind}:{r['id']}", "kind": kind, "text": norm(r["output"])})
    return rows


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


def score(rows):
    import torch
    code = find("variants.py").parent
    sys.path.insert(0, str(code))
    from transformers import BertTokenizer, BertTokenizerFast
    from model import ChunkedBERTClassifier
    from variants import VariantClassifier, TABLE10
    from baly import build
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = json.load(open(find("outlet_vocab.json")))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    specs = [
        ("chunkybert", lambda: ChunkedBERTClassifier(num_classes=3), ("chunkybert_final_epoch.pt", "chunkybert_5ep.pt"), True, False),
        ("no_classifier", lambda: VariantClassifier(num_classes=3, **TABLE10["No Classifier"]),
         ("best_variant_no_classifier.pt", "no_classifier_1ep.pt"), True, False),
        ("baly_random", lambda: build(3), ("bert_random.pt",), False, True),
        ("baly_media", lambda: build(3), ("bert_media.pt",), False, True),
        ("baly_aa", lambda: build(3), ("bert_aa_media.pt",), False, True),
        ("baly_outlet", lambda: build(len(vocab)), ("bert_medium.pt",), False, False),
    ]
    for name, mk, wnames, chunked, nonstrict in specs:
        t0 = time.time()
        m = mk()
        m.load_state_dict(torch.load(find(*wnames), map_location="cpu", weights_only=True), strict=not nonstrict)
        m.eval().to(dev)
        tok = BertTokenizer.from_pretrained("bert-base-uncased") if chunked else BertTokenizerFast.from_pretrained("bert-base-uncased")
        max_len, bs = (5000, 16) if chunked else (512, 64)
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
    generate()
    rows = texts()
    print(f"{len(rows)} texts to score", flush=True)
    vad(rows)
    score(rows)
    print("ALL DONE", flush=True)
