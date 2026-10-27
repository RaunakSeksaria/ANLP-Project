"""QA-based evaluation of the team's LED neutralisers (RaunakSeksaria/anlp-neutral-led,
side-neutral-full/lambda{0,035}).

The generator rewrites ONE partisan article into a neutral one. Input format and decoding follow
the training code (origin/manas src/data/triplet_dataset.py) and the checkpoint's training config:
"title </s> body", 512 input tokens, greedy, 512 new tokens.

Stages (each skips work whose output already exists, so a rerun resumes):
  gen   generate rewrites with all three models, for
          - DiverseSumm: every source article (245 stories x 10 = 2,450)
          - own test split: the 1,775 held-out partisan articles the λ models were built with
            (Raunak's side_neutral_full/prepared/test/baseline.jsonl, shipped as led_test.jsonl).
            NEW DATA ONLY: nothing from stories_10k / own_test.
  nli   Eval 1 — DiverseSumm QA coverage. Every reference answer is tied to the article it was
        extracted from; an answer counts as covered by a text if an NLI model says some chunk of
        that text entails it. Scored on the original article (control) and on each rewrite.
  qe    Eval 2 — QuestEval on the own test split: precision (questions from the rewrite, answered
        on the source = faithfulness) and recall (questions from the source, answered on the
        rewrite = coverage). Raunak's per-article Nemotron neutral rewrite (the LED training
        target, `target_text`) is scored the same way as a reference system.
"""
import argparse, json, math, os, re, sys, time
from pathlib import Path

REPO = os.environ.get("LED_REPO", "RaunakSeksaria/anlp-neutral-led")   # local copy for smoke tests
MODELS = {"lambda0": "side-neutral-full/lambda0",
          "lambda035": "side-neutral-full/lambda035"}
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
GEN = dict(num_beams=1, do_sample=False, max_new_tokens=512)
MAX_IN = 512
SETS = {"own", "diversesumm"}


def find(name):
    for root in ("/kaggle/input", str(Path(__file__).resolve().parent), "."):
        for d, dirs, files in os.walk(root):
            if name in files:
                return Path(d) / name
    raise FileNotFoundError(name)


def fmt(title, text):                      # training format; </s> is the LED sep_token
    return f"{' '.join(title.split())} </s> {' '.join(text.split())}"


def plain(title, text):                    # what scorers read: no model-specific separator
    return f"{title}\n\n{text}"


OWN_FILE = "led_test.jsonl"     # --own-file: led_long1k.jsonl for the long1k set


def load_inputs(limit):
    ds = json.load(open(find("diverse_summ.json")))
    own = [json.loads(l) for l in open(find(OWN_FILE))]
    if limit:
        ds, own = ds[:max(1, limit // 10)], own[:limit]
    ds_items = [{"id": a["aid"], "eid": s["eid"], "text": fmt(a["title"], a["content"])}
                for s in ds for a in s["articles"]]
    own_items = [{"id": r["input_article_id"], "text": fmt(r["input_title"], r["input_text"])} for r in own]
    return ds, own, {"diversesumm": ds_items, "own": own_items}


# ---------------------------------------------------------------------------- generation
def gen_worker(gpu, jobs, items, out, cfg=None):
    global MAX_IN
    if cfg:                                  # spawned workers do not inherit main()'s settings
        MAX_IN = cfg["max_in"]; GEN.update(max_new_tokens=cfg["max_new"])
    print(f"[gpu{gpu}] MAX_IN {MAX_IN}, max_new_tokens {GEN['max_new_tokens']}", flush=True)
    import torch
    from transformers import AutoTokenizer, LEDForConditionalGeneration
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"      # cpu: local smoke tests only
    for model_key, ds_key in jobs:
        path = out / f"gen_{ds_key}_{model_key}.jsonl"
        if path.exists():
            print(f"[gpu{gpu}] skip {path.name}", flush=True); continue
        tok = AutoTokenizer.from_pretrained(REPO, subfolder=MODELS[model_key])
        model = LEDForConditionalGeneration.from_pretrained(REPO, subfolder=MODELS[model_key])
        model = (model.to(dev).half() if dev != "cpu" else model).eval()
        rows = items[ds_key]
        lens = [len(tok(r["text"], truncation=True, max_length=MAX_IN)["input_ids"]) for r in rows]
        order = sorted(range(len(rows)), key=lambda i: -lens[i])
        res, t0, i = {}, time.time(), 0

        def run(idx):
            enc = tok([rows[j]["text"] for j in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=MAX_IN).to(dev)
            g = torch.zeros_like(enc["input_ids"]); g[:, 0] = 1
            try:
                with torch.no_grad():
                    ids = model.generate(**enc, global_attention_mask=g, **GEN)
                return tok.batch_decode(ids, skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                h = len(idx) // 2
                return run(idx[:h]) + run(idx[h:])

        while i < len(order):
            L = lens[order[i]]
            bs = max(1, min(32, 16384 // max(512, math.ceil(L / 512) * 512)))
            idx = order[i:i + bs]
            for j, o in zip(idx, run(idx)):
                res[j] = o
            i += bs
            if (i // bs) % 20 == 0:
                print(f"[gpu{gpu}] {path.name} {i}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        with open(path, "w") as f:
            for j, r in enumerate(rows):
                f.write(json.dumps({"id": r["id"], "output": res[j], "input_tokens": lens[j],
                                    "truncated": lens[j] >= MAX_IN}) + "\n")
        print(f"[gpu{gpu}] wrote {path.name} in {(time.time()-t0)/60:.1f} min", flush=True)
        del model; torch.cuda.empty_cache()


def stage_gen(items, out):
    import torch, torch.multiprocessing as mp
    jobs = [(m, d) for d in ("own", "diversesumm") if d in SETS for m in MODELS]     # test articles first
    n = max(1, torch.cuda.device_count())
    split = [jobs[g::n] for g in range(n)]
    cfg = {"max_in": MAX_IN, "max_new": GEN["max_new_tokens"]}
    if n == 1:
        return gen_worker(0, split[0], items, out, cfg)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=gen_worker, args=(g, split[g], items, out, cfg)) for g in range(n)]
    [p.start() for p in ps]; [p.join() for p in ps]
    bad = [p.exitcode for p in ps if p.exitcode]
    if bad:
        raise RuntimeError(f"generation worker failed: {bad}")


def read_gen(out, ds_key, model_key):
    return {r["id"]: r for r in map(json.loads, open(out / f"gen_{ds_key}_{model_key}.jsonl"))}


# ---------------------------------------------------------------------------- eval 1: NLI coverage
SENT = re.compile(r"(?<=[.!?])\s+")


def chunks(text, tok, max_tok=320):
    out, cur, n = [], [], 0
    for s in SENT.split(text.strip()):
        k = len(tok(s, add_special_tokens=False)["input_ids"])
        if cur and n + k > max_tok:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += k
    if cur:
        out.append(" ".join(cur))
    return out or [""]


def stage_nli(ds, out):
    import torch, pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dst = out / "diversesumm_items.csv"
    if dst.exists():
        print("skip nli", flush=True); return
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"           # cpu: local smoke tests only
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev)
    model = (model.half() if dev == "cuda" else model).eval()
    ent = {k.lower(): v for k, v in model.config.label2id.items()}["entailment"]

    texts = {"original": {a["aid"]: plain(a["title"], a["content"]) for s in ds for a in s["articles"]}}
    for m in MODELS:
        texts[m] = {k: v["output"] for k, v in read_gen(out, "diversesumm", m).items()}
    items = [{"eid": s["eid"], "q": qi, "g": gi, "aid": it["aid"], "answer": it["answer"]}
             for s in ds for qi, qa in enumerate(s["question_answers"])
             for gi, grp in enumerate(qa["answer_groups"]) for it in grp]
    have = set.intersection(*(set(d) for d in texts.values()))
    n0 = len(items)
    items = [it for it in items if it["aid"] in have]      # DiverseSumm: 2 answers cite an article not in the release
    print(f"[nli] dropped {n0 - len(items)} of {n0} answer items whose article is missing", flush=True)

    chunk_cache = {(sysk, aid): chunks(t, tok) for sysk, d in texts.items() for aid, t in d.items()}
    pairs = sorted({(c, it["answer"]) for it in items for sysk in texts
                    for c in chunk_cache[(sysk, it["aid"])]}, key=lambda p: len(p[0]))
    print(f"[nli] {len(items)} answer items, {len(pairs)} unique (chunk, answer) pairs", flush=True)
    score, t0 = {}, time.time()
    for i in range(0, len(pairs), 64):
        b = pairs[i:i + 64]
        enc = tok([p[0] for p in b], [p[1] for p in b], return_tensors="pt", padding=True,
                  truncation="only_first", max_length=512).to(dev)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits.float(), -1)[:, ent].cpu().tolist()
        score.update(zip(b, p))
        if (i // 64) % 200 == 0:
            print(f"[nli] {i}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)
    for it in items:
        for sysk in texts:
            it[f"p_{sysk}"] = max(score[(c, it["answer"])] for c in chunk_cache[(sysk, it["aid"])])
    pd.DataFrame(items).to_csv(dst, index=False)
    print(f"[nli] wrote {dst.name} ({(time.time()-t0)/60:.1f} min)", flush=True)


# ---------------------------------------------------------------------------- eval 2: QuestEval
def qe_worker(gpu, systems, own, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import datasets, evaluate
    datasets.load_metric = evaluate.load            # removed from `datasets`; same interface
    from questeval import DIR as QE_DIR
    os.makedirs(os.path.join(QE_DIR, "logs"), exist_ok=True)   # listed at init; may be absent
    from questeval.questeval_metric import QuestEval

    class QE(QuestEval):
        """Keep precision and recall instead of only their mean."""
        def _calculate_score_from_logs(self, hyp_log, compared_logs):
            c = compared_logs[0]
            if c["text"] == "" or hyp_log["text"] == "":
                p = r = 0.0
            else:
                r = self._base_score(hyp_log, c)     # source questions answered on output
                p = self._base_score(c, hyp_log)     # output questions answered on source
            self.pr.append((p, r))
            return (p + r) / 2

    qe = QE(task="text2text")

    def t5s(m):                                     # every API_T2T wrapper in qe.models (nested dicts)
        if isinstance(m, dict):
            for v in m.values():
                yield from t5s(v)
        elif m is not None and hasattr(m, "model"):
            yield m.model
    # The 2021 T5 checkpoints define no decoder_start_token_id, and QuestEval calls generate(...,
    # decoder_start_token_id=None), which transformers 5 rejects. Use T5's own start token (pad) instead.
    for t5 in t5s(qe.models):
        def gen(*args, _g=t5.generate, _start=t5.config.pad_token_id, **kw):
            if kw.get("decoder_start_token_id") is None:
                kw["decoder_start_token_id"] = _start
            return _g(*args, **kw)
        t5.generate = gen
    qe.log_dir = str(out / f"qe_cache_gpu{gpu}"); os.makedirs(qe.log_dir, exist_ok=True)
    qe.hash_files = set(os.listdir(qe.log_dir))
    src = [plain(r["input_title"], r["input_text"]) for r in own]
    for sysk in systems:
        dst = out / f"questeval_{sysk}.jsonl"
        if dst.exists():
            print(f"[qe gpu{gpu}] skip {dst.name}", flush=True); continue
        if sysk == "nemotron":
            hyp = [r["target_text"] for r in own]
        else:
            g = read_gen(out, "own", sysk); hyp = [g[r["input_article_id"]]["output"] for r in own]
        qe.pr, t0 = [], time.time()
        res = qe.corpus_questeval(hypothesis=hyp, sources=src, batch_size=256)
        with open(dst, "w") as f:
            for r, s, (p, rc) in zip(own, res["ex_level_scores"], qe.pr):
                f.write(json.dumps({"id": r["input_article_id"], "questeval": s, "precision": p, "recall": rc}) + "\n")
        print(f"[qe gpu{gpu}] {sysk}: {res['corpus_score']:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)


def stage_qe(own, out):
    import torch, torch.multiprocessing as mp
    systems = ["nemotron", *MODELS]
    n = max(1, torch.cuda.device_count())
    split = [systems[g::n] for g in range(n)]
    if n == 1:
        return qe_worker(0, split[0], own, out)
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=qe_worker, args=(g, split[g], own, out)) for g in range(n)]
    [p.start() for p in ps]; [p.join() for p in ps]
    bad = [p.exitcode for p in ps if p.exitcode]
    if bad:
        raise RuntimeError(f"questeval worker failed: {bad}")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["gen", "nli", "qe", "all"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: N own examples, N/10 stories")
    ap.add_argument("--reuse", default=None, help="dir with earlier outputs to copy in (resume)")
    ap.add_argument("--own-file", default="led_test.jsonl", help="own test set (led_test.jsonl | led_long1k.jsonl)")
    ap.add_argument("--sets", default="own,diversesumm", help="generation: which inputs")
    ap.add_argument("--max-in", type=int, default=512, help="LED input tokens (training used 512)")
    ap.add_argument("--max-new", type=int, default=512, help="max new tokens")
    a = ap.parse_args()
    global OWN_FILE, SETS, MAX_IN
    OWN_FILE, SETS, MAX_IN = a.own_file, set(a.sets.split(",")), a.max_in
    GEN["max_new_tokens"] = a.max_new
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.reuse:
        import shutil
        for f in Path(a.reuse).rglob("gen_*.jsonl"):
            if not (out / f.name).exists():
                shutil.copy(f, out / f.name)
    ds, own, items = load_inputs(a.limit)
    own_recs = [json.loads(l) for l in open(find(OWN_FILE))][:len(items["own"])]
    t0 = time.time()
    if a.stage in ("gen", "all"):
        stage_gen(items, out)
    if a.stage in ("nli", "all"):
        stage_nli(ds, out)
    if a.stage in ("qe", "all"):
        stage_qe(own_recs, out)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
