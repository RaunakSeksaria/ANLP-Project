"""Fluency and hallucination checks on the E2 long test split (default evaluation set from 2026-10-30).

Texts scored, per test article: the original body, its Nemotron rewrite (the target) and the rewrites of the LED
neutralisers lambda0 / lambda035 / first_token (`gen_own_<model>.jsonl` from `qa_eval.py --stage gen`).

Fluency (every system; the original is the baseline):
  rep3        share of word 3-grams that repeat an earlier 3-gram in the same text (loops, copy-paste)
  cola        mean P(acceptable) of the text's sentences, textattack/roberta-base-CoLA; cola_ok = share > 0.5
  ppl         GPT-2-large perplexity of the text (first 1,024 tokens)
  lt_*        LanguageTool matches per 100 words: grammar rules only, and all rules except style/typography/spelling;
              lt_spelling separately (proper nouns inflate it)
Hallucination (rewrites only; the source is the original "title. body"):
  entities    spaCy NER on the rewrite; an entity is new if its normalised text does not occur in the source.
              Reported for named entities (person, org, place, group, event, law, ...) and numbers / dates
  minicheck   lytang/MiniCheck-Flan-T5-Large: P(supported by the source) per rewrite sentence;
              unsupported = P < 0.5

    python quality_eval.py build                                  # local: inputs/e2long_test.jsonl
    python quality_eval.py fluency       --out out --systems ...  # GPU
    python quality_eval.py hallucination --out out --systems ...  # GPU
    python quality_eval.py languagetool  --out out                # CPU, needs Java
    python quality_eval.py report        --out out
"""
import argparse, gzip, json, os, re, time
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = "e2long_test.jsonl"
SYSTEMS = ["original", "nemotron", "lambda0", "lambda035", "first_token"]
LIMIT = None                    # --limit: smoke tests
NAMED = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "LAW", "PRODUCT", "WORK_OF_ART"}
NUMERIC = {"DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL"}
norm = lambda t: " ".join(re.sub(r"[^a-z0-9]+", " ", t.lower().replace("'s ", " ")).split())


def find(name):
    for root in ("/kaggle/input", str(HERE), "."):
        for d, _, files in os.walk(root):
            if name in files:
                return Path(d) / name
    raise FileNotFoundError(name)


def build():
    """E2 long test split, minus articles that are in an LED training split (E2 short train), in qa_eval's format."""
    led_train = set()
    for side in ("left", "right"):
        for l in open(ROOT / "data/short_dedup" / f"{side}_neutral_dedup.jsonl"):
            r = json.loads(l)
            if r["split"] == "train":
                led_train.add(norm(r[f"{side}_original"]["body"]))
    out, dropped = [], 0
    for side in ("left", "right"):
        for l in gzip.open(ROOT / "data/long_10k" / side / "test.jsonl.gz", "rt"):
            r = json.loads(l); o = r[f"{side}_original"]
            if norm(o["body"]) in led_train:
                dropped += 1; continue
            out.append({"input_article_id": f"{r['story_id']}:{side}", "input_ideology": side, "event_id": r["story_id"],
                        "split": "test", "publisher": o["publisher"], "input_title": " ".join(o["title"].split()),
                        "input_text": " ".join(o["body"].split()), "target_text": " ".join(r[f"{side}_neutralised"].split())})
    (HERE / "inputs").mkdir(exist_ok=True)
    with open(HERE / "inputs" / INPUT, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} articles ({dropped} dropped: in LED training data)")


def load(out, systems):
    """{system: [(id, text, source)]} for the requested systems."""
    recs = [json.loads(l) for l in open(find(INPUT))][:LIMIT]
    src = {r["input_article_id"]: f"{r['input_title']}. {r['input_text']}" for r in recs}
    res = {}
    for s in systems:
        if s == "original":
            t = {r["input_article_id"]: r["input_text"] for r in recs}
        elif s == "nemotron":
            t = {r["input_article_id"]: r["target_text"] for r in recs}
        else:
            t = {r["id"]: " ".join(r["output"].split()) for r in map(json.loads, open(Path(out) / f"gen_own_{s}.jsonl"))}
        res[s] = [(i, t[i], src[i]) for i in src]
    return res


def sentences(text):
    import nltk
    try:
        return [s for s in nltk.sent_tokenize(text) if s.strip()] or [text]
    except LookupError:
        nltk.download("punkt", quiet=True); nltk.download("punkt_tab", quiet=True)
        return sentences(text)


def rep3(text):
    w = text.lower().split()
    g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def fluency(out, systems, batch=64):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ctok = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    cola = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to(dev).half().eval() \
        if dev == "cuda" else AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").eval()
    gtok = AutoTokenizer.from_pretrained("gpt2-large")
    gpt = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16 if dev == "cuda" else None).to(dev).eval()
    for s, items in load(out, systems).items():
        path = Path(out) / f"fluency_{s}.jsonl"
        if path.exists():
            print(f"skip {path.name}", flush=True); continue
        t0 = time.time()
        sents = [(k, x) for k, (_, text, _) in enumerate(items) for x in sentences(text)]
        p_ok = []
        with torch.no_grad():
            for i in range(0, len(sents), batch):
                enc = ctok([x for _, x in sents[i:i + batch]], truncation=True, max_length=128, padding=True,
                           return_tensors="pt").to(dev)
                p_ok += torch.softmax(cola(**enc).logits.float(), -1)[:, 1].tolist()      # label 1 = acceptable
            per = [[] for _ in items]
            for (k, _), p in zip(sents, p_ok):
                per[k].append(p)
            ppl = []
            for _, text, _ in items:                    # one text per pass: no padding effects on perplexity
                ids = gtok(text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(dev)
                ppl.append(float(torch.exp(gpt(ids, labels=ids).loss.float())) if ids.shape[1] > 1 else float("nan"))
        with open(path, "w") as f:
            for (i, text, _), ps, pp in zip(items, per, ppl):
                f.write(json.dumps({"id": i, "words": len(text.split()), "sents": len(ps), "rep3": rep3(text),
                                    "cola": mean(ps), "cola_ok": mean(p > 0.5 for p in ps), "ppl": pp}) + "\n")
        print(f"fluency {s}: {len(items)} texts in {(time.time() - t0) / 60:.1f} min", flush=True)


def new_entities(nlp, items):
    rows = []
    for (i, text, src), doc in zip(items, nlp.pipe([t for _, t, _ in items], batch_size=64)):
        s = f" {norm(src)} "
        ents = [(e.label_, e.text) for e in doc.ents if e.label_ in NAMED | NUMERIC and norm(e.text)]
        new = [(lab, t) for lab, t in ents if f" {norm(t)} " not in s]
        rows.append({"named": sum(l in NAMED for l, _ in ents), "named_new": sum(l in NAMED for l, _ in new),
                     "numeric": sum(l in NUMERIC for l, _ in ents), "numeric_new": sum(l in NUMERIC for l, _ in new),
                     "new": new})
    return rows


def hallucination(out, systems):
    import spacy, torch
    from minicheck.minicheck import MiniCheck
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    mc = MiniCheck(model_name="flan-t5-large", batch_size=32, cache_dir=os.environ.get("HF_HOME"))
    for s, items in load(out, [x for x in systems if x != "original"]).items():
        path = Path(out) / f"halluc_{s}.jsonl"
        if path.exists():
            print(f"skip {path.name}", flush=True); continue
        t0 = time.time()
        ents = new_entities(nlp, items)
        claims = [(k, x) for k, (_, text, _) in enumerate(items) for x in sentences(text)]
        _, probs, _, _ = mc.score(docs=[items[k][2] for k, _ in claims], claims=[x for _, x in claims])
        per = [[] for _ in items]
        for (k, _), p in zip(claims, probs):
            per[k].append(float(p))
        with open(path, "w") as f:
            for (i, _, _), e, ps in zip(items, ents, per):
                f.write(json.dumps({"id": i, **e, "sents": len(ps), "unsupported": sum(p < 0.5 for p in ps),
                                    "support": ps}) + "\n")
        print(f"hallucination {s}: {len(items)} texts, {len(claims)} sentences in {(time.time() - t0) / 60:.1f} min",
              flush=True)
        torch.cuda.empty_cache()


def languagetool(out, systems, workers=6):
    import language_tool_python
    from concurrent.futures import ThreadPoolExecutor
    tool = language_tool_python.LanguageTool("en-US")
    skip = {"STYLE", "TYPOGRAPHY", "CASING", "REDUNDANCY", "PLAIN_ENGLISH", "TYPOS"}   # TYPOS: mostly names in news
    for s, items in load(out, systems).items():
        path = Path(out) / f"languagetool_{s}.jsonl"
        if path.exists():
            print(f"skip {path.name}", flush=True); continue
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda it: tool.check(it[1]), items))
        with open(path, "w") as f:
            for (i, text, _), ms in zip(items, res):
                w = max(1, len(text.split()))
                cats = [m.category for m in ms]
                f.write(json.dumps({"id": i, "words": w, "lt_grammar": 100 * sum(c == "GRAMMAR" for c in cats) / w,
                                    "lt_errors": 100 * sum(c not in skip for c in cats) / w,
                                    "lt_spelling": 100 * sum(c == "TYPOS" for c in cats) / w, "rules": [m.rule_id for m in ms], "cats": cats}) + "\n")
        print(f"languagetool {s}: {len(items)} texts in {(time.time() - t0) / 60:.1f} min", flush=True)
    tool.close()


def report(out):
    out = Path(out)
    rd = lambda p: [json.loads(l) for l in open(p)] if p.exists() else None
    med = lambda xs: sorted(xs)[len(xs) // 2]
    res = {}
    print("| System | n | Words | Repeated 3-grams % | CoLA acceptable % | GPT-2-L perplexity (median) "
          "| LT grammar / 100 w | LT errors / 100 w |\n|---|---|---|---|---|---|---|---|")
    for s in SYSTEMS:
        F, L = rd(out / f"fluency_{s}.jsonl"), rd(out / f"languagetool_{s}.jsonl")
        if not F:
            continue
        r = {"n": len(F), "words": mean(x["words"] for x in F), "rep3": 100 * mean(x["rep3"] for x in F),
             "cola_ok": 100 * mean(x["cola_ok"] for x in F), "ppl_median": med([x["ppl"] for x in F])}
        if L:
            r |= {"lt_grammar": mean(x["lt_grammar"] for x in L), "lt_errors": mean(x["lt_errors"] for x in L)}
        res[s] = r
        lt = f"{r['lt_grammar']:.2f} | {r['lt_errors']:.2f}" if L else "— | —"
        print(f"| {s} | {r['n']} | {r['words']:.0f} | {r['rep3']:.2f} | {r['cola_ok']:.1f} | {r['ppl_median']:.1f} | {lt} |")
    print("\n| System | n | Unsupported sentences % (MiniCheck) | Texts with ≥ 1 unsupported % | New named entities % "
          "| New numbers/dates % | Texts with a new named entity % |\n|---|---|---|---|---|---|---|")
    for s in SYSTEMS[1:]:
        H = rd(out / f"halluc_{s}.jsonl")
        if not H:
            continue
        r = {"n": len(H), "unsupported_sent": 100 * sum(x["unsupported"] for x in H) / sum(x["sents"] for x in H),
             "texts_unsupported": 100 * mean(x["unsupported"] > 0 for x in H),
             "named_new": 100 * sum(x["named_new"] for x in H) / max(1, sum(x["named"] for x in H)),
             "numeric_new": 100 * sum(x["numeric_new"] for x in H) / max(1, sum(x["numeric"] for x in H)),
             "texts_named_new": 100 * mean(x["named_new"] > 0 for x in H)}
        res.setdefault(s, {}).update(r)
        print(f"| {s} | {r['n']} | {r['unsupported_sent']:.1f} | {r['texts_unsupported']:.1f} | {r['named_new']:.1f} | "
              f"{r['numeric_new']:.1f} | {r['texts_named_new']:.1f} |")
    json.dump(res, open(out / "summary.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "fluency", "hallucination", "languagetool", "report"])
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--systems", nargs="+", default=SYSTEMS)
    ap.add_argument("--limit", type=int, default=None, help="first N articles (smoke test)")
    a = ap.parse_args()
    global LIMIT
    LIMIT = a.limit
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage == "build":
        build()
    elif a.stage == "report":
        report(a.out)
    else:
        globals()[a.stage](a.out, a.systems)


if __name__ == "__main__":
    main()
