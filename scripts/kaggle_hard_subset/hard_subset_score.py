"""Kaggle GPU kernel: the `score` step of scripts/make_hard_subset.py, all 4 v2 seeds, both releases.

Pushed with:  kaggle kernels push -p scripts/kaggle_hard_subset
Output:       /kaggle/working/scores.jsonl          original article bodies (key = md5(body))
              /kaggle/working/scores_targets.jsonl  generated targets, <stance>_neutralised (key = md5(text))
              both in the format of runs/hard/scores.jsonl
              ({"key": ..., "windows": n, "logits": {"seed42": [L, C, R], ...}}); copy them to runs/hard/ and
              run `make_hard_subset.py select` locally. Version 1 of the kernel wrote scores.jsonl only.

Data comes from the HF mirror of the releases (byte-identical to the local files per manifest.json);
the classifier from Harshit's Kaggle dataset `harshitlalwani/politics-ideology-classifier`. OUTLET_RE is
copied verbatim from scripts/compare_variants.py so masking, and therefore keys and logits, match.
"""
import gzip, hashlib, importlib.util, io, json, re, sys, time, urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/RaunakSeksaria/anlp-project-datasets/resolve/main/data"
CONFIGS = {"dedup": "reframing_{st}", "full1024": "long/reframing_{st}"}
SPLITS = ("train", "validation", "test")
OUTLETS = [r"Associated Press", r"\bAP\b", r"The Hill", r"USA Today", r"CNN", r"Fox News", r"\bFox\b",
           r"Breitbart(?: News)?", r"Huff(?:ington )?Post", r"HuffPost", r"New York Times", r"\bNYT\b",
           r"Washington Post", r"Washington Times", r"Daily Kos"]
OUTLET_RE = re.compile("|".join(OUTLETS), re.I)

clf = next(p.parent for p in Path("/kaggle/input").rglob("politics_classifier.py"))
spec = importlib.util.spec_from_file_location("politics_classifier", clf / "politics_classifier.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

texts, counts = {"scores": {}, "scores_targets": {}}, {}
for rel, cfg in CONFIGS.items():
    for st in ("left", "right"):
        for sp in SPLITS:
            raw = urllib.request.urlopen(f"{HF}/{cfg.format(st=st)}/{sp}.jsonl.gz").read()
            rows = [json.loads(l) for l in gzip.open(io.BytesIO(raw), "rt", encoding="utf-8")]
            counts[f"{rel}/{st}/{sp}"] = len(rows)
            for r in rows:
                for out, text in (("scores", r[f"{st}_original"]["body"]), ("scores_targets", r[f"{st}_neutralised"])):
                    texts[out][hashlib.md5(text.encode()).hexdigest()] = text
print(json.dumps(counts), *(f"\n{len(v):,} unique texts -> {k}.jsonl" for k, v in texts.items()), flush=True)

weights = sorted(clf.glob("seed*_best.pt"))
models, _ = pc.load(weights)
tok = pc.tokenizer()
device = next(models[0].parameters()).device
print(f"{len(weights)} seeds on {device}", flush=True)

for out, bodies in texts.items():
    items = sorted(bodies.items(), key=lambda kv: len(kv[1]))
    t = time.time()
    wins = pc.to_windows(tok, [OUTLET_RE.sub("<mask>", b) for _, b in items])
    print(f"{out}: {sum(map(len, wins)):,} windows", flush=True)
    L = []
    for w, m in zip(weights, models):
        L.append(pc.article_logits(m, wins, tok.pad_token_id, device))
        print(f"  {w.name} done, {time.time() - t:.0f}s", flush=True)

    with open(f"/kaggle/working/{out}.jsonl", "w", encoding="utf-8") as f:
        for i, (k, _) in enumerate(items):
            f.write(json.dumps({"key": k, "windows": len(wins[i]),
                                "logits": {w.name.split("_")[0]: L[j][i].round(4).tolist()
                                           for j, w in enumerate(weights)}}) + "\n")
    print(f"wrote /kaggle/working/{out}.jsonl", file=sys.stderr)
