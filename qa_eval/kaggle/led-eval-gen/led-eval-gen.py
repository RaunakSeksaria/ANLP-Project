# Kernel B: LED generation, then classifiers on the rewrites, then NLI coverage (DiverseSumm).
#   1. qa_eval.py --stage gen : λ0 / λ0.35 on led_test (1,775 held-out side_neutral_full test articles) and DiverseSumm (2,450)
#   2. ChunkyBERT + No-Classifier on the 3,550 led_test rewrites (rows keyed by the same text hash
#      build_texts.py uses, so they join texts.jsonl after it is rebuilt locally)
#   3. qa_eval.py --stage nli
# Each step is independent of a later one failing; outputs land in /kaggle/working/out and /scores.
import hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

W = Path("/kaggle/working")
OUT = W / "out"


def find(name, under="/kaggle/input"):
    for d, _, files in os.walk(under):
        if name in files:
            return Path(d) / name
    return None


def step(name, fn):
    t0 = time.time()
    print(f"===== {name}", flush=True)
    try:
        fn()
        print(f"===== {name} OK ({(time.time()-t0)/60:.1f} min)", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"===== {name} FAILED ({(time.time()-t0)/60:.1f} min)", flush=True)


for f in ("qa_eval.py", "classify.py", "model.py", "variants.py"):
    shutil.copy(find(f), W / f)
OUT.mkdir(exist_ok=True)
for d, _, files in os.walk("/kaggle/input"):         # reuse rewrites shipped in led-eval-data: gen then skips
    for f in files:
        if f.startswith("gen_") and f.endswith(".jsonl") and not (OUT / f).exists():
            shutil.copy(Path(d) / f, OUT / f); print("reusing", f, flush=True)


def gen():
    subprocess.run([sys.executable, str(W / "qa_eval.py"), "--stage", "gen", "--out", str(OUT)], check=True, cwd=W)


def classify_rewrites():
    norm = lambda s: " ".join(s.split())
    rows = []
    for lam in ("lambda0", "lambda035"):
        for g in map(json.loads, open(OUT / f"gen_own_{lam}.jsonl")):
            t = norm(g["output"])
            rows.append({"h": hashlib.sha1(t.encode()).hexdigest()[:16], "kind": f"led_{lam}", "text": t})
    with open(W / "texts_led.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pts = [Path(d) / x for d, _, fs in os.walk("/kaggle/input") for x in fs if x.endswith(".pt")]
    cb = next(p for p in pts if "political-bias" in str(p) or p.name in ("chunkybert_final_epoch.pt", "chunkybert_5ep.pt"))
    nc = next(p for p in pts if "no_classifier" in p.name)
    args = [sys.executable, str(W / "classify.py"), "--texts", str(W / "texts_led.jsonl"), "--out", str(W / "scores")]
    ps = [subprocess.Popen(args + ["--model", "chunkybert", "--gpu", "0", "--weights", str(cb)]),
          subprocess.Popen(args + ["--model", "no_classifier", "--gpu", "1", "--weights", str(nc)])]
    codes = [p.wait() for p in ps]
    if any(codes):
        raise RuntimeError(f"classify exit codes {codes}")


def nli():
    subprocess.run([sys.executable, str(W / "qa_eval.py"), "--stage", "nli", "--out", str(OUT)], check=True, cwd=W)


step("generation", gen)
step("classifiers on rewrites", classify_rewrites)
step("nli coverage", nli)
print("ALL STEPS FINISHED", flush=True)
