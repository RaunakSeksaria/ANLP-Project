# Kernel LG (long1k): LED λ0 / λ0.35 rewrite the 2,000 long1k articles with the full article as input
# (max 1,024 input tokens; the models were trained at 512) and up to 1,024 new tokens, greedy; then the four
# classifiers score the 4,000 rewrites (rows keyed by the text hash build_texts.py uses).
# Outputs: /kaggle/working/out/gen_own_{lambda0,lambda035}.jsonl, /kaggle/working/scores/*.jsonl
import hashlib, json, os, shlex, shutil, subprocess, sys, time, traceback
from pathlib import Path

W = Path("/kaggle/working")
OUT = W / "out"

def find(name, under="/kaggle/input"):
    for d, _, files in os.walk(under):
        if name in files:
            return Path(d) / name
    return None


def weights():
    pts = [Path(d) / f for d, _, fs in os.walk("/kaggle/input") for f in fs if f.endswith(".pt")]
    print("weights found:", *pts, sep="\n  ", flush=True)
    pick = lambda pred: next(p for p in pts if pred(p))
    return {"chunkybert": pick(lambda p: p.name in ("chunkybert_final_epoch.pt", "chunkybert_5ep.pt")),
            "no_classifier": pick(lambda p: "no_classifier" in p.name),
            "politics_alls": pick(lambda p: p.name == "alls_politics_s1_lr2e-05.pt"),
            "roberta_alls": pick(lambda p: p.name == "alls_roberta_s1_lr2e-05.pt")}


def classify_all(texts, out):
    """4 classifiers, two lanes (one per T4): ChunkyBERT -> POLITICS-AllS, No-Classifier -> RoBERTa-AllS."""
    w = weights()
    cls = lambda m, g: [sys.executable, str(W / "classify.py"), "--texts", str(texts), "--out", str(out),
                        "--model", m, "--gpu", str(g), "--weights", str(w[m])]
    pol = lambda n, enc, g: [sys.executable, str(W / "score_politics.py"), "--texts", str(texts), "--out", str(out),
                             "--name", n, "--encoder", enc, "--gpu", str(g), "--weights", str(w[n])]
    lanes = [" ; ".join(map(shlex.join, [cls("chunkybert", 0), pol("politics_alls", "launch/POLITICS", 0)])),
             " ; ".join(map(shlex.join, [cls("no_classifier", 1), pol("roberta_alls", "roberta-base", 1)]))]
    codes = [p.wait() for p in [subprocess.Popen(l, shell=True) for l in lanes]]
    print("classifier lanes exit", codes, flush=True)
    return codes


def step(name, fn):
    t0 = time.time(); print(f"===== {name}", flush=True)
    try:
        fn(); print(f"===== {name} OK ({(time.time()-t0)/60:.1f} min)", flush=True)
    except Exception:
        traceback.print_exc(); print(f"===== {name} FAILED ({(time.time()-t0)/60:.1f} min)", flush=True)


for f in ("qa_eval.py", "classify.py", "model.py", "variants.py", "score_politics.py"):
    shutil.copy(find(f), W / f)
OUT.mkdir(exist_ok=True)


def gen():
    subprocess.run([sys.executable, str(W / "qa_eval.py"), "--stage", "gen", "--own-file", "led_long1k.jsonl",
                    "--sets", "own", "--max-in", "1024", "--max-new", "1024", "--out", str(OUT)], check=True, cwd=W)


def classify_rewrites():
    norm = lambda s: " ".join(s.split())
    with open(W / "texts_led.jsonl", "w") as f:
        for lam in ("lambda0", "lambda035"):
            for g in map(json.loads, open(OUT / f"gen_own_{lam}.jsonl")):
                t = norm(g["output"])
                f.write(json.dumps({"h": hashlib.sha1(t.encode()).hexdigest()[:16], "kind": f"led_{lam}", "text": t}) + "\n")
    if any(classify_all(W / "texts_led.jsonl", W / "scores")):
        raise RuntimeError("a classifier lane failed")


step("generation (1,024-token input)", gen)
step("classifiers on rewrites", classify_rewrites)
print("ALL STEPS FINISHED", flush=True)
