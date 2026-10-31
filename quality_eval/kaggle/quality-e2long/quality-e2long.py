# Kaggle kernel: fluency + hallucination on the E2 long test split (1,864 articles; quality_eval/README.md).
#   0. smoke test of both GPU stages on 8 articles; the full run starts only if it passes
#   1. LED lambda0 / lambda035 / first_token rewrite every test article (qa_eval.py --stage gen; "title </s> body",
#      1,024 input tokens, greedy, up to 1,024 new tokens)
#   2. two lanes, one per T4:
#        GPU0: POLITICS v2 on the LED rewrites -> fluency (original, nemotron, lambda0) -> hallucination (nemotron, lambda0)
#        GPU1: fluency (lambda035, first_token) -> hallucination (lambda035, first_token)
#   3. report
# Outputs in /kaggle/working/out. LanguageTool runs locally on the downloaded rewrites.
import json, os, shutil, subprocess, sys, time
from pathlib import Path

W, OUT = Path("/kaggle/working"), Path("/kaggle/working/out")
OUT.mkdir(exist_ok=True)
PY = sys.executable
T0 = time.time()


def find(name):
    for d, _, files in os.walk("/kaggle/input"):
        if name in files:
            return Path(d) / name
    raise FileNotFoundError(name)


def sh(cmd, **kw):
    print(f"\n===== {' '.join(map(str, cmd))[:160]}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    return subprocess.run(list(map(str, cmd)), cwd=W, **kw)


sh([PY, "-m", "pip", "install", "-q", "--no-deps",
    "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@b58b9fa69acbd1015ec970fa65dd752413a053d2"], check=True)
sh([PY, "-m", "spacy", "download", "en_core_web_sm"], check=True)
sh([PY, "-c", "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"], check=True)
for f in ("quality_eval.py", "qa_eval.py", "score_texts.py", "politics_classifier.py"):
    shutil.copy(find(f), W / f)
QE = [PY, "-u", W / "quality_eval.py"]

# 0. smoke
for stage, systems in (("fluency", ["original", "nemotron"]), ("hallucination", ["nemotron"])):
    if sh(QE + [stage, "--out", W / "smoke", "--systems", *systems, "--limit", "8"]).returncode:
        sys.exit(f"smoke test failed at {stage}; not starting the full run")
sh(QE + ["report", "--out", W / "smoke"])

# 1. generation
sh([PY, "-u", W / "qa_eval.py", "--stage", "gen", "--own-file", "e2long_test.jsonl", "--sets", "own",
    "--max-in", "1024", "--max-new", "1024", "--out", OUT], check=True)
with open(W / "led_texts.jsonl", "w") as f:
    for m in ("lambda0", "lambda035", "first_token"):
        for r in map(json.loads, open(OUT / f"gen_own_{m}.jsonl")):
            f.write(json.dumps({"id": f"{r['id']}:{m}", "text": " ".join(r["output"].split())}) + "\n")

# 2. lanes
weights = [find(f"seed{s}_best.pt") for s in (42, 43, 44, 45)]
politics = [PY, "-u", W / "score_texts.py", "--weights", *weights, "--calibration", find("calibration.json"),
            "--texts", W / "led_texts.jsonl", "--out", OUT / "led_politics.csv"]
lanes = [[politics,
          QE + ["fluency", "--out", OUT, "--systems", "original", "nemotron", "lambda0"],
          QE + ["hallucination", "--out", OUT, "--systems", "nemotron", "lambda0"]],
         [QE + ["fluency", "--out", OUT, "--systems", "lambda035", "first_token"],
          QE + ["hallucination", "--out", OUT, "--systems", "lambda035", "first_token"]]]
procs = [subprocess.Popen(" ; ".join(" ".join(f"'{x}'" for x in c) for c in lane), shell=True, cwd=W,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}) for g, lane in enumerate(lanes)]
print("lanes exit", [p.wait() for p in procs], flush=True)

# 3. report
sh(QE + ["report", "--out", OUT])
print(f"ALL DONE in {(time.time() - T0) / 60:.1f} min")
