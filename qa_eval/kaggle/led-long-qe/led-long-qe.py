# Kernel LQ (long1k): QuestEval on a seeded 300-article sample of long1k (150 events x left + right, random.Random(7)), for {nemotron (Raunak target), λ0, λ0.35}; LED rewrites from kernel LG (1,024-token input). Needs kernel B's generations:
# upload them into led-eval-data (gen_own_*.jsonl) before pushing. Runs a --limit 5 smoke test first;
# the full run only starts if the smoke test passes.
import os, shutil, subprocess, sys
from pathlib import Path

W = Path("/kaggle/working")


def find(name, under="/kaggle/input"):
    for d, _, files in os.walk(under):
        if name in files:
            return Path(d) / name
    return None


subprocess.run([sys.executable, "-m", "pip", "install", "-q", "questeval", "--no-deps"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unidecode", "evaluate", "bert_score", "spacy"], check=True)
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
shutil.copy(find("qa_eval.py"), W / "qa_eval.py")
gens = W / "gens"; gens.mkdir(exist_ok=True)
for lam in ("lambda0", "lambda035"):
    shutil.copy(find(f"gen_own_{lam}.jsonl"), gens / f"gen_own_{lam}.jsonl")

qa = [sys.executable, str(W / "qa_eval.py"), "--stage", "qe", "--reuse", str(gens), "--own-file", "led_long1k_qe300.jsonl"]
# resume: systems already scored (shipped in led-long-data) are copied into out/ so qa_eval skips them; run 1 had both
# workers loaded at once and the λ0 one was OOM-killed (-9), so a lone remaining system also avoids that
(W / "out").mkdir(exist_ok=True)
for d, _, fs in os.walk("/kaggle/input"):
    for f in fs:
        if f.startswith("questeval_") and f.endswith(".jsonl"):
            shutil.copy(Path(d) / f, W / "out" / f); print("reusing", f, flush=True)
smoke = subprocess.run(qa + ["--out", str(W / "smoke"), "--limit", "5"], cwd=W)
if smoke.returncode:
    sys.exit("QuestEval smoke test failed; not starting the full run")
subprocess.run(qa + ["--out", str(W / "out")], check=True, cwd=W)
