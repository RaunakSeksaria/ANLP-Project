# Kernel Q: QuestEval on led_test (new data only) for {nemotron (Raunak target), λ0, λ0.35}. Needs kernel B's generations:
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

qa = [sys.executable, str(W / "qa_eval.py"), "--stage", "qe", "--reuse", str(gens)]
smoke = subprocess.run(qa + ["--out", str(W / "smoke"), "--limit", "5"], cwd=W)
if smoke.returncode:
    sys.exit("QuestEval smoke test failed; not starting the full run")
subprocess.run(qa + ["--out", str(W / "out")], check=True, cwd=W)
