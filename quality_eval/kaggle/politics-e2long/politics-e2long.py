# Kaggle kernel: POLITICS v2 (calibrated 4-seed ensemble, 512-token sliding windows) on every E2 long text:
# 19,922 original bodies + their 19,922 Nemotron rewrites (evaluation/long10k/texts.jsonl). One half per T4.
# Output: /kaggle/working/e2long_politics.csv (id, per-seed logits, ensemble P(L, C, R), windows).
import os, shutil, subprocess, sys
from pathlib import Path

W = Path("/kaggle/working")


def find(name):
    for d, _, files in os.walk("/kaggle/input"):
        if name in files:
            return Path(d) / name
    raise FileNotFoundError(name)


for f in ("score_texts.py", "politics_classifier.py"):
    shutil.copy(find(f), W / f)
weights = [str(find(f"seed{s}_best.pt")) for s in (42, 43, 44, 45)]
lines = open(find("texts.jsonl")).readlines()          # e2long_texts.jsonl.gz, unpacked by Kaggle
half = (len(lines) + 1) // 2
procs = []
for g, part in enumerate((lines[:half], lines[half:])):
    (W / f"part{g}.jsonl").write_text("".join(part))
    cmd = [sys.executable, "-u", str(W / "score_texts.py"), "--weights", *weights, "--calibration",
           str(find("calibration.json")), "--texts", str(W / f"part{g}.jsonl"), "--out", str(W / f"part{g}.csv")]
    procs.append(subprocess.Popen(cmd, cwd=W, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}))
codes = [p.wait() for p in procs]
if any(codes):
    sys.exit(f"scoring failed: {codes}")
a, b = (W / "part0.csv").read_text(), (W / "part1.csv").read_text()
(W / "e2long_politics.csv").write_text(a + b.split("\n", 1)[1])
for g in (0, 1):
    (W / f"part{g}.jsonl").unlink(); (W / f"part{g}.csv").unlink()
print("done:", len(lines), "texts")
