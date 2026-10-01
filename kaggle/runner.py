"""Kaggle runner: side-neutral LED joint training, one lambda per GPU on T4 x2.

MODE="smoke" trains 1 epoch on a 96/256/16 slice (~12 min; the trainer needs >=256 val).
MODE="bench" trains on the 9k split for 10 min and logs optimizer steps per minute.
MODE="full" trains 4 epochs on every pairable article (12,484/1,778/1,775, ~9.5h),
then evaluates each best checkpoint on test.
Everything lands in /kaggle/working/runs/<run>/.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODE = "full"
RUNS = [("lambda0", 0.0), ("lambda035", 0.35)]  # run i goes to GPU i
TRANSFORMERS_VERSION = "5.16.1"  # what the Ada runs used
SETTINGS = {
    "smoke": {"data": "side_neutral_smoke256", "max_epochs": 1, "patience": 1,
              "train_deadline_h": 0.3, "total_deadline_h": 0.4},
    "bench": {"data": "side_neutral_9k", "max_epochs": 5, "patience": 2,
              "train_deadline_h": 0.17, "total_deadline_h": 0.17},
    "full": {"data": "side_neutral_full", "max_epochs": 4, "patience": 2,
             "train_deadline_h": 10.5, "total_deadline_h": 11.6},
}[MODE]
EVAL_BATCH_SIZE = 4
WANDB_ENTITY = "manasagrawal206-iiit-hyderabad"
WANDB_PROJECT = {"smoke": "anlp-kaggle-smoke", "bench": "anlp-kaggle-smoke",
                 "full": "anlp-project"}[MODE]

START = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
UNPACKED = Path("/kaggle/tmp/unpacked")


def log(msg):
    print(f"[{(time.time() - START) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd):
    log("$ " + cmd)
    subprocess.run(cmd, shell=True, check=False)


def locate(relative):
    """Find a path inside the attached dataset, unzipping archives if Kaggle did not."""
    for root in (INPUT, UNPACKED):
        hits = sorted(root.rglob(relative)) if root.exists() else []
        if hits:
            return hits[0]
    for archive in sorted(INPUT.rglob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(UNPACKED / archive.stem)
    hits = sorted(UNPACKED.rglob(relative))
    if not hits:
        raise FileNotFoundError(relative)
    return hits[0]


def tail(path, lines=4):
    if not path.exists():
        return "(no log yet)"
    text = path.read_text(errors="replace").replace("\r", "\n")
    kept = [l for l in text.splitlines() if l.strip() and "padded from" not in l]
    return "\n".join(l[:240] for l in kept[-lines:])


def steps_done(log_path):
    """Optimizer steps finished so far; step_metrics.jsonl gets one line per step."""
    path = log_path.parent / "train" / "step_metrics.jsonl"
    return sum(1 for _ in path.open()) if path.exists() else 0


def supervise(procs, deadline_s, label):
    """Wait for (name, Popen, log) triples; terminate any still alive at the deadline."""
    last_report = 0.0
    while any(p.poll() is None for _, p, _ in procs):
        now = time.time() - START
        if now > deadline_s:
            for name, p, _ in procs:
                if p.poll() is None:
                    log(f"{label} deadline hit, terminating {name}")
                    p.send_signal(signal.SIGTERM)
            time.sleep(60)
            for _, p, _ in procs:
                if p.poll() is None:
                    p.kill()
            break
        if now - last_report > (600 if MODE == "full" else 60):
            last_report = now
            for name, _, path in procs:
                log(f"{label} {name} optimizer_steps={steps_done(path)}:\n{tail(path)}")
            sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader")
        time.sleep(15)
    for name, p, path in procs:
        log(f"{label} {name} exit={p.poll()}\n{tail(path, 8)}")


def wandb_key():
    """Kaggle Secret WANDB_API_KEY if attached, else the private anlp-wandb-key dataset."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        pass
    try:
        return locate("wandb_key.txt").read_text().strip()
    except FileNotFoundError:
        return None


def main():
    sh("nvidia-smi -L")
    import torch
    import transformers
    log(f"python {sys.version.split()[0]} torch {torch.__version__} "
        f"transformers {transformers.__version__} gpus {torch.cuda.device_count()}")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        sh(f"{sys.executable} -m pip install -q transformers=={TRANSFORMERS_VERSION}")

    code_root = locate("src/training/train_joint.py").parents[2]
    data_dir = locate(f"{SETTINGS['data']}/prepared/manifest.json").parent
    log(f"code {code_root}\ndata {data_dir}")

    gpus = torch.cuda.device_count()
    runs = RUNS[:max(gpus, 1)]
    if gpus < len(RUNS):
        log(f"only {gpus} GPU(s); running {[r[0] for r in runs]} only")
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "2"}
    key = wandb_key()
    if key:
        env["WANDB_API_KEY"] = key
    log(f"wandb {'enabled: ' + WANDB_ENTITY + '/' + WANDB_PROJECT if key else 'disabled (no key)'}")

    procs = []
    for gpu, (name, lam) in enumerate(runs):
        out = WORK / "runs" / name
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", str(code_root / "src/scripts/train_joint.py"),
               "--data-dir", str(data_dir), "--output-dir", str(out / "train"),
               "--model-name", "allenai/led-base-16384",
               "--train-batch-size", "1", "--eval-batch-size", str(EVAL_BATCH_SIZE),
               "--gradient-accumulation-steps", "8",
               "--max-epochs", str(SETTINGS["max_epochs"]),
               "--early-stopping-patience", str(SETTINGS["patience"]),
               "--learning-rate", "3e-5", "--weight-decay", "0.01", "--warmup-ratio", "0.10",
               "--max-grad-norm", "1.0", "--max-input-length", "512",
               "--max-target-length", "512", "--generation-max-new-tokens", "512",
               "--triplet-lambda", str(lam), "--triplet-margin", "0.2",
               "--attention-window", "512", "--seed", "42", "--device", "cuda",
               "--precision", "fp32", "--overwrite-output-dir"]
        cmd += (["--wandb-entity", WANDB_ENTITY, "--wandb-project", WANDB_PROJECT,
                 "--wandb-run-name", f"kaggle-{SETTINGS['data']}-{name}"]
                if key else ["--disable-wandb"])
        log(f"launch {name} on GPU {gpu}")
        handle = (out / "train.log").open("w")
        procs.append((name, subprocess.Popen(
            cmd, stdout=handle, stderr=subprocess.STDOUT,
            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}), out / "train.log"))
    supervise(procs, SETTINGS["train_deadline_h"] * 3600, "train")
    if MODE == "bench":
        log("bench done; triplets/s = 8 * delta(optimizer_steps) / delta(seconds) above")
        return

    evals = []
    for gpu, (name, _) in enumerate(runs):
        best = WORK / "runs" / name / "train" / "best"
        if not (best / "joint_state.json").exists():
            log(f"{name}: no best checkpoint, skipping test eval")
            continue
        path = WORK / "runs" / name / "test_eval.log"
        cmd = [sys.executable, "-u", str(code_root / "src/scripts/evaluate_joint.py"),
               "--data-dir", str(data_dir), "--checkpoint", str(best),
               "--output-dir", str(WORK / "runs" / name / "test_eval"), "--split", "test",
               "--eval-batch-size", str(EVAL_BATCH_SIZE), "--device", "cuda",
               "--overwrite-output-dir"]
        evals.append((name, subprocess.Popen(
            cmd, stdout=path.open("w"), stderr=subprocess.STDOUT,
            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}), path))
    supervise(evals, SETTINGS["total_deadline_h"] * 3600, "test")

    # Optimizer state (~1.3 GB per checkpoint) and the last-epoch model are not
    # needed downstream; keep outputs to the best model plus metrics.
    for name, _ in runs:
        train_dir = WORK / "runs" / name / "train"
        shutil.rmtree(train_dir / "last", ignore_errors=True)
        for state in train_dir.rglob("trainer_state.pt"):
            state.unlink()
    sh(f"du -sh {WORK}/runs/*/train/best 2>/dev/null; du -sh {WORK}")
    log("done")


if __name__ == "__main__":
    main()
