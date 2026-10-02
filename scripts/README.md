# scripts/

Launchers and upload jobs around `training/`. None of them is needed to read
or run the code locally.

| path | what it does |
| --- | --- |
| `ada/ada_train_side_neutral.sh` | SLURM job for the IIIT-H Ada cluster (RTX 2080 Ti): trains one joint run with the published settings, then evaluates the best checkpoint on test. `LAMBDA`, `DATA` and `RUN` select the run; `DRY_RUN=1` checks the environment and data only. Builds `.venv` inside the job on first use. |
| `kaggle/runner.py` | Kaggle kernel (2× T4): trains two runs in parallel, one per GPU, with deadlines that keep a session under Kaggle's 12 h limit, then evaluates each best checkpoint on test. Set `MODE` and `RUNS` at the top; `kernel-metadata.json` lists the Kaggle datasets it mounts (the `training/` code and a prepared split). |
| `kaggle/hf_upload/upload_to_hf.py` | Uploads best checkpoints with their metrics to `RaunakSeksaria/anlp-neutral-led` as `<experiment>/<run>/`, refuses to overwrite an existing run, and rebuilds the model card from every run in the repo. On Kaggle it reads the outputs of the training kernel named in `kernel-metadata.json`; elsewhere pass `--experiment` and `--run NAME=DIR` with `HF_TOKEN` set. |

```bash
# Ada (from the repo root, after splits/build.sh)
LAMBDA=0.35 DATA=dataset/side_neutral_full_publisher/prepared \
  RUN=ada-side_neutral_full_publisher-lambda035 sbatch scripts/ada/ada_train_side_neutral.sh

# Kaggle
KAGGLE_API_TOKEN=... uvx kaggle kernels push -p scripts/kaggle

# Hugging Face, from wherever the run folder is
HF_TOKEN=... uv run python scripts/kaggle/hf_upload/upload_to_hf.py \
    --experiment side-neutral-full-publisher --run lambda035=runs/ada/<run>_<jobid>
```

Ada notes: log in through `ada-gw1`, which allows 1 CPU and 2 GB per user, so
never build environments or import torch there; use `sinteractive` or `sbatch`.
Use `--mem-per-cpu`, never `--mem`, and `--constraint=2080ti`.
