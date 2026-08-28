#!/bin/bash
# One-time setup, run on the Ada LOGIN node (it has network; compute nodes may not).
#
#   bash scripts/ada_setup.sh
#
# Creates a venv, installs torch + transformers, and pre-downloads BART-large so
# the compute node never needs the network. Takes ~10 min, mostly the ~3 GB of
# CUDA wheels.

set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME" logs runs

# --- python env ---------------------------------------------------------------
# Ada usually exposes python via modules; adjust if your allocation differs.
# module load u18/python/3.10.2 || true
# module load u18/cuda/12.1     || true

if [ ! -d .venv-ada ]; then
    python3 -m venv .venv-ada
fi
source .venv-ada/bin/activate

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.40" "numpy<3" scikit-learn sentencepiece

# --- prefetch the model -------------------------------------------------------
python - <<'PY'
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
name = "facebook/bart-large"
AutoTokenizer.from_pretrained(name)
AutoModelForSeq2SeqLM.from_pretrained(name)
print(f"cached {name}")
PY

# --- build the data (CPU, seconds) --------------------------------------------
python scripts/prepare_data.py

echo
echo "setup done. Smoke-test on the login node (CPU, ~2 min):"
echo "  source .venv-ada/bin/activate"
echo "  python scripts/train_neus.py --run-name smoke --epochs 1 --limit-train 8 \\"
echo "      --batch-size 2 --grad-accum 1 --precision fp32 --num-beams 1"
echo
echo "then:  sbatch scripts/ada_run.sh"
