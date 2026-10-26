#!/bin/bash
#SBATCH -A research
#SBATCH -p u22
#SBATCH --qos=medium
#SBATCH -c 36
#SBATCH --mem-per-cpu=2900M
#SBATCH --time=12:00:00
#SBATCH --job-name=bignews-heldout
#SBATCH --output=logs/%x-%j.out
# Download the public POLITICS release (BIGNEWS, BIGNEWSBLN, BIGNEWSALIGN; Google Drive folders) into
# node-local /scratch, then sample BIGNEWS articles POLITICS never trained on. Only results come home.
#
#   sbatch pipeline.sbatch                                        # download + check
#   sbatch --export=ALL,SKIP_DOWNLOAD=1 pipeline.sbatch           # reuse /scratch (same node only)
#   sbatch --export=ALL,SELFTEST=1 -c 4 --time=00:15:00 pipeline.sbatch
set -euo pipefail
cd ~/bignews_heldout
echo "node $(hostname)  python $(python3 --version 2>&1)  cpus ${SLURM_CPUS_PER_TASK:-?}  start $(date -u +%FT%TZ)"
if [ "${SELFTEST:-0}" = 1 ]; then python3 -u check_heldout.py --selftest; exit 0; fi

WORK=/scratch/$USER/bignews; mkdir -p "$WORK"; df -h /scratch | tail -1
declare -A FOLDERS=(
  [BIGNEWS]=https://drive.google.com/drive/folders/1DO7o0IqYm2kPD9RyJroWeLT2WoMw5oRj
  [BIGNEWSBLN]=https://drive.google.com/drive/folders/1rJhlGkNkpT-sBpjowlc0mrTGs-SNhdRL
  [BIGNEWSALIGN]=https://drive.google.com/drive/folders/16DxAK0tYQVPd1oMsmzYo1jgP9Cv90sqB
)
if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  [ -x .venv/bin/gdown ] || { python3 -m venv .venv && .venv/bin/pip -q install gdown; }
  # smallest first, so a Drive quota/permission problem shows up before the long transfers
  for name in BIGNEWSALIGN BIGNEWSBLN BIGNEWS; do
    t0=$(date +%s)
    echo "DOWNLOAD START $name $(date -u +%T)"
    .venv/bin/gdown --folder "${FOLDERS[$name]}" -O "$WORK/$name" --continue 2>&1 | grep -vE "^\s*[0-9]+%|it/s\]" || true
    dt=$(( $(date +%s) - t0 )); sz=$(du -sm "$WORK/$name" | cut -f1)
    echo "DOWNLOAD DONE $name: ${sz} MB in ${dt}s = $(awk -v s=$sz -v t=$dt 'BEGIN{printf "%.1f", s/(t>0?t:1)}') MB/s"
    ls -la "$WORK/$name"
  done
fi
for name in BIGNEWS BIGNEWSBLN BIGNEWSALIGN; do
  n=$(find "$WORK/$name" -name '*.json' | wc -l); echo "$name: $n json files"; [ "$n" -gt 0 ] || { echo "MISSING $name"; exit 1; }
done

mkdir -p results
echo "CHECK START $(date -u +%T)"
python3 -u check_heldout.py --bignews "$WORK/BIGNEWS" --bln "$WORK/BIGNEWSBLN" \
  --align "$(find "$WORK/BIGNEWSALIGN" -name '*.json' | head -1)" --out "$WORK/out" \
  --per-outlet "${PER_OUTLET:-5000}" --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 ))
cp "$WORK/out/report.json" "$WORK/out/flagged.jsonl" results/
gzip -c "$WORK/out/clean_heldout.jsonl" > results/clean_heldout.jsonl.gz
echo "CHECK DONE $(date -u +%T)"; ls -la results
