#!/bin/bash
# CPU follow-up: linear probe + readout ladder on exported grid embeddings.
# GPU array 1673885 already wrote the .npz files; its in-job probes with
# --n-perm 100 are very slow. This job runs --n-perm 0 with thread caps so
# figures can land sooner. Re-run later with --n-perm 100 if needed.
#
#   sbatch scripts/run_readout_probes_cpu.sbatch
#
#SBATCH --job-name=sb-probe-cpu
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/probe_cpu_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/probe_cpu_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

OUT=results/probe.results/session_readout_power
EMB=$OUT/embeddings_grid
PIX=$EMB/pixels32.npz
mkdir -p results/probe.results/jobs "$OUT"

echo "host=$(hostname) date=$(date -Is) PROBE_CPU"
[[ -f "$PIX" ]] || { echo "missing $PIX"; exit 1; }

shopt -s nullglob
NPZS=("$EMB"/qwen3.5-4b_*.npz "$EMB"/qwen3.5-9b_*.npz \
      "$EMB"/qwen3-vl-8b_*.npz "$EMB"/smolvlm_*.npz)
echo "n_npz=${#NPZS[@]}"

for NPZ in "${NPZS[@]}"; do
  BASE=$(basename "$NPZ" .npz)
  DEST="$OUT/probe_${BASE}.json"
  if [[ -f "$DEST" ]]; then
    echo "skip existing $DEST"
    continue
  fi
  echo "===== probe $BASE $(date -Is) ====="
  python playgrounds/linear_probe.py \
    --emb "$NPZ" \
    --pixels "$PIX" \
    --n-perm 0 \
    --out "$DEST"
done

echo "===== figures $(date -Is) ====="
python analysis_pipe/readout_power_figure.py --probe-dir "$OUT" || \
  echo "WARN: figure script failed (may need more models)"

echo "ALL DONE $(date -Is)"
ls -la "$OUT"/probe_*.json 2>/dev/null || true
