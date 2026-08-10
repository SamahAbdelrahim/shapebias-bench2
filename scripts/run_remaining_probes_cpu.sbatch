#!/bin/bash
# Finish linear probes for any grid NPZ that lacks a probe JSON.
# CPU-only: the NPZ export already happened on GPU; probing does not need one.
#
#   sbatch --dependency=afterany:1678529 scripts/run_remaining_probes_cpu.sbatch
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

SESS=results/probe.results/session_readout_power
OUT="$SESS/embeddings_grid"
PIX="$OUT/pixels32.npz"

echo "host=$(hostname) date=$(date -Is) PROBE-CPU"
ANY_FAIL=0
for NPZ in "$OUT"/*.npz; do
  base=$(basename "$NPZ" .npz)
  [[ "$base" == "pixels32" ]] && continue
  js="$SESS/probe_${base}.json"
  if [[ -s "$js" ]]; then
    echo "SKIP $base"
    continue
  fi
  echo "===== probe $base ====="
  set +e
  python playgrounds/linear_probe.py \
    --emb "$NPZ" \
    --pixels "$PIX" \
    --n-perm 100 \
    --out "$js"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: probe failed for $base (rc=$rc)"
    ANY_FAIL=1
  fi
done

echo "===== readout_power figures ====="
python analysis_pipe/readout_power_figure.py \
  --probe-dir "$SESS" || echo "WARN: figure rebuild failed"

echo "DONE probe-cpu $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
