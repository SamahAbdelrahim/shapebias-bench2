#!/bin/bash
# After Smith ladder arrays finish: PriDe, summary CSV, HTML report.
#
#   sbatch --dependency=afterany:<smith_job> scripts/run_smith_finalize.sbatch
#   # or once the array is done:
#   bash scripts/run_smith_finalize.sbatch
#
#SBATCH --job-name=sb-smith-fin
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/smith_finalize_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/smith_finalize_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1

GEN=results/model.results/session_smith_ladder
LOGIT=results/model.results/session_smith_ladder_logit

echo "===== PriDe ====="
if ls "$LOGIT"/*.csv >/dev/null 2>&1; then
  python analysis_pipe/full_grid_pride.py \
    --session "$LOGIT" \
    --out results/data/smith_ladder_pride.csv
else
  echo "WARN: no logit CSVs; skipping pride"
fi

echo "===== generation summary ====="
if ls "$GEN"/*.csv >/dev/null 2>&1; then
  python analysis_pipe/full_grid_summary.py \
    --session "$GEN" \
    --prefix smith_ladder \
    --out-dir results/data
else
  echo "WARN: no generation CSVs; skipping summary"
fi

echo "===== HTML report ====="
python scripts/build_smith_report.py

echo "DONE smith finalize $(date -Is)"
