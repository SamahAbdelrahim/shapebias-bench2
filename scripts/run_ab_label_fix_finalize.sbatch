#!/bin/bash
#SBATCH --job-name=sb-ab-post
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_finalize_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_finalize_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_DATASET=stimuli_pipe/stimuli_per_stl_packages/stimuli_A_auto_contrast

BENCH=results/playground.results/session_2026-07-30_farmshare
SMITH=results/playground.results/session_2026-07-30_smith_farmshare
PROBE_OUT=results/probe.results/session_2026-07-30_farmshare

echo "host=$(hostname) date=$(date -Is)"

echo "===== PriDe: benchmark AB session ====="
python playgrounds/playground_pride_debias.py \
  --session "$BENCH" \
  --out-prefix results/playground.results/prompt_pride_debias_2026-07-30

echo "===== PriDe: Smith AB session ====="
python playgrounds/playground_pride_debias.py \
  --session "$SMITH" \
  --out-prefix results/playground.results/smith_prompt_pride_debias_2026-07-30

echo "===== Probe experiment — AB label set only (fixed Image A/B slots) ====="
mkdir -p "$PROBE_OUT"
python playgrounds/probe_experiment.py \
  --label-sets ab \
  --out-prefix "$PROBE_OUT/probe_experiment_ab_fixed"

echo "===== Per-report HTML ====="
python scripts/build_playground_results_html.py

echo "===== Master interpretation report ====="
python scripts/build_master_interpretation_report.py

echo "DONE finalize $(date -Is)"
