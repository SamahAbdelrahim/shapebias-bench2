#!/bin/bash
# Embedding robustness for the full-grid paper figures:
#   1) stratified sample (n=114) of the novel-object texture grid
#   2) Smith probe (n=30)
#   3) Geirhos cue-conflict (n=30)
# One array task per model; %4 respects the gpu QOS cap.
#
#   sbatch scripts/run_full_grid_embedding_v1a.sbatch
#
#SBATCH --job-name=sb-grid-emb
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/grid_emb_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/grid_emb_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/probe.results/jobs results/probe.results/session_full_grid_v1a
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
SESS=results/probe.results/session_full_grid_v1a
GRID_PKG=stimuli_texture_grid_v1_scratch
SMITH=previous-lit-stimuli/smith_stimuli
CUE=previous-lit-stimuli/geirhos_cueconflict

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL EMBED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ANY_FAIL=0

run_one() {
  local OUT_PREFIX="$1"
  shift
  echo "===== $OUT_PREFIX ====="
  set +e
  python playgrounds/embedding_robust.py \
    --models "$MODEL" \
    --out-prefix "$OUT_PREFIX" \
    "$@"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: $OUT_PREFIX exited $rc"
    ANY_FAIL=1
  fi
}

# Per-model files so concurrent array tasks do not race on one JSON.
run_one "$SESS/embedding_grid_${MODEL}" \
  --grid-pkg "$GRID_PKG" --stim-set stimuli_A_auto_contrast --n-stimuli 114 --seed 0

run_one "$SESS/embedding_smith_${MODEL}" \
  --smith-probe "$SMITH" --n-stimuli 30 --seed 0

if [[ -d "$CUE" ]]; then
  run_one "$SESS/embedding_cueconflict_${MODEL}" \
    --cue-conflict "$CUE" --n-stimuli 30 --seed 0
else
  echo "WARN: cue-conflict dir missing at $CUE; skipping"
fi

echo "DONE embed $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
