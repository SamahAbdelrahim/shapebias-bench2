#!/bin/bash
#SBATCH --job-name=sb-q35-27b
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/q35_27b_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/q35_27b_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) date=$(date -Is)"
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), 'n_gpu=', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

N=30
NEW=qwen3.5-27b
# qwen3.5-27b runs in 4-bit (nf4) on a single GPU (~16 GB); the bf16 2-GPU
# shard was numerically unstable (nan) for this Gated-DeltaNet hybrid. Passing
# the already-complete models alongside NEW lets --resume skip them and append
# only NEW to the shared logs.
NUMERIC_MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b qwen3-vl-8b qwen3.5-9b "$NEW")
AB_MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b "$NEW")

# The comparison HTML reads the July-17 shared session logs.
SESSION="results/playground.results/session_2026-07-17_farmshare"

# --- Numeric 1/2 three-way: append qwen3.5-27b to the shared logs ---
for COND in no_word_similarity no_word_category noun_label; do
  echo "===== NUMERIC CONDITION: $COND ====="
  WORD_ARGS=()
  SUFFIX=""
  if [ "$COND" = noun_label ]; then
    WORD_ARGS=(--word shiple)
    SUFFIX="_shiple"
  fi
  for ORDER in shape_first texture_first; do
    OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${SUFFIX}.txt"
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${NUMERIC_MODELS[@]}" \
      --out "$OUT"
  done
done

# --- A/B three-way: append qwen3.5-27b to the shared logs ---
for COND in no_word_similarity_AB no_word_category_AB noun_label_AB; do
  echo "===== A/B CONDITION: $COND ====="
  WORD_ARGS=()
  SUFFIX=""
  if [ "$COND" = noun_label_AB ]; then
    WORD_ARGS=(--word shiple)
    SUFFIX="_shiple"
  fi
  for ORDER in shape_first texture_first; do
    OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${SUFFIX}.txt"
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${AB_MODELS[@]}" \
      --out "$OUT"
  done
done

echo "DONE $(date -Is)"
python scripts/build_playground_results_html.py
echo "HTML rebuild finished $(date -Is)"
