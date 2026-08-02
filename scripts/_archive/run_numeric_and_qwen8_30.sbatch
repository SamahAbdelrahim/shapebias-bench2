#!/bin/bash
#SBATCH --job-name=sb-num-q8
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/numeric_q8_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/numeric_q8_%j.err

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
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

N=30
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b qwen3-vl-8b)
SESSION="results/playground.results/session_$(date +%F)_farmshare"
mkdir -p "$SESSION"

# Numeric 1/2 three-way on all seven models.
for COND in no_word_similarity no_word_category noun_label; do
  echo "===== NUMERIC CONDITION: $COND ====="
  WORD_ARGS=()
  if [ "$COND" = noun_label ]; then
    WORD_ARGS=(--word shiple)
  fi
  for ORDER in shape_first texture_first; do
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${MODELS[@]}"
  done
done

# Fill only qwen3-vl-8b's missing A/B cells; the other six models are complete.
for COND in no_word_similarity_AB no_word_category_AB noun_label_AB; do
  echo "===== QWEN8 A/B CONDITION: $COND ====="
  WORD_ARGS=()
  if [ "$COND" = noun_label_AB ]; then
    WORD_ARGS=(--word shiple)
  fi
  for ORDER in shape_first texture_first; do
    WORD_SUFFIX=""
    if [ "$COND" = noun_label_AB ]; then
      WORD_SUFFIX="_shiple"
    fi
    OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${WORD_SUFFIX}_qwen3-vl-8b.txt"
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models qwen3-vl-8b \
      --out "$OUT"
  done
done

echo "DONE $(date -Is)"
python scripts/build_playground_results_html.py
echo "HTML rebuild finished $(date -Is)"
