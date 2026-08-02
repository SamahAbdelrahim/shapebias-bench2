#!/bin/bash
#SBATCH --job-name=sb-smith30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/smith30_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/smith30_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Pin the session date so a midnight rollover cannot split one ladder.
export RESULTS_SESSION_DATE=2026-07-25
export RESULTS_SESSION_HOST=farmshare

SMITH_DIR=previous-lit-stimuli/smith_stimuli
SESSION=results/playground.results/session_2026-07-25_smith_farmshare
N=30
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)

echo "host=$(hostname) date=$(date -Is)"
echo "SESSION=$SESSION RESULTS_SESSION_DATE=$RESULTS_SESSION_DATE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
mkdir -p "$SESSION"

run_condition() {
  local COND="$1"
  shift
  local WORD_ARGS=("$@")
  local WORD_SUFFIX=""
  if [[ " ${WORD_ARGS[*]} " == *" --word "* ]]; then
    # --word <name> sits after --word in WORD_ARGS
    local prev=""
    for arg in "${WORD_ARGS[@]}"; do
      if [[ "$prev" == "--word" ]]; then
        WORD_SUFFIX="_${arg}"
        break
      fi
      prev="$arg"
    done
  fi
  echo "===== SMITH CONDITION: $COND ====="
  for ORDER in shape_first texture_first; do
    local OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${WORD_SUFFIX}.txt"
    echo "----- $ORDER -> $OUT -----"
    python playgrounds/smoke_test_playground.py --resume \
      --smith-stimuli "$SMITH_DIR" \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${MODELS[@]}" \
      --out "$OUT"
  done
}

# Numeric 1/2 ladder
for COND in no_word_similarity no_word_category; do
  run_condition "$COND"
done
run_condition noun_label --word shiple

# A/B ladder
for COND in no_word_similarity_AB no_word_category_AB; do
  run_condition "$COND"
done
run_condition noun_label_AB --word shiple

echo "===== PriDe / swap on Smith logs ====="
python playgrounds/playground_pride_debias.py \
  --session "$SESSION" \
  --out-prefix results/playground.results/smith_prompt_pride_debias_2026-07-25

echo "===== Rebuild HTML ====="
python scripts/build_playground_results_html.py

echo "DONE $(date -Is)"
