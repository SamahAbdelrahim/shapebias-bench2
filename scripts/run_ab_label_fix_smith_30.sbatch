#!/bin/bash
#SBATCH --job-name=sb-ab-smith
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_smith_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_smith_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export RESULTS_SESSION_DATE=2026-07-30
export RESULTS_SESSION_HOST=farmshare

SMITH_DIR=previous-lit-stimuli/smith_stimuli
SESSION=results/playground.results/session_2026-07-30_smith_farmshare
N=30
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)
ANY_FAIL=0

echo "host=$(hostname) date=$(date -Is)"
echo "SESSION=$SESSION SMITH=$SMITH_DIR"
python -c "from evaluation_pipe.models.local_models.qwen35 import Qwen35_27B as M; print('27b 4bit=', M._quantization_4bit, 'device_map=', M._device_map)"
mkdir -p "$SESSION"

run_condition() {
  local COND="$1"
  shift
  local WORD_ARGS=("$@")
  local WORD_SUFFIX=""
  if [[ " ${WORD_ARGS[*]} " == *" --word "* ]]; then
    local prev=""
    for arg in "${WORD_ARGS[@]}"; do
      if [[ "$prev" == "--word" ]]; then
        WORD_SUFFIX="_${arg}"
        break
      fi
      prev="$arg"
    done
  fi
  echo "===== SMITH AB: $COND ====="
  for ORDER in shape_first texture_first; do
    local OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${WORD_SUFFIX}.txt"
    echo "----- $ORDER -> $OUT -----"
    set +e
    python playgrounds/smoke_test_playground.py --resume \
      --smith-stimuli "$SMITH_DIR" \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${MODELS[@]}" \
      --out "$OUT"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "WARN: smoke exited $rc for $COND $ORDER (continuing; check Failures in log)"
      ANY_FAIL=1
    fi
  done
}

for COND in no_word_similarity_AB no_word_category_AB; do
  run_condition "$COND"
done
run_condition noun_label_AB --word shiple

echo "DONE smith AB $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
