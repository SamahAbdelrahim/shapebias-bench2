#!/bin/bash
#SBATCH --job-name=sb-ab-bench
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_benchmark_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/jobs/ab_fix_benchmark_%j.err

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

export RESULTS_SESSION_DATE=2026-07-30
export RESULTS_SESSION_HOST=farmshare

SESSION=results/playground.results/session_2026-07-30_farmshare
N=30
source scripts/model_ladder.sh
# Sudo-word generality keeps its own smaller subset; it is not a scale ladder.
WORD_MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b qwen3-vl-8b)
WORDS=(clapher plailass procation adinefults)
ANY_FAIL=0

echo "host=$(hostname) date=$(date -Is)"
echo "SESSION=$SESSION (AB label fix rerun — Image A/B slots match choice text)"
python -c "from evaluation_pipe.eval_core import PROMPT_TEMPLATES as P; print('no_word_category_AB =', repr(P['no_word_category_AB']))"
python -c "from evaluation_pipe.models.local_models.qwen35 import Qwen35_27B as M; print('27b 4bit=', M._quantization_4bit, 'device_map=', M._device_map)"
mkdir -p "$SESSION"

run_condition() {
  local COND="$1"
  shift
  local ACTIVE_MODELS=("$@")
  # Optional trailing --word NAME after model list
  local MODEL_END=${#ACTIVE_MODELS[@]}
  for i in "${!ACTIVE_MODELS[@]}"; do
    if [[ "${ACTIVE_MODELS[$i]}" == "--word" ]]; then
      MODEL_END=$i
      break
    fi
  done
  local MODEL_LIST=("${ACTIVE_MODELS[@]:0:$MODEL_END}")
  local WORD_ARGS=()
  if (( MODEL_END < ${#ACTIVE_MODELS[@]} )); then
    WORD_ARGS=("${ACTIVE_MODELS[@]:$MODEL_END}")
  fi
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
  echo "===== BENCHMARK AB: $COND models=${MODEL_LIST[*]} ====="
  for ORDER in shape_first texture_first; do
    local OUT="$SESSION/playground_smoke_${N}trials_${ORDER}_${COND}${WORD_SUFFIX}.txt"
    echo "----- $ORDER -> $OUT -----"
    set +e
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      "${WORD_ARGS[@]}" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${MODEL_LIST[@]}" \
      --out "$OUT"
    local rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "WARN: smoke exited $rc for $COND $ORDER (continuing; check Failures in log)"
      ANY_FAIL=1
    fi
  done
}

# Three-way AB ladder (benchmark stimuli, revised category wording)
for COND in no_word_similarity_AB no_word_category_AB; do
  run_condition "$COND" "${MODELS[@]}"
done
run_condition noun_label_AB "${MODELS[@]}" --word shiple

# Sudo-word generality — AB only (numeric logs remain in Jul-17 session)
for WORD in "${WORDS[@]}"; do
  run_condition noun_label_AB "${WORD_MODELS[@]}" --word "$WORD"
done

echo "DONE benchmark AB $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
