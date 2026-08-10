#!/bin/bash
# Linda Smith probe set: generation + logit-forced for the full 14-model ladder.
#
# Matches the existing embedding protocol: n=30, seed=0 from
# previous-lit-stimuli/smith_stimuli. Six cells × both orders, same as the
# grid / cue-conflict ladder. Color match is the texture foil.
#
# Per model: 30 × 2 orders × 6 cells = 360 trials per readout, 720 total.
#
#   sbatch scripts/run_smith_ladder.sbatch
#
# Model list: scripts/model_ladder.sh
#
#SBATCH --job-name=sb-smith
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-13%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/smith_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/smith_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SMITH=previous-lit-stimuli/smith_stimuli
GEN_SESSION=results/model.results/session_smith_ladder
LOGIT_SESSION=results/model.results/session_smith_ladder_logit
FIXED_WORD=shiple
N=30
SEED=0

source scripts/model_ladder.sh
CONDITIONS=(
  no_word_similarity
  no_word_similarity_AB
  no_word_category
  no_word_category_AB
  noun_label
  noun_label_AB
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$GEN_SESSION" "$LOGIT_SESSION"

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL SMITH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ANY_FAIL=0

# swap_correct off to match grid / cue-conflict logit sessions.
for MODE in 2afc logit_forced; do
  if [[ "$MODE" == "2afc" ]]; then SESSION="$GEN_SESSION"; else SESSION="$LOGIT_SESSION"; fi

  for COND in "${CONDITIONS[@]}"; do
    WORD_ARGS=()
    SUFFIX=""
    if [[ "$COND" == noun_label* ]]; then
      WORD_ARGS=(--word "$FIXED_WORD")
      SUFFIX="_${FIXED_WORD}"
    fi

    OUT="$SESSION/${MODEL}__${COND}${SUFFIX}.csv"
    OUT_ARGS=(-o "$OUT")
    if [[ -s "$OUT" ]]; then
      OUT_ARGS=(--resume "$OUT")
      echo "----- $MODE $COND: resuming $OUT ($(( $(wc -l < "$OUT") - 1 )) rows) -----"
    else
      echo "----- $MODE $COND -> $OUT -----"
    fi

    START=$(date +%s)
    set +e
    python scripts/run_local.py \
      --models "$MODEL" \
      --ordering both \
      --prompt-condition "$COND" \
      --decision-mode "$MODE" \
      --smith-probe "$SMITH" \
      --num-stimuli "$N" \
      --seed "$SEED" \
      "${WORD_ARGS[@]}" \
      "${OUT_ARGS[@]}"
    rc=$?
    set -e
    END=$(date +%s)

    if [[ $rc -ne 0 ]]; then
      echo "WARN: $MODEL $MODE $COND exited $rc after $((END-START))s (continuing)"
      ANY_FAIL=1
      continue
    fi
    echo "OK $MODEL $MODE $COND rows=$(( $(wc -l < "$OUT") - 1 )) elapsed_s=$((END-START))"
  done
done

echo "DONE smith $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
