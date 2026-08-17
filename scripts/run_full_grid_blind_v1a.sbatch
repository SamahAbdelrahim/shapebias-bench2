#!/bin/bash
# Blind baseline for the full texture grid: the same 6 prompt cells and both
# orderings, but every stimulus image replaced by a blind variant.
#
#   IMAGE_MODE=blank            mid-grey fields (default)
#   IMAGE_MODE=phase_scramble   amplitude-preserving phase scrambles
#
# This is the language-prior control the prompt grid lacks (Fu et al. style):
# a cell whose blind shape rate matches its sighted rate is measuring the
# language prior, not vision. Especially relevant to the AB cells, whose image
# tracking collapses in fig1.
#
#   sbatch scripts/run_full_grid_blind_v1a.sbatch                       # blank
#   IMAGE_MODE=phase_scramble sbatch --export=ALL scripts/run_full_grid_blind_v1a.sbatch
#
#SBATCH --job-name=sb-grid-blind
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-13%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_blind_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_blind_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
# Prefer the hub/ layout; older TRANSFORMERS_CACHE=$HF_HOME made some
# nodes miss snapshots that live under $HF_HOME/hub/.
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Fail fast with a clear message if a weight is missing from the shared cache.
export HF_HUB_DISABLE_TELEMETRY=1

IMAGE_MODE="${IMAGE_MODE:-blank}"
GRID_PKG=stimuli_texture_grid_v1_scratch
STIM_SET=stimuli_A_auto_contrast
SESSION="results/model.results/session_full_grid_v1a_blind_${IMAGE_MODE}"
FIXED_WORD=shiple

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
mkdir -p "$SESSION"

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL image_mode=$IMAGE_MODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ANY_FAIL=0

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
    echo "----- $COND: resuming $OUT ($(( $(wc -l < "$OUT") - 1 )) rows present) -----"
  else
    echo "----- $COND -> $OUT -----"
  fi

  START=$(date +%s)
  set +e
  python scripts/run_local.py \
    --models "$MODEL" \
    --ordering both \
    --prompt-condition "$COND" \
    --grid-pkg "$GRID_PKG" \
    --stim-set "$STIM_SET" \
    --image-mode "$IMAGE_MODE" \
    "${WORD_ARGS[@]}" \
    "${OUT_ARGS[@]}"
  rc=$?
  set -e
  END=$(date +%s)

  if [[ $rc -ne 0 ]]; then
    echo "WARN: $MODEL $COND exited $rc after $((END-START))s (continuing)"
    ANY_FAIL=1
    continue
  fi
  ROWS=$(( $(wc -l < "$OUT") - 1 ))
  echo "OK $MODEL $COND rows=$ROWS elapsed_s=$((END-START))"
done

echo "DONE $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
