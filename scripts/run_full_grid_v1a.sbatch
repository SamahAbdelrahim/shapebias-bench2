#!/bin/bash
# Full texture-grid evaluation: stimuli_texture_grid_v1, mode A (1,140 trials).
#
# One array task per model, six cells per task, both orderings per cell:
#   1,140 stimuli x 2 orders x 6 cells = 13,680 trials per model.
# The cells match the 30-set playground design (similarity / category / noun,
# each numeric and A/B), so the two scales are directly comparable.
#
# The gpu QOS caps this account at gres/gpu=4 and 4 running jobs, so %4 is the
# cluster limit rather than a throttle choice. Each cell writes its own CSV and
# is restarted with --resume, so a preempted or timed-out task loses nothing.
#
#   sbatch scripts/run_full_grid_v1a.sbatch          # all 9 models
#   sbatch --array=7 scripts/run_full_grid_v1a.sbatch  # just qwen3.5-9b
#
#SBATCH --job-name=sb-grid-v1a
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_v1a_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_v1a_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Scratch stage of the mode-A grid (see session README for how to rebuild it).
# 13,680 trials/model over NFS home is the I/O pattern that stalled oat-01 before.
GRID_PKG=stimuli_texture_grid_v1_scratch
STIM_SET=stimuli_A_auto_contrast
SESSION=results/model.results/session_full_grid_v1a
FIXED_WORD=shiple

MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)
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

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
from evaluation_pipe.eval_core import load_stimuli_grid
r = load_stimuli_grid('$GRID_PKG', '$STIM_SET')
print('grid trials:', len(r),
      '| shapes', len({x[\"stl_id\"] for x in r}),
      '| textures', len({x[\"texture_set\"] for x in r}))
"

ANY_FAIL=0

for COND in "${CONDITIONS[@]}"; do
  WORD_ARGS=()
  SUFFIX=""
  if [[ "$COND" == noun_label* ]]; then
    WORD_ARGS=(--word "$FIXED_WORD")
    SUFFIX="_${FIXED_WORD}"
  fi

  OUT="$SESSION/${MODEL}__${COND}${SUFFIX}.csv"
  # An existing CSV means a previous attempt got partway; --resume skips its rows.
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
    "${WORD_ARGS[@]}" \
    "${OUT_ARGS[@]}"
  rc=$?
  set -e
  END=$(date +%s)

  if [[ $rc -ne 0 ]]; then
    # Keep going: one bad cell should not cost the other five on this GPU.
    echo "WARN: $MODEL $COND exited $rc after $((END-START))s (continuing)"
    ANY_FAIL=1
    continue
  fi
  ROWS=$(( $(wc -l < "$OUT") - 1 ))
  echo "OK $MODEL $COND rows=$ROWS elapsed_s=$((END-START))"
done

echo "DONE $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
