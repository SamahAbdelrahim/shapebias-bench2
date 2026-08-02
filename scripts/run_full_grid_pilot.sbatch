#!/bin/bash
# Timing / plumbing pilot for the full texture grid (stimuli_texture_grid_v1, mode A).
# One fast model, two cells, a 40-stimulus subsample. Measures seconds per trial so
# scripts/run_full_grid_v1a.sbatch can be sized before committing GPU days.
#SBATCH --job-name=sb-grid-pilot
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_pilot_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_pilot_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GRID_PKG=stimuli_texture_grid_v1
STIM_SET=stimuli_A_auto_contrast
MODEL=qwen3-vl-2b
N=40
SESSION=results/model.results/session_full_grid_v1a_pilot
mkdir -p "$SESSION"

echo "host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "
from evaluation_pipe.eval_core import load_stimuli_grid
r = load_stimuli_grid('$GRID_PKG', '$STIM_SET')
print('grid trials available:', len(r),
      '| shapes', len({x[\"stl_id\"] for x in r}),
      '| textures', len({x[\"texture_set\"] for x in r}))
"

for COND in no_word_similarity noun_label; do
  WORD_ARGS=()
  SUFFIX=""
  if [[ "$COND" == noun_label* ]]; then
    WORD_ARGS=(--word shiple)
    SUFFIX="_shiple"
  fi
  OUT="$SESSION/${MODEL}__${COND}${SUFFIX}.csv"
  echo "===== PILOT $COND -> $OUT ====="
  START=$(date +%s)
  python scripts/run_local.py \
    --models "$MODEL" \
    --ordering both \
    --prompt-condition "$COND" \
    --grid-pkg "$GRID_PKG" \
    --stim-set "$STIM_SET" \
    --num-stimuli "$N" \
    "${WORD_ARGS[@]}" \
    -o "$OUT"
  END=$(date +%s)
  ROWS=$(( $(wc -l < "$OUT") - 1 ))
  echo "TIMING cond=$COND rows=$ROWS elapsed_s=$((END-START)) sec_per_trial=$(python -c "print(round($((END-START))/max($ROWS,1),3))")"
done

echo "DONE pilot $(date -Is)"
