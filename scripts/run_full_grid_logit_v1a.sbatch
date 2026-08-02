#!/bin/bash
# Logit-forced rerun of the full texture grid (v1A) for PriDe / order-swap.
# Same 9 models × 6 cells × 1,140 stimuli × both orders as the generation run.
# Writes to session_full_grid_v1a_logit/ so generation CSVs are untouched.
#
#   sbatch scripts/run_full_grid_logit_v1a.sbatch
#
#SBATCH --job-name=sb-grid-logit
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-8%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_logit_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_logit_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GRID_PKG=stimuli_texture_grid_v1_scratch
STIM_SET=stimuli_A_auto_contrast
SESSION=results/model.results/session_full_grid_v1a_logit
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

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL LOGIT"
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
    echo "----- $COND: resuming $OUT -----"
  else
    echo "----- $COND -> $OUT -----"
  fi

  set +e
  python scripts/run_local.py \
    --models "$MODEL" \
    --ordering both \
    --prompt-condition "$COND" \
    --decision-mode logit_forced \
    --grid-pkg "$GRID_PKG" \
    --stim-set "$STIM_SET" \
    "${WORD_ARGS[@]}" \
    "${OUT_ARGS[@]}"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "WARN: $MODEL $COND exited $rc"
    ANY_FAIL=1
  else
    echo "OK $MODEL $COND rows=$(( $(wc -l < "$OUT") - 1 ))"
  fi
done

echo "DONE logit $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
