#!/bin/bash
# Behavioral evaluation on the two edited Geirhos cue-conflict triad sets.
#
# Both sets share the same 1,280 reference images and differ only in what the
# two answer options are:
#   cc_triads             stylized foils; all three images isolated on white
#   decomposition_triads  the two source photographs the reference was built from
# The decomposition set carries a low-level confound: reference and shape_match
# are both isolated objects on white while texture_match is a cluttered
# photograph, so a model preferring the matching image format looks shape-biased
# for reasons unrelated to shape. Treat cc_triads as the primary measure.
#
# Both run on the same fixed 320-triad subset (20 per Geirhos class, seed 0,
# manifest in results/data/cueconflict_triad_subset_n320.csv), so the two sets,
# the two readouts and the embedding pass are paired trial for trial. The 16
# within-class triads (dog1-dog2) in that subset are congruent rather than
# conflicting; they stay in and are identified by the stl_id / texture_set
# columns, which carry the shape and texture exemplar of every trial.
#
# Per model: 320 triads x 2 orders x 6 cells x 2 sets = 7,680 trials per
# readout, 15,360 in total.
#
#   sbatch scripts/run_cueconflict_triads.sbatch
#   sbatch --array=13 scripts/run_cueconflict_triads.sbatch   # just qwen3.5-27b
#
# Model list and array-index mapping: scripts/model_ladder.sh
#
#SBATCH --job-name=sb-cctriad
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-13%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/cctriad_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/cctriad_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Symlinks into the /scratch stage; these PNGs are not on NFS home.
TRIAD_ROOT=previous-lit-stimuli/cue-conflict
GEN_SESSION=results/model.results/session_cueconflict_triads
LOGIT_SESSION=results/model.results/session_cueconflict_triads_logit
FIXED_WORD=shiple

source scripts/model_ladder.sh
SETS=(cc_triads decomposition_triads)
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

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for S in "${SETS[@]}"; do
  python -c "
from evaluation_pipe.eval_core import load_stimuli_triad_dir
r = load_stimuli_triad_dir('$TRIAD_ROOT/${S}_n320')
print('$S:', len(r), 'triads |',
      len({x['stl_id'] for x in r}), 'shape ids |',
      len({x['texture_set'] for x in r}), 'texture ids')
"
done

ANY_FAIL=0

# Generation first, then logit-forced over the same cells, so a task that hits
# the wall clock resumes into whichever readout it was partway through.
# swap_correct stays off to match session_full_grid_v1a_logit.
for MODE in 2afc logit_forced; do
  if [[ "$MODE" == "2afc" ]]; then SESSION="$GEN_SESSION"; else SESSION="$LOGIT_SESSION"; fi

  for S in "${SETS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
      WORD_ARGS=()
      SUFFIX=""
      if [[ "$COND" == noun_label* ]]; then
        WORD_ARGS=(--word "$FIXED_WORD")
        SUFFIX="_${FIXED_WORD}"
      fi

      OUT="$SESSION/${MODEL}__${S}__${COND}${SUFFIX}.csv"
      OUT_ARGS=(-o "$OUT")
      if [[ -s "$OUT" ]]; then
        OUT_ARGS=(--resume "$OUT")
        echo "----- $MODE $S $COND: resuming $OUT ($(( $(wc -l < "$OUT") - 1 )) rows) -----"
      else
        echo "----- $MODE $S $COND -> $OUT -----"
      fi

      START=$(date +%s)
      set +e
      python scripts/run_local.py \
        --models "$MODEL" \
        --ordering both \
        --prompt-condition "$COND" \
        --decision-mode "$MODE" \
        --triad-dir "$TRIAD_ROOT/${S}_n320" \
        "${WORD_ARGS[@]}" \
        "${OUT_ARGS[@]}"
      rc=$?
      set -e
      END=$(date +%s)

      if [[ $rc -ne 0 ]]; then
        # Keep going: one bad cell should not cost the other 23 on this GPU.
        echo "WARN: $MODEL $MODE $S $COND exited $rc after $((END-START))s (continuing)"
        ANY_FAIL=1
        continue
      fi
      echo "OK $MODEL $MODE $S $COND rows=$(( $(wc -l < "$OUT") - 1 )) elapsed_s=$((END-START))"
    done
  done
done

echo "DONE $MODEL $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
