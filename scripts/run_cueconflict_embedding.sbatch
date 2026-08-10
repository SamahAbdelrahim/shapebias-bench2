#!/bin/bash
# Vision-tower embedding readout on the two edited cue-conflict triad sets.
#
# Runs the same fixed 320-triad subset the behavioral job uses, so cosine and
# behavioral measures are paired trial for trial rather than sampled apart.
# Retrieval@1 depends on gallery size, so these n=320 numbers are not comparable
# to the n=30 Smith / Geirhos runs or the n=114 grid run; the cosine shape rate
# is per-triplet and does compare.
#
# The triad builder supplies Geirhos class labels, which turns on the
# label-level control (retrieval_mesh_at1 / retrieval_texset_at1) against a
# 1/16 chance baseline.
#
# One sequential job rather than a 14-task array: a model needs only 960 forward
# passes per set here, and the gpu QOS allows 32 submitted jobs in total, which
# the behavioral array and the grid backfill already mostly spend.
#
#   sbatch scripts/run_cueconflict_embedding.sbatch
#
#SBATCH --job-name=sb-ccemb
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/ccemb_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/ccemb_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TRIAD_ROOT=previous-lit-stimuli/cue-conflict
SESS=results/probe.results/session_cueconflict_triads
N=320

source scripts/model_ladder.sh
SETS=(cc_triads decomposition_triads)
mkdir -p results/probe.results/jobs "$SESS"

echo "host=$(hostname) date=$(date -Is) models=${#MODELS[@]} EMBED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

ANY_FAIL=0
for MODEL in "${MODELS[@]}"; do
  for S in "${SETS[@]}"; do
    OUT="$SESS/embedding_${S}_${MODEL}"
    # Per-model files, and skip anything already written so a requeue is cheap.
    if [[ -s "${OUT}.json" ]]; then
      echo "===== $S / $MODEL: exists, skipping ====="
      continue
    fi
    echo "===== $S / $MODEL ====="
    START=$(date +%s)
    set +e
    python playgrounds/embedding_robust.py \
      --models "$MODEL" \
      --triad-dir "$TRIAD_ROOT/${S}_n320" \
      --n-stimuli "$N" \
      --seed 0 \
      --out-prefix "$OUT"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "WARN: $S $MODEL exited $rc after $(( $(date +%s) - START ))s (continuing)"
      ANY_FAIL=1
    else
      echo "OK $S $MODEL elapsed_s=$(( $(date +%s) - START ))"
    fi
  done
done

echo "DONE ccemb $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
