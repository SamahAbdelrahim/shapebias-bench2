#!/bin/bash
# Persist vision-encoder embeddings for every cell of the texture grid, so the
# linear-probe / read-out-power analysis can run on CPU afterwards.
#
# Answers the read-out-power objection: cosine weights all dimensions equally,
# so it cannot distinguish "shape absent from the encoder" from "shape present
# but low-weighted". A probe can. See playgrounds/linear_probe.py.
#
# Full 14-model ladder (scripts/model_ladder.sh). Models that already have
# ${MODEL}_*.npz under the out-dir are skipped (the original 4-model pass).
#
#   sbatch scripts/run_grid_embedding_export.sbatch
#   sbatch --array=3 scripts/run_grid_embedding_export.sbatch  # one model
#
#SBATCH --job-name=sb-emb-export
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-13%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/emb_export_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/emb_export_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SESS=results/probe.results/session_readout_power
OUT="$SESS/embeddings_grid"
mkdir -p results/probe.results/jobs "$OUT"

source scripts/model_ladder.sh
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
GRID_PKG=stimuli_texture_grid_v1_scratch

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL EXPORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Skip GPU export when this model already has at least one readout NPZ
# (the original 4-model pass wrote proj_mean / vit_* for those rungs).
EXISTING=( "$OUT/${MODEL}_"*.npz )
if [[ -e "${EXISTING[0]}" ]]; then
  echo "SKIP export: ${MODEL} already has NPZ under $OUT"
else
  # 1,140 grid cells x 3 roles; content-hash dedupe reduces unique forwards.
  python playgrounds/embedding_export.py \
    --models "$MODEL" \
    --grid-pkg "$GRID_PKG" \
    --stim-set stimuli_A_auto_contrast \
    --out-dir "$OUT"
fi

echo "DONE export $MODEL $(date -Is)"

# Probe is CPU-only. Re-run for any NPZ that lacks a matching probe JSON.
for NPZ in "$OUT/${MODEL}"_*.npz; do
  [[ -e "$NPZ" ]] || continue
  BASE=$(basename "$NPZ" .npz)
  PROBE_JSON="$SESS/probe_${BASE}.json"
  if [[ -s "$PROBE_JSON" ]]; then
    echo "SKIP probe: $PROBE_JSON exists"
    continue
  fi
  echo "===== probe $BASE ====="
  python playgrounds/linear_probe.py \
    --emb "$NPZ" \
    --pixels "$OUT/pixels32.npz" \
    --n-perm 100 \
    --out "$PROBE_JSON" || echo "WARN: probe failed for $BASE"
done

echo "ALL DONE $MODEL $(date -Is)"
