#!/bin/bash
# Persist vision-encoder embeddings for every cell of the texture grid, so the
# linear-probe / read-out-power analysis can run on CPU afterwards.
#
# Answers the read-out-power objection: cosine weights all 1,152 dimensions
# equally, so it cannot distinguish "shape absent from the encoder" from "shape
# present but low-weighted". A probe can. See playgrounds/linear_probe.py.
#
# Four models carry the argument: qwen3.5-4b and qwen3.5-9b (the only clean
# within-model dissociations), qwen3-vl-8b (the model the original claim was
# built on), and smolvlm (strongest texture lean). One array task per model.
#
#   sbatch scripts/run_grid_embedding_export.sbatch
#
#SBATCH --job-name=sb-emb-export
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

MODELS=(qwen3.5-4b qwen3.5-9b qwen3-vl-8b smolvlm)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
GRID_PKG=stimuli_texture_grid_v1_scratch

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL EXPORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1,140 grid cells x 3 roles. The script hashes file contents and reuses an
# embedding when two roles resolve to the same bytes, so the actual number of
# forward passes is printed at start-up.
python playgrounds/embedding_export.py \
  --models "$MODEL" \
  --grid-pkg "$GRID_PKG" \
  --stim-set stimuli_A_auto_contrast \
  --out-dir "$OUT"

echo "DONE export $MODEL $(date -Is)"

# The probe itself is CPU-only. Run it here so results land in the same session,
# or sync $OUT back and run locally:
#   python playgrounds/linear_probe.py --emb <npz> --pixels $OUT/pixels32.npz \
#       --out results/probe.results/session_readout_power/probe_<model>_<readout>.json
for NPZ in "$OUT/${MODEL}"_*.npz; do
  [[ -e "$NPZ" ]] || continue
  BASE=$(basename "$NPZ" .npz)
  echo "===== probe $BASE ====="
  python playgrounds/linear_probe.py \
    --emb "$NPZ" \
    --pixels "$OUT/pixels32.npz" \
    --n-perm 100 \
    --out "$SESS/probe_${BASE}.json" || echo "WARN: probe failed for $BASE"
done

echo "ALL DONE $MODEL $(date -Is)"
