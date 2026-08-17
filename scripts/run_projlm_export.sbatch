#!/bin/bash
# Export the LM-facing projected-token locus (proj_lm) for the full ladder.
#
# The historical export's proj_mean is byte-identical to vit_last_mean: the
# path it took returned the vision tower's last hidden state, not the
# projector output. extract_reps now takes get_image_features().pooler_output,
# which is the projected token stream the language model actually consumes
# (Qwen visual merger / InternVL multi_modal_projector / SmolVLM connector).
#
# Focal models (qwen3.5-4b, qwen3.5-9b) also save 32 proj_lm tokens per
# reference image for concept-erasure fitting (playgrounds/intervene.py).
#
# After the GPU export, the probe ladder and per-triad margins for the new
# locus run on the same node (CPU only).
#
#   sbatch scripts/run_projlm_export.sbatch
#   sbatch --array=11 scripts/run_projlm_export.sbatch   # one model
#
#SBATCH --job-name=sb-projlm
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-13%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/projlm_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/projlm_%A_%a.err

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

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL PROJ_LM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TOKEN_ARGS=()
case "$MODEL" in
  qwen3.5-4b|qwen3.5-9b) TOKEN_ARGS=(--token-save 32) ;;
esac

if [[ -s "$OUT/${MODEL}_proj_lm_mean.npz" ]]; then
  echo "SKIP export: ${MODEL}_proj_lm_mean.npz exists"
else
  python playgrounds/embedding_export.py \
    --models "$MODEL" \
    --grid-pkg "$GRID_PKG" \
    --stim-set stimuli_A_auto_contrast \
    --out-dir "$OUT" \
    --reps proj_lm "${TOKEN_ARGS[@]}"
fi

echo "DONE export $MODEL $(date -Is)"

NPZ="$OUT/${MODEL}_proj_lm_mean.npz"
if [[ -s "$NPZ" ]]; then
  # Margins only on the GPU job. The full probe ladder (C-tuning +
  # permutations) can burn hours on a GPU node; run it separately on CPU
  # via playgrounds/linear_probe.py if a probe JSON is needed.
  python playgrounds/margin_export.py --models "$MODEL" || echo "WARN: margins failed"
fi

echo "ALL DONE $MODEL $(date -Is)"
