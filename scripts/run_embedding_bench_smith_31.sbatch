#!/bin/bash
#SBATCH --job-name=sb-embed-fill
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/embed_bench_smith_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/jobs/embed_bench_smith_%j.err

# Fill missing vision-tower embeddings for fig 7b (benchmark vs Smith):
#   1. qwen3.5-9b + 27b on novel/benchmark stimuli (IMAGE_DATASET)
#   2. full 9-model ladder on Linda Smith probe set
set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/probe.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export IMAGE_DATASET=stimuli_pipe/stimuli_per_stl_packages/stimuli_A_auto_contrast

SESSION=results/probe.results/session_2026-07-31_farmshare
SMITH_DIR=previous-lit-stimuli/smith_stimuli
FILL_MODELS=(qwen3.5-9b qwen3.5-27b)
LADDER=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)

echo "host=$(hostname) date=$(date -Is)"
echo "SESSION=$SESSION IMAGE_DATASET=$IMAGE_DATASET SMITH=$SMITH_DIR"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from evaluation_pipe.models.local_models.qwen35 import Qwen35_27B as M; print('27b 4bit=', M._quantization_4bit, 'device_map=', M._device_map)"
test -d "$SMITH_DIR"
mkdir -p "$SESSION"

echo "===== 1/2 benchmark fill: ${FILL_MODELS[*]} ====="
python playgrounds/embedding_robust.py \
  --models "${FILL_MODELS[@]}" \
  --n-stimuli 30 \
  --out-prefix "$SESSION/embedding_robust_fill_large"

echo "===== 2/2 Smith probe: ${LADDER[*]} ====="
python playgrounds/embedding_robust.py \
  --models "${LADDER[@]}" \
  --n-stimuli 30 \
  --smith-probe "$SMITH_DIR" \
  --out-prefix "$SESSION/embedding_smith_probe"

echo "DONE $(date -Is)"
echo "Rerun: python analysis_pipe/playground_figures.py"
