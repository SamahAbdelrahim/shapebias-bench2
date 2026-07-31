#!/bin/bash
#SBATCH --job-name=sb-catAB-rerun
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/catAB_rerun_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/catAB_rerun_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) date=$(date -Is)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from evaluation_pipe.eval_core import PROMPT_TEMPLATES as P; print('no_word_category_AB =', repr(P['no_word_category_AB']))"

N=30
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)

for ORDER in shape_first texture_first; do
  echo "===== NEW no_word_category_AB ($ORDER) ====="
  python playgrounds/smoke_test_playground.py \
    --prompt-condition no_word_category_AB \
    --order "$ORDER" \
    --n-trials "$N" \
    --models "${MODELS[@]}"
done

echo "DONE $(date -Is)"
