#!/bin/bash
#SBATCH --job-name=sb-smoke-dual
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# oat-01: home AFS import stalls (jobs 1645294, 1645299). Successful smoke used oat-02.
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/smoke_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/smoke_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results
export PYTHONUNBUFFERED=1

# Home AFS HF cache is pathologically slow for weight loads (job 1645294: ~1–2
# tensors/s on SmolVLM). Scratch already has the smoke models.
export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) date=$(date -Is)"
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Dual-path smoke: shape_first then texture_first (option 1C + 2C)
# Default out path includes PLAYGROUND_PROMPT_CONDITION (no_word_category_AB).
python playgrounds/smoke_test_playground.py --resume --order shape_first --n-trials 5
python playgrounds/smoke_test_playground.py --resume --order texture_first --n-trials 5

echo "DONE $(date -Is)"
