#!/bin/bash
#SBATCH --job-name=sb-cat-rerun
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/cat_rerun_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/cat_rerun_%j.err

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
python -c "from evaluation_pipe.eval_core import PROMPT_TEMPLATES as P; print('no_word_category =', repr(P['no_word_category']))"

N=30
# Full ladder present in the category section (qwen3.5-27b is 4-bit on 1 GPU).
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3-vl-8b qwen3.5-0.8b qwen3.5-4b qwen3.5-9b qwen3.5-27b)

# Fresh run of the revised numeric no_word_category prompt. Default --out writes
# to today's session dir, keeping the July-17 old-prompt logs as the baseline.
for ORDER in shape_first texture_first; do
  echo "===== NEW no_word_category ($ORDER) ====="
  python playgrounds/smoke_test_playground.py \
    --prompt-condition no_word_category \
    --order "$ORDER" \
    --n-trials "$N" \
    --models "${MODELS[@]}"
done

echo "DONE $(date -Is)"
