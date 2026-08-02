#!/bin/bash
#SBATCH --job-name=sb-shiple30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/shiple30_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/shiple30_%j.err

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
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Same playground contract as the two completed n=30 conditions:
# 6 models, 30 trials, both orders, two_pass + one_pass, shared local system prompt.
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b)
for ORDER in shape_first texture_first; do
  python playgrounds/smoke_test_playground.py --resume \
    --prompt-condition noun_label_AB \
    --word shiple \
    --order "$ORDER" \
    --n-trials 30 \
    --models "${MODELS[@]}"
done

echo "DONE $(date -Is)"
python scripts/build_playground_results_html.py
echo "HTML rebuild finished $(date -Is)"
