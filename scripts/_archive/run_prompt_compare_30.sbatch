#!/bin/bash
#SBATCH --job-name=sb-prompt30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# oat-01: home AFS import stalls (see MEMORY.md). Prefer other GPU nodes.
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/prompt30_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/prompt30_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results
export PYTHONUNBUFFERED=1

# Prefer scratch HF cache (home AFS weight loads are pathologically slow).
export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) date=$(date -Is)"
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Powered comparison: same 6 local models, 30 trials × 2 orders × 2 no-word AB prompts.
# System prompt: LOCAL_VLM_SYSTEM_PROMPT (logged by smoke_test_playground.py).
N=30
MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b)

for COND in no_word_similarity_AB no_word_category_AB; do
  echo "===== PROMPT CONDITION: $COND ====="
  for ORDER in shape_first texture_first; do
    python playgrounds/smoke_test_playground.py --resume \
      --prompt-condition "$COND" \
      --order "$ORDER" \
      --n-trials "$N" \
      --models "${MODELS[@]}"
  done
done

echo "DONE $(date -Is)"

# Rebuild HTML reports (n=30 page appears when both prompt logs are complete).
python scripts/build_playground_results_html.py
echo "HTML rebuild finished $(date -Is)"
