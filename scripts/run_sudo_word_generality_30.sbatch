#!/bin/bash
#SBATCH --job-name=sb-words30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/words30_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/words30_%j.err

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

MODELS=(smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b qwen3-vl-8b)
WORDS=(clapher plailass procation adinefults)

# Shiple is already complete. Run the other four curated sudo words under both
# label sets, preserving the same playground path/model/order contract.
for WORD in "${WORDS[@]}"; do
  for COND in noun_label noun_label_AB; do
    echo "===== WORD: $WORD · CONDITION: $COND ====="
    for ORDER in shape_first texture_first; do
      python playgrounds/smoke_test_playground.py --resume \
        --prompt-condition "$COND" \
        --word "$WORD" \
        --order "$ORDER" \
        --n-trials 30 \
        --models "${MODELS[@]}"
    done
  done
done

echo "DONE $(date -Is)"
python playgrounds/playground_pride_debias.py
python scripts/build_playground_results_html.py
echo "Analysis + HTML rebuild finished $(date -Is)"
