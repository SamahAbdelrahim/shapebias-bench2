#!/bin/bash
#SBATCH --job-name=sb-embed35
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# oat-01: home AFS import stalls (see MEMORY.md).
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/probe.results/embed_fill_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/probe.results/embed_fill_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/probe.results/session_2026-07-17_farmshare
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "host=$(hostname) date=$(date -Is)"
echo "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Fill the three playground models that lack July-10 encoder probes.
# Same 30 novel stimuli + Geirhos familiar-category positive control.
MODELS=(smolvlm qwen3.5-0.8b qwen3.5-4b)
SESSION=results/probe.results/session_2026-07-17_farmshare
CC=/scratch/users/samahabd/stimuli/geirhos_cueconflict

python playgrounds/embedding_robust.py \
  --models "${MODELS[@]}" \
  --n-stimuli 30 \
  --out-prefix "$SESSION/embedding_robust_fill"

python playgrounds/embedding_robust.py \
  --models "${MODELS[@]}" \
  --n-stimuli 30 \
  --cue-conflict "$CC" \
  --out-prefix "$SESSION/embedding_cueconflict_fill"

python playgrounds/embedding_readout.py \
  --models "${MODELS[@]}" \
  --n-stimuli 30 \
  --out-prefix "$SESSION/embedding_readout_fill"

python scripts/build_vision_vs_language_report.py
echo "DONE $(date -Is)"
