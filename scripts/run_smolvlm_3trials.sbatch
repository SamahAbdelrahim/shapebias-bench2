#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=smol3
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/smolvlm_3trials.log
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/smolvlm_3trials.log

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
echo "HOST=$(hostname) START=$(date)" | tee /home/users/samahabd/shapebias-bench2/results/smolvlm_3trials.progress
module load cuda/12.9.0 || true
echo "after module" | tee -a results/smolvlm_3trials.progress
source .venv/bin/activate
echo "after venv python=$(which python)" | tee -a results/smolvlm_3trials.progress
export IMAGE_DATASET=stimuli_pipe/stimuli_per_stl_packages/stimuli_A_auto_contrast
export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PLAYGROUND_NUM_TRIALS=3
export PLAYGROUND_MODEL=smolvlm
export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
unset HF_TOKEN HF_API_TOKEN || true
nvidia-smi -L | tee -a results/smolvlm_3trials.progress
echo "importing torch..." | tee -a results/smolvlm_3trials.progress
python -u -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)'
echo "running smoke..." | tee -a results/smolvlm_3trials.progress
python -u scripts/run_local_playground_smoke.py
echo "END=$(date)" | tee -a results/smolvlm_3trials.progress
