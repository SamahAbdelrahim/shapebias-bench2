#!/bin/bash
#SBATCH --job-name=sb-smoke-dual
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/playground.results/smoke_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/playground.results/smoke_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/playground.results

echo "host=$(hostname) date=$(date -Is)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Dual-path smoke: shape_first then texture_first (option 1C + 2C)
python playgrounds/smoke_test_playground.py --order shape_first --n-trials 5
python playgrounds/smoke_test_playground.py --order texture_first --n-trials 5

echo "DONE $(date -Is)"
