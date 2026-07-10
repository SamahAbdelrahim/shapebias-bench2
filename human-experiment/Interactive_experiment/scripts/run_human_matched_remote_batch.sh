#!/usr/bin/env bash
# Run human-matched remote evaluation for all REMOTE_MODELS keys, both stimulus
# packages (v1 + v2), with explicit CSV paths under results/model.results/human_matched/.
#
# Prerequisites:
#   - Python 3 with deps: pip install -r requirements.txt (or at least Pillow python-dotenv openai)
#   - Hugging Face token in .env (HF_TOKEN / HF_API_TOKEN / HUGGING_FACE) or exported
#   - If system python is minimal: use Conda (see below) or apt install python3-pip python3-venv
#
# Usage (from repo root):
#   ./scripts/run_human_matched_remote_batch.sh
# Optional env:
#   PYTHON=python3.12  STIM_SET=stimuli_A_auto_contrast  WORKERS=6
#   MODELS="qwen3.5-9b llama4-scout"   # space-separated keys; default is ALL remote keys via loop below
#
# Interpreter resolution (if PYTHON is unset):
#   1. $CONDA_PREFIX/bin/python when conda activate has run
#   2. ~/miniconda3/envs/r-env/bin/python if that file exists
#   3. python3

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/r-env/bin/python" ]]; then
  PY="${HOME}/miniconda3/envs/r-env/bin/python"
else
  PY="python3"
fi
echo "Using Python: $($PY -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY (import check failed)")"
STIM_SET="${STIM_SET:-stimuli_A_auto_contrast}"
WORKERS="${WORKERS:-6}"
OUTDIR="$ROOT/results/model.results/human_matched"
mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Default: all registry names (must match run_remote.py --models all)
if [[ -z "${MODELS:-}" ]]; then
  MODEL_LIST=(all)
else
  read -r -a MODEL_LIST <<< "$MODELS"
fi

for PKG in stimuli_unique_texture_per_stl_v1 stimuli_unique_texture_per_stl_v2; do
  SUF="${PKG##*_}"
  OUT="$OUTDIR/remote_human_${SUF}_${TS}_models.csv"
  echo "=== $PKG -> $OUT ==="
  "$PY" scripts/run_remote.py \
    --eval-mode human_matched \
    --stim-pkg "$PKG" \
    --stim-set "$STIM_SET" \
    --ordering both \
    --models "${MODEL_LIST[@]}" \
    --workers "$WORKERS" \
    -o "$OUT"
done

echo "Done. Outputs under $OUTDIR"
