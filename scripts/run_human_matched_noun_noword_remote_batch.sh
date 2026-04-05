#!/usr/bin/env bash
# Paired human-matched runs: noun_label then no_word_category with the same
# stimulus shuffle as noun (via --human-matched-stim-condition noun_label).
# Default models match the first human-matched remote quintet.
#
# From repo root:
#   ./scripts/run_human_matched_noun_noword_remote_batch.sh
#
# Optional env:
#   PYTHON=...  STIM_SET=stimuli_A_auto_contrast  WORKERS=6
#   HUMAN_EVAL_SEED=model_eval  TRIAL_LIMIT=   # empty = script default (30 in run_remote)
#   MODELS="qwen3.5-9b qwen3.5-27b ..."        # space-separated keys

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
echo "Using Python: $($PY -c 'import sys; print(sys.executable)' 2>/dev/null || echo "$PY")"

STIM_SET="${STIM_SET:-stimuli_A_auto_contrast}"
WORKERS="${WORKERS:-6}"
HUMAN_EVAL_SEED="${HUMAN_EVAL_SEED:-model_eval}"
OUTDIR="$ROOT/results/model.results/human_matched"
mkdir -p "$OUTDIR"
TS="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${MODELS:-}" ]]; then
  MODELS="qwen3.5-9b qwen3.5-27b qwen3.5-35b-a3b qwen3.5-122b-a10b llama4-scout"
fi
read -r -a MODEL_LIST <<< "$MODELS"

EXTRA=()
if [[ -n "${TRIAL_LIMIT:-}" ]]; then
  EXTRA+=(--trial-limit "$TRIAL_LIMIT")
fi

for PKG in stimuli_unique_texture_per_stl_v1 stimuli_unique_texture_per_stl_v2; do
  SUF="${PKG##*_}"
  OUT_NOUN="$OUTDIR/remote_human_${SUF}_noun_${TS}_models.csv"
  OUT_NOWORD="$OUTDIR/remote_human_${SUF}_noword_${TS}_models.csv"

  echo "=== $PKG noun_label -> $OUT_NOUN ==="
  "$PY" scripts/run_remote.py \
    --eval-mode human_matched \
    --stim-pkg "$PKG" \
    --stim-set "$STIM_SET" \
    --ordering both \
    --prompt-condition noun_label \
    --human-eval-seed "$HUMAN_EVAL_SEED" \
    --models "${MODEL_LIST[@]}" \
    --workers "$WORKERS" \
    "${EXTRA[@]}" \
    -o "$OUT_NOUN"

  echo "=== $PKG no_word_category (stim shuffle = noun_label) -> $OUT_NOWORD ==="
  "$PY" scripts/run_remote.py \
    --eval-mode human_matched \
    --stim-pkg "$PKG" \
    --stim-set "$STIM_SET" \
    --ordering both \
    --prompt-condition no_word_category \
    --human-matched-stim-condition noun_label \
    --human-eval-seed "$HUMAN_EVAL_SEED" \
    --models "${MODEL_LIST[@]}" \
    --workers "$WORKERS" \
    "${EXTRA[@]}" \
    -o "$OUT_NOWORD"
done

echo "Done. Paired outputs share timestamp $TS under $OUTDIR"
