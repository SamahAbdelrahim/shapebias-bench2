#!/bin/bash
set -euo pipefail

# ── Load .env from repo root ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# ── Configuration ──
MODELS="smolvlm internvl qwen3-vl-2b qwen3-vl-4b qwen3.5-0.8b qwen3.5-4b" # space-separated model names to evaluate (use "all" for all registered models)
REPEATS=1 # number of passses through the data for each ordering
TEMPERATURE=0.0 # 0 is deterministic, higher values add more randomness
RESULTS_DIR="${RESULTS_DIR:-results}" # from .env or default to "results"
OUTPUT="${RESULTS_DIR}/data/local_eval.csv" # output file for results

# Fresh output file
rm -f "$OUTPUT"

# ── Run shape_first + texture_first (via --ordering both) ──
echo ""
echo "========================================================"
echo "  Run: ordering=both  models=$MODELS  repeats=$REPEATS  temp=$TEMPERATURE"
echo "========================================================"
echo ""
conda run --no-capture-output -n hackathon python scripts/run_local.py --models $MODELS --ordering both --repeats "$REPEATS" --temperature "$TEMPERATURE" -o "$OUTPUT"

# ── Run random ordering ──
echo ""
echo "========================================================"
echo "  Run: ordering=random  models=$MODELS  repeats=$REPEATS  temp=$TEMPERATURE"
echo "========================================================"
echo ""
conda run --no-capture-output -n hackathon python scripts/run_local.py --models $MODELS --ordering random --repeats "$REPEATS" --temperature "$TEMPERATURE" -o "$OUTPUT"

echo ""
echo "Done. Results: $OUTPUT"
