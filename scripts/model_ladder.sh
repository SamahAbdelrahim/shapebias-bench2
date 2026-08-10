#!/bin/bash
# Single source of truth for the local model ladder. Source it, do not run it:
#
#   source scripts/model_ladder.sh
#
# Order is family-grouped and ascending in parameter count inside each family,
# so a SLURM array index maps onto a position on the scale ladder.
#
# Adding or removing an entry shifts every later array index. Each runner that
# indexes MODELS with $SLURM_ARRAY_TASK_ID must set --array to 0-N%4 where
# N = ${#MODELS[@]} - 1. Run `bash scripts/model_ladder.sh` to print the
# current index mapping.
#
# Registered in evaluation_pipe.models but deliberately not on the ladder:
#   qwen3-vl-32b     does not fit bf16 on one 48 GB L40S (MEMORY.md 2026-07-23)
#   tinyllava        not imported by local_models/__init__.py, deprecated
#   levante-runtime  delegating wrapper, not a scale rung
#
# Every model here must be present in $HF_HOME, since the runners set
# HF_HUB_OFFLINE=1 and will fail on a cache miss rather than download.

MODELS=(
  smolvlm-256m
  smolvlm
  internvl
  internvl-2b
  internvl-8b
  internvl-14b
  qwen3-vl-2b
  qwen3-vl-4b
  qwen3-vl-8b
  qwen3.5-0.8b
  qwen3.5-2b
  qwen3.5-4b
  qwen3.5-9b
  qwen3.5-27b
)

print_model_ladder() {
  for i in "${!MODELS[@]}"; do
    printf '  %2d  %s\n' "$i" "${MODELS[$i]}"
  done
}

# Executed rather than sourced: show the index mapping.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "${#MODELS[@]} models (use --array=0-$(( ${#MODELS[@]} - 1 ))%4)"
  print_model_ladder
fi
