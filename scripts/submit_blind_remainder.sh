#!/bin/bash
# Submit the remaining blind-baseline arrays as QOS submit slots free up.
# Priority: scramble for focal models, then blank for the rest, then scramble rest.
set -u
cd /home/users/samahabd/shapebias-bench2

submit_when_possible() {
  local desc="$1"; shift
  while true; do
    out=$("$@" 2>&1)
    if [[ $? -eq 0 ]]; then
      echo "SUBMITTED $desc: $out"
      return 0
    fi
    if [[ "$out" != *QOS* ]]; then
      echo "FAILED $desc: $out"
      return 1
    fi
    sleep 300
  done
}

submit_when_possible "scramble qwen3.5 (9-13)" \
  sbatch --parsable --array=9-13%4 --export=ALL,IMAGE_MODE=phase_scramble scripts/run_full_grid_blind_v1a.sbatch
submit_when_possible "blank rest (0-8)" \
  sbatch --parsable --array=0-8%4 scripts/run_full_grid_blind_v1a.sbatch
submit_when_possible "scramble rest (0-8)" \
  sbatch --parsable --array=0-8%4 --export=ALL,IMAGE_MODE=phase_scramble scripts/run_full_grid_blind_v1a.sbatch
echo "ALL_BLIND_SUBMITTED"
