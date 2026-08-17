#!/bin/bash
set -u
cd /home/users/samahabd/shapebias-bench2
source scripts/model_ladder.sh
need=()
for i in $(seq 0 13); do
  m="${MODELS[$i]}"
  n=$(ls results/model.results/session_full_grid_v1a_blind_phase_scramble/${m}__*.csv 2>/dev/null | wc -l)
  [[ $n -lt 6 ]] && need+=($i)
done
[[ ${#need[@]} -eq 0 ]] && { echo "scramble complete"; exit 0; }
arr=$(IFS=,; echo "${need[*]}")
while true; do
  out=$(sbatch --parsable --array=${arr}%4 --export=ALL,IMAGE_MODE=phase_scramble scripts/run_full_grid_blind_v1a.sbatch 2>&1)
  if [[ $? -eq 0 ]]; then echo "SUBMITTED scramble: $out"; exit 0; fi
  [[ "$out" == *QOS* ]] || { echo "FAILED: $out"; exit 1; }
  sleep 300
done
