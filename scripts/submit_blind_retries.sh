#!/bin/bash
# Resubmit the three blind tasks that failed on a transient cache error,
# retrying while the gpu QOS submit cap is full.
set -u
cd /home/users/samahabd/shapebias-bench2
submit() {
  local desc="$1"; shift
  while true; do
    out=$("$@" 2>&1) && { echo "SUBMITTED $desc: $out"; return 0; }
    [[ "$out" == *QOS* ]] || { echo "FAILED $desc: $out"; return 1; }
    sleep 300
  done
}
submit "blank retry (2b,9b)" sbatch --parsable --array=10,12 scripts/run_full_grid_blind_v1a.sbatch
submit "scramble retry (2b)" sbatch --parsable --array=10 --export=ALL,IMAGE_MODE=phase_scramble scripts/run_full_grid_blind_v1a.sbatch
echo "ALL_RETRIES_SUBMITTED"
