#!/usr/bin/bash
# Submit benchmark + Smith AB reruns in parallel, then postprocess when both finish.
set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
mkdir -p results/playground.results/jobs

J1=$(sbatch --parsable scripts/run_ab_label_fix_benchmark_30.sbatch)
J2=$(sbatch --parsable scripts/run_ab_label_fix_smith_30.sbatch)
J3=$(sbatch --parsable --dependency=afterok:${J1}:${J2} scripts/run_ab_label_fix_finalize.sbatch)

echo "Submitted benchmark AB rerun: job ${J1}"
echo "Submitted Smith AB rerun:     job ${J2}"
echo "Submitted finalize (after both): job ${J3}"
echo "Monitor: squeue -u \$USER"
echo "Logs: results/playground.results/jobs/ab_fix_*"
