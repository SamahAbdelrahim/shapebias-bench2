#!/bin/bash
# After the cue-conflict triad arrays finish: merge the per-model embedding
# JSONs and run PriDe over the logit CSVs for each of the two sets.
#
#   sbatch --dependency=afterany:<behav_job>:<emb_job> scripts/run_cueconflict_finalize.sbatch
#   # or once both arrays are done:
#   bash scripts/run_cueconflict_finalize.sbatch
#
#SBATCH --job-name=sb-ccfin
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/ccfinalize_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/ccfinalize_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1

LOGIT_SESSION=results/model.results/session_cueconflict_triads_logit

echo "===== merge per-model embedding JSONs ====="
python - <<'PY'
import json
from pathlib import Path

sess = Path("results/probe.results/session_cueconflict_triads")
for stem in ("embedding_cc_triads", "embedding_decomposition_triads"):
    merged = {"models": {}, "config": {"merged_from": []}}
    for path in sorted(sess.glob(f"{stem}_*.json")):
        data = json.loads(path.read_text())
        for k, v in (data.get("models") or {}).items():
            merged["models"][k] = v
        merged["config"]["merged_from"].append(path.name)
        if "config" in data and not merged["config"].get("n_stimuli"):
            merged["config"].update({k: data["config"][k] for k in data["config"]
                                     if k != "models"})
    out = sess / f"{stem}.json"
    if merged["models"]:
        out.write_text(json.dumps(merged, indent=2))
        print(f"wrote {out} ({len(merged['models'])} models)")
    else:
        print(f"skip {out}: no per-model files yet")
PY

echo "===== PriDe per set ====="
# full_grid_pride.py globs a session dir, so each set gets its own staging dir
# of symlinks; otherwise the two sets would collapse into one cell key.
for SET in cc_triads decomposition_triads; do
  STAGE="$LOGIT_SESSION/_pride_$SET"
  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  n=0
  for f in "$LOGIT_SESSION"/*__"$SET"__*.csv; do
    [[ -e "$f" ]] || continue
    ln -sf "$(realpath "$f")" "$STAGE/$(basename "$f")"
    n=$((n+1))
  done
  if [[ $n -eq 0 ]]; then
    echo "WARN: no logit CSVs for $SET; skipping pride"
    rm -rf "$STAGE"
    continue
  fi
  echo "----- $SET ($n cells) -----"
  python analysis_pipe/full_grid_pride.py \
    --session "$STAGE" \
    --out "results/data/cueconflict_${SET}_pride.csv"
  rm -rf "$STAGE"
done

echo "DONE cueconflict finalize $(date -Is)"
