#!/bin/bash
# After logit + embedding array jobs finish: merge embedding JSONs, run PriDe
# readout, rebuild full-grid figures and HTML.
#
#   sbatch --dependency=afterok:<logit_job>,<emb_job> scripts/run_full_grid_finalize.sbatch
#   # or run interactively once both arrays are done:
#   bash scripts/run_full_grid_finalize.sbatch
#
#SBATCH --job-name=sb-grid-fin
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_finalize_%j.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/grid_finalize_%j.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
export PYTHONUNBUFFERED=1

SESS=results/probe.results/session_full_grid_v1a

echo "===== merge per-model embedding JSONs ====="
python - <<'PY'
import json
from pathlib import Path

sess = Path("results/probe.results/session_full_grid_v1a")
for stem in ("embedding_grid", "embedding_smith", "embedding_cueconflict"):
    merged = {"models": {}, "config": {"merged_from": []}}
    for path in sorted(sess.glob(f"{stem}_*.json")):
        data = json.loads(path.read_text())
        models = data.get("models") or {}
        for k, v in models.items():
            merged["models"][k] = v
        merged["config"]["merged_from"].append(path.name)
        if "config" in data and not merged["config"].get("n_stimuli"):
            merged["config"].update({k: data["config"][k] for k in data["config"]
                                    if k not in ("models",)})
    out = sess / f"{stem.replace('embedding_', 'embedding_')}.json"
    # canonical names expected by full_grid_figures.py
    if stem == "embedding_grid":
        out = sess / "embedding_grid.json"
    elif stem == "embedding_smith":
        out = sess / "embedding_smith_probe.json"
    else:
        out = sess / "embedding_cueconflict.json"
    if merged["models"]:
        out.write_text(json.dumps(merged, indent=2))
        print(f"wrote {out} ({len(merged['models'])} models)")
    else:
        print(f"skip {out}: no per-model files yet")
PY

echo "===== PriDe from logit CSVs ====="
if ls results/model.results/session_full_grid_v1a_logit/*.csv >/dev/null 2>&1; then
  python analysis_pipe/full_grid_pride.py
else
  echo "WARN: no logit CSVs yet; skipping pride"
fi

echo "===== summary + figures + HTML ====="
python analysis_pipe/full_grid_summary.py
python analysis_pipe/full_grid_figures.py
python scripts/build_full_grid_report.py

echo "DONE finalize $(date -Is)"
