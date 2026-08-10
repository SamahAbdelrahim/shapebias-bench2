#!/usr/bin/env python3
"""Summarize cue-conflict triad behavioral sessions (cc_triads / decomposition).

Stages per-set CSV symlinks so full_grid_summary.py does not merge the two
sets into one cell key, then writes summary + by-shape CSVs under results/data/.

Run:  .venv/bin/python analysis_pipe/cueconflict_summary.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION = REPO / "results" / "model.results" / "session_cueconflict_triads"
OUT_DIR = REPO / "results" / "data"
SETS = ("cc_triads", "decomposition_triads")


def main() -> int:
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    summary = REPO / "analysis_pipe" / "full_grid_summary.py"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for stim_set in SETS:
        stage = SESSION / f"_summary_{stim_set}"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        n = 0
        for path in sorted(SESSION.glob(f"*__{stim_set}__*.csv")):
            (stage / path.name).symlink_to(path.resolve())
            n += 1
        if n == 0:
            print(f"WARN: no CSVs for {stim_set}")
            shutil.rmtree(stage)
            continue
        print(f"===== {stim_set} ({n} cells) =====")
        subprocess.run(
            [
                str(py), str(summary),
                "--session", str(stage),
                "--prefix", f"cueconflict_{stim_set}",
                "--out-dir", str(OUT_DIR),
            ],
            check=True,
            cwd=str(REPO),
        )
        shutil.rmtree(stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
