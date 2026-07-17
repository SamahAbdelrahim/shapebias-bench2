#!/usr/bin/env python3
"""Export session JSON results to tidy CSVs (results/probe.results/session_*/csv/)."""

import csv
import json
from pathlib import Path

SESSION = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "probe.results"
    / "session_2026-07-10_farmshare"
)
OUT = SESSION / "csv"
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def export_probe(json_path: Path, prefix: str) -> None:
    d = json.loads(json_path.read_text())
    metrics_rows, trial_rows = [], []
    for model, cells in d["models"].items():
        if not isinstance(cells, dict):
            continue
        for cell, data in cells.items():
            if not isinstance(data, dict) or "metrics" not in data:
                continue
            cond, label = cell.split("|")
            metrics_rows.append({"model": model, "condition": cond, "label_set": label, **data["metrics"]})
            for t in data.get("per_stim", []):
                trial_rows.append({"model": model, "condition": cond, "label_set": label, **t})
    write_csv(f"{prefix}_metrics.csv", metrics_rows)
    write_csv(f"{prefix}_per_trial.csv", trial_rows)


def export_embedding(json_path: Path, prefix: str) -> None:
    d = json.loads(json_path.read_text())
    rows = []
    for model, mo in d["models"].items():
        for rep, e in mo.get("reps", {}).items():
            rows.append(
                {
                    "model": model,
                    "readout": rep,
                    "dim": e["dim"],
                    "n": e["n"],
                    "shape_rate_raw": e["raw"]["shape_rate"],
                    "shape_rate_centered": e["centered"]["shape_rate"],
                    "ci95_lo": e["centered"]["ci95"][0],
                    "ci95_hi": e["centered"]["ci95"][1],
                    "margin_raw": e["raw"]["mean_margin"],
                    "margin_centered": e["centered"]["mean_margin"],
                    "retrieval_shape_at1": e.get("retrieval_shape_at1"),
                    "retrieval_texture_at1": e.get("retrieval_texture_at1"),
                }
            )
    write_csv(f"{prefix}.csv", rows)


export_probe(SESSION / "probe_experiment.json", "probe_experiment")
export_probe(SESSION / "probe_scaling_noun.json", "probe_scaling_noun")
export_embedding(SESSION / "embedding_robust.json", "embedding_robust")
export_embedding(SESSION / "embedding_cueconflict.json", "embedding_cueconflict")

# embedding_readout.json is 20 MB (contains raw embeddings) and lives on scratch only
readout = Path("/scratch/users/samahabd/sb_results/embedding_readout.json")
if readout.exists():
    d = json.loads(readout.read_text())
    rows = [
        {
            "model": m,
            "extraction": mo.get("extraction"),
            "embed_dim": mo.get("embed_dim"),
            "n": mo.get("n"),
            "shape_rate_raw": mo.get("embed_shape_rate_raw"),
            "shape_rate_centered": mo.get("embed_shape_rate_centered"),
            "margin_raw": mo.get("mean_margin_raw"),
            "margin_centered": mo.get("mean_margin_centered"),
        }
        for m, mo in d["models"].items()
        if isinstance(mo, dict) and "error" not in mo
    ]
    write_csv("embedding_readout_summary.csv", rows)
