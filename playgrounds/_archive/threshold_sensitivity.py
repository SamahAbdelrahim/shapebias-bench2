"""Gate-threshold sensitivity analysis for the image-tracking validity gate.

The validity gate requires image tracking >= 0.70 before any shape rate is
interpreted (see analysis_pipe/src/validity_gates.R and REPORT.md section 3).
0.70 is a researcher degree of freedom: this script asks whether the headline
conclusions change as the threshold moves.

Inputs
------
1. results/probe.results/session_2026-07-10_farmshare/probe_scaling_noun.json
   Per-trial data (30 stimuli x both orders) for the 7-model scaling ladder,
   noun_label condition, numeric and A/B label sets. Bootstrap over stimuli.
2. The 24-cell probe experiment (6 models x 2 conditions x 2 label sets).
   Per-trial JSON was written to FarmShare scratch and is not in this checkout,
   so cell-level metrics are embedded below as recorded in
   farmshare/probe-experiment-results.canvas.tsx (source of record:
   results/probe_experiment.txt on FarmShare). Threshold sweep on point
   estimates only for these cells; no bootstrap.

Outputs (results/probe.results/analysis/)
-----------------------------------------
threshold_sensitivity_cells.csv   per cell: tracking, Wilson 95% CI, shape rates,
                                  pass/fail at each threshold, bootstrap P(pass)
threshold_sensitivity_summary.csv per threshold: n cells passing, which cells,
                                  min/max shape rate among passing cells
threshold_sensitivity.md          narrative summary
"""

import csv
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION = REPO / "results/probe.results/session_2026-07-10_farmshare"
OUT = REPO / "results/probe.results/analysis"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
N_BOOT = 5000
random.seed(7)

# 24-cell probe experiment, cell-level metrics transcribed from the canvas
# (model, condition, label_set, parse, gen_tracking, log_tracking,
#  gen_shape, swap_logit_shape, gen_first_rate)
PROBE_CELLS = [
    ("smolvlm", "noun_label", "numeric", 1.0, 0.0, 0.0, 0.5, 0.49, 1.0),
    ("smolvlm", "noun_label", "AB", 1.0, 0.03, 0.0, 0.52, 0.51, 0.02),
    ("smolvlm", "no_word", "numeric", 1.0, 0.07, 0.0, 0.5, 0.5, 0.07),
    ("smolvlm", "no_word", "AB", 1.0, 0.0, 0.0, 0.5, 0.49, 1.0),
    ("internvl", "noun_label", "numeric", 1.0, 0.0, 0.1, 0.5, 0.47, 1.0),
    ("internvl", "noun_label", "AB", 1.0, 0.0, 0.0, 0.5, 0.49, 1.0),
    ("internvl", "no_word", "numeric", 1.0, 0.07, 0.03, 0.5, 0.5, 0.03),
    ("internvl", "no_word", "AB", 1.0, 0.07, 0.2, 0.47, 0.49, 0.97),
    ("qwen3-vl-2b", "noun_label", "numeric", 1.0, 0.53, 0.57, 0.47, 0.51, 0.43),
    ("qwen3-vl-2b", "noun_label", "AB", 1.0, 0.0, 0.03, 0.5, 0.51, 1.0),
    ("qwen3-vl-2b", "no_word", "numeric", 1.0, 0.07, 0.0, 0.53, 0.51, 0.03),
    ("qwen3-vl-2b", "no_word", "AB", 1.0, 0.0, 0.0, 0.5, 0.48, 1.0),
    ("qwen3-vl-4b", "noun_label", "numeric", 1.0, 0.63, 0.63, 0.72, 0.74, 0.68),
    ("qwen3-vl-4b", "noun_label", "AB", 1.0, 0.3, 0.3, 0.65, 0.68, 0.85),
    ("qwen3-vl-4b", "no_word", "numeric", 1.0, 0.1, 0.07, 0.52, 0.5, 0.05),
    ("qwen3-vl-4b", "no_word", "AB", 1.0, 0.07, 0.07, 0.53, 0.55, 0.03),
    ("qwen3.5-0.8b", "noun_label", "numeric", 1.0, 0.0, 0.0, 0.5, 0.54, 1.0),
    ("qwen3.5-0.8b", "noun_label", "AB", 1.0, 0.0, 0.0, 0.5, 0.52, 1.0),
    ("qwen3.5-0.8b", "no_word", "numeric", 1.0, 0.23, 0.7, 0.62, 0.56, 0.12),
    ("qwen3.5-0.8b", "no_word", "AB", 1.0, 0.3, 0.7, 0.65, 0.53, 0.25),
    ("qwen3.5-4b", "noun_label", "numeric", 1.0, 0.77, 0.0, 0.82, 0.5, 0.62),
    ("qwen3.5-4b", "noun_label", "AB", 1.0, 0.13, 0.0, 0.57, 0.51, 0.93),
    ("qwen3.5-4b", "no_word", "numeric", 1.0, 0.13, 0.0, 0.57, 0.5, 0.07),
    ("qwen3.5-4b", "no_word", "AB", 1.0, 0.03, 0.0, 0.52, 0.5, 0.02),
]


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_scaling_cells():
    data = json.loads((SESSION / "probe_scaling_noun.json").read_text())
    cells = []
    for model, blocks in data["models"].items():
        for block_key, block in blocks.items():
            condition, label_set = block_key.split("|")
            per_stim = block["per_stim"]
            tracked = [int(s["sf_gen_shape"] == s["tf_gen_shape"]) for s in per_stim]
            shape_sw = [
                (int(s["sf_gen_shape"]) + int(s["tf_gen_shape"])) / 2 for s in per_stim
            ]
            logit_sw = [(s["sf_log_shape"] + s["tf_log_shape"]) / 2 for s in per_stim]
            cells.append(
                {
                    "run": "scaling_2026-07-10",
                    "model": model,
                    "condition": condition,
                    "label_set": label_set,
                    "n": len(per_stim),
                    "tracked": tracked,
                    "shape_sw": shape_sw,
                    "logit_sw": logit_sw,
                    "metrics": block["metrics"],
                }
            )
    return cells


def bootstrap_pass_prob(tracked, thr):
    n = len(tracked)
    hits = 0
    for _ in range(N_BOOT):
        sample = [tracked[random.randrange(n)] for _ in range(n)]
        if sum(sample) / n >= thr:
            hits += 1
    return hits / N_BOOT


def main():
    scaling = load_scaling_cells()

    rows = []
    for c in scaling:
        n = c["n"]
        k = sum(c["tracked"])
        trk = k / n
        lo, hi = wilson_ci(k, n)
        row = {
            "run": c["run"],
            "model": c["model"],
            "condition": c["condition"],
            "label_set": c["label_set"],
            "n": n,
            "gen_tracking": round(trk, 3),
            "tracking_ci_lo": round(lo, 3),
            "tracking_ci_hi": round(hi, 3),
            "gen_shape_swapavg": round(sum(c["shape_sw"]) / n, 3),
            "logit_shape_swapavg": round(sum(c["logit_sw"]) / n, 3),
        }
        for thr in THRESHOLDS:
            row[f"pass_{thr:.2f}"] = int(trk >= thr)
            row[f"bootP_{thr:.2f}"] = round(bootstrap_pass_prob(c["tracked"], thr), 3)
        rows.append(row)

    for (model, condition, label_set, parse, gtrk, ltrk, gshp, lshp, gfirst) in PROBE_CELLS:
        k = round(gtrk * 30)
        lo, hi = wilson_ci(k, 30)
        row = {
            "run": "probe_2026-07-10",
            "model": model,
            "condition": condition,
            "label_set": label_set,
            "n": 30,
            "gen_tracking": gtrk,
            "tracking_ci_lo": round(lo, 3),
            "tracking_ci_hi": round(hi, 3),
            "gen_shape_swapavg": gshp,
            "logit_shape_swapavg": lshp,
        }
        for thr in THRESHOLDS:
            row[f"pass_{thr:.2f}"] = int(gtrk >= thr)
            row[f"bootP_{thr:.2f}"] = ""  # no per-trial data in this checkout
        rows.append(row)

    cells_path = OUT / "threshold_sensitivity_cells.csv"
    with cells_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-threshold summary over all cells
    summary = []
    for thr in THRESHOLDS:
        passing = [r for r in rows if r[f"pass_{thr:.2f}"] == 1]
        ids = "; ".join(
            f"{r['model']}|{r['condition']}|{r['label_set']}|{r['run'].split('_')[0]}"
            for r in passing
        )
        shp = [r["gen_shape_swapavg"] for r in passing]
        summary.append(
            {
                "threshold": thr,
                "n_cells_passing": len(passing),
                "n_cells_total": len(rows),
                "min_shape_among_passing": min(shp) if shp else "",
                "max_shape_among_passing": max(shp) if shp else "",
                "passing_cells": ids,
            }
        )
    summary_path = OUT / "threshold_sensitivity_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Narrative
    lines = [
        "# Gate-threshold sensitivity analysis",
        "",
        "Question: do the headline conclusions depend on the 0.70 image-tracking",
        "threshold? Sweep 0.50 to 0.90 over all 38 cells (24 probe cells, cell-level",
        "only; 14 scaling cells, per-trial bootstrap, 5000 resamples, n=30 stimuli).",
        "",
        "## Cells passing by threshold",
        "",
        "| threshold | cells passing | passing cells (model, condition, labels) |",
        "|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['threshold']:.2f} | {s['n_cells_passing']}/{s['n_cells_total']} | {s['passing_cells']} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "See threshold_sensitivity_cells.csv for per-cell Wilson CIs and bootstrap",
        "pass probabilities. Interpretation notes are written by the analyst, not",
        "this script; the script only computes.",
    ]
    (OUT / "threshold_sensitivity.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {cells_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {OUT / 'threshold_sensitivity.md'}")
    # Console: key cells at key thresholds
    for r in rows:
        if r["run"].startswith("scaling") and r["label_set"] == "numeric":
            print(
                f"{r['model']:<14} trk={r['gen_tracking']:.2f} "
                f"CI[{r['tracking_ci_lo']:.2f},{r['tracking_ci_hi']:.2f}] "
                f"shape={r['gen_shape_swapavg']:.2f} "
                f"bootP(pass@.70)={r['bootP_0.70']}"
            )


if __name__ == "__main__":
    main()
