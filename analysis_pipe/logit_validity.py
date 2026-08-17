#!/usr/bin/env python3
"""Validity gating on the logit path, computed the same way as on the generation path.

``full_grid_summary.py`` gates cells on generation-path tracking, and
``full_grid_pride.py`` reports swap / PriDe shape rates without asking whether the
logit path passes a gate of its own. So a cell can be called invalid from its
generated text while its option probabilities still track image content, and
nothing in the pipeline says so.

This script recomputes, per model x prompt cell, from
``session_full_grid_v1a_logit``:

  logit_tracking   fraction of stimuli whose forced-choice argmax is unchanged
                   when the two options swap position. Same definition as
                   ``summarize_cell`` in full_grid_summary.py, applied to the
                   logit decision instead of the parsed generation.
  logit_shape      shape rate from the logit argmax, both orders pooled.
  swap_shape_rate  fraction of stimuli whose swap-averaged P(shape) exceeds 0.5.
                   This is the column to compare against logit_shape or against
                   a generation shape rate, since all three are decision rates.
  swap_mean_p_shape
                   mean swap-averaged P(shape). Keeps confidence, so it sits
                   nearer 0.5 than the rate whenever the model is undecided;
                   the two are not interchangeable.
  margin           mean |P(shape) - 0.5|, how far from indifferent the two
                   option probabilities are.
  option_mass      mean of prob_1_abs + prob_2_abs, the total probability the
                   model puts on the two option tokens.
  degenerate_rate  fraction of trials whose two option probabilities are
                   saturated (one of them within 1e-6 of 0).

Both swap columns divide by the option mass, so they read P(shape | the answer is
one of the two options), which is what a forced choice asks. Averaging the
unnormalised probabilities instead pulls a model that puts little mass on either
option token toward 0 whatever its preference; that is the source of the
exactly-0.000 SmolVLM values flagged in REPORT.md section 7, and ``option_mass``
is the column that shows it.

Output: results/data/full_grid_logit_validity.csv

Run:  .venv/bin/python analysis_pipe/logit_validity.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO / "results" / "model.results" / "session_full_grid_v1a_logit"
DEFAULT_OUT = REPO / "results" / "data" / "full_grid_logit_validity.csv"
TRACKING_GATE = 0.70
DECIDED = ("shape", "texture")
SATURATED = 1e-6


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _p_shape(row: dict) -> float | None:
    """P(shape match) for one trial from the absolute option probabilities."""
    try:
        p1 = float(row["prob_1_abs"])
        p2 = float(row["prob_2_abs"])
    except (KeyError, TypeError, ValueError):
        return None
    total = p1 + p2
    if total <= 0:
        return None
    # a_is names the content shown in option 1, so option 1 is the shape match
    # only when a_is == "shape".
    p_first = p1 / total
    return p_first if row.get("a_is") == "shape" else 1.0 - p_first


def summarize_cell(rows: list[dict]) -> dict:
    decided = [r for r in rows if r.get("choice") in DECIDED]

    by_stim: dict[str, dict[str, str]] = defaultdict(dict)
    for r in decided:
        by_stim[r.get("stim_id", "")][r.get("ordering", "")] = r["choice"]
    track_vals = [
        1.0 if picks["shape_first"] == picks["texture_first"] else 0.0
        for picks in by_stim.values()
        if "shape_first" in picks and "texture_first" in picks
    ]
    tracking = _mean(track_vals)

    n_shape = sum(1 for r in decided if r["choice"] == "shape")
    logit_shape = n_shape / len(decided) if decided else float("nan")
    lo, hi = _wilson_ci(n_shape, len(decided))
    pos_first = _mean([1.0 if r["choice"] == r.get("a_is") else 0.0 for r in decided])

    # Continuous swap correction: average each stimulus with its position-flipped
    # twin before averaging over stimuli, so a side preference cancels.
    p_by_stim: dict[str, dict[str, float]] = defaultdict(dict)
    margins: list[float] = []
    masses: list[float] = []
    n_saturated = 0
    for r in rows:
        p = _p_shape(r)
        if p is None:
            continue
        margins.append(abs(p - 0.5))
        try:
            p1, p2 = float(r["prob_1_abs"]), float(r["prob_2_abs"])
            masses.append(p1 + p2)
            if min(p1, p2) <= SATURATED:
                n_saturated += 1
        except (KeyError, TypeError, ValueError):
            pass
        p_by_stim[r.get("stim_id", "")][r.get("ordering", "")] = p
    swap_vals = [
        0.5 * (ps["shape_first"] + ps["texture_first"])
        for ps in p_by_stim.values()
        if "shape_first" in ps and "texture_first" in ps
    ]

    return {
        "n_rows": len(rows),
        "n_decided": len(decided),
        "n_stimuli": len(by_stim),
        "n_paired": len(track_vals),
        "logit_tracking": tracking,
        "logit_gate_pass": (not math.isnan(tracking)) and tracking >= TRACKING_GATE,
        "logit_shape": logit_shape,
        "logit_shape_ci_lo": lo,
        "logit_shape_ci_hi": hi,
        "swap_shape_rate": _mean([1.0 if p > 0.5 else 0.0 for p in swap_vals]),
        "swap_mean_p_shape": _mean(swap_vals),
        "logit_pos_first": pos_first,
        "margin": _mean(margins),
        "option_mass": _mean(masses),
        "degenerate_rate": n_saturated / len(rows) if rows else float("nan"),
    }


def read_session(session_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(session_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {session_dir}")
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = read_session(args.session)
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r.get("model", ""), r.get("prompt_condition", ""), r.get("word", ""))].append(r)

    out = []
    for (model, cond, word), rs in sorted(cells.items()):
        out.append({"model": model, "prompt_condition": cond, "word": word,
                    **summarize_cell(rs)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    n_pass = sum(1 for r in out if r["logit_gate_pass"])
    print(f"Logit-path validity: {len(out)} cells, {n_pass} pass tracking >= {TRACKING_GATE}")
    flagged = [r for r in out if r["degenerate_rate"] > 0.5]
    if flagged:
        print(f"  {len(flagged)} cells with >50% saturated option probabilities:")
        for r in flagged:
            print(f"    {r['model']:<15} {r['prompt_condition']:<24} "
                  f"degenerate={r['degenerate_rate']:.2f} "
                  f"swap_rate={r['swap_shape_rate']:.3f} "
                  f"swap_meanP={r['swap_mean_p_shape']:.3f}")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
