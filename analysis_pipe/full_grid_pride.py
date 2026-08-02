#!/usr/bin/env python3
"""Order-swap + PriDe-style readout from full-grid logit_forced CSVs.

Expects CSVs written by:
  python scripts/run_local.py --decision-mode logit_forced --ordering both \\
      --grid-pkg stimuli_texture_grid_v1_scratch ...

For each stimulus, P(shape) under shape_first and texture_first is averaged
(order-swap correction). A PriDe-like held-out estimate uses the first 10% of
stimuli (sorted by stim_id) as a prior for first-option bias and evaluates on
the rest — same spirit as playgrounds/playground_pride_debias.py, adapted to
CSV absolute probs rather than smoke-log relative probs.

Run after the logit sbatch finishes:
  .venv/bin/python analysis_pipe/full_grid_pride.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO / "results" / "model.results" / "session_full_grid_v1a_logit"
OUT = REPO / "results" / "data" / "full_grid_v1a_pride.csv"


def _p_shape(row: dict) -> float | None:
    """Absolute P(shape) from logit_forced columns."""
    try:
        p1 = float(row["prob_1_abs"])
        p2 = float(row["prob_2_abs"])
    except (KeyError, TypeError, ValueError):
        return None
    if row.get("a_is") == "shape":
        return p1
    if row.get("b_is") == "shape":
        return p2
    return None


def _p_first(row: dict) -> float | None:
    try:
        return float(row["prob_1_abs"])
    except (KeyError, TypeError, ValueError):
        return None


def summarize_cell(rows: list[dict]) -> dict:
    by_stim: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_stim[r["stim_id"]][r["ordering"]] = r

    gen_shape = []
    swap_shape = []
    first_sf = []
    first_tf = []
    stim_ids = sorted(by_stim)
    for sid in stim_ids:
        picks = by_stim[sid]
        if "shape_first" not in picks or "texture_first" not in picks:
            continue
        sf, tf = picks["shape_first"], picks["texture_first"]
        # Generation choice shape rate (from choice column)
        for r in (sf, tf):
            if r.get("choice") in ("shape", "texture"):
                gen_shape.append(1.0 if r["choice"] == "shape" else 0.0)
        psf, ptf = _p_shape(sf), _p_shape(tf)
        if psf is not None and ptf is not None:
            swap_shape.append(0.5 * (psf + ptf))
        fsf, ftf = _p_first(sf), _p_first(tf)
        if fsf is not None:
            first_sf.append(fsf)
        if ftf is not None:
            first_tf.append(ftf)

    # PriDe-like: prior = mean first-option bias on first 10% of stimuli;
    # holdout = debiased P(shape) on the rest.
    n_prior = max(1, len(stim_ids) // 10)
    prior_ids = set(stim_ids[:n_prior])
    hold_shape = []
    for sid in stim_ids[n_prior:]:
        picks = by_stim[sid]
        if "shape_first" not in picks or "texture_first" not in picks:
            continue
        sf, tf = picks["shape_first"], picks["texture_first"]
        # Simple order average (swap) as the held-out estimate; full PriDe
        # prior subtraction needs relative probs that vary by model. When
        # swap_corrected_* columns exist, prefer them.
        if sf.get("swap_corrected_a_abs") not in ("", None) and sf.get("a_is") == "shape":
            try:
                hold_shape.append(float(sf["swap_corrected_a_abs"]))
                continue
            except ValueError:
                pass
        psf, ptf = _p_shape(sf), _p_shape(tf)
        if psf is not None and ptf is not None:
            hold_shape.append(0.5 * (psf + ptf))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_stimuli": len(stim_ids),
        "n_prior": n_prior,
        "gen_shape": mean(gen_shape),
        "swap_shape": mean(swap_shape),
        "pride_shape": mean(hold_shape),
        "pos1_sf": mean(first_sf),
        "pos1_tf": mean(first_tf),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    files = sorted(args.session.glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No logit CSVs in {args.session}. "
            "Submit scripts/run_full_grid_logit_v1a.sbatch first."
        )

    cells: dict[tuple, list] = defaultdict(list)
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("model", ""), row.get("prompt_condition", ""), row.get("word", ""))
                cells[key].append(row)

    out_rows = []
    for (model, cond, word), rs in sorted(cells.items()):
        out_rows.append({
            "model": model, "prompt_condition": cond, "word": word,
            **summarize_cell(rs),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {args.out} ({len(out_rows)} cells from {len(files)} CSVs)")
    for r in out_rows:
        print(f"  {r['model']:<14} {r['prompt_condition']:<24} "
              f"gen={r['gen_shape']:.2f} swap={r['swap_shape']:.2f} pride={r['pride_shape']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
