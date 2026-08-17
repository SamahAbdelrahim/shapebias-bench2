#!/usr/bin/env python3
"""Order-swap + PriDe-style readout from full-grid logit_forced CSVs.

Expects CSVs written by:
  python scripts/run_local.py --decision-mode logit_forced --ordering both \\
      --grid-pkg stimuli_texture_grid_v1_scratch ...

For each stimulus, P(shape) under shape_first and texture_first is averaged
(order-swap correction). PriDe estimates the first-option token prior on a
seeded random 10% of stimuli, divides each held-out option probability by its
token prior, renormalises, and scores the corrected choice. This is the same
correction as ``playgrounds/pride_debias.py``, adapted to the full-grid CSVs.

Every corrected estimate is written twice, because the two forms are not
interchangeable and were compared across scales once already:

  ``*_shape_rate``    fraction of units whose corrected P(shape) exceeds 0.5.
                      Comparable with a generation shape rate or a logit argmax.
  ``*_mean_p_shape``  mean corrected P(shape). Keeps confidence, and sits closer
                      to 0.5 than the rate whenever the model is undecided.

Run after the logit sbatch finishes:
  .venv/bin/python analysis_pipe/full_grid_pride.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO / "results" / "model.results" / "session_full_grid_v1a_logit"
OUT = REPO / "results" / "data" / "full_grid_v1a_pride.csv"
EPS = 1e-6
PRIOR_SEED = 0
# Below this the estimated option prior is one-sided enough that dividing by it
# amplifies whatever noise is left rather than removing a prior.
PRIOR_DEGENERATE = 0.02


def _clamp(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def _option_probs(row: dict) -> tuple[float, float] | None:
    """P(first), P(second), conditioned on one of the two option tokens."""
    try:
        p1 = float(row["prob_1_abs"])
        p2 = float(row["prob_2_abs"])
    except (KeyError, TypeError, ValueError):
        return None
    mass = p1 + p2
    if mass <= 0 or not math.isfinite(mass):
        return None
    return p1 / mass, p2 / mass


def _p_shape(row: dict) -> float | None:
    """P(shape | one of the two options) from logit_forced columns."""
    probs = _option_probs(row)
    if probs is None:
        return None
    p1, p2 = probs
    if row.get("a_is") == "shape":
        return p1
    if row.get("b_is") == "shape":
        return p2
    return None


def _p_first(row: dict) -> float | None:
    probs = _option_probs(row)
    return probs[0] if probs is not None else None


def _estimate_prior_first(pairs: list[tuple[dict, dict]]) -> float:
    """Geometric-mean option-ID prior from both content permutations."""
    log_first, log_second = [], []
    for sf, tf in pairs:
        for row in (sf, tf):
            probs = _option_probs(row)
            if probs is None:
                continue
            p1, p2 = probs
            log_first.append(math.log(_clamp(p1)))
            log_second.append(math.log(_clamp(p2)))
    if not log_first:
        return 0.5
    m_first = sum(log_first) / len(log_first)
    m_second = sum(log_second) / len(log_second)
    return math.exp(m_first) / (math.exp(m_first) + math.exp(m_second))


def _pride_p_shape(row: dict, prior_first: float) -> float | None:
    """PriDe: P(content) proportional to P(observed option ID) / prior(ID)."""
    probs = _option_probs(row)
    if probs is None:
        return None
    p1, p2 = probs
    a = _clamp(p1) / _clamp(prior_first)
    b = _clamp(p2) / _clamp(1.0 - prior_first)
    corrected_first = a / (a + b)
    if row.get("a_is") == "shape":
        return corrected_first
    if row.get("b_is") == "shape":
        return 1.0 - corrected_first
    return None


def summarize_cell(rows: list[dict]) -> dict:
    by_stim: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_stim[r["stim_id"]][r["ordering"]] = r

    argmax_shape = []
    swap_ps = []
    first_sf = []
    first_tf = []
    stim_ids = sorted(by_stim)
    for sid in stim_ids:
        picks = by_stim[sid]
        if "shape_first" not in picks or "texture_first" not in picks:
            continue
        sf, tf = picks["shape_first"], picks["texture_first"]
        # ``choice`` in a logit session is the argmax of the two option
        # probabilities, not a generated answer.
        for r in (sf, tf):
            if r.get("choice") in ("shape", "texture"):
                argmax_shape.append(1.0 if r["choice"] == "shape" else 0.0)
        psf, ptf = _p_shape(sf), _p_shape(tf)
        if psf is not None and ptf is not None:
            swap_ps.append(0.5 * (psf + ptf))
        fsf, ftf = _p_first(sf), _p_first(tf)
        if fsf is not None:
            first_sf.append(fsf)
        if ftf is not None:
            first_tf.append(ftf)

    # PriDe: estimate the option-ID prior on 10% of stimuli, then correct each
    # single-order observation in the held-out 90%. The sample is random rather
    # than the head of the sorted ids, which would be a handful of shapes:
    # stim_id is "<stl_id>/<texture>" and sorts lexicographically by shape.
    n_prior = max(1, len(stim_ids) // 10)
    prior_ids = set(random.Random(PRIOR_SEED).sample(stim_ids, n_prior))
    prior_pairs = []
    for sid in sorted(prior_ids):
        picks = by_stim[sid]
        if "shape_first" in picks and "texture_first" in picks:
            prior_pairs.append((picks["shape_first"], picks["texture_first"]))
    prior_first = _estimate_prior_first(prior_pairs)
    pride_sf, pride_tf = [], []
    for sid in stim_ids:
        if sid in prior_ids:
            continue
        picks = by_stim[sid]
        if "shape_first" not in picks or "texture_first" not in picks:
            continue
        sf, tf = picks["shape_first"], picks["texture_first"]
        psf = _pride_p_shape(sf, prior_first)
        ptf = _pride_p_shape(tf, prior_first)
        if psf is not None:
            pride_sf.append(psf)
        if ptf is not None:
            pride_tf.append(ptf)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def rate(xs):
        return mean([1.0 if p > 0.5 else 0.0 for p in xs])

    return {
        "n_stimuli": len(stim_ids),
        "n_prior": n_prior,
        "logit_argmax_shape": mean(argmax_shape),
        "swap_shape_rate": rate(swap_ps),
        "swap_mean_p_shape": mean(swap_ps),
        "prior_first": prior_first,
        "prior_degenerate": (prior_first < PRIOR_DEGENERATE
                             or prior_first > 1.0 - PRIOR_DEGENERATE),
        "pride_shape_rate": rate(pride_sf + pride_tf),
        "pride_sf_shape_rate": rate(pride_sf),
        "pride_tf_shape_rate": rate(pride_tf),
        "pride_mean_p_shape": mean(pride_sf + pride_tf),
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
    print("Rates are P(corrected P(shape) > 0.5); meanP keeps confidence.")
    for r in out_rows:
        flag = "  <- degenerate prior" if r["prior_degenerate"] else ""
        print(f"  {r['model']:<14} {r['prompt_condition']:<24} "
              f"argmax={r['logit_argmax_shape']:.2f} "
              f"swapRate={r['swap_shape_rate']:.2f} "
              f"prideRate={r['pride_shape_rate']:.2f} "
              f"(swapMeanP={r['swap_mean_p_shape']:.2f} "
              f"prideMeanP={r['pride_mean_p_shape']:.2f} "
              f"prior={r['prior_first']:.2f}){flag}")
    n_deg = sum(1 for r in out_rows if r["prior_degenerate"])
    if n_deg:
        print(f"{n_deg} cell(s) have an option prior within {PRIOR_DEGENERATE} of 0 or 1; "
              "their PriDe estimate divides by an almost-zero prior.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
