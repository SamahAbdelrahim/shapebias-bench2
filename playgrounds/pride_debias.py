#!/usr/bin/env python3
"""Point 5 — raw vs corrected shape preference, with a PriDe-style debiaser.

Reads the per-trial logit records saved by probe_experiment.py /
probe_scaling_noun (fields sf_log_shape / tf_log_shape = P(shape) among the two
option tokens under shape-first / texture-first ordering) and reports, side by
side, for every model x condition x label-set cell:

  gen_raw_sf / gen_raw_tf : generation shape rate read from ONE order only
                            (what a naive, uncounterbalanced study reports)
  gen_swap                : generation shape rate averaged over both orders
  log_raw_sf / log_raw_tf : mean P(shape) from one order only
  log_swap                : arithmetic mean over orders (our default correction)
  log_fullperm            : full-permutation debias (Zheng et al. 2024) —
                            geometric mean of P over both content permutations
  log_pride               : PriDe (Zheng et al. 2024) — option-ID prior is
                            estimated on the first K stimuli (both orders),
                            then each remaining stimulus is debiased from a
                            SINGLE observation (P_obs / prior, renormalized).
                            Reported for sf-only and tf-only observations.

Rates are the fraction of stimuli with debiased P(shape) > 0.5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

EPS = 1e-6


def _clamp(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def fullperm_p_shape(sf_ps: float, tf_ps: float) -> float:
    """Geometric-mean debias over the two content permutations."""
    ls = 0.5 * (math.log(_clamp(sf_ps)) + math.log(_clamp(tf_ps)))
    lt = 0.5 * (math.log(_clamp(1 - sf_ps)) + math.log(_clamp(1 - tf_ps)))
    return math.exp(ls) / (math.exp(ls) + math.exp(lt))


def estimate_prior_first(trials: list[dict]) -> float:
    """PriDe prior P(first option ID), estimated over trials x permutations.

    Under sf the first option holds shape (P(first)=sf_log_shape); under tf it
    holds texture (P(first)=1-tf_log_shape).
    """
    logs = []
    for t in trials:
        logs.append(math.log(_clamp(t["sf_log_shape"])))
        logs.append(math.log(_clamp(1 - t["tf_log_shape"])))
    m_first = sum(logs) / len(logs)
    logs2 = []
    for t in trials:
        logs2.append(math.log(_clamp(1 - t["sf_log_shape"])))
        logs2.append(math.log(_clamp(t["tf_log_shape"])))
    m_second = sum(logs2) / len(logs2)
    return math.exp(m_first) / (math.exp(m_first) + math.exp(m_second))


def pride_p_shape(p_obs_first: float, shape_is_first: bool, prior_first: float) -> float:
    """Debias one single-order observation: P_deb(content) ∝ P_obs(id)/prior(id)."""
    a = _clamp(p_obs_first) / _clamp(prior_first)
    b = _clamp(1 - p_obs_first) / _clamp(1 - prior_first)
    p_first_content = a / (a + b)
    return p_first_content if shape_is_first else 1 - p_first_content


def analyze_cell(per_stim: list[dict], n_est: int) -> dict:
    n = len(per_stim)
    est, test = per_stim[:n_est], per_stim[n_est:]

    rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    gen_raw_sf = rate([t["sf_gen_shape"] for t in per_stim])
    gen_raw_tf = rate([t["tf_gen_shape"] for t in per_stim])
    gen_swap = 0.5 * (gen_raw_sf + gen_raw_tf)

    log_raw_sf = rate([t["sf_log_shape"] > 0.5 for t in per_stim])
    log_raw_tf = rate([t["tf_log_shape"] > 0.5 for t in per_stim])
    log_swap = rate(
        [0.5 * (t["sf_log_shape"] + t["tf_log_shape"]) > 0.5 for t in per_stim]
    )
    log_fullperm = rate(
        [fullperm_p_shape(t["sf_log_shape"], t["tf_log_shape"]) > 0.5 for t in per_stim]
    )

    prior_first = estimate_prior_first(est)
    pride_sf = rate(
        [pride_p_shape(t["sf_log_shape"], True, prior_first) > 0.5 for t in test]
    )
    pride_tf = rate(
        [pride_p_shape(1 - t["tf_log_shape"], False, prior_first) > 0.5 for t in test]
    )

    return {
        "n": n,
        "n_est": n_est,
        "prior_first": prior_first,
        "gen_raw_sf": gen_raw_sf,
        "gen_raw_tf": gen_raw_tf,
        "gen_swap": gen_swap,
        "log_raw_sf": log_raw_sf,
        "log_raw_tf": log_raw_tf,
        "log_swap": log_swap,
        "log_fullperm": log_fullperm,
        "log_pride_sf": pride_sf,
        "log_pride_tf": pride_tf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/probe.results/session_2026-07-10_farmshare/probe_experiment.json",
            "results/probe.results/session_2026-07-10_farmshare/probe_scaling_noun.json",
        ],
    )
    ap.add_argument("--n-est", type=int, default=10, help="stimuli used for PriDe prior")
    ap.add_argument(
        "--out-prefix",
        default="results/probe.results/session_2026-07-10_farmshare/pride_debias",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    rows = []
    for inp in args.inputs:
        path = Path(inp) if Path(inp).is_absolute() else root / inp
        d = json.loads(path.read_text())
        src = path.stem
        for model, cells in d["models"].items():
            if not isinstance(cells, dict):
                continue
            for cell, data in cells.items():
                if not isinstance(data, dict) or "per_stim" not in data:
                    continue
                cond, label = cell.split("|")
                res = analyze_cell(data["per_stim"], args.n_est)
                gate = data.get("metrics", {}).get("gate_pass", False)
                rows.append({"source": src, "model": model, "condition": cond, "label_set": label, "gate": gate, **res})

    out_csv = root / f"{args.out_prefix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = []
    hdr = (
        f"{'source':22} {'model':13} {'cond':10} {'lbl':7} {'gate':4} "
        f"{'gRawSF':>6} {'gRawTF':>6} {'gSwap':>6} | "
        f"{'lRawSF':>6} {'lRawTF':>6} {'lSwap':>6} {'lPerm':>6} {'PriSF':>6} {'PriTF':>6} {'prior1':>6}"
    )
    lines.append("Raw vs corrected shape preference (rates; PriDe: prior on first "
                 f"{args.n_est} stimuli, debias single-order obs on the rest)")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        lines.append(
            f"{r['source']:22} {r['model']:13} {r['condition'][:10]:10} {r['label_set']:7} "
            f"{'PASS' if r['gate'] else 'fail':4} "
            f"{r['gen_raw_sf']:6.2f} {r['gen_raw_tf']:6.2f} {r['gen_swap']:6.2f} | "
            f"{r['log_raw_sf']:6.2f} {r['log_raw_tf']:6.2f} {r['log_swap']:6.2f} "
            f"{r['log_fullperm']:6.2f} {r['log_pride_sf']:6.2f} {r['log_pride_tf']:6.2f} "
            f"{r['prior_first']:6.2f}"
        )

    # flag cells where a naive single-order read is misleading (|raw-0.5|>=0.15
    # while the full-permutation estimate stays within 0.10 of chance)
    lines.append("")
    lines.append("Cells where a single-order raw read would mislead (raw >=0.15 from chance, fullperm within 0.10):")
    for r in rows:
        for tag in ("log_raw_sf", "log_raw_tf"):
            if abs(r[tag] - 0.5) >= 0.15 and abs(r["log_fullperm"] - 0.5) <= 0.10:
                lines.append(
                    f"  {r['source']} {r['model']} {r['condition']}|{r['label_set']}: "
                    f"{tag}={r[tag]:.2f} -> fullperm={r['log_fullperm']:.2f}"
                )

    text = "\n".join(lines)
    out_txt = root / f"{args.out_prefix}.txt"
    out_txt.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nCSV: {out_csv}\nText: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
