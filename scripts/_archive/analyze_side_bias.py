#!/usr/bin/env python3
"""Analyze side/position bias from evaluation CSV output.

This script is model-agnostic and works with CSVs produced by:
  - scripts/run_local.py
  - scripts/run_remote.py

It focuses on:
  1) Option-id bias ("1" vs "2")
  2) First-vs-second image bias across counterbalanced orderings
  3) Ordering-dependent shape-choice shift (shape_first vs texture_first)

Usage:
  python scripts/analyze_side_bias.py \
    --input results/local_20260409_123456.csv \
    --model smolvlm
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _norm_row(raw: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        nk = (k or "").strip().lstrip("\ufeff")
        if not nk:
            continue
        out[nk] = (v or "").strip()
    return out


def _load_rows(path: Path, model: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = _norm_row(raw)
            if row.get("model") != model:
                continue
            if row.get("ordering") not in {"shape_first", "texture_first"}:
                continue
            rows.append(row)
    return rows


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "N/A"
    return f"{100.0 * x:.1f}%"


def _fmt_abs_pct_diff(a: float, b: float) -> str:
    if math.isnan(a) or math.isnan(b):
        return "N/A"
    return f"{100.0 * abs(a - b):.1f}%"


def _simple_bias_label(
    second_rate: float,
    order_gap: float,
    *,
    second_rate_strong: float = 0.60,
    second_rate_moderate: float = 0.55,
    order_gap_strong: float = 0.25,
    order_gap_moderate: float = 0.15,
) -> str:
    if math.isnan(second_rate) or math.isnan(order_gap):
        return "insufficient_data"
    if second_rate >= second_rate_strong or order_gap >= order_gap_strong:
        return "strong_side_bias"
    if second_rate >= second_rate_moderate or order_gap >= order_gap_moderate:
        return "moderate_side_bias"
    return "low_side_bias"


def _second_pick(row: dict[str, str]) -> bool | None:
    choice = row.get("choice")
    ordering = row.get("ordering")
    if choice not in {"shape", "texture"}:
        return None
    if ordering == "shape_first":
        # [ref, shape, texture] -> second option means texture.
        return choice == "texture"
    if ordering == "texture_first":
        # [ref, texture, shape] -> second option means shape.
        return choice == "shape"
    return None


def _analysis(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    decisive = [r for r in rows if r.get("choice") in {"shape", "texture"}]
    sf = [r for r in decisive if r.get("ordering") == "shape_first"]
    tf = [r for r in decisive if r.get("ordering") == "texture_first"]

    n_total = len(rows)
    n_decisive = len(decisive)
    n_unclear = n_total - n_decisive

    # Option-id bias ("1" vs "2"), if parsed_answer is present/usable.
    n_ans = 0
    n_1 = 0
    n_2 = 0
    for r in decisive:
        ans = r.get("parsed_answer")
        if ans in {"1", "2"}:
            n_ans += 1
            if ans == "1":
                n_1 += 1
            else:
                n_2 += 1

    option2_rate = _ratio(n_2, n_ans)

    # Side bias by spatial position: did model pick the second image?
    n_second = 0
    n_second_den = 0
    for r in decisive:
        picked_second = _second_pick(r)
        if picked_second is None:
            continue
        n_second_den += 1
        if picked_second:
            n_second += 1
    second_rate = _ratio(n_second, n_second_den)

    sf_shape = sum(1 for r in sf if r.get("choice") == "shape")
    tf_shape = sum(1 for r in tf if r.get("choice") == "shape")
    sf_rate = _ratio(sf_shape, len(sf))
    tf_rate = _ratio(tf_shape, len(tf))
    order_gap = abs(sf_rate - tf_rate) if sf and tf else float("nan")
    raw_shape_rate = _ratio(sf_shape + tf_shape, len(sf) + len(tf))

    # Counterbalanced de-biasing:
    # average shape rate across the two orderings to cancel first/second heuristics.
    # If a model always picks first, sf_rate=1 and tf_rate=0 => adjusted=0.5.
    if sf and tf:
        adjusted_shape_rate = 0.5 * (sf_rate + tf_rate)
    else:
        adjusted_shape_rate = float("nan")
    adjusted_shape_bias_index = (
        adjusted_shape_rate - 0.5 if not math.isnan(adjusted_shape_rate) else float("nan")
    )

    label = _simple_bias_label(second_rate, order_gap)

    return {
        "n_total": n_total,
        "n_decisive": n_decisive,
        "n_unclear": n_unclear,
        "unclear_rate": _ratio(n_unclear, n_total),
        "n_answered_1or2": n_ans,
        "option1_count": n_1,
        "option2_count": n_2,
        "option2_rate": option2_rate,
        "n_second_denom": n_second_den,
        "second_pick_count": n_second,
        "second_pick_rate": second_rate,
        "n_shape_first": len(sf),
        "n_texture_first": len(tf),
        "raw_shape_rate_decisive": raw_shape_rate,
        "shape_rate_shape_first": sf_rate,
        "shape_rate_texture_first": tf_rate,
        "order_gap_abs": order_gap,
        "adjusted_shape_rate_counterbalanced": adjusted_shape_rate,
        "adjusted_shape_bias_index": adjusted_shape_bias_index,
        "side_bias_label": label,
    }


def _suggestions(result: dict[str, float | int | str]) -> list[str]:
    label = str(result["side_bias_label"])
    suggestions: list[str] = []

    if label == "strong_side_bias":
        suggestions.extend(
            [
                "Use only `--ordering both` during measurement; never evaluate with one fixed ordering.",
                "Increase repeats (`--repeats 3`) and aggregate by majority vote per (stimulus, word) to reduce order noise.",
                "Run a label-swap control prompt (same image order, swap identifiers) to separate side bias from token bias.",
                "Raise prompt rigidity: ask for exactly one character and reject any non-{1,2} output with retry.",
            ]
        )
    elif label == "moderate_side_bias":
        suggestions.extend(
            [
                "Keep counterbalancing (`--ordering both`) and report side-bias metrics with every shape-bias score.",
                "Add a calibration check on a held-out split and track drift of second-pick rate over time.",
                "Use multiple wording templates and average them to reduce prompt-form sensitivity.",
            ]
        )
    else:
        suggestions.extend(
            [
                "Side bias looks low; continue tracking `second_pick_rate` and `order_gap_abs` as guardrail metrics.",
                "Still keep both orderings in production evaluation to avoid silent regressions.",
            ]
        )

    return suggestions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Path to evaluation CSV")
    ap.add_argument("--model", required=True, help="Model key in CSV (e.g. smolvlm)")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")

    rows = _load_rows(args.input, args.model)
    if not rows:
        raise SystemExit(
            f"No rows found for model={args.model!r} with ordering in "
            f"{{shape_first, texture_first}} in {args.input}"
        )

    result = _analysis(rows)
    suggestions = _suggestions(result)

    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print("")
    print("Core counts")
    print(f"  total rows:      {result['n_total']}")
    print(f"  decisive rows:   {result['n_decisive']}")
    print(f"  unclear rows:    {result['n_unclear']} ({_fmt_pct(float(result['unclear_rate']))})")
    print("")
    print("Option-id and side metrics")
    print(
        "  answered 1/2:    "
        f"{result['n_answered_1or2']} | "
        f"1-count={result['option1_count']} 2-count={result['option2_count']} "
        f"(2-rate={_fmt_pct(float(result['option2_rate']))})"
    )
    print(
        "  second picks:    "
        f"{result['second_pick_count']}/{result['n_second_denom']} "
        f"({_fmt_pct(float(result['second_pick_rate']))})"
    )
    print("")
    print("Ordering effect")
    print(
        "  raw shape rate (decisive):  "
        f"{_fmt_pct(float(result['raw_shape_rate_decisive']))}"
    )
    print(
        "  shape rate (shape_first):   "
        f"{_fmt_pct(float(result['shape_rate_shape_first']))} "
        f"(n={result['n_shape_first']})"
    )
    print(
        "  shape rate (texture_first): "
        f"{_fmt_pct(float(result['shape_rate_texture_first']))} "
        f"(n={result['n_texture_first']})"
    )
    print(
        "  |delta shape rate|:         "
        f"{_fmt_abs_pct_diff(float(result['shape_rate_shape_first']), float(result['shape_rate_texture_first']))}"
    )
    print("")
    print("Adjusted (counterbalanced) metric")
    print(
        "  smolvlm2_adjusted_score:    "
        f"{_fmt_pct(float(result['adjusted_shape_rate_counterbalanced']))} "
        "(average of per-ordering shape rates)"
    )
    adj_idx = float(result["adjusted_shape_bias_index"])
    if math.isnan(adj_idx):
        idx_txt = "N/A"
    else:
        idx_txt = f"{adj_idx:+.3f} (0 = no shape-vs-texture preference after de-biasing)"
    print(f"  adjusted_shape_bias_index:  {idx_txt}")
    print("")
    print(f"Bias label: {result['side_bias_label']}")
    print("")
    print("Suggestions")
    for i, s in enumerate(suggestions, start=1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()
