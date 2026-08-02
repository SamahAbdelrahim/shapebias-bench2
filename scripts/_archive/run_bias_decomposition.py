#!/usr/bin/env python3
"""Run a quick bias decomposition suite for local VLMs.

This script helps separate:
  1) lexical token bias (e.g., prefers "1")
  2) first-option positional bias (prefers first candidate regardless of label set)
  3) weak visual grounding fallback (bias drops when forced to compare first)

Usage:
  source .venv/bin/activate
  python scripts/run_bias_decomposition.py --model smolvlm --device cuda
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_pipe.eval_core import load_stimuli, load_words  # noqa: E402
from evaluation_pipe.models import create_model  # noqa: E402


@dataclass(frozen=True)
class Condition:
    name: str
    label_set: str  # "12" | "ab" | "lr"
    use_word: bool
    forced_grounding: bool


CONDITIONS: tuple[Condition, ...] = (
    Condition("baseline_12", "12", True, False),
    Condition("no_word_12", "12", False, False),
    Condition("label_ab", "ab", True, False),
    Condition("left_right", "lr", True, False),
    Condition("forced_grounding_12", "12", True, True),
)

ORDERINGS = ("shape_first", "texture_first")


def _prompt(cond: Condition, word: str) -> str:
    if cond.name == "baseline_12":
        return (
            f"The first image is a {word}. "
            f"Which of the following two images (1 or 2) is also a {word}? "
            "Your response must be exactly one character: 1 or 2. No other text."
        )
    if cond.name == "no_word_12":
        return (
            "See this object in the first image. "
            "Can you find another one of the two (1 or 2)? "
            "Your response must be exactly one character: 1 or 2. No other text."
        )
    if cond.name == "label_ab":
        return (
            f"The first image is a {word}. "
            f"Which of the two candidate images (A or B) is also a {word}? "
            "Your response must be exactly one character: A or B. No other text."
        )
    if cond.name == "left_right":
        return (
            f"The first image is a {word}. "
            f"Which candidate image matches it better: left or right? "
            "Your response must be exactly one word: left or right."
        )
    if cond.name == "forced_grounding_12":
        return (
            f"The first image is a {word}. Compare candidate 1 and candidate 2 to the first image. "
            "Then output your final answer on a new line in this format exactly: FINAL: 1 or FINAL: 2"
        )
    raise ValueError(f"Unhandled condition: {cond.name}")


def _parse_12(raw: str, *, forced: bool) -> str | None:
    t = raw.strip().lower()
    if forced:
        m = re.search(r"final:\s*([12])\b", t)
        if m:
            return m.group(1)
    has_1 = bool(re.search(r"\b1\b", t))
    has_2 = bool(re.search(r"\b2\b", t))
    if has_1 and has_2:
        return None
    if has_1:
        return "1"
    if has_2:
        return "2"
    return None


def _parse_ab(raw: str) -> str | None:
    t = raw.strip().lower()
    has_a = bool(re.search(r"\ba\b", t))
    has_b = bool(re.search(r"\bb\b", t))
    if has_a and has_b:
        return None
    if has_a:
        return "a"
    if has_b:
        return "b"
    return None


def _parse_lr(raw: str) -> str | None:
    t = raw.strip().lower()
    has_l = bool(re.search(r"\bleft\b", t))
    has_r = bool(re.search(r"\bright\b", t))
    if has_l and has_r:
        return None
    if has_l:
        return "left"
    if has_r:
        return "right"
    return None


def _parse_label(cond: Condition, raw: str) -> str | None:
    if cond.label_set == "12":
        return _parse_12(raw, forced=cond.forced_grounding)
    if cond.label_set == "ab":
        return _parse_ab(raw)
    if cond.label_set == "lr":
        return _parse_lr(raw)
    return None


def _labels_for_ordering(cond: Condition, ordering: str) -> tuple[str, str]:
    if cond.label_set == "12":
        return ("1", "2")
    if cond.label_set == "ab":
        return ("A", "B")
    if cond.label_set == "lr":
        return ("left", "right")
    raise ValueError(f"Unknown label set: {cond.label_set}")


def _choice_from_label(
    parsed: str | None,
    label_first: str,
    label_second: str,
    ordering: str,
) -> tuple[str, str]:
    if parsed is None:
        return ("unclear", "unclear")
    if parsed.lower() == label_first.lower():
        return ("first", "shape" if ordering == "shape_first" else "texture")
    if parsed.lower() == label_second.lower():
        return ("second", "texture" if ordering == "shape_first" else "shape")
    return ("unclear", "unclear")


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


def _summary(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    conds = sorted({r["condition"] for r in rows})
    for cond in conds:
        cr = [r for r in rows if r["condition"] == cond]
        dec = [r for r in cr if r["picked_position"] in {"first", "second"}]
        sf = [r for r in dec if r["ordering"] == "shape_first"]
        tf = [r for r in dec if r["ordering"] == "texture_first"]

        first = sum(1 for r in dec if r["picked_position"] == "first")
        second = sum(1 for r in dec if r["picked_position"] == "second")
        shape_sf = sum(1 for r in sf if r["picked_semantic"] == "shape")
        shape_tf = sum(1 for r in tf if r["picked_semantic"] == "shape")
        sf_rate = _rate(shape_sf, len(sf))
        tf_rate = _rate(shape_tf, len(tf))
        gap = abs(sf_rate - tf_rate) if sf and tf else float("nan")

        out.append(
            {
                "condition": cond,
                "n_total": len(cr),
                "n_decisive": len(dec),
                "decisive_rate": _rate(len(dec), len(cr)),
                "first_rate": _rate(first, len(dec)),
                "second_rate": _rate(second, len(dec)),
                "shape_rate_shape_first": sf_rate,
                "shape_rate_texture_first": tf_rate,
                "order_gap_abs": gap,
            }
        )
    return out


def _interpret(summary_rows: list[dict]) -> list[str]:
    by_cond = {r["condition"]: r for r in summary_rows}
    notes: list[str] = []

    b12 = by_cond.get("baseline_12")
    ab = by_cond.get("label_ab")
    lr = by_cond.get("left_right")
    fg = by_cond.get("forced_grounding_12")

    if b12 and ab and lr:
        b_first = float(b12["first_rate"])
        ab_first = float(ab["first_rate"])
        lr_first = float(lr["first_rate"])
        if b_first >= 0.8 and ab_first >= 0.8 and lr_first >= 0.8:
            notes.append(
                "Dominant first-option positional bias: high first-choice rate across numeric, letter, and left/right labels."
            )
        elif b_first >= 0.8 and ab_first < 0.7 and lr_first < 0.7:
            notes.append(
                "Dominant lexical numeric-token bias: strong in 1/2 framing but weaker in A/B and left/right."
            )
        else:
            notes.append(
                "Mixed bias profile: label set changes the preference, suggesting both positional and token-level effects."
            )

    if b12 and fg:
        b_first = float(b12["first_rate"])
        fg_first = float(fg["first_rate"])
        if b_first - fg_first >= 0.15:
            notes.append(
                "Forced grounding reduces first-option bias, suggesting weak visual grounding contributes to default-to-first behavior."
            )
        elif fg_first >= 0.8:
            notes.append(
                "Forced grounding does not reduce bias materially; behavior looks like a strong prior rather than prompt underspecification."
            )

    if not notes:
        notes.append("No strong decomposition signal; increase sample size (`--num-stimuli` and `--num-words`) for clearer diagnosis.")
    return notes


def _build_md_report(
    *,
    model: str,
    device: str,
    num_stimuli: int,
    num_words: int,
    output_csv: Path,
    summary_rows: list[dict],
    notes: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Bias Decomposition Report")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Stimuli sampled: `{num_stimuli}`")
    lines.append(f"- Words sampled: `{num_words}`")
    lines.append(f"- Detailed rows CSV: `{output_csv}`")
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    lines.append(
        "| Condition | Decisive rate | First rate | Second rate | Shape rate (shape_first) | Shape rate (texture_first) | \\|Δshape\\| |"
    )
    lines.append(
        "|-----------|---------------|------------|-------------|---------------------------|-----------------------------|-----------|"
    )
    for r in summary_rows:
        lines.append(
            f"| `{r['condition']}` | {_fmt_pct(float(r['decisive_rate']))} | "
            f"{_fmt_pct(float(r['first_rate']))} | {_fmt_pct(float(r['second_rate']))} | "
            f"{_fmt_pct(float(r['shape_rate_shape_first']))} | {_fmt_pct(float(r['shape_rate_texture_first']))} | "
            f"{_fmt_pct(float(r['order_gap_abs']))} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="smolvlm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-stimuli", type=int, default=3)
    ap.add_argument("--num-words", type=int, default=10)
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "smolvlm_bias_decomposition_rows.csv",
    )
    ap.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Optional markdown report path (default: <output stem>.md)",
    )
    args = ap.parse_args()

    words = load_words()[: args.num_words]
    stimuli = load_stimuli(num_stimuli=args.num_stimuli)
    model = create_model(args.model, device=args.device)
    rows: list[dict] = []

    try:
        for cond in CONDITIONS:
            print(f"\n=== Condition: {cond.name} ===")
            for stim in stimuli:
                for w in words:
                    for ordering in ORDERINGS:
                        if ordering == "shape_first":
                            img1 = stim["shape_match"]
                            img2 = stim["texture_match"]
                        else:
                            img1 = stim["texture_match"]
                            img2 = stim["shape_match"]

                        prompt = _prompt(cond, w["name"])
                        resp = model.generate(
                            images=[stim["reference"], img1, img2],
                            prompt=prompt,
                            max_new_tokens=96,
                            temperature=0.0,
                        )

                        parsed = _parse_label(cond, resp.raw_text)
                        l1, l2 = _labels_for_ordering(cond, ordering)
                        picked_position, picked_semantic = _choice_from_label(parsed, l1, l2, ordering)
                        rows.append(
                            {
                                "model": args.model,
                                "condition": cond.name,
                                "stim_id": stim["stim_id"],
                                "word": w["name"],
                                "word_type": w["type"],
                                "ordering": ordering,
                                "label_first": l1,
                                "label_second": l2,
                                "raw_text": resp.raw_text,
                                "parsed_label": parsed or "",
                                "picked_position": picked_position,
                                "picked_semantic": picked_semantic,
                                "generation_time_s": f"{resp.generation_time_s:.3f}",
                            }
                        )
    finally:
        model.unload()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "condition",
        "stim_id",
        "word",
        "word_type",
        "ordering",
        "label_first",
        "label_second",
        "raw_text",
        "parsed_label",
        "picked_position",
        "picked_semantic",
        "generation_time_s",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary_rows = _summary(rows)
    notes = _interpret(summary_rows)
    print("\n=== Bias decomposition summary ===")
    for r in summary_rows:
        print(
            f"- {r['condition']:20s} "
            f"decisive={_fmt_pct(float(r['decisive_rate'])):>7s} "
            f"first={_fmt_pct(float(r['first_rate'])):>7s} "
            f"second={_fmt_pct(float(r['second_rate'])):>7s} "
            f"|Δshape|={_fmt_pct(float(r['order_gap_abs'])):>7s}"
        )

    print("\n=== Interpretation ===")
    for note in notes:
        print(f"- {note}")

    report_md = args.report_md if args.report_md is not None else args.output.with_suffix(".md")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_txt = _build_md_report(
        model=args.model,
        device=args.device,
        num_stimuli=args.num_stimuli,
        num_words=args.num_words,
        output_csv=args.output,
        summary_rows=summary_rows,
        notes=notes,
    )
    report_md.write_text(report_txt, encoding="utf-8")

    print(f"\nWrote detailed rows to: {args.output}")
    print(f"Wrote markdown report to: {report_md}")


if __name__ == "__main__":
    main()

