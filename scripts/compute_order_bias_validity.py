#!/usr/bin/env python3
"""Compute order/position-bias validity from standardized local benchmark CSV.

This diagnostic targets models that may be deciding based on option position
rather than image-content matching in the 2AFC setup.

Default input:
  results/model.results/benchmark_standardized_rerun/local_eval_standardized.csv

Outputs:
  - results/data/order_bias_validity_local_standardized.csv
  - updates interpret/order_bias_validity.md auto section

Usage:
  python scripts/compute_order_bias_validity.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "model.results"
    / "benchmark_standardized_rerun"
    / "local_eval_standardized.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "data" / "order_bias_validity_local_standardized.csv"
DEFAULT_MD = REPO_ROOT / "interpret" / "order_bias_validity.md"

MARKER_START = "<!-- AUTO_ORDER_VALIDITY_START -->"
MARKER_END = "<!-- AUTO_ORDER_VALIDITY_END -->"

T_TRACK_VALID = 0.70
T_TRACK_BORDER = 0.50
T_WORD = 0.20
T_PARSE = 0.97


def _load_cvhm():
    path = REPO_ROOT / "scripts" / "compute_validity_human_matched.py"
    spec = importlib.util.spec_from_file_location("_cvhm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_keys(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in row.items():
        nk = (k or "").strip().lstrip("\ufeff")
        if not nk:
            continue
        out[nk] = (v or "").strip()
    return out


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for raw in csv.DictReader(f):
            r = _norm_keys(raw)
            if not r.get("model") or not r.get("ordering"):
                continue
            if r.get("choice") not in {"shape", "texture", "unclear"}:
                continue
            rows.append(r)
    return rows


def summarize_model(
    model: str,
    model_rows: list[dict[str, str]],
    gap_valid_max: float,
    gap_borderline_max: float,
    first_valid_band: tuple[float, float],
    first_borderline_band: tuple[float, float],
) -> dict[str, str]:
    sf = [r for r in model_rows if r.get("ordering") == "shape_first"]
    tf = [r for r in model_rows if r.get("ordering") == "texture_first"]

    sf_shape = sum(1 for r in sf if r.get("choice") == "shape")
    sf_total = len(sf)
    tf_shape = sum(1 for r in tf if r.get("choice") == "shape")
    tf_total = len(tf)
    sf_rate = (sf_shape / sf_total) if sf_total else float("nan")
    tf_rate = (tf_shape / tf_total) if tf_total else float("nan")
    gap = abs(sf_rate - tf_rate) if sf_total and tf_total else float("nan")

    # "First option" means:
    # - shape_first: choosing shape == picked option #1
    # - texture_first: choosing texture == picked option #1
    first_picks = sf_shape + sum(1 for r in tf if r.get("choice") == "texture")
    first_total = sf_total + tf_total
    first_rate = (first_picks / first_total) if first_total else float("nan")

    if gap != gap or first_rate != first_rate:
        label = "invalid"
    elif gap <= gap_valid_max and first_valid_band[0] <= first_rate <= first_valid_band[1]:
        label = "valid"
    elif gap <= gap_borderline_max and first_borderline_band[0] <= first_rate <= first_borderline_band[1]:
        label = "borderline"
    else:
        label = "invalid"

    return {
        "model": model,
        "n_shape_first": str(sf_total),
        "n_texture_first": str(tf_total),
        "shape_rate_shape_first": f"{sf_rate:.6f}" if sf_rate == sf_rate else "",
        "shape_rate_texture_first": f"{tf_rate:.6f}" if tf_rate == tf_rate else "",
        "order_gap_abs": f"{gap:.6f}" if gap == gap else "",
        "first_option_rate": f"{first_rate:.6f}" if first_rate == first_rate else "",
        "order_validity_label": label,
    }


def label_benchmark(track: float, word: float, parse: float) -> str:
    if track != track or word != word or parse != parse:
        return "invalid"
    if track >= T_TRACK_VALID and word >= T_WORD and parse >= T_PARSE:
        return "valid"
    if track >= T_TRACK_BORDER:
        return "borderline"
    return "invalid"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "n_pairs",
        "image_tracking_rate",
        "n_groups",
        "word_sensitivity_rate",
        "n_trials",
        "unclear_rate",
        "retry_rate",
        "parse_quality",
        "validity_label",
        "n_shape_first",
        "n_texture_first",
        "shape_rate_shape_first",
        "shape_rate_texture_first",
        "order_gap_abs",
        "first_option_rate",
        "order_validity_label",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _sort_key(row: dict[str, str]) -> tuple[int, int, float, str]:
    order = {"valid": 0, "borderline": 1, "invalid": 2}
    gap = float(row.get("order_gap_abs") or "999")
    return (
        order.get(row.get("validity_label", "invalid"), 9),
        order.get(row["order_validity_label"], 9),
        gap,
        row["model"],
    )


def _pc(x: str) -> str:
    if not x:
        return "N/A"
    return f"{100 * float(x):.1f}%"


def _pass(rate: float, threshold: float, *, at_floor: bool = False) -> str:
    if rate >= threshold:
        if at_floor and abs(rate - threshold) < 1e-9:
            return f"yes (at {rate:.2f})"
        return "yes"
    return f"no ({rate:.2f})"


def _norm_gap(observed: float, threshold: float) -> float:
    return (threshold - observed) / threshold


def _primary_barrier(track: float, word: float, parse: float) -> str:
    if track >= T_TRACK_VALID and word >= T_WORD and parse >= T_PARSE:
        return "—"
    gaps: list[tuple[float, str]] = []
    if track < T_TRACK_VALID:
        gaps.append((_norm_gap(track, T_TRACK_VALID), "image tracking"))
    if word < T_WORD:
        gaps.append((_norm_gap(word, T_WORD), "word sensitivity"))
    if parse < T_PARSE:
        gaps.append((_norm_gap(parse, T_PARSE), "parse quality"))
    gaps.sort(key=lambda x: -x[0])
    if len(gaps) == 1:
        name = gaps[0][1]
        if name == "image tracking" and track >= T_TRACK_BORDER:
            return "**Image tracking** (short of 0.70 only)"
        return f"**{name.title()}**"
    primary = gaps[0]
    out = [f"**{primary[1].title()}** (largest gap vs valid bar)"]
    if len(gaps) > 1 and gaps[1][0] >= 0.85 * primary[0] and gaps[1][0] > 0.05:
        out.append(f"**{gaps[1][1].title()}** second")
    return "; ".join(out)


def _r_note(label: str, track: float, word: float, parse: float) -> str:
    if label == "valid":
        return "All gates pass."
    if label == "borderline":
        bits = ["Tier = borderline because tracking ≥0.50."]
        miss = []
        if track < T_TRACK_VALID:
            miss.append("image tracking below 0.70")
        if word < T_WORD:
            miss.append("word sensitivity below 0.20")
        if parse < T_PARSE:
            miss.append("parse quality below 0.97")
        if miss:
            bits.append("Blocks valid: " + "; ".join(miss) + ".")
        return " ".join(bits)
    bits = ["Invalid tier: tracking &lt;0.50."]
    extra = []
    if word < T_WORD:
        extra.append("word sensitivity below 0.20")
    if parse < T_PARSE:
        extra.append("parse quality below 0.97 (often unclear answers)")
    if extra:
        bits.append("Also: " + "; ".join(extra) + ".")
    return " ".join(bits)


def build_md(rows: list[dict[str, str]], *, gap_valid_max: float, gap_borderline_max: float) -> str:
    n = len(rows)
    n_val_main = sum(1 for r in rows if r["validity_label"] == "valid")
    n_bor_main = sum(1 for r in rows if r["validity_label"] == "borderline")
    n_inv_main = sum(1 for r in rows if r["validity_label"] == "invalid")
    n_valid = sum(1 for r in rows if r["order_validity_label"] == "valid")
    n_border = sum(1 for r in rows if r["order_validity_label"] == "borderline")
    n_inv = sum(1 for r in rows if r["order_validity_label"] == "invalid")

    lines: list[str] = []
    lines.append("*Auto-generated by `python scripts/compute_order_bias_validity.py`.*")
    lines.append("")
    lines.append("### Failure analysis: which criterion blocks “valid”?")
    lines.append("")
    lines.append(
        "For each model, the table uses the **valid** thresholds (0.70 / 0.20 / 0.97) "
        "from `interpret/models_validity.md`. A checkmark means the metric meets that bar. "
        "**Primary barrier to valid** is the failed criterion with the **largest normalized shortfall** "
        "(for each failed metric: (threshold - observed rate) / threshold). "
        "The table then appends position-bias diagnostics and an order-bias label."
    )
    lines.append("")
    lines.append("*Main-gate columns mirror `interpret/models_validity.md`; extra columns are appended.*")
    lines.append("")
    lines.append("### Per-model validity + position bias")
    lines.append("")
    lines.append(
        "| Model | Main label | Image track ≥0.70 | Word sens. ≥0.20 | Parse ≥0.97 | "
        "Primary barrier to “valid” | Note on R label | "
        "Order label | Shape% (shape_first) | Shape% (texture_first) | |Δshape%| | First-option rate |"
    )
    lines.append(
        "|-------|------------|-------------------|------------------|-------------|----------------------------|-------------------|-------------|----------------------|------------------------|----------|-------------------|"
    )
    for r in sorted(rows, key=_sort_key):
        t = float(r["image_tracking_rate"])
        w = float(r["word_sensitivity_rate"])
        p = float(r["parse_quality"])
        w_at_floor = w >= T_WORD and abs(w - T_WORD) < 1e-9
        barrier = _primary_barrier(t, w, p)
        note = _r_note(r["validity_label"], t, w, p)
        lines.append(
            f"| `{r['model']}` | {r['validity_label']} | {_pass(t, T_TRACK_VALID)} | "
            f"{_pass(w, T_WORD, at_floor=w_at_floor)} | {_pass(p, T_PARSE)} | {barrier} | {note} | "
            f"{r['order_validity_label']} | "
            f"{_pc(r['shape_rate_shape_first'])} | {_pc(r['shape_rate_texture_first'])} | "
            f"{_pc(r['order_gap_abs'])} | {_pc(r['first_option_rate'])} |"
        )
    lines.append("")
    lines.append(
        "**Reading the table:** For models that are **`invalid`**, the **R rule** is always "
        "“image tracking below 0.50.” The **primary barrier** column answers: "
        "*if you wanted to reach the **valid** profile, which metric is furthest below target?*"
    )
    lines.append("")
    lines.append("### Updated outcomes (main validity)")
    lines.append("")
    lines.append(f"- **Valid models:** {n_val_main} / {n}")
    lines.append(f"- **Borderline models:** {n_bor_main} / {n}")
    lines.append(f"- **Invalid models:** {n_inv_main} / {n}")
    lines.append("")
    lines.append("### Updated outcomes (order-bias validity)")
    lines.append("")
    lines.append(f"- **Valid models:** {n_valid} / {n}")
    lines.append(f"- **Borderline models:** {n_border} / {n}")
    lines.append(f"- **Invalid models:** {n_inv} / {n}")
    lines.append("")
    lines.append("### Thresholds used")
    lines.append("")
    lines.append("- **Main validity** (same as `interpret/models_validity.md`):")
    lines.append("  - `valid` if image tracking ≥0.70, word sensitivity ≥0.20, parse quality ≥0.97")
    lines.append("  - `borderline` if tracking ≥0.50 but not fully valid")
    lines.append("  - `invalid` otherwise")
    lines.append("")
    lines.append("- **Order-bias validity** (additional gate):")
    lines.append(
        f"  - `valid`: `|Δshape%| <= {100*gap_valid_max:.0f}%` and first-option rate in `[40%, 60%]`"
    )
    lines.append(
        f"  - `borderline`: `|Δshape%| <= {100*gap_borderline_max:.0f}%` and first-option rate in `[30%, 70%]`"
    )
    lines.append("  - `invalid`: otherwise")
    lines.append("")
    return "\n".join(lines)


def patch_md(md_path: Path, section: str) -> None:
    txt = md_path.read_text(encoding="utf-8")
    pat = re.compile(re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END), re.DOTALL)
    repl = MARKER_START + "\n" + section + MARKER_END
    if not pat.search(txt):
        raise SystemExit(f"Markers not found in {md_path}: {MARKER_START} ... {MARKER_END}")
    md_path.write_text(pat.sub(repl, txt, count=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--gap-valid-max", type=float, default=0.15)
    ap.add_argument("--gap-borderline-max", type=float, default=0.35)
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")

    rows = load_rows(args.input)
    by_model: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    cvhm = _load_cvhm()
    disamb = [dict(r) for r in rows]
    cvhm.disambiguate_stim(disamb)
    track = cvhm.compute_image_tracking(disamb)
    word = cvhm.compute_word_sensitivity(disamb)
    parse = cvhm.compute_parse_quality(disamb)

    summary: list[dict[str, str]] = []
    for model, mrows in sorted(by_model.items()):
        order_part = summarize_model(
            model,
            mrows,
            args.gap_valid_max,
            args.gap_borderline_max,
            first_valid_band=(0.40, 0.60),
            first_borderline_band=(0.30, 0.70),
        )
        np, tr = track.get(model, (0, float("nan")))
        ng, wr = word.get(model, (0, float("nan")))
        nt, ur, rr, pq = parse.get(model, (0, 0.0, 0.0, float("nan")))
        order_part.update(
            {
                "n_pairs": str(np),
                "image_tracking_rate": str(tr),
                "n_groups": str(ng),
                "word_sensitivity_rate": str(wr),
                "n_trials": str(nt),
                "unclear_rate": str(ur),
                "retry_rate": str(rr),
                "parse_quality": str(pq),
                "validity_label": label_benchmark(
                    tr if tr == tr else -1.0,
                    wr if wr == wr else -1.0,
                    pq if pq == pq else -1.0,
                ),
            }
        )
        summary.append(order_part)

    write_csv(args.output, summary)
    section = build_md(summary, gap_valid_max=args.gap_valid_max, gap_borderline_max=args.gap_borderline_max)
    patch_md(args.md, section)

    print(f"Wrote {args.output} ({len(summary)} models)")
    print(f"Updated {args.md}")


if __name__ == "__main__":
    main()

