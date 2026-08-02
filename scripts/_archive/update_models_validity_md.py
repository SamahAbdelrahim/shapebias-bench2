#!/usr/bin/env python3
"""Regenerate auto sections of interpret/models_validity.md.

- **Single merged table:** ``results/data/model_validity_summary.csv`` (R pipeline on full canonical).
- **Split (recommended):** ``python scripts/compute_model_validity_split.py`` then
  ``python scripts/update_models_validity_md.py --from-split`` — word benchmark (local + remote)
  and no-word **trio** dedup, matching the main experimental sequence.

Thresholds match ``analysis_pipe/src/validity_gates.R``.

Usage (from repo root):
    python scripts/update_models_validity_md.py
    python scripts/update_models_validity_md.py --csv results/data/model_validity_summary.csv
    python scripts/compute_model_validity_split.py && python scripts/update_models_validity_md.py --from-split
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "results" / "data" / "model_validity_summary.csv"
DEFAULT_MD = REPO_ROOT / "interpret" / "models_validity.md"
DATA_DIR = REPO_ROOT / "results" / "data"
SPLIT_CSV_WORD = DATA_DIR / "model_validity_summary_word.csv"
SPLIT_CSV_NO_WORD_TRIO = DATA_DIR / "model_validity_summary_no_word_trio.csv"

T_TRACK_VALID = 0.70
T_TRACK_BORDER = 0.50
T_WORD = 0.20
T_PARSE = 0.97

MARKER_START = "<!-- AUTO_VALIDITY_SECTION_START -->"
MARKER_END = "<!-- AUTO_VALIDITY_SECTION_END -->"


def _fmt_rate(x: float) -> str:
    return f"{x:.2f}"


def pass_cell(ok: bool, rate: float, thresh: float, *, at_floor: bool = False) -> str:
    if ok:
        if at_floor and abs(rate - thresh) < 1e-9:
            return f"yes (at {_fmt_rate(rate)})"
        return "yes"
    return f"no ({_fmt_rate(rate)})"


def normalized_gaps(t: float, w: float, p: float) -> tuple[tuple[float, str], ...]:
    gaps: list[tuple[float, str]] = []
    if t < T_TRACK_VALID:
        gaps.append(((T_TRACK_VALID - t) / T_TRACK_VALID, "image tracking"))
    if w < T_WORD:
        gaps.append(((T_WORD - w) / T_WORD, "word sensitivity"))
    if p < T_PARSE:
        gaps.append(((T_PARSE - p) / T_PARSE, "parse quality"))
    return tuple(sorted(gaps, key=lambda x: -x[0]))


def primary_barrier(t: float, w: float, p: float) -> str:
    pretty = {
        "image tracking": "Image tracking",
        "word sensitivity": "Word sensitivity",
        "parse quality": "Parse quality",
    }
    if t >= T_TRACK_VALID and w >= T_WORD and p >= T_PARSE:
        return "—"
    gaps = list(normalized_gaps(t, w, p))
    if not gaps:
        return "—"
    # Single failed criterion
    if len(gaps) == 1:
        name = gaps[0][1]
        if name == "image tracking" and t >= T_TRACK_BORDER:
            return "**Image tracking** (short of 0.70 only)"
        return f"**{pretty[name]}**"
    primary_name = gaps[0][1]
    primary_g = gaps[0][0]
    parts = [f"**{pretty[primary_name]}** (largest gap vs valid bar)"]
    g2 = gaps[1][0]
    if g2 >= 0.85 * primary_g and g2 > 0.05:
        parts.append(f"**{pretty[gaps[1][1]]}** second")
    return "; ".join(parts)


def normalized_gaps_hm(t: float, p: float) -> tuple[tuple[float, str], ...]:
    gaps: list[tuple[float, str]] = []
    if t < T_TRACK_VALID:
        gaps.append(((T_TRACK_VALID - t) / T_TRACK_VALID, "image tracking"))
    if p < T_PARSE:
        gaps.append(((T_PARSE - p) / T_PARSE, "parse quality"))
    return tuple(sorted(gaps, key=lambda x: -x[0]))


def primary_barrier_hm(t: float, p: float) -> str:
    pretty = {
        "image tracking": "Image tracking",
        "parse quality": "Parse quality",
    }
    if t >= T_TRACK_VALID and p >= T_PARSE:
        return "—"
    gaps = list(normalized_gaps_hm(t, p))
    if not gaps:
        return "—"
    if len(gaps) == 1:
        name = gaps[0][1]
        if name == "image tracking" and t >= T_TRACK_BORDER:
            return "**Image tracking** (short of 0.70 only)"
        return f"**{pretty[name]}**"
    primary_name = gaps[0][1]
    primary_g = gaps[0][0]
    parts = [f"**{pretty[primary_name]}** (largest gap vs valid bar)"]
    g2 = gaps[1][0]
    if g2 >= 0.85 * primary_g and g2 > 0.05:
        parts.append(f"**{pretty[gaps[1][1]]}** second")
    return "; ".join(parts)


def r_note_hm(label: str, t: float, p: float) -> str:
    if label == "valid":
        return (
            "Image tracking ≥0.70 and parse ≥0.97. "
            "Word sensitivity gate N/A (one word per stimulus in human-matched protocol)."
        )
    if label == "borderline":
        bits = ["Tier = borderline because tracking ≥0.50."]
        miss = []
        if t < T_TRACK_VALID:
            miss.append("image tracking below 0.70")
        if p < T_PARSE:
            miss.append("parse quality below 0.97")
        if miss:
            bits.append("Blocks valid: " + "; ".join(miss) + ".")
        bits.append("Word sensitivity gate N/A for this protocol.")
        return " ".join(bits)
    bits = ["Invalid tier: tracking &lt;0.50."]
    extra = []
    if p < T_PARSE:
        extra.append("parse quality below 0.97 (often unclear answers)")
    if extra:
        bits.append("Also: " + "; ".join(extra) + ".")
    bits.append("Word sensitivity gate N/A for this protocol.")
    return " ".join(bits)


def r_note(label: str, t: float, w: float, p: float) -> str:
    if label == "valid":
        return "All gates pass."
    if label == "borderline":
        bits = ["Tier = borderline because tracking ≥0.50."]
        miss = []
        if t < T_TRACK_VALID:
            miss.append("image tracking below 0.70")
        if w < T_WORD:
            miss.append("word sensitivity below 0.20")
        if p < T_PARSE:
            miss.append("parse quality below 0.97")
        if miss:
            bits.append("Blocks valid: " + "; ".join(miss) + ".")
        return " ".join(bits)
    # invalid
    bits = ["Invalid tier: tracking &lt;0.50."]
    extra = []
    if w < T_WORD:
        extra.append("word sensitivity below 0.20")
    if p < T_PARSE:
        extra.append("parse quality below 0.97 (often unclear answers)")
    if extra:
        bits.append("Also: " + "; ".join(extra) + ".")
    return " ".join(bits)


def sort_key(row: dict) -> tuple[int, str]:
    order = {"valid": 0, "borderline": 1, "invalid": 2}
    return (order.get(row["validity_label"], 9), row["model"])


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_markdown(rows: list[dict], *, skip_word_sensitivity_gate: bool = False) -> str:
    lines: list[str] = []
    n = len(rows)
    n_valid = sum(1 for r in rows if r["validity_label"] == "valid")
    n_border = sum(1 for r in rows if r["validity_label"] == "borderline")
    n_inv = sum(1 for r in rows if r["validity_label"] == "invalid")

    lines.append("### Failure analysis: which criterion blocks “valid”?")
    lines.append("")
    if skip_word_sensitivity_gate:
        lines.append(
            "Human-matched protocol: **word sensitivity** as in `validity_gates.R` is **not applicable** "
            "(each stimulus–ordering pair has a single pseudo-word). **Valid** requires image tracking ≥0.70 "
            "and parse quality ≥0.97. **Borderline** if tracking ≥0.50 but not fully valid; **invalid** if "
            "tracking &lt;0.50."
        )
    else:
        lines.append(
            "For each model, the table uses the **valid** thresholds (0.70 / 0.20 / 0.97). "
            "A checkmark means the metric meets that bar. **Primary barrier to valid** is the failed "
            "criterion with the **largest normalized shortfall** "
            "(for each failed metric: (threshold - observed rate) / threshold). "
            "If a second metric is close behind, it is listed. "
            "For **`valid`** models, all three pass and there is no barrier."
        )
    lines.append("")
    lines.append(
        "*This block is generated by `scripts/update_models_validity_md.py` "
        "(single-table or `--from-split` mode).*"
    )
    lines.append("")
    lines.append(
        "| Model | Label | Image track ≥0.70 | Word sens. ≥0.20 | Parse ≥0.97 | "
        "Primary barrier to “valid” | Note on R label |"
    )
    lines.append(
        "|-------|-------|-------------------|------------------|-------------|"
        "----------------------------|-------------------|"
    )

    for r in sorted(rows, key=sort_key):
        model = r["model"]
        label = r["validity_label"]
        t = float(r["image_tracking_rate"])
        w = float(r["word_sensitivity_rate"])
        p = float(r["parse_quality"])
        pt = t >= T_TRACK_VALID
        pw = w >= T_WORD
        pp = p >= T_PARSE
        wc = pass_cell(pt, t, T_TRACK_VALID)
        if skip_word_sensitivity_gate:
            wc_w = "N/A (1 word / stimulus)"
        else:
            at_word_floor = pw and abs(w - T_WORD) < 1e-6
            if at_word_floor:
                wc_w = pass_cell(pw, w, T_WORD, at_floor=True)
            else:
                wc_w = pass_cell(pw, w, T_WORD)
        wc_p = pass_cell(pp, p, T_PARSE)
        if skip_word_sensitivity_gate:
            barrier = primary_barrier_hm(t, p)
            note = r_note_hm(label, t, p)
        else:
            barrier = primary_barrier(t, w, p)
            note = r_note(label, t, w, p)
        lines.append(
            f"| `{model}` | {label} | {wc} | {wc_w} | {wc_p} | {barrier} | {note} |"
        )

    lines.append("")
    if skip_word_sensitivity_gate:
        lines.append(
            "**Reading the table:** **Invalid** means image tracking &lt;0.50. **Borderline** means "
            "tracking ≥0.50 but image tracking or parse quality still below the **valid** thresholds."
        )
    else:
        lines.append(
            "**Reading the table:** For models that are **`invalid`**, the **R rule** is always "
            "“image tracking below 0.50.” The **primary barrier** column answers a different question: "
            "*if you wanted to reach the **valid** profile, which metric is furthest below its target?* "
            "That is usually still **image tracking** for low-tracking models, but for some runs "
            "**parse quality** (unclear / non-`1`/`2` answers) can show the largest gap relative to 0.97 "
            "even though the assigned label is driven by tracking &lt;0.50."
        )
    lines.append("")
    lines.append("### Updated outcomes")
    lines.append("")
    lines.append(f"- **Valid models:** {n_valid} / {n}")
    lines.append(f"- **Borderline models:** {n_border} / {n}")
    lines.append(f"- **Invalid models:** {n_inv} / {n}")
    lines.append("")
    lines.append("#### Per-model labels")
    lines.append("")
    for r in sorted(rows, key=sort_key):
        m = r["model"]
        lab = r["validity_label"]
        if lab == "valid":
            lines.append(f"- `{m}` -> **{lab}**")
        elif lab == "borderline":
            lines.append(f"- `{m}` -> **{lab}**")
        else:
            lines.append(f"- `{m}` -> {lab}")
    lines.append("")
    return "\n".join(lines)


def build_split_markdown(
    sections: list[tuple[str, str, str, list[dict]]],
) -> str:
    """Build auto section from (title, subset_note, metrics_csv_note, rows) tuples."""
    parts: list[str] = []
    parts.append(
        "*Auto-generated: run `python scripts/compute_model_validity_split.py` then "
        "`python scripts/update_models_validity_md.py --from-split`.*"
    )
    parts.append("")
    for title, subset_note, csv_note, rows in sections:
        parts.append(f"### {title}")
        parts.append("")
        parts.append(subset_note)
        parts.append("")
        parts.append(f"*Metrics CSV: `{csv_note}`.*")
        parts.append("")
        parts.append(build_markdown(rows, skip_word_sensitivity_gate=False))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def patch_md(md_path: Path, new_section: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END),
        re.DOTALL,
    )
    replacement = MARKER_START + "\n" + new_section + MARKER_END
    if not pattern.search(text):
        raise SystemExit(
            f"Markers not found in {md_path}. Expected {MARKER_START!r} ... {MARKER_END!r}"
        )
    updated = pattern.sub(replacement, text, count=1)
    md_path.write_text(updated, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to model_validity_summary.csv")
    ap.add_argument("--md", type=Path, default=DEFAULT_MD, help="Path to models_validity.md")
    ap.add_argument(
        "--from-split",
        action="store_true",
        help="Two tables: word benchmark (local+remote) + no-word trio dedup",
    )
    ap.add_argument("--split-word-csv", type=Path, default=SPLIT_CSV_WORD)
    ap.add_argument("--split-no-word-trio-csv", type=Path, default=SPLIT_CSV_NO_WORD_TRIO)
    ap.add_argument(
        "--skip-word-sensitivity-gate",
        action="store_true",
        help="Human-matched style: word column N/A; barriers/notes use tracking+parse only",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print generated section only")
    args = ap.parse_args()

    if args.from_split:
        for label, p in (
            ("word benchmark", args.split_word_csv),
            ("no-word trio", args.split_no_word_trio_csv),
        ):
            if not p.is_file():
                raise SystemExit(
                    f"Split CSV not found ({label}): {p}\n"
                    "Run: python scripts/compute_model_validity_split.py"
                )
        word_rows = load_rows(args.split_word_csv)
        trio_rows = load_rows(args.split_no_word_trio_csv)
        section = build_split_markdown(
            [
                (
                    "Word condition (noun-label benchmark: local + remote)",
                    "Merged trial rows from `results/model.results/local_eval.csv` and "
                    "`results/model.results/remote_all_fixed.csv` (noun-label / default word protocol only).",
                    "results/data/model_validity_summary_word.csv",
                    word_rows,
                ),
                (
                    "No-word — diagnostic trio (deduplicated)",
                    "Trial rows from `results/model.results/no_word_full_remote_trio_dedup.csv` "
                    "(benchmark-matched no-word control on the diagnostic trio; "
                    "see `interpret/no_word_trio_interim_report.md`).",
                    "results/data/model_validity_summary_no_word_trio.csv",
                    trio_rows,
                ),
            ]
        )
        if args.dry_run:
            print(section)
            return
        patch_md(args.md, section)
        print(
            f"Updated {args.md} from --from-split "
            f"({len(word_rows)} + {len(trio_rows)} model summary rows)"
        )
        return

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    rows = load_rows(args.csv)
    section = build_markdown(rows, skip_word_sensitivity_gate=args.skip_word_sensitivity_gate)

    if args.dry_run:
        print(section)
        return

    patch_md(args.md, section)
    print(f"Updated {args.md} from {args.csv} ({len(rows)} models)")


if __name__ == "__main__":
    main()
