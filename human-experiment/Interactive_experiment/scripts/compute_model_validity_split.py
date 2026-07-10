#!/usr/bin/env python3
"""Compute benchmark validity summaries aligned with the main run sequence.

1. **Word (noun-label) benchmark — local + remote:** merges ``local_eval.csv`` and
   ``remote_all_fixed.csv`` (noun-label trials only).
2. **No-word diagnostic trio:** ``no_word_full_remote_trio_dedup.csv`` (matched benchmark
   protocol; see ``interpret/no_word_trio_interim_report.md``).

Writes:

- ``results/data/model_validity_summary_word.csv``
- ``results/data/model_validity_summary_no_word_trio.csv``

Gates match ``analysis_pipe/src/validity_gates.R``. Human-matched remote validity is
documented separately in ``interpret/human_matched_validity.md``.

Usage (repo root):
    python scripts/compute_model_validity_split.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_RESULTS = REPO_ROOT / "results" / "model.results"
DATA_DIR = REPO_ROOT / "results" / "data"

PATH_LOCAL = MODEL_RESULTS / "local_eval.csv"
PATH_REMOTE_WORD = MODEL_RESULTS / "remote_all_fixed.csv"
PATH_NO_WORD_TRIO = MODEL_RESULTS / "no_word_full_remote_trio_dedup.csv"

OUT_WORD = DATA_DIR / "model_validity_summary_word.csv"
OUT_NO_WORD_TRIO = DATA_DIR / "model_validity_summary_no_word_trio.csv"

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


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_prompt_condition(row: dict) -> str:
    pc = (row.get("prompt_condition") or "").strip()
    return pc if pc else "noun_label"


def rows_word_benchmark() -> list[dict]:
    """Noun-label rows from local + remote fixed benchmark CSVs."""
    out: list[dict] = []
    for path, default_pc in (
        (PATH_LOCAL, "noun_label"),
        (PATH_REMOTE_WORD, "noun_label"),
    ):
        if not path.is_file():
            continue
        for r in load_csv(path):
            pc = normalize_prompt_condition(r)
            if pc != "noun_label":
                continue
            rr = dict(r)
            rr["prompt_condition"] = "noun_label"
            rr["source_file"] = path.name
            out.append(rr)
    return out


def rows_no_word_trio() -> list[dict]:
    if not PATH_NO_WORD_TRIO.is_file():
        return []
    out = []
    for r in load_csv(PATH_NO_WORD_TRIO):
        rr = dict(r)
        rr["source_file"] = PATH_NO_WORD_TRIO.name
        out.append(rr)
    return out


def label_benchmark(t: float, w: float, p: float) -> str:
    if t != t or w != w or p != p:
        return "invalid"
    if t >= T_TRACK_VALID and w >= T_WORD and p >= T_PARSE:
        return "valid"
    if t >= T_TRACK_BORDER:
        return "borderline"
    return "invalid"


def merge_validity_benchmark(
    models: list[str],
    track: dict[str, tuple[int, float]],
    word: dict[str, tuple[int, float]],
    parse: dict[str, tuple[int, float, float, float]],
) -> list[dict[str, str]]:
    rows_out: list[dict[str, str]] = []
    for m in sorted(models):
        np, tr = track.get(m, (0, float("nan")))
        ng, wr = word.get(m, (0, float("nan")))
        nt, ur, rr, pq = parse.get(m, (0, 0.0, 0.0, float("nan")))
        tr_e = tr if tr == tr else -1.0
        wr_e = wr if wr == wr else -1.0
        pq_e = pq if pq == pq else -1.0
        rows_out.append(
            {
                "model": m,
                "n_pairs": str(np),
                "image_tracking_rate": str(tr),
                "n_groups": str(ng),
                "word_sensitivity_rate": str(wr),
                "n_trials": str(nt),
                "unclear_rate": str(ur),
                "retry_rate": str(rr),
                "parse_quality": str(pq),
                "validity_label": label_benchmark(tr_e, wr_e, pq_e),
            }
        )
    return rows_out


def summarize_rows(rows: list[dict], cvhm) -> list[dict[str, str]]:
    if not rows:
        return []
    block = [dict(r) for r in rows]
    cvhm.disambiguate_stim(block)
    models = sorted({r["model"] for r in block})
    track = cvhm.compute_image_tracking(block)
    word = cvhm.compute_word_sensitivity(block)
    parse = cvhm.compute_parse_quality(block)
    return merge_validity_benchmark(models, track, word, parse)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
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
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    cvhm = _load_cvhm()

    word_rows = rows_word_benchmark()
    trio_rows = rows_no_word_trio()

    if not word_rows and not PATH_LOCAL.is_file() and not PATH_REMOTE_WORD.is_file():
        print("Error: need at least one of local_eval.csv or remote_all_fixed.csv", file=sys.stderr)
        sys.exit(1)

    summ_word = summarize_rows(word_rows, cvhm)
    summ_trio = summarize_rows(trio_rows, cvhm)

    write_summary(OUT_WORD, summ_word)
    write_summary(OUT_NO_WORD_TRIO, summ_trio)

    print(f"Wrote {OUT_WORD} ({len(summ_word)} models, {len(word_rows)} trial rows)")
    print(f"Wrote {OUT_NO_WORD_TRIO} ({len(summ_trio)} models, {len(trio_rows)} trial rows)")
    if not PATH_NO_WORD_TRIO.is_file():
        print(f"Warning: missing {PATH_NO_WORD_TRIO} — no-word trio summary is empty.", file=sys.stderr)


if __name__ == "__main__":
    main()
