#!/usr/bin/env python3
"""Compute validity gates for human-matched remote CSVs (same rules as validity_gates.R).

Loads v1 + v2 batches, disambiguates ``stim_id`` with ``stim_pkg`` for the combined
metrics so pairs from different packages never collide.

Usage (repo root):
    python scripts/compute_validity_human_matched.py
    python scripts/compute_validity_human_matched.py \\
        results/model.results/human_matched/a.csv results/.../b.csv

Writes:
    results/data/model_validity_summary_human_matched.csv  (v1+v2 combined)
    results/data/model_validity_summary_human_matched_v1.csv
    results/data/model_validity_summary_human_matched_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HM_DIR = REPO_ROOT / "results" / "model.results" / "human_matched"
DATA_DIR = REPO_ROOT / "results" / "data"

T_TRACK_VALID = 0.70
T_TRACK_BORDER = 0.50
T_WORD = 0.20
T_PARSE = 0.97

FNAME_RE = re.compile(
    r"^remote_human_(v[12])_(\d{8}_\d{6})_models\.csv$"
)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def disambiguate_stim(rows: list[dict]) -> None:
    for r in rows:
        pkg = r.get("stim_pkg") or ""
        sid = r.get("stim_id") or ""
        r["stim_id"] = f"{pkg}/{sid}" if pkg else sid


def _is_deterministic_order_row(r: dict) -> bool:
    """Legacy benchmark CSVs used ``order_method=fixed``; current runs use ``deterministic``."""
    om = (r.get("order_method") or "").strip()
    return om in ("deterministic", "fixed")


def compute_image_tracking(rows: list[dict]) -> dict[str, tuple[int, float]]:
    """Per model: (n_pairs, image_tracking_rate)."""
    # key -> {ordering: parsed_answer}
    buckets: dict[tuple[str, str, str], dict[str, str]] = defaultdict(dict)
    for r in rows:
        if not _is_deterministic_order_row(r):
            continue
        o = r.get("ordering")
        if o not in ("shape_first", "texture_first"):
            continue
        key = (r["model"], r["stim_id"], r["word"])
        pa = (r.get("parsed_answer") or "").strip()
        buckets[key][o] = pa

    by_model: dict[str, list[int]] = defaultdict(list)
    for key, d in buckets.items():
        if len(d.get("shape_first", "")) != 1 or len(d.get("texture_first", "")) != 1:
            continue
        sf = d["shape_first"]
        tf = d["texture_first"]
        if not sf or not tf:
            continue
        model = key[0]
        track = 1 if (sf == "1" and tf == "2") or (sf == "2" and tf == "1") else 0
        by_model[model].append(track)

    out: dict[str, tuple[int, float]] = {}
    for m, vals in by_model.items():
        n = len(vals)
        out[m] = (n, sum(vals) / n if n else float("nan"))
    return out


def compute_word_sensitivity(rows: list[dict]) -> dict[str, tuple[int, float]]:
    """Per model: (n_groups, word_sensitivity_rate)."""
    # (model, stim_id, ordering) -> set of choices
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in rows:
        ch = (r.get("choice") or "").strip()
        if ch not in ("shape", "texture"):
            continue
        key = (r["model"], r["stim_id"], r.get("ordering") or "")
        groups[key].add(ch)

    by_model: dict[str, list[int]] = defaultdict(list)
    for key, choices in groups.items():
        sens = 1 if len(choices) > 1 else 0
        by_model[key[0]].append(sens)

    out: dict[str, tuple[int, float]] = {}
    for m, vals in by_model.items():
        n = len(vals)
        out[m] = (n, sum(vals) / n if n else float("nan"))
    return out


def compute_parse_quality(rows: list[dict]) -> dict[str, tuple[int, float, float, float]]:
    """Per model: (n_trials, unclear_rate, retry_rate, parse_quality)."""
    by_model: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in rows:
        ch = (r.get("choice") or "").strip()
        try:
            att = int(r.get("attempts") or 0)
        except ValueError:
            att = 0
        by_model[r["model"]].append((ch, att))

    out: dict[str, tuple[int, float, float, float]] = {}
    for m, pairs in by_model.items():
        n = len(pairs)
        unclear = sum(1 for ch, _ in pairs if ch == "unclear") / n if n else 0.0
        retry = sum(1 for _, a in pairs if a > 1) / n if n else 0.0
        pq = 1.0 - unclear
        out[m] = (n, unclear, retry, pq)
    return out


def label_human_matched(t: float, p: float) -> str:
    """Validity tier for human-matched protocol (see interpret/human_matched_validity.md).

    Word sensitivity as defined in ``validity_gates.R`` is **not applicable**: each
    (stimulus, ordering) appears with a **single** pseudo-word, so the rate is
    structurally 0. Use **image tracking + parse quality** only.
    """
    if t >= T_TRACK_VALID and p >= T_PARSE:
        return "valid"
    if t >= T_TRACK_BORDER:
        return "borderline"
    return "invalid"


def merge_validity(
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
                "validity_label": label_human_matched(
                    tr if tr == tr else -1.0,
                    pq if pq == pq else -1.0,
                ),
            }
        )
    return rows_out


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


def find_latest_pair() -> tuple[Path, Path] | None:
    by_ts: dict[str, dict[str, Path]] = defaultdict(dict)
    if not HM_DIR.is_dir():
        return None
    for p in HM_DIR.iterdir():
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        vx, ts = m.group(1), m.group(2)
        by_ts[ts][vx] = p
    for ts in sorted(by_ts.keys(), reverse=True):
        d = by_ts[ts]
        if "v1" in d and "v2" in d:
            return d["v1"], d["v2"]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional CSV paths; default = latest v1+v2 pair in human_matched/",
    )
    args = ap.parse_args()

    if args.inputs:
        paths = [Path(p).resolve() for p in args.inputs]
        for p in paths:
            if not p.is_file():
                raise SystemExit(f"Missing file: {p}")
    else:
        pair = find_latest_pair()
        if not pair:
            raise SystemExit(
                f"No remote_human_v1_* / v2_* pair found under {HM_DIR}. "
                "Pass CSV paths explicitly."
            )
        paths = list(pair)
        print(f"Using latest pair: {paths[0].name} + {paths[1].name}")

    combined: list[dict] = []
    for p in paths:
        rows = load_csv(p)
        combined.extend(rows)

    rows_v1 = [r for r in combined if r.get("stim_pkg", "").endswith("_v1")]
    rows_v2 = [r for r in combined if r.get("stim_pkg", "").endswith("_v2")]

    def run_block(block: list[dict]) -> list[dict[str, str]]:
        b = [dict(r) for r in block]
        disambiguate_stim(b)
        models = sorted({r["model"] for r in b})
        track = compute_image_tracking(b)
        word = compute_word_sensitivity(b)
        parse = compute_parse_quality(b)
        return merge_validity(models, track, word, parse)

    summ_combined = run_block(combined)
    summ_v1 = run_block(rows_v1) if rows_v1 else []
    summ_v2 = run_block(rows_v2) if rows_v2 else []

    write_summary(DATA_DIR / "model_validity_summary_human_matched.csv", summ_combined)
    print(f"Wrote {DATA_DIR / 'model_validity_summary_human_matched.csv'}")
    if summ_v1:
        write_summary(DATA_DIR / "model_validity_summary_human_matched_v1.csv", summ_v1)
        print(f"Wrote {DATA_DIR / 'model_validity_summary_human_matched_v1.csv'}")
    if summ_v2:
        write_summary(DATA_DIR / "model_validity_summary_human_matched_v2.csv", summ_v2)
        print(f"Wrote {DATA_DIR / 'model_validity_summary_human_matched_v2.csv'}")

    print(f"Combined trial rows: {len(combined)}")
    for r in summ_combined:
        print(
            f"  {r['model']:20s}  track={float(r['image_tracking_rate']):.3f}  "
            f"word={float(r['word_sensitivity_rate']):.3f}  "
            f"parse={float(r['parse_quality']):.3f}  -> {r['validity_label']}"
        )


if __name__ == "__main__":
    main()
