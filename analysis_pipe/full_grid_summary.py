#!/usr/bin/env python3
"""Readout for the full texture-grid evaluation (session_full_grid_v1a).

The grid runs through ``scripts/run_local.py`` and writes CSVs, while the 30-set
playground writes text logs read by ``playground_figures.py`` and the canonical
R pipeline reads a fixed list of legacy filenames. Neither picks these up, so
this script produces the equivalent readout from the session CSVs.

Metrics follow ``playgrounds/probe_experiment.py`` so the 1,140-trial grid and
the 30-set results are read on the same scale:

  tracking   fraction of stimuli whose shape/texture pick is unchanged when the
             two options swap position. Below the 0.70 gate the model is
             answering by position, and its shape rate means nothing.
  shape      fraction of decided trials choosing the shape match, both orders
             pooled.
  pos_first  fraction of decided trials choosing whichever option was shown
             first. Near 1.0 or 0.0 is a position lock.

The grid adds two breakdowns the 30-set design could not support: shape rate by
STL shape (30 levels) and by texture (38 levels).

Usage:
    python analysis_pipe/full_grid_summary.py
    python analysis_pipe/full_grid_summary.py --session results/model.results/session_full_grid_v1a_pilot
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSION = REPO_ROOT / "results" / "model.results" / "session_full_grid_v1a"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "data"
TRACKING_GATE = 0.70
DECIDED = ("shape", "texture")


def _fmt(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  n/a"
    return f"{x:5.2f}"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval; the grid's n is large enough that normal CIs also
    work, but this stays consistent with the 30-set figures."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_session(session_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(session_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {session_dir}")
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["_source"] = path.name
                rows.append(row)
    return rows


def cell_key(row: dict) -> tuple[str, str, str]:
    return (row.get("model", ""), row.get("prompt_condition", ""), row.get("word", ""))


def summarize_cell(rows: list[dict]) -> dict:
    decided = [r for r in rows if r.get("choice") in DECIDED]

    # Tracking needs both orders for the same stimulus, so pair on stim_id.
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
    shape_rate = n_shape / len(decided) if decided else float("nan")
    lo, hi = _wilson_ci(n_shape, len(decided))
    pos_first = _mean([1.0 if r["choice"] == r.get("a_is") else 0.0 for r in decided])

    return {
        "n_rows": len(rows),
        "n_decided": len(decided),
        "n_stimuli": len(by_stim),
        "n_paired": len(track_vals),
        "parse_rate": len(decided) / len(rows) if rows else float("nan"),
        "tracking": tracking,
        "gate_pass": (not math.isnan(tracking)) and tracking >= TRACKING_GATE,
        "shape_rate": shape_rate,
        "shape_ci_lo": lo,
        "shape_ci_hi": hi,
        "pos_first": pos_first,
    }


def breakdown(rows: list[dict], field: str) -> list[dict]:
    """Shape rate per level of ``field`` (stl_id or texture_set), per cell."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("choice") not in DECIDED:
            continue
        level = r.get(field, "")
        if not level:
            continue
        groups[(*cell_key(r), level)].append(r)

    out = []
    for (model, cond, word, level), rs in sorted(groups.items()):
        n_shape = sum(1 for r in rs if r["choice"] == "shape")
        lo, hi = _wilson_ci(n_shape, len(rs))
        out.append({
            "model": model, "prompt_condition": cond, "word": word,
            "grouping": field, "level": level,
            "n_decided": len(rs), "shape_rate": n_shape / len(rs),
            "shape_ci_lo": lo, "shape_ci_hi": hi,
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION,
                        help=f"Session directory of run_local CSVs (default: {DEFAULT_SESSION})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Where to write the summary CSVs (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--prefix", default=None,
                        help="Output filename prefix (default: the session directory name)")
    args = parser.parse_args()

    rows = read_session(args.session)
    prefix = args.prefix or args.session.name.replace("session_", "")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[cell_key(r)].append(r)

    summary = []
    for (model, cond, word), rs in sorted(cells.items()):
        summary.append({"model": model, "prompt_condition": cond, "word": word,
                        **summarize_cell(rs)})

    print(f"Session: {args.session}")
    print(f"Rows:    {len(rows)} across {len(cells)} model x condition cells\n")
    header = (f"{'model':<15} {'condition':<24} {'word':<8} "
              f"{'n':>6} {'parse':>6} {'track':>6} {'gate':>5} {'shape':>6} "
              f"{'95% CI':>14} {'pos1st':>7}")
    print(header)
    print("-" * len(header))
    for s in summary:
        gate = "PASS" if s["gate_pass"] else "fail"
        ci = f"[{s['shape_ci_lo']:.2f},{s['shape_ci_hi']:.2f}]"
        print(f"{s['model']:<15} {s['prompt_condition']:<24} {s['word'] or '-':<8} "
              f"{s['n_decided']:>6} {_fmt(s['parse_rate']):>6} {_fmt(s['tracking']):>6} "
              f"{gate:>5} {_fmt(s['shape_rate']):>6} {ci:>14} {_fmt(s['pos_first']):>7}")

    n_pass = sum(1 for s in summary if s["gate_pass"])
    print(f"\nGate (tracking >= {TRACKING_GATE}): {n_pass}/{len(summary)} cells pass. "
          "Shape rate is interpretable only in those.")

    summary_path = args.out_dir / f"{prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    bd = breakdown(rows, "stl_id") + breakdown(rows, "texture_set")
    bd_path = args.out_dir / f"{prefix}_by_shape_texture.csv"
    if bd:
        with open(bd_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bd[0].keys()))
            w.writeheader()
            w.writerows(bd)
        n_shapes = len({r["level"] for r in bd if r["grouping"] == "stl_id"})
        n_tex = len({r["level"] for r in bd if r["grouping"] == "texture_set"})
        print(f"\nBreakdown: {n_shapes} shapes x {n_tex} textures -> {bd_path}")
    else:
        print("\nBreakdown skipped: no stl_id / texture_set columns "
              "(these CSVs predate the grid run).")

    print(f"Summary:   {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
