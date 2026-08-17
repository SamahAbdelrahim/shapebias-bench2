#!/usr/bin/env python3
"""Per-triad model shape rates for the novel grid, so model items can be joined
to the human item means.

``full_grid_summary.py`` aggregates each cell to one number and breaks it down by
shape and by texture, but never by triad, so the human item means in
``results/data/human_item_means.csv`` (keyed on stim_id) had nothing to join to.
This writes one row per model x prompt cell x stim_id.

Output: results/data/full_grid_item_rates.csv

Run:  .venv/bin/python analysis_pipe/item_rates.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = REPO / "results" / "model.results" / "session_full_grid_v1a"
DEFAULT_OUT = REPO / "results" / "data" / "full_grid_item_rates.csv"
DECIDED = ("shape", "texture")


def build(session: Path) -> list[dict]:
    files = sorted(session.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {session}")
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("choice") not in DECIDED:
                    continue
                key = (r.get("model", ""), r.get("prompt_condition", ""),
                       r.get("stim_id", ""))
                groups[key].append(r)

    out = []
    for (model, cond, stim_id), rs in sorted(groups.items()):
        n_shape = sum(1 for r in rs if r["choice"] == "shape")
        first = rs[0]
        out.append({
            "model": model,
            "prompt_condition": cond,
            "stim_id": stim_id,
            "stl_id": first.get("stl_id", ""),
            "texture_set": first.get("texture_set", ""),
            "n_decided": len(rs),
            "shape_rate": n_shape / len(rs),
        })
    return out


def load_or_build(out: Path = DEFAULT_OUT, session: Path = DEFAULT_SESSION) -> list[dict]:
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    rows = build(session)
    write(rows, out)
    return [{k: str(v) for k, v in r.items()} for r in rows]


def write(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    rows = build(args.session)
    write(rows, args.out)
    n_items = len({r["stim_id"] for r in rows})
    n_cells = len({(r["model"], r["prompt_condition"]) for r in rows})
    print(f"  {n_items} triads x {n_cells} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
