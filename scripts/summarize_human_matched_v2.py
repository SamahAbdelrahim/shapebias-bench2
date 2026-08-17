#!/usr/bin/env python3
"""Summarize matched_v2 human trials when R/Quarto is unavailable.

Mirrors analysis_pipe/src/human_analysis.R for the matched_v2 path:
  catch exclusion (max_errors=1), set x condition rates with Wilson CIs,
  item means, and the between-participant position check.

Writes:
  results/data/human_catch_by_participant.csv
  results/data/human_summary_by_set.csv
  results/data/human_item_means.csv
  results/data/human_position_check.csv

Example:
  .venv/bin/python scripts/summarize_human_matched_v2.py \\
    --input results/human.results/full_grid/samah.shape_bias_human_trials.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "results" / "data"
DEFAULT_INPUT = (
    REPO / "results" / "human.results" / "full_grid" / "samah.shape_bias_human_trials.csv"
)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "t", "yes"}


def to_float(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def participant_id(row: dict) -> str:
    return f"{row.get('prolific_pid', '')}|{row.get('study_id', '')}|{row.get('session_id', '')}"


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        design = row.get("design") or row.get("raw_trial.design") or ""
        if design and design != "matched_v2":
            continue
        cleaned = {
            "prolific_pid": row.get("prolific_pid") or row.get("raw_trial.prolific_pid") or "",
            "study_id": row.get("study_id") or row.get("raw_trial.study_id") or "",
            "session_id": row.get("session_id") or row.get("raw_trial.session_id") or "",
            "condition": row.get("condition") or row.get("raw_trial.condition") or "",
            "design": design or "matched_v2",
            "stim_set_name": row.get("stim_set_name") or row.get("raw_trial.stim_set_name") or "",
            "pool_version": row.get("pool_version") or row.get("raw_trial.pool_version") or "",
            "stim_id": row.get("stim_id") or row.get("raw_trial.stim_id") or "",
            "stl_id": row.get("stl_id") or row.get("raw_trial.stl_id") or "",
            "texture_set": row.get("texture_set") or row.get("raw_trial.texture_set") or "",
            "ordering": row.get("ordering") or row.get("raw_trial.ordering") or "",
            "ordering_group": row.get("ordering_group") or row.get("raw_trial.ordering_group") or "",
            "is_catch": as_bool(row.get("is_catch", row.get("raw_trial.is_catch"))),
            "catch_correct": as_bool(row.get("catch_correct", row.get("raw_trial.catch_correct"))),
            "choice": (row.get("choice") or "").strip().lower(),
            "response_key": str(row.get("response_key") or row.get("raw_trial.response") or ""),
            "rt_ms": to_float(row.get("rt_ms") or row.get("raw_trial.rt")),
            "trial_index": row.get("trial_index") or row.get("raw_trial.trial_index") or "",
        }
        cleaned["participant_id"] = participant_id(cleaned)
        out.append(cleaned)
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize_catch(rows: list[dict], max_errors: int = 1) -> list[dict]:
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["is_catch"]:
            by_pid[row["participant_id"]].append(row)

    # Also include participants with zero logged catch rows.
    for row in rows:
        by_pid.setdefault(row["participant_id"], [])

    out = []
    for pid, catches in sorted(by_pid.items()):
        sample = next((r for r in rows if r["participant_id"] == pid), {})
        catch_trials = len(catches)
        catch_passed = sum(1 for r in catches if r["catch_correct"])
        catch_errors = catch_trials - catch_passed
        # Missing catch rows count as incomplete; exclude if below expected 4
        # only via error rule when rows exist. Incomplete sessions are flagged.
        out.append({
            "participant_id": pid,
            "prolific_pid": sample.get("prolific_pid", ""),
            "study_id": sample.get("study_id", ""),
            "session_id": sample.get("session_id", ""),
            "condition": sample.get("condition", ""),
            "ordering_group": sample.get("ordering_group", ""),
            "catch_trials": catch_trials,
            "catch_passed": catch_passed,
            "catch_errors": catch_errors,
            "excluded": catch_errors > max_errors,
            "n_total_rows": sum(1 for r in rows if r["participant_id"] == pid),
            "n_test_rows": sum(1 for r in rows if r["participant_id"] == pid and not r["is_catch"]),
        })
    out.sort(key=lambda r: (-r["catch_errors"], r["participant_id"]))
    return out


def apply_exclusions(rows: list[dict], catch_summary: list[dict]) -> list[dict]:
    excluded = {r["participant_id"] for r in catch_summary if r["excluded"]}
    return [
        r for r in rows
        if r["participant_id"] not in excluded and not r["is_catch"]
    ]


def summarize_by_set(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["stim_set_name"], row["condition"])].append(row)

    out = []
    for (stim_set, condition), items in sorted(groups.items()):
        n_shape = sum(1 for r in items if r["choice"] == "shape")
        n_texture = sum(1 for r in items if r["choice"] == "texture")
        n_unclear = sum(1 for r in items if r["choice"] == "unclear")
        n = len(items)
        n_decided = n_shape + n_texture
        lo, hi = wilson_ci(n_shape, n_decided)
        rts = [r["rt_ms"] for r in items if not math.isnan(r["rt_ms"])]
        out.append({
            "stim_set_name": stim_set,
            "condition": condition,
            "participants": len({r["participant_id"] for r in items}),
            "trials": n,
            "unique_stimuli": len({r["stim_id"] for r in items}),
            "shape_prop": n_shape / n if n else float("nan"),
            "shape_lo": lo,
            "shape_hi": hi,
            "texture_prop": n_texture / n if n else float("nan"),
            "unclear_rate": n_unclear / n if n else float("nan"),
            "median_rt_ms": statistics.median(rts) if rts else float("nan"),
            "mean_rt_ms": statistics.fmean(rts) if rts else float("nan"),
        })
    return out


def summarize_items(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["stim_set_name"], row["condition"], row["stim_id"],
            row["stl_id"], row["texture_set"],
        )
        groups[key].append(row)

    out = []
    for key, items in sorted(groups.items()):
        stim_set, condition, stim_id, stl_id, texture_set = key
        shape_first = [r for r in items if r["ordering"] == "shape_first"]
        texture_first = [r for r in items if r["ordering"] == "texture_first"]
        shape_prop = mean_eq(items, "shape")
        shape_prop_sf = mean_eq(shape_first, "shape")
        shape_prop_tf = mean_eq(texture_first, "shape")
        content = (
            1.0 - abs(shape_prop_sf - shape_prop_tf)
            if shape_first and texture_first
            else float("nan")
        )
        out.append({
            "stim_set_name": stim_set,
            "condition": condition,
            "stim_id": stim_id,
            "stl_id": stl_id,
            "texture_set": texture_set,
            "n": len(items),
            "shape_prop": shape_prop,
            "n_shape_first": len(shape_first),
            "shape_prop_shape_first": shape_prop_sf,
            "n_texture_first": len(texture_first),
            "shape_prop_texture_first": shape_prop_tf,
            "option1_rate": mean_key(items, "1"),
            "content_consistency": content,
        })
    return out


def mean_eq(items: list[dict], choice: str) -> float:
    if not items:
        return float("nan")
    return sum(1 for r in items if r["choice"] == choice) / len(items)


def mean_key(items: list[dict], key: str) -> float:
    if not items:
        return float("nan")
    return sum(1 for r in items if r["response_key"] == key) / len(items)


def summarize_position(item_means: list[dict], gate: float = 0.70) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in item_means:
        groups[(row["stim_set_name"], row["condition"])].append(row)

    out = []
    for (stim_set, condition), items in sorted(groups.items()):
        both = [
            r for r in items
            if r["n_shape_first"] > 0 and r["n_texture_first"] > 0
            and not math.isnan(r["content_consistency"])
        ]
        above = sum(1 for r in both if r["content_consistency"] >= gate)
        consist = [r["content_consistency"] for r in both]
        opt1 = [r["option1_rate"] for r in items if not math.isnan(r["option1_rate"])]
        out.append({
            "stim_set_name": stim_set,
            "condition": condition,
            "items": len(items),
            "items_with_both_orderings": len(both),
            "mean_content_consistency": statistics.fmean(consist) if consist else float("nan"),
            "items_above_gate": above,
            "mean_option1_rate": statistics.fmean(opt1) if opt1 else float("nan"),
            "prop_items_above_gate": (above / len(both)) if both else float("nan"),
        })
    return out


def audit_attention(rows: list[dict], catch_summary: list[dict]) -> dict:
    """Extra attention-check diagnostics beyond the exclusion table."""
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pid[row["participant_id"]].append(row)

    catch_n_dist = defaultdict(int)
    error_dist = defaultdict(int)
    choice_on_catch = defaultdict(int)
    incomplete = []
    zero_catch = []
    for pid, trials in by_pid.items():
        catches = [t for t in trials if t["is_catch"]]
        tests = [t for t in trials if not t["is_catch"]]
        catch_n_dist[len(catches)] += 1
        errors = sum(1 for t in catches if not t["catch_correct"])
        error_dist[errors] += 1
        for t in catches:
            choice_on_catch[t["choice"] or "(empty)"] += 1
        if len(tests) != 27 or len(catches) != 4:
            incomplete.append({
                "participant_id": pid,
                "n_test": len(tests),
                "n_catch": len(catches),
                "n_total": len(trials),
                "condition": trials[0]["condition"] if trials else "",
            })
        if len(catches) == 0:
            zero_catch.append(pid)

    excluded = [r for r in catch_summary if r["excluded"]]
    return {
        "n_sessions": len(by_pid),
        "catch_trials_per_session": dict(sorted(catch_n_dist.items())),
        "catch_error_counts": dict(sorted(error_dist.items())),
        "catch_choice_counts": dict(choice_on_catch),
        "n_excluded": len(excluded),
        "excluded_pids": [r["prolific_pid"] for r in excluded],
        "excluded_detail": excluded,
        "incomplete_or_odd_sessions": incomplete,
        "zero_catch_sessions": zero_catch,
        "pass_all_4": error_dist.get(0, 0),
        "fail_exactly_1": error_dist.get(1, 0),
        "fail_2_plus": sum(v for k, v in error_dist.items() if k >= 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-errors", type=int, default=1)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No matched_v2 rows in {args.input}")

    catch_summary = summarize_catch(rows, max_errors=args.max_errors)
    audit = audit_attention(rows, catch_summary)
    kept = apply_exclusions(rows, catch_summary)
    by_set = summarize_by_set(kept)
    items = summarize_items(kept)
    position = summarize_position(items)

    write_csv(
        args.out_dir / "human_catch_by_participant.csv",
        catch_summary,
        [
            "participant_id", "prolific_pid", "study_id", "session_id",
            "condition", "ordering_group", "catch_trials", "catch_passed",
            "catch_errors", "excluded", "n_total_rows", "n_test_rows",
        ],
    )
    write_csv(
        args.out_dir / "human_summary_by_set.csv",
        by_set,
        [
            "stim_set_name", "condition", "participants", "trials",
            "unique_stimuli", "shape_prop", "shape_lo", "shape_hi",
            "texture_prop", "unclear_rate", "median_rt_ms", "mean_rt_ms",
        ],
    )
    write_csv(
        args.out_dir / "human_item_means.csv",
        items,
        [
            "stim_set_name", "condition", "stim_id", "stl_id", "texture_set",
            "n", "shape_prop", "n_shape_first", "shape_prop_shape_first",
            "n_texture_first", "shape_prop_texture_first", "option1_rate",
            "content_consistency",
        ],
    )
    write_csv(
        args.out_dir / "human_position_check.csv",
        position,
        [
            "stim_set_name", "condition", "items", "items_with_both_orderings",
            "mean_content_consistency", "items_above_gate",
            "mean_option1_rate", "prop_items_above_gate",
        ],
    )

    print("=== attention checks ===")
    print(f"sessions: {audit['n_sessions']}")
    print(f"catch trials per session: {audit['catch_trials_per_session']}")
    print(f"catch error counts (0=all pass): {audit['catch_error_counts']}")
    print(f"catch choices: {audit['catch_choice_counts']}")
    print(
        f"pass all 4: {audit['pass_all_4']}; "
        f"fail exactly 1 (kept): {audit['fail_exactly_1']}; "
        f"fail 2+ (excluded): {audit['fail_2_plus']}"
    )
    print(f"excluded: {audit['n_excluded']}")
    if audit["excluded_detail"]:
        for r in audit["excluded_detail"]:
            print(
                f"  EXCLUDE {r['prolific_pid']} cond={r['condition']} "
                f"errors={r['catch_errors']}/{r['catch_trials']} "
                f"rows={r['n_total_rows']}"
            )
    if audit["incomplete_or_odd_sessions"]:
        print(f"odd session lengths ({len(audit['incomplete_or_odd_sessions'])}):")
        for r in audit["incomplete_or_odd_sessions"]:
            print(
                f"  {r['participant_id'][:24]}... "
                f"test={r['n_test']} catch={r['n_catch']} total={r['n_total']} "
                f"cond={r['condition']}"
            )
    else:
        print("all sessions have 27 test + 4 catch trials")

    print("\n=== retained sample (after catch exclusion) ===")
    print(f"participants kept: {len({r['participant_id'] for r in kept})}")
    print(f"test trials kept: {len(kept)}")
    for row in by_set:
        print(
            f"  {row['stim_set_name']} / {row['condition']}: "
            f"shape={row['shape_prop']:.3f} "
            f"[{row['shape_lo']:.3f}, {row['shape_hi']:.3f}] "
            f"n_part={row['participants']} n_trial={row['trials']} "
            f"stimuli={row['unique_stimuli']} median_rt={row['median_rt_ms']:.0f}ms"
        )
    print("\n=== position check ===")
    for row in position:
        print(
            f"  {row['stim_set_name']} / {row['condition']}: "
            f"content_consistency={row['mean_content_consistency']:.3f} "
            f"above_gate={row['items_above_gate']}/{row['items_with_both_orderings']} "
            f"({row['prop_items_above_gate']:.3f}) "
            f"option1={row['mean_option1_rate']:.3f}"
        )
    print(f"\nwrote CSVs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
