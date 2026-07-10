#!/usr/bin/env python3
"""Rerun failed/unclear trials from a previous remote evaluation CSV.

Reads the CSV, identifies rows with choice='unclear', reruns those specific
trials, and replaces the failed rows in the CSV with the new results.

Usage:
    python scripts/rerun_failed.py results/remote_20260326_150727.csv
    python scripts/rerun_failed.py results/remote_20260326_150727.csv --workers 15
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    CSV_FIELDS,
    ENV_PATH,
    HUMAN_STIM_PACKAGES,
    RANDOM_SEED,
    load_stimuli,
    load_stimuli_human_package,
    make_prompt,
    resolve_stim_set_name,
    run_with_retry,
)
from scripts.run_remote import REMOTE_MODELS, run_remote

load_dotenv(ENV_PATH)


def main():
    parser = argparse.ArgumentParser(description="Rerun failed/unclear trials from a CSV")
    parser.add_argument("csv_file", help="Path to the CSV with failed trials")
    parser.add_argument("--workers", type=int, default=15,
                        help="Number of parallel workers (default: 15)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    random.seed(args.seed)
    csv_path = Path(args.csv_file)

    # Read all rows
    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))

    # Find failed rows
    failed = [(i, row) for i, row in enumerate(all_rows) if row["choice"] == "unclear"]
    print(f"Total rows: {len(all_rows)}")
    print(f"Failed/unclear: {len(failed)}")

    if not failed:
        print("Nothing to rerun!")
        return

    # Count by model
    from collections import Counter
    model_counts = Counter(row["model"] for _, row in failed)
    for m, c in model_counts.most_common():
        print(f"  {m}: {c} failed")

    # Load all stimuli (we need to look up by stim_id)
    sample = next((r for r in all_rows if r.get("stim_id")), None)
    if (
        sample
        and sample.get("eval_mode") == "human_matched"
        and sample.get("stim_pkg") in HUMAN_STIM_PACKAGES
    ):
        pkg = sample["stim_pkg"]
        stim_set = resolve_stim_set_name(sample.get("stim_set") or None)
        stimuli = load_stimuli_human_package(pkg, stim_set)
        print(f"Loaded human_matched stimuli: {pkg}/{stim_set} ({len(stimuli)} items)")
    else:
        stimuli = load_stimuli()
    stim_by_id = {s["stim_id"]: s for s in stimuli}

    # Rerun failed trials
    print(f"\nRerunning {len(failed)} trials with {args.workers} workers...")
    results_lock = threading.Lock()
    fixed = 0
    still_failed = 0

    def rerun_one(idx_row):
        idx, row = idx_row
        model_key = row["model"]
        stim_id = row["stim_id"]
        word = row["word"]
        ordering = row["ordering"]
        a_is = row["a_is"]
        b_is = row["b_is"]

        if model_key not in REMOTE_MODELS:
            return idx, row, False  # can't rerun non-remote model

        stim = stim_by_id.get(stim_id)
        if stim is None:
            return idx, row, False

        ref = stim["reference"]
        if ordering == "shape_first":
            images = [ref, stim["shape_match"], stim["texture_match"]]
        else:
            images = [ref, stim["texture_match"], stim["shape_match"]]

        pc = row.get("prompt_condition") or "noun_label"
        prompt = make_prompt(word, prompt_condition=pc)

        def run_fn(imgs, p, _mk=model_key):
            return run_remote(_mk, imgs, p, temperature=float(row.get("temperature") or 0.0))

        res = run_with_retry(run_fn, images, prompt)
        answer = res.get("parsed_answer")
        if answer == "1":
            choice = a_is
        elif answer == "2":
            choice = b_is
        else:
            choice = "unclear"

        # Build updated row
        new_row = dict(row)
        new_row["raw_text"] = res["raw_text"]
        new_row["parsed_answer"] = res.get("parsed_answer", "")
        new_row["choice"] = choice
        new_row["generation_time_s"] = res["generation_time_s"]
        new_row["num_tokens_generated"] = res.get("num_tokens_generated", "")
        new_row["attempts"] = res.get("attempts", "")

        return idx, new_row, choice != "unclear"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(rerun_one, item): item for item in failed}

        for future in as_completed(futures):
            idx, new_row, was_fixed = future.result()
            with results_lock:
                all_rows[idx] = new_row
                if was_fixed:
                    fixed += 1
                    status = "FIXED"
                else:
                    still_failed += 1
                    status = "STILL UNCLEAR"
                print(f"  [{fixed + still_failed}/{len(failed)}] "
                      f"stim={new_row['stim_id']} word={new_row['word']} "
                      f"{new_row['ordering']} -> {new_row['raw_text']!r} ({status})")

    # Write updated CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="",
        )
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in CSV_FIELDS} for r in all_rows)

    print(f"\nDone. Fixed {fixed}/{len(failed)}, still unclear: {still_failed}")
    print(f"Updated CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
