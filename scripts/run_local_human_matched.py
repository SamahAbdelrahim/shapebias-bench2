#!/usr/bin/env python3
"""Run human-matched shape-bias evaluation with local (GPU) models.

This script mirrors the human-matched protocol data flow used by remote runs:
- unique-texture stimulus packages (v1 / v2)
- deterministic pseudo-word generation (or no-word placeholders)
- same trial-format CSV metadata for traceability

Unlike ``scripts/run_remote.py``, inference is done with local wrappers from
``evaluation_pipe.models``.

Usage (repo root):
    python scripts/run_local_human_matched.py \
      --models qwen3-vl-4b qwen3.5-4b \
      --stim-pkg stimuli_unique_texture_per_stl_v1 \
      --ordering both --device cuda
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    DEFAULT_DEVICE,
    DEFAULT_HUMAN_TRIAL_LIMIT,
    ENV_PATH,
    HUMAN_MATCHED_REMOTE_CSV_SUBDIR,
    HUMAN_STIM_PACKAGES,
    MAX_TOKENS_LOCAL,
    add_common_args,
    build_unique_human_words,
    human_eval_seed_text,
    human_matched_csv_meta,
    load_completed_trial_keys,
    load_stimuli_human_package,
    maybe_sample_stimuli_human,
    print_summary,
    resolve_output_path,
    resolve_stim_set_name,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)


def run_local(model, images: list[Image.Image], prompt: str, temperature: float = 0.0) -> dict:
    from evaluation_pipe.models.base import ModelResponse

    resp: ModelResponse = model.generate(
        images=images,
        prompt=prompt,
        max_new_tokens=MAX_TOKENS_LOCAL,
        temperature=temperature,
    )
    return {
        "raw_text": resp.raw_text,
        "generation_time_s": round(resp.generation_time_s, 2),
        "model_name": resp.model_name,
        "num_tokens_generated": resp.num_tokens_generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Local model keys or 'all'.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Device (default: {DEFAULT_DEVICE})")
    parser.add_argument(
        "--ordering",
        required=True,
        choices=["shape_first", "texture_first", "random", "both"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", default=None, metavar="CSV")
    parser.add_argument(
        "--prompt-condition",
        default="noun_label",
        choices=["noun_label", "no_word_category"],
        help="Prompt variant (default noun_label).",
    )
    parser.add_argument(
        "--stim-pkg",
        required=True,
        help=f"Human package: one of {sorted(HUMAN_STIM_PACKAGES)}",
    )
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=None,
        help=f"Max stimuli after deterministic shuffle (default {DEFAULT_HUMAN_TRIAL_LIMIT}; 0=all).",
    )
    parser.add_argument(
        "--human-eval-seed",
        default="model_eval",
        help="Seed namespace for deterministic words/stimulus subset.",
    )
    parser.add_argument(
        "--human-matched-stim-condition",
        default=None,
        choices=["noun_label", "no_word_category"],
        metavar="CONDITION",
        help="Condition string only for stimulus subset/shuffle seed; defaults to --prompt-condition.",
    )
    parser.add_argument("--word-mode", default="sudo_only", choices=["sudo_only", "mixed"])
    parser.add_argument("--word-min-len", type=int, default=4)
    parser.add_argument("--word-max-len", type=int, default=8)
    parser.add_argument("--sudo-threshold", type=float, default=0.62)
    add_common_args(parser)
    args = parser.parse_args()

    if args.stim_pkg not in HUMAN_STIM_PACKAGES:
        print(f"Error: --stim-pkg must be one of {sorted(HUMAN_STIM_PACKAGES)}")
        sys.exit(1)
    if args.num_stimuli is not None:
        print("Warning: --num-stimuli is ignored for human_matched (use --trial-limit).")
    if args.word_max_len < args.word_min_len or args.word_min_len < 1:
        print("Error: need 1 <= word_min_len <= word_max_len")
        sys.exit(1)

    random.seed(args.seed)

    from evaluation_pipe.models import create_model, list_models

    available = list_models()
    model_names: list[str] = []
    for m in args.models:
        if m == "all":
            model_names.extend(available)
        else:
            if m not in available:
                print(f"Error: unknown local model '{m}'. Available: {available}")
                sys.exit(1)
            model_names.append(m)

    stim_set_resolved = resolve_stim_set_name(args.stim_set)
    trial_limit = DEFAULT_HUMAN_TRIAL_LIMIT if args.trial_limit is None else args.trial_limit
    stim_cond = args.human_matched_stim_condition or args.prompt_condition

    full_stimuli = load_stimuli_human_package(args.stim_pkg, stim_set_resolved)
    base_seed_stim = human_eval_seed_text(
        args.human_eval_seed,
        stim_set_resolved,
        args.stim_pkg,
        stim_cond,
        args.word_mode,
    )
    base_seed = human_eval_seed_text(
        args.human_eval_seed,
        stim_set_resolved,
        args.stim_pkg,
        args.prompt_condition,
        args.word_mode,
    )
    stimuli = maybe_sample_stimuli_human(full_stimuli, trial_limit, f"{base_seed_stim}|stimuli")

    if args.prompt_condition == "noun_label":
        words = build_unique_human_words(
            len(stimuli),
            f"{base_seed}|words",
            mode=args.word_mode,
            word_min_len=args.word_min_len,
            word_max_len=args.word_max_len,
            sudo_threshold=args.sudo_threshold,
        )
    else:
        words = [{"name": f"__no_word__{i + 1}", "type": "no_word", "length": 0} for i in range(len(stimuli))]
    stim_pairs = list(zip(stimuli, words))

    csv_meta = human_matched_csv_meta(
        stim_pkg=args.stim_pkg,
        stim_set=stim_set_resolved,
        human_word_seed=base_seed,
        stimulus_shuffle_condition=stim_cond,
        word_mode=args.word_mode,
        word_min_len=args.word_min_len,
        word_max_len=args.word_max_len,
        sudo_threshold=args.sudo_threshold,
        trial_limit=trial_limit,
    )

    print(f"Models:      {model_names}")
    print("Eval mode:   human_matched (local)")
    print(f"Device:      {args.device}")
    print(f"Ordering:    {args.ordering}")
    print(f"Condition:   {args.prompt_condition}")
    print(f"Repeats:     {args.repeats}")
    print(f"Temperature: {args.temperature}")
    print(
        f"Stimuli:     {len(stimuli)} (pkg={args.stim_pkg}, set={stim_set_resolved}, "
        f"trial_limit={csv_meta['trial_limit']})"
    )
    print(f"Word seed:   {csv_meta['human_word_seed']}")
    if stim_cond != args.prompt_condition:
        print(f"Stim seed:   uses condition={stim_cond} in shuffle (prompt_condition={args.prompt_condition})")
    if args.prompt_condition == "noun_label":
        print(
            f"Words:       {len(words)} (mode={args.word_mode}, "
            f"len {args.word_min_len}-{args.word_max_len}, sudo_threshold={args.sudo_threshold})"
        )
    else:
        print(f"No-word placeholders: {len(words)}")
    ord_mult = 2 if args.ordering == "both" else 1
    trials_per = len(stim_pairs) * args.repeats * ord_mult
    print(f"Trials/model: {len(stim_pairs)} pairs x {args.repeats} x {ord_mult} = {trials_per}")
    print()

    if args.resume:
        output_path = Path(args.resume)
        done_keys = load_completed_trial_keys(output_path)
        if output_path.exists():
            print(f"Resuming from {output_path} — {len(done_keys)} trials already done")
        else:
            print(f"Warning: --resume file {output_path} not found, starting fresh")
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = resolve_output_path(
            args.output,
            prefix="local_human",
            default_subdir=HUMAN_MATCHED_REMOTE_CSV_SUBDIR,
        )
        done_keys = set()

    all_results: list[dict] = []

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Local model: {model_key}")
        print(f"{'='*60}")

        model = create_model(model_key, device=args.device)
        print(f"  Loaded: {model.name}")

        def run_fn(images, prompt, _m=model):
            return run_local(_m, images, prompt, temperature=args.temperature)

        skipped = 0
        for repeat in range(1, args.repeats + 1):
            for stim, w in stim_pairs:
                if args.ordering == "both":
                    check_ords = ["shape_first", "texture_first"]
                elif args.ordering == "random":
                    check_ords = ["shape_first", "texture_first"]
                else:
                    check_ords = [args.ordering]

                all_done = all(
                    (model_key, stim["stim_id"], w["name"], o, str(repeat)) in done_keys
                    for o in check_ords
                )
                if all_done:
                    skipped += 1
                    continue

                word, word_type, word_length = w["name"], w["type"], w["length"]
                print(f"  Stimulus {stim['stim_id']:>4s} (word={word}, type={word_type}, len={word_length})")
                trial_results = run_trial(
                    run_fn,
                    stim,
                    word,
                    word_type,
                    word_length,
                    ordering=args.ordering,
                    prompt_condition=args.prompt_condition,
                )
                for r in trial_results:
                    r["model"] = model_key
                    r["repeat"] = repeat
                    r["temperature"] = args.temperature
                    r.update(csv_meta)
                    print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                    all_results.append(r)
                    done_keys.add((r["model"], r["stim_id"], r["word"], r["ordering"], str(r["repeat"])))
                write_results(trial_results, output_path, append=True, quiet=True)

        if skipped:
            print(f"  Skipped {skipped} already-completed stimulus-word tasks")
        model.unload()
        print(f"  Unloaded {model_key}")

    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()

