#!/usr/bin/env python3
"""Benchmark local evaluation with standardized prompts (isolated copy).

Patches registered local VLM wrappers **at runtime** by setting
``_system_prompt = REMOTE_UNIFORM_SYSTEM_PROMPT`` on each Transformers
wrapper class (same system line as human-matched remote). Generate and
``score_choices`` both read that attribute, so they stay consistent.
User prompts still come from ``run_trial`` / ``make_prompt`` (noun_label by default).

Default output:
``results/model.results/benchmark_standardized_rerun/local_eval_standardized.csv``

Usage (from repo root):
    python evaluation_pipe/benchmark_standardized_rerun/run_local_benchmark_standardized.py \\
        --models all --ordering both --repeats 1

Resume after timeout (same stim set, seed, ordering, repeats, and models as the
interrupted run); results append to the CSV given by ``--resume``:
    python ... --models qwen3-vl-2b --ordering both --repeats 1 \\
        --resume results/model.results/benchmark_standardized_rerun/local_eval_standardized.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation_pipe.eval_core import (
    BENCHMARK_STIM_PACKAGE,
    DEFAULT_DEVICE,
    ENV_PATH,
    MAX_TOKENS_LOCAL,
    REMOTE_UNIFORM_SYSTEM_PROMPT,
    add_common_args,
    benchmark_csv_meta,
    load_completed_trial_keys,
    load_stimuli,
    load_words,
    print_summary,
    resolve_stim_set_name,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)

DEFAULT_LOCAL_CSV = (
    REPO_ROOT / "results" / "model.results" / "benchmark_standardized_rerun" / "local_eval_standardized.csv"
)

# Six models from the eleven-model matrix that have local wrappers in this repo.
STANDARDIZED_LOCAL_MODEL_KEYS = frozenset(
    {
        "internvl",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3.5-0.8b",
        "qwen3.5-4b",
        "smolvlm",
    }
)


def _apply_standardized_system_prompts() -> None:
    """Point all local wrappers at REMOTE_UNIFORM for generate + score_choices."""
    from evaluation_pipe.models.local_models import internvl as internvl_mod
    from evaluation_pipe.models.local_models import qwen as qwen_mod
    from evaluation_pipe.models.local_models import qwen35 as qwen35_mod
    from evaluation_pipe.models.local_models import smolvlm as smolvlm_mod

    u = REMOTE_UNIFORM_SYSTEM_PROMPT
    qwen35_mod._Qwen35Base._system_prompt = u
    smolvlm_mod.SmolVLM._system_prompt = u
    internvl_mod.InternVL._system_prompt = u
    qwen_mod._Qwen3VLBase._system_prompt = u


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
    _apply_standardized_system_prompts()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local registry keys, or 'all' for the six standardized local models.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Device (default: {DEFAULT_DEVICE})")
    parser.add_argument(
        "--ordering",
        required=True,
        choices=["shape_first", "texture_first", "random", "both"],
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CSV",
        help="Append to this CSV and skip (model, stim_id, word, ordering, repeat) already present.",
    )
    add_common_args(parser)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(DEFAULT_LOCAL_CSV)

    random.seed(args.seed)

    from evaluation_pipe.models import create_model, list_models

    available = set(list_models())
    model_names: list[str] = []
    for m in args.models:
        if m == "all":
            for k in sorted(STANDARDIZED_LOCAL_MODEL_KEYS & available):
                model_names.append(k)
        else:
            if m not in available:
                print(f"Error: unknown local model '{m}'. Available: {sorted(available)}")
                sys.exit(1)
            if m not in STANDARDIZED_LOCAL_MODEL_KEYS:
                print(
                    f"Warning: '{m}' is outside the six-model standardized set; "
                    f"still running with patched prompts."
                )
            model_names.append(m)

    if not model_names:
        print("Error: no models to run (check --models all vs registry).")
        sys.exit(1)

    words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    stim_set_label = resolve_stim_set_name(args.stim_set)
    csv_meta = benchmark_csv_meta(stim_set_label)

    print("Standardized local benchmark (evaluation_pipe/benchmark_standardized_rerun)")
    print(f"System prompt: REMOTE_UNIFORM_SYSTEM_PROMPT (runtime-patched local wrappers)")
    print(f"Output:        {args.resume or args.output}")
    print(f"Models:        {model_names}")
    print(f"Device:        {args.device}")
    print(f"Ordering:      {args.ordering}")
    print(f"Stimuli:       {len(stimuli)} from {BENCHMARK_STIM_PACKAGE}/{stim_set_label}")
    ord_mult = 2 if args.ordering == "both" else 1
    trials_per = len(stimuli) * len(words) * args.repeats * ord_mult
    print(f"Trials/model:  {trials_per}")
    print()

    if args.resume:
        output_path = Path(args.resume)
        if output_path.exists() and output_path.stat().st_size > 0:
            done_keys = load_completed_trial_keys(output_path)
            print(f"Resuming from {output_path} — {len(done_keys)} row-keys already done")
        else:
            done_keys = set()
            print(f"Warning: --resume file {output_path} not found or empty, starting fresh")
            output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
            if args.repeats > 1:
                print(f"\n  --- Repeat {repeat}/{args.repeats} ---")
            for stim in stimuli:
                for w in words:
                    word, word_type, word_length = w["name"], w["type"], w["length"]
                    if args.ordering == "both":
                        check_ords = ["shape_first", "texture_first"]
                    elif args.ordering == "random":
                        check_ords = ["shape_first", "texture_first"]
                    else:
                        check_ords = [args.ordering]
                    if all(
                        (model_key, stim["stim_id"], word, o, str(repeat)) in done_keys
                        for o in check_ords
                    ):
                        skipped += 1
                        continue

                    print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                    trial_results = run_trial(
                        run_fn,
                        stim,
                        word,
                        word_type,
                        word_length,
                        ordering=args.ordering,
                    )
                    for r in trial_results:
                        r["model"] = model_key
                        r["repeat"] = repeat
                        r["temperature"] = args.temperature
                        r.update(csv_meta)
                        print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                        all_results.append(r)
                        done_keys.add(
                            (
                                r["model"],
                                r["stim_id"],
                                r["word"],
                                r["ordering"],
                                str(r["repeat"]),
                            )
                        )
                    write_results(trial_results, output_path, append=True, quiet=True)

        if skipped:
            print(f"  Skipped {skipped} already-completed stimulus–word blocks")

        model.unload()
        print(f"  Unloaded {model_key}")

    if output_path.exists() and output_path.stat().st_size > 0:
        with open(output_path, newline="", encoding="utf-8-sig") as f:
            summary_rows = list(csv.DictReader(f))
        if summary_rows:
            print_summary(summary_rows, model_names)
        else:
            print_summary(all_results, model_names)
    else:
        print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
