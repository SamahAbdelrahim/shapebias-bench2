#!/usr/bin/env python3
"""Shape-bias evaluation for local (GPU-based) VLMs.

Usage:
    # Run a single model with one ordering
    python scripts/run_local.py --models smolvlm --ordering shape_first

    # Run multiple local models
    python scripts/run_local.py --models smolvlm internvl --ordering texture_first

    # Run all registered local models
    python scripts/run_local.py --models all --ordering shape_first

    # Multiple repeats with temperature
    python scripts/run_local.py --models smolvlm --ordering shape_first --repeats 3 --temperature 0.7

    # Append results to existing CSV
    python scripts/run_local.py --models smolvlm --ordering shape_first -o results/run.csv
    python scripts/run_local.py --models smolvlm --ordering texture_first -o results/run.csv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    BENCHMARK_STIM_PACKAGE,
    DEFAULT_DEVICE,
    ENV_PATH,
    MAX_TOKENS_LOCAL,
    add_common_args,
    benchmark_csv_meta,
    load_stimuli,
    load_words,
    print_summary,
    resolve_output_path,
    resolve_stim_set_name,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)


# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------
def run_local(model, images: list[Image.Image], prompt: str,
              temperature: float = 0.0) -> dict:
    from evaluation_pipe.models.base import ModelResponse
    resp: ModelResponse = model.generate(
        images=images, prompt=prompt,
        max_new_tokens=MAX_TOKENS_LOCAL, temperature=temperature,
    )
    return {
        "raw_text": resp.raw_text,
        "generation_time_s": round(resp.generation_time_s, 2),
        "model_name": resp.model_name,
        "num_tokens_generated": resp.num_tokens_generated,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run shape-bias evaluation (local GPU models)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate. Use 'all' for all registered local models.")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"Device for local models (default: {DEFAULT_DEVICE})")
    parser.add_argument("--ordering", required=True,
                        choices=["shape_first", "texture_first", "random", "both"],
                        help="Trial ordering: shape_first, texture_first, random, or both")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per trial (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    add_common_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    from evaluation_pipe.models import create_model, list_models

    # Resolve model list
    available = list_models()
    model_names = []
    for m in args.models:
        if m == "all":
            model_names.extend(available)
        else:
            if m not in available:
                print(f"Error: unknown local model '{m}'. Available: {available}")
                sys.exit(1)
            model_names.append(m)

    # Load stimuli and words
    words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    stim_set_label = resolve_stim_set_name(args.stim_set)
    csv_meta = benchmark_csv_meta(stim_set_label)
    print(f"Models:      {model_names}")
    print(f"Device:      {args.device}")
    print(f"Ordering:    {args.ordering}")
    print(f"Repeats:     {args.repeats}")
    print(f"Temperature: {args.temperature}")
    print(f"Stimuli:     {len(stimuli)} from {BENCHMARK_STIM_PACKAGE}/{stim_set_label}")
    print(f"Words:       {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
    ord_mult = 2 if args.ordering == "both" else 1
    trials_per = len(stimuli) * len(words) * args.repeats * ord_mult
    print(f"Trials per model: {len(stimuli)} x {len(words)} x {args.repeats} repeats x {ord_mult} orderings = {trials_per}")
    print()

    output_path = resolve_output_path(args.output, prefix="local")
    all_results = []

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Local model: {model_key}")
        print(f"{'='*60}")

        model = create_model(model_key, device=args.device)
        print(f"  Loaded: {model.name}")

        def run_fn(images, prompt, _m=model):
            return run_local(_m, images, prompt, temperature=args.temperature)

        for repeat in range(1, args.repeats + 1):
            if args.repeats > 1:
                print(f"\n  --- Repeat {repeat}/{args.repeats} ---")
            for stim in stimuli:
                for w in words:
                    word, word_type, word_length = w["name"], w["type"], w["length"]
                    print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                    trial_results = run_trial(run_fn, stim, word, word_type, word_length,
                                              ordering=args.ordering)
                    for r in trial_results:
                        r["model"] = model_key
                        r["repeat"] = repeat
                        r["temperature"] = args.temperature
                        r.update(csv_meta)
                        print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                        all_results.append(r)
                    # Save incrementally after each stimulus+word trial
                    write_results(trial_results, output_path, append=True, quiet=True)

        model.unload()
        print(f"  Unloaded {model_key}")

    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
