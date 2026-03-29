#!/usr/bin/env python3
"""Shape-bias evaluation for remote (API-based) VLMs.

Usage:
    # Run a single model with one ordering
    python scripts/run_remote.py --models qwen3.5-9b --ordering shape_first

    # Run multiple models
    python scripts/run_remote.py --models qwen3.5-9b llama4-scout --ordering both

    # Run all remote models
    python scripts/run_remote.py --models all --ordering both

    # Multiple repeats with temperature
    python scripts/run_remote.py --models qwen3.5-9b --ordering shape_first --repeats 3 --temperature 0.7

    # Control parallelism
    python scripts/run_remote.py --models qwen3.5-9b --ordering both --workers 10

    # Append results to existing CSV
    python scripts/run_remote.py --models qwen3.5-9b --ordering shape_first -o results/run.csv
    python scripts/run_remote.py --models qwen3.5-9b --ordering texture_first -o results/run.csv

    # Run no-word control condition (reduced pilot)
    python scripts/run_remote.py --models all --ordering both --prompt-condition no_word_category --no-word-mode reduced -o results/no_word_pilot_remote.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_pipe.eval_core import (
    ENV_PATH,
    MAX_TOKENS_REMOTE,
    TEMPERATURE,
    add_common_args,
    load_stimuli,
    load_words,
    print_summary,
    resolve_output_path,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)

# ===========================================================================
# REMOTE MODEL REGISTRY
# ===========================================================================
REMOTE_MODELS = {
    "qwen3.5-9b":       {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-9B"},
    "qwen3.5-27b":      {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-27B:featherless-ai"},
    "qwen3.5-35b-a3b":  {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-35B-A3B"},
    "qwen3.5-122b-a10b":{"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-122B-A10B"},
    "llama4-scout":     {"provider": "huggingface-groq",  "model_id": "meta-llama/llama-4-scout-17b-16e-instruct"},
    # "llama4-maverick":{"provider": "huggingface-sambanova", "model_id": "Llama-4-Maverick-17B-128E-Instruct"},
}

PROVIDER_BASE_URLS = {
    "huggingface":           "https://router.huggingface.co/v1",
    "huggingface-groq":      "https://router.huggingface.co/groq/openai/v1",
    "huggingface-sambanova": "https://router.huggingface.co/sambanova/v1",
}


# ---------------------------------------------------------------------------
# Remote inference helpers
# ---------------------------------------------------------------------------
def image_to_base64_url(img: Image.Image, fmt: str = "JPEG",
                        max_side: int = 768, jpeg_quality: int = 85) -> str:
    # Resize/compress to avoid provider request-body limits (HTTP 413).
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    if fmt.upper() == "JPEG":
        img = img.convert("RGB")

    buf = BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs.update({"quality": jpeg_quality, "optimize": True})
    img.save(buf, format=fmt, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def build_messages(images: list[Image.Image], prompt: str) -> list[dict]:
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(img)},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def run_remote(model_name: str, images: list[Image.Image], prompt: str) -> dict:
    from openai import OpenAI

    cfg = REMOTE_MODELS[model_name]
    base_url = PROVIDER_BASE_URLS[cfg["provider"]]
    hf_token = (
        os.environ.get("HUGGING_FACE")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not hf_token:
        raise RuntimeError(
            "Missing API token. Set one of HUGGING_FACE, HF_API_TOKEN, HF_TOKEN, or OPENAI_API_KEY."
        )

    client = OpenAI(api_key=hf_token, base_url=base_url, timeout=60.0)
    messages = build_messages(images, prompt)

    # Disable thinking mode for Qwen3.5 models to avoid wasting tokens.
    # Keep max_tokens low (128) to cap runaway thinking if the provider
    # ignores the disable flag — better to get a truncated/empty response
    # and retry than to burn thousands of thinking tokens.
    extra = {}
    max_tok = MAX_TOKENS_REMOTE
    if "qwen" in cfg["model_id"].lower():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg["model_id"],
        messages=messages,
        max_tokens=max_tok,
        temperature=TEMPERATURE,
        **extra,
    )
    elapsed = time.perf_counter() - start

    choice = response.choices[0]
    raw_text = (choice.message.content or "").strip()
    tokens = response.usage.completion_tokens if response.usage else None

    return {
        "raw_text": raw_text,
        "generation_time_s": round(elapsed, 2),
        "model_name": cfg["model_id"],
        "num_tokens_generated": tokens,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run shape-bias evaluation (remote API models)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate. Use 'all' for all remote models.")
    parser.add_argument("--ordering", required=True,
                        choices=["shape_first", "texture_first", "random", "both"],
                        help="Trial ordering: shape_first, texture_first, random, or both")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Number of repeats per trial (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0 = greedy)")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    parser.add_argument("--resume", default=None, metavar="CSV",
                        help="Resume from a partial CSV — skip already-completed trials and append new results")
    parser.add_argument("--prompt-condition", default="noun_label",
                        choices=["noun_label", "no_word_category"],
                        help="Prompt variant to use (default: noun_label)")
    parser.add_argument("--no-word-mode", default="matched",
                        choices=["matched", "reduced"],
                        help="For no_word_category: matched keeps full word-loop trial count, reduced runs one pass per stimulus.")
    add_common_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)

    token_present = bool(
        os.environ.get("HUGGING_FACE")
        or os.environ.get("HF_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not token_present:
        print("Error: no API token found. Set HUGGING_FACE, HF_API_TOKEN, HF_TOKEN, or OPENAI_API_KEY.")
        sys.exit(1)

    # Resolve model list
    model_names = []
    for m in args.models:
        if m == "all":
            model_names.extend(REMOTE_MODELS.keys())
        else:
            if m not in REMOTE_MODELS:
                print(f"Error: unknown remote model '{m}'. Available: {list(REMOTE_MODELS.keys())}")
                sys.exit(1)
            model_names.append(m)

    # Load stimuli and words (or no-word placeholders)
    if args.prompt_condition == "no_word_category" and args.no_word_mode == "reduced":
        words = [{"name": "__no_word__", "type": "no_word", "length": 0}]
    elif args.prompt_condition == "no_word_category":
        # Matched mode preserves trial counts by reusing word slots.
        words = [
            {"name": f"no_word_slot_{i+1}", "type": "no_word", "length": 0}
            for i in range(len(load_words()))
        ]
    else:
        words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    stim_set_label = args.stim_set or "env/default"
    ord_mult = 2 if args.ordering == "both" else 1
    trials_per = len(stimuli) * len(words) * args.repeats * ord_mult

    print(f"Models:      {model_names}")
    print(f"Ordering:    {args.ordering}")
    print(f"Condition:   {args.prompt_condition} ({args.no_word_mode})")
    print(f"Repeats:     {args.repeats}")
    print(f"Temperature: {args.temperature}")
    print(f"Workers:     {args.workers}")
    print(f"Stimuli:     {len(stimuli)} from {stim_set_label}")
    if args.prompt_condition == "noun_label":
        print(f"Words:       {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
    else:
        print(f"No-word slots: {len(words)} (mode={args.no_word_mode})")
    print(f"Trials per model: {len(stimuli)} x {len(words)} x {args.repeats} repeats x {ord_mult} orderings = {trials_per}")
    print()

    # Handle --resume: load completed trials and use that CSV for appending
    done_keys: set[tuple] = set()
    if args.resume:
        output_path = Path(args.resume)
        if output_path.exists():
            with open(output_path) as f:
                for row in csv.DictReader(f):
                    # Key: (model, stim_id, word, ordering, repeat)
                    done_keys.add((
                        row["model"], row["stim_id"], row["word"],
                        row["ordering"], row.get("repeat", "1"),
                    ))
            print(f"Resuming from {output_path} — {len(done_keys)} trials already done")
        else:
            print(f"Warning: --resume file {output_path} not found, starting fresh")
    else:
        output_path = resolve_output_path(args.output, prefix="remote")

    all_results = []
    write_lock = threading.Lock()

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Remote model: {model_key}")
        print(f"{'='*60}")

        # Build list of tasks, skipping already-completed ones
        tasks = []
        skipped = 0
        for repeat in range(1, args.repeats + 1):
            for stim in stimuli:
                for w in words:
                    # Check if all orderings for this task are done
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
                    else:
                        tasks.append((repeat, stim, w))

        if skipped:
            print(f"  Skipped {skipped} already-completed tasks")

        completed = 0
        total = len(tasks) * ord_mult  # total trial rows

        def process_task(task):
            repeat, stim, w = task
            word, word_type, word_length = w["name"], w["type"], w["length"]

            def run_fn(images, prompt, _mk=model_key):
                return run_remote(_mk, images, prompt)

            trial_results = run_trial(run_fn, stim, word, word_type, word_length,
                                      ordering=args.ordering,
                                      prompt_condition=args.prompt_condition)
            for r in trial_results:
                r["model"] = model_key
                r["repeat"] = repeat
                r["temperature"] = args.temperature
            return trial_results

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(process_task, t): t for t in tasks}

            for future in as_completed(future_to_task):
                trial_results = future.result()
                with write_lock:
                    for r in trial_results:
                        print(f"  Stim {r['stim_id']:>3s} word={r['word']:12s} "
                              f"{r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                        all_results.append(r)
                    completed += len(trial_results)
                    # Incremental save
                    write_results(trial_results, output_path, append=True, quiet=True)

                    # Progress
                    print(f"  [{completed}/{total}] done")

    print(f"\nAll results saved to {output_path}")
    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
