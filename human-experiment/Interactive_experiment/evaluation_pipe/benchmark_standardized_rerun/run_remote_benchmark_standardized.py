#!/usr/bin/env python3
"""Benchmark-only remote evaluation with standardized prompts (isolated copy).

Uses the same system + user prompts as human-matched remote (eval_core). Writes by default to
``results/model.results/benchmark_standardized_rerun/remote_all_fixed_standardized.csv``.

Does not include human_matched mode (see scripts/run_remote.py for that).

Usage (from repo root):
    python evaluation_pipe/benchmark_standardized_rerun/run_remote_benchmark_standardized.py \\
        --models all --ordering both --workers 8
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation_pipe.eval_core import (
    BENCHMARK_STIM_PACKAGE,
    ENV_PATH,
    MAX_TOKENS_REMOTE,
    REMOTE_UNIFORM_SYSTEM_PROMPT,
    TEMPERATURE,
    add_common_args,
    benchmark_csv_meta,
    build_openai_compatible_vision_messages,
    load_stimuli,
    load_words,
    print_summary,
    resolve_stim_set_name,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)

DEFAULT_REMOTE_CSV = (
    REPO_ROOT / "results" / "model.results" / "benchmark_standardized_rerun" / "remote_all_fixed_standardized.csv"
)

# Ten remote-served models from the eleven-model benchmark matrix (InternVL is local-only here).
STANDARDIZED_REMOTE_MODEL_KEYS = frozenset(
    {
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3.5-0.8b",
        "qwen3.5-4b",
        "smolvlm",
        "qwen3.5-9b",
        "qwen3.5-27b",
        "qwen3.5-35b-a3b",
        "qwen3.5-122b-a10b",
        "llama4-scout",
    }
)

REMOTE_MODELS = {
    "qwen3-vl-2b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3-vl-4b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
        "endpoint_base_url_env": "HF_ENDPOINT_QWEN3_VL_4B_BASE_URL",
        "endpoint_model_env": "HF_ENDPOINT_QWEN3_VL_4B_MODEL",
    },
    "qwen3.5-0.8b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-0.8B",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3.5-4b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-4B",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "smolvlm": {
        "provider": "huggingface",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3.5-9b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-9B",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3.5-27b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-27B:featherless-ai",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3.5-35b-a3b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3.5-122b-a10b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3.5-122B-A10B",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "llama4-scout": {
        "provider": "huggingface-groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
}

PROVIDER_BASE_URLS = {
    "huggingface": "https://router.huggingface.co/v1",
    "huggingface-groq": "https://router.huggingface.co/groq/openai/v1",
    "huggingface-sambanova": "https://router.huggingface.co/sambanova/openai/v1",
}


def _openai_base_url_for_config(cfg: dict) -> str:
    env_key = cfg.get("endpoint_base_url_env")
    if env_key:
        raw = os.environ.get(env_key, "").strip()
        if raw:
            raw = raw.rstrip("/")
            if not raw.endswith("/v1"):
                raw = f"{raw}/v1"
            return raw
    return PROVIDER_BASE_URLS[cfg["provider"]]


def _openai_model_for_config(cfg: dict) -> str:
    env_key = cfg.get("endpoint_model_env")
    if env_key:
        override = os.environ.get(env_key, "").strip()
        if override:
            return override
    return cfg["model_id"]


def image_to_base64_url(img: Image.Image, fmt: str = "JPEG", max_side: int = 768, jpeg_quality: int = 85) -> str:
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


def build_messages(
    images: list[Image.Image],
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict]:
    return build_openai_compatible_vision_messages(
        images,
        prompt,
        image_to_url=image_to_base64_url,
        system_prompt=system_prompt,
    )


def run_remote(model_name: str, images: list[Image.Image], prompt: str, temperature: float | None = None) -> dict:
    from openai import OpenAI

    cfg = REMOTE_MODELS[model_name]
    system_prompt = cfg.get("system_prompt")
    base_url = _openai_base_url_for_config(cfg)
    api_model = _openai_model_for_config(cfg)
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
    messages = build_messages(images, prompt, system_prompt=system_prompt)

    extra = {}
    max_tok = MAX_TOKENS_REMOTE
    using_dedicated_endpoint = bool(
        cfg.get("endpoint_base_url_env") and os.environ.get(cfg["endpoint_base_url_env"], "").strip()
    )
    if "qwen" in cfg["model_id"].lower() and not using_dedicated_endpoint:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    start = time.perf_counter()
    temp = TEMPERATURE if temperature is None else temperature
    response = client.chat.completions.create(
        model=api_model,
        messages=messages,
        max_tokens=max_tok,
        temperature=temp,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model keys, or 'all' for the ten remote registry models in this package.",
    )
    parser.add_argument(
        "--ordering",
        required=True,
        choices=["shape_first", "texture_first", "random", "both"],
        help="Trial ordering",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resume", default=None, metavar="CSV")
    parser.add_argument(
        "--prompt-condition",
        default="noun_label",
        choices=["noun_label", "no_word_category"],
        help="Default noun_label for word benchmark; no_word for controls.",
    )
    parser.add_argument(
        "--no-word-mode",
        default="matched",
        choices=["matched", "reduced"],
    )
    add_common_args(parser)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(DEFAULT_REMOTE_CSV)

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

    model_names: list[str] = []
    for m in args.models:
        if m == "all":
            model_names.extend(sorted(STANDARDIZED_REMOTE_MODEL_KEYS))
        else:
            if m not in REMOTE_MODELS:
                print(f"Error: unknown model '{m}'. Available: {sorted(REMOTE_MODELS)}")
                sys.exit(1)
            model_names.append(m)

    stim_set_resolved = resolve_stim_set_name(args.stim_set)

    if args.prompt_condition == "no_word_category" and args.no_word_mode == "reduced":
        words = [{"name": "__no_word__", "type": "no_word", "length": 0}]
    elif args.prompt_condition == "no_word_category":
        words = [
            {"name": f"no_word_slot_{i+1}", "type": "no_word", "length": 0}
            for i in range(len(load_words()))
        ]
    else:
        words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    stim_pairs = [(s, w) for s in stimuli for w in words]
    csv_meta = benchmark_csv_meta(stim_set_resolved)

    stim_set_label = stim_set_resolved
    ord_mult = 2 if args.ordering == "both" else 1
    trials_per = len(stim_pairs) * args.repeats * ord_mult

    print("Standardized remote benchmark (evaluation_pipe/benchmark_standardized_rerun)")
    print(f"Default prompts: REMOTE_UNIFORM_SYSTEM_PROMPT + noun_label templates (eval_core)")
    print(f"Output:        {args.output}")
    print(f"Models:        {model_names}")
    print(f"Ordering:      {args.ordering}")
    print(f"Condition:     {args.prompt_condition} ({args.no_word_mode})")
    print(f"Stimuli:       {len(stimuli)} from {BENCHMARK_STIM_PACKAGE}/{stim_set_label}")
    print(f"Trials/model:  {trials_per}")
    print()

    done_keys: set[tuple] = set()
    output_path = Path(args.output)
    if args.resume:
        output_path = Path(args.resume)
        if output_path.exists():
            with open(output_path) as f:
                for row in csv.DictReader(f):
                    done_keys.add(
                        (
                            row["model"],
                            row["stim_id"],
                            row["word"],
                            row["ordering"],
                            row.get("repeat", "1"),
                        )
                    )
            print(f"Resuming from {output_path} — {len(done_keys)} trials already done")
        else:
            print(f"Warning: --resume file {output_path} not found, starting fresh")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    write_lock = threading.Lock()

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Remote model: {model_key}")
        print(f"{'='*60}")

        tasks = []
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
                else:
                    tasks.append((repeat, stim, w))

        if skipped:
            print(f"  Skipped {skipped} already-completed tasks")

        completed = 0
        total = len(tasks) * ord_mult

        def process_task(task):
            repeat, stim, w = task
            word, word_type, word_length = w["name"], w["type"], w["length"]

            def run_fn(images, prompt, _mk=model_key):
                return run_remote(_mk, images, prompt, temperature=args.temperature)

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
            return trial_results

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(process_task, t): t for t in tasks}

            for future in as_completed(future_to_task):
                trial_results = future.result()
                with write_lock:
                    for r in trial_results:
                        print(
                            f"  Stim {r['stim_id']:>3s} word={r['word']:12s} "
                            f"{r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}"
                        )
                        all_results.append(r)
                    completed += len(trial_results)
                    write_results(trial_results, output_path, append=True, quiet=True)
                    print(f"  [{completed}/{total}] done")

    print(f"\nAll results saved to {output_path}")
    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
