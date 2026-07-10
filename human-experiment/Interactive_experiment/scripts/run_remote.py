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

    # Human-matched eval (v1/v2 stimuli + JS-parity words; deterministic --ordering)
    python scripts/run_remote.py --eval-mode human_matched --stim-pkg stimuli_unique_texture_per_stl_v1 \\
        --ordering both --models qwen3.5-9b
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
    BENCHMARK_STIM_PACKAGE,
    DEFAULT_HUMAN_TRIAL_LIMIT,
    ENV_PATH,
    HUMAN_MATCHED_REMOTE_CSV_SUBDIR,
    HUMAN_STIM_PACKAGES,
    MAX_TOKENS_REMOTE,
    REMOTE_UNIFORM_SYSTEM_PROMPT,
    TEMPERATURE,
    add_common_args,
    benchmark_csv_meta,
    build_openai_compatible_vision_messages,
    build_unique_human_words,
    human_eval_seed_text,
    human_matched_csv_meta,
    load_stimuli,
    load_stimuli_human_package,
    load_words,
    maybe_sample_stimuli_human,
    print_summary,
    resolve_output_path,
    resolve_stim_set_name,
    run_trial,
    write_results,
)

load_dotenv(ENV_PATH)

# ===========================================================================
# REMOTE MODEL REGISTRY
# ===========================================================================
REMOTE_MODELS = {
    # Previously local-only VLMs (same HF IDs as local wrappers); remote uses
    # REMOTE_UNIFORM_SYSTEM_PROMPT — verify each ID on your HF router account.
    #
    # InternVL: local default `OpenGVLab/InternVL3-1B-hf` is often **not** exposed on
    # the default HF Inference router (400 model_not_supported). Add an entry here when
    # your account lists a served InternVL / multi-modal ID.
    "qwen3-vl-2b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
    },
    "qwen3-vl-4b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "system_prompt": REMOTE_UNIFORM_SYSTEM_PROMPT,
        # Optional dedicated Inference Endpoint (see .env.example + evaluation_pipe/README.md)
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
    # "llama4-maverick":{"provider": "huggingface-sambanova", "model_id": "...", "system_prompt": None},
}

PROVIDER_BASE_URLS = {
    "huggingface":           "https://router.huggingface.co/v1",
    "huggingface-groq":      "https://router.huggingface.co/groq/openai/v1",
    "huggingface-sambanova": "https://router.huggingface.co/sambanova/v1",
}


def _openai_base_url_for_config(cfg: dict) -> str:
    """Router default, or dedicated HF Inference Endpoint from env (OpenAI-compatible /v1)."""
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


def build_messages(
    images: list[Image.Image],
    prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict]:
    """Same vision layout as local VLMs (labeled slots + task prompt from eval_core)."""
    return build_openai_compatible_vision_messages(
        images,
        prompt,
        image_to_url=image_to_base64_url,
        system_prompt=system_prompt,
    )


def run_remote(model_name: str, images: list[Image.Image], prompt: str,
               temperature: float | None = None) -> dict:
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

    # Disable thinking mode for Qwen3.5 models to avoid wasting tokens.
    # Keep max_tokens low (128) to cap runaway thinking if the provider
    # ignores the disable flag — better to get a truncated/empty response
    # and retry than to burn thousands of thinking tokens.
    extra = {}
    max_tok = MAX_TOKENS_REMOTE
    # Router-only: Qwen chat templates may honor enable_thinking; dedicated endpoints often omit it.
    using_dedicated_endpoint = bool(
        cfg.get("endpoint_base_url_env")
        and os.environ.get(cfg["endpoint_base_url_env"], "").strip()
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
    parser.add_argument("--eval-mode", default="benchmark",
                        choices=["benchmark", "human_matched"],
                        help="benchmark: stimuli_per_stl_packages + WORD_PAIRS. "
                             "human_matched: v1/v2 unique-texture packages + human word generator.")
    parser.add_argument("--stim-pkg", default=None,
                        help="Required for human_matched: stimuli_unique_texture_per_stl_v1 or v2.")
    parser.add_argument("--trial-limit", type=int, default=None,
                        help="human_matched: max stimuli after shuffle (default 30; 0 = all). Ignored for benchmark.")
    parser.add_argument("--human-eval-seed", default="model_eval",
                        help="human_matched: replaces participant id block in seedText (default model_eval).")
    parser.add_argument(
        "--human-matched-stim-condition",
        default=None,
        choices=["noun_label", "no_word_category"],
        metavar="CONDITION",
        help="human_matched only: condition string used only for stimulus subsample/shuffle seed "
             "(default: same as --prompt-condition). For paired noun vs no-word runs, pass "
             "noun_label on no_word_category jobs so stimulus order matches the noun-label batch.",
    )
    parser.add_argument("--word-mode", default="sudo_only",
                        choices=["sudo_only", "mixed"],
                        help="human_matched: sudo_only (default) or mixed sudo/random.")
    parser.add_argument("--word-min-len", type=int, default=4,
                        help="human_matched: min generated word length (default 4).")
    parser.add_argument("--word-max-len", type=int, default=8,
                        help="human_matched: max generated word length (default 8).")
    parser.add_argument("--sudo-threshold", type=float, default=0.62,
                        help="human_matched: min English bigram score for sudo words (default 0.62).")
    add_common_args(parser)
    args = parser.parse_args()

    if args.human_matched_stim_condition and args.eval_mode != "human_matched":
        print("Error: --human-matched-stim-condition requires --eval-mode human_matched")
        sys.exit(1)

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

    stim_set_resolved = resolve_stim_set_name(args.stim_set)
    csv_meta: dict[str, str]

    if args.eval_mode == "human_matched":
        if not args.stim_pkg or args.stim_pkg not in HUMAN_STIM_PACKAGES:
            print(
                f"Error: --eval-mode human_matched requires --stim-pkg in "
                f"{sorted(HUMAN_STIM_PACKAGES)}"
            )
            sys.exit(1)
        if args.num_stimuli is not None:
            print("Warning: --num-stimuli is ignored in human_matched (use --trial-limit).")
        trial_limit = (
            DEFAULT_HUMAN_TRIAL_LIMIT if args.trial_limit is None else args.trial_limit
        )
        if args.word_max_len < args.word_min_len or args.word_min_len < 1:
            print("Error: need 1 <= word_min_len <= word_max_len")
            sys.exit(1)

        full_stimuli = load_stimuli_human_package(args.stim_pkg, stim_set_resolved)
        stim_cond = (
            args.human_matched_stim_condition
            if args.human_matched_stim_condition is not None
            else args.prompt_condition
        )
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
        stimuli = maybe_sample_stimuli_human(
            full_stimuli, trial_limit, f"{base_seed_stim}|stimuli"
        )
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
            words = [
                {"name": f"__no_word__{i + 1}", "type": "no_word", "length": 0}
                for i in range(len(stimuli))
            ]
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
    else:
        if args.stim_pkg:
            print("Warning: --stim-pkg is ignored unless --eval-mode human_matched.")
        if args.trial_limit is not None:
            print("Warning: --trial-limit is ignored unless --eval-mode human_matched.")
        # Load stimuli and words (or no-word placeholders)
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

    print(f"Models:      {model_names}")
    print(f"Eval mode:   {args.eval_mode}")
    print(f"Ordering:    {args.ordering}")
    print(f"Condition:   {args.prompt_condition} ({args.no_word_mode})")
    print(f"Repeats:     {args.repeats}")
    print(f"Temperature: {args.temperature}")
    print(f"Workers:     {args.workers}")
    if args.eval_mode == "human_matched":
        print(
            f"Stimuli:     {len(stimuli)} (pkg={args.stim_pkg}, set={stim_set_label}, "
            f"trial_limit={csv_meta['trial_limit']})"
        )
        print(f"Word seed:   {csv_meta['human_word_seed']}")
        if stim_cond != args.prompt_condition:
            print(
                f"Stim seed:   uses condition={stim_cond} in shuffle (prompt_condition={args.prompt_condition})"
            )
        if args.prompt_condition == "noun_label":
            print(
                f"Words:       {len(words)} (mode={args.word_mode}, "
                f"len {args.word_min_len}-{args.word_max_len}, sudo_threshold={args.sudo_threshold})"
            )
        else:
            print(f"No-word placeholders: {len(words)}")
    else:
        print(f"Stimuli:     {len(stimuli)} from {BENCHMARK_STIM_PACKAGE}/{stim_set_label}")
        if args.prompt_condition == "noun_label":
            print(f"Words:       {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
        else:
            print(f"No-word slots: {len(words)} (mode={args.no_word_mode})")
    print(f"Trials per model: {len(stim_pairs)} pairs x {args.repeats} repeats x {ord_mult} orderings = {trials_per}")
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
        prefix = "remote_human" if args.eval_mode == "human_matched" else "remote"
        hm_sub = HUMAN_MATCHED_REMOTE_CSV_SUBDIR if args.eval_mode == "human_matched" else None
        output_path = resolve_output_path(args.output, prefix=prefix, default_subdir=hm_sub)

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
            for stim, w in stim_pairs:
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
                return run_remote(_mk, images, prompt, temperature=args.temperature)

            trial_results = run_trial(run_fn, stim, word, word_type, word_length,
                                      ordering=args.ordering,
                                      prompt_condition=args.prompt_condition)
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
