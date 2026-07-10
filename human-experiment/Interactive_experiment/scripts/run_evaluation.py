#!/usr/bin/env python3
"""Unified shape-bias evaluation script for local and remote VLMs.

Runs 2AFC trials across stimuli with counterbalanced position ordering
(shape-first and texture-first) and novel word prompts.

Usage:
    # Run a single remote model
    python scripts/run_evaluation.py --models qwen3.5-9b

    # Run multiple remote models
    python scripts/run_evaluation.py --models qwen3.5-9b llama4-scout

    # Run a local model (requires GPU)
    python scripts/run_evaluation.py --models smolvlm --device cuda

    # Run all available remote models
    python scripts/run_evaluation.py --models all-remote

    # Run all available local models
    python scripts/run_evaluation.py --models all-local --device cuda

    # Limit number of stimuli
    python scripts/run_evaluation.py --models qwen3.5-9b --num-stimuli 5

    # Specify output file
    python scripts/run_evaluation.py --models qwen3.5-9b -o results/my_run.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import os
import random
import sys
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# ===========================================================================
# PATHS
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = REPO_ROOT / "stimuli_pipe" / "stimuli_per_stl_packages"
RESULTS_DIR = REPO_ROOT / "results"
ENV_PATH = REPO_ROOT / ".env"

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
MAX_RETRIES = 5               # retries per trial if answer is ambiguous/empty
MAX_TOKENS_REMOTE = 128       # max tokens for remote API calls (thinking disabled)
MAX_TOKENS_LOCAL = 128        # max tokens for local model generation
TEMPERATURE = 0.0             # sampling temperature (0.0 = greedy/deterministic)
RANDOM_SEED = 42              # default random seed for reproducibility
DEFAULT_STIM_SET = "stimuli_A_auto_contrast"
DEFAULT_DEVICE = "cuda"

# ===========================================================================
# WORD PAIRS — 5 pseudo (sudo) + 5 random, length-matched
# ===========================================================================
WORD_PAIRS = [
    # (sudo_word,    random_word,   length)
    ("shiple",       "afnafq",      6),
    ("clapher",      "ieyiccw",     7),
    ("plailass",     "orvufaig",    8),
    ("procation",    "qahftrxck",   9),
    ("adinefults",   "cgchqjjfgy",  10),
]

# ===========================================================================
# REMOTE MODEL REGISTRY
# ===========================================================================
REMOTE_MODELS = {
    "qwen3.5-9b":       {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-9B"},
    "qwen3.5-27b":      {"provider": "huggingface",      "model_id": "Qwen/Qwen3.5-27B"},
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

# ===========================================================================
# PROMPT TEMPLATE
# ===========================================================================
PROMPT_TEMPLATE = (
    "The first image is a {word}. "
    "Which of the following two images (A or B) is also a {word}? "
    "Answer with just 'A' or 'B'."
)

# ===========================================================================
# CSV OUTPUT FIELDS
# ===========================================================================
CSV_FIELDS = [
    "model", "model_name", "stim_id", "word", "word_type", "word_length",
    "ordering", "a_is", "b_is", "raw_text", "parsed_answer", "choice",
    "generation_time_s", "num_tokens_generated", "attempts",
]

# ===========================================================================
# Setup
# ===========================================================================
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(ENV_PATH)


# ---------------------------------------------------------------------------
# Word list
# ---------------------------------------------------------------------------
def load_words() -> list[dict]:
    """Load the curated length-matched word pairs as a flat list."""
    words = []
    for sudo, rand, length in WORD_PAIRS:
        words.append({"name": sudo,  "type": "sudo",   "length": length})
        words.append({"name": rand,  "type": "random", "length": length})
    return words


def make_prompt(word: str) -> str:
    return PROMPT_TEMPLATE.format(word=word)


# ---------------------------------------------------------------------------
# Stimuli loading
# ---------------------------------------------------------------------------
def load_stimuli(stim_set: str = DEFAULT_STIM_SET,
                 num_stimuli: int | None = None) -> list[dict]:
    stim_base = STIMULI_DIR / stim_set
    stim_dirs = sorted([d for d in stim_base.iterdir() if d.is_dir()],
                       key=lambda d: int(d.name))

    if num_stimuli is not None:
        stim_dirs = random.sample(stim_dirs, min(num_stimuli, len(stim_dirs)))

    stimuli = []
    for d in stim_dirs:
        stimuli.append({
            "stim_id": d.name,
            "reference": Image.open(d / "reference.png").convert("RGB"),
            "shape_match": Image.open(d / "shape_match.png").convert("RGB"),
            "texture_match": Image.open(d / "texture_match.png").convert("RGB"),
        })
    return stimuli


# ---------------------------------------------------------------------------
# Remote model inference
# ---------------------------------------------------------------------------
def image_to_base64_url(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
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
    hf_token = os.environ.get("HUGGING_FACE") or os.environ.get("HF_API_TOKEN")

    client = OpenAI(api_key=hf_token, base_url=base_url)
    messages = build_messages(images, prompt)

    # Disable thinking mode for Qwen3.5 models to avoid wasting tokens.
    # Also bump max_tokens as a safety net — if thinking isn't fully
    # disabled by the provider, the thinking tokens eat into the budget.
    extra = {}
    max_tok = MAX_TOKENS_REMOTE
    if "qwen" in cfg["model_id"].lower():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        max_tok = max(max_tok, 8192)

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
# Local model inference
# ---------------------------------------------------------------------------
def run_local(model, images: list[Image.Image], prompt: str) -> dict:
    from evaluation_pipe.models.base import ModelResponse
    resp: ModelResponse = model.generate(
        images=images, prompt=prompt,
        max_new_tokens=MAX_TOKENS_LOCAL, temperature=TEMPERATURE,
    )
    return {
        "raw_text": resp.raw_text,
        "generation_time_s": round(resp.generation_time_s, 2),
        "model_name": resp.model_name,
        "num_tokens_generated": resp.num_tokens_generated,
    }


# ---------------------------------------------------------------------------
# Answer parsing with retry
# ---------------------------------------------------------------------------
def parse_answer(raw_text: str) -> str | None:
    upper = raw_text.upper()
    has_a = "A" in upper
    has_b = "B" in upper
    if has_a and has_b:
        return None
    if has_a:
        return "A"
    if has_b:
        return "B"
    return None


def run_with_retry(run_fn, images: list[Image.Image], prompt: str) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        result = run_fn(images, prompt)
        answer = parse_answer(result["raw_text"])
        if answer is not None:
            result["parsed_answer"] = answer
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty: {result['raw_text']!r}")

    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


# ---------------------------------------------------------------------------
# Single trial (both orderings)
# ---------------------------------------------------------------------------
def run_trial(run_fn, stimulus: dict, word: str, word_type: str,
              word_length: int = 0) -> list[dict]:
    """Run one stimulus in both orderings. Returns two result dicts."""
    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word)
    results = []

    # Ordering 1: A=shape, B=texture
    res = run_with_retry(run_fn, [ref, shape, texture], prompt)
    answer = res.get("parsed_answer")
    if answer == "A":
        choice = "shape"
    elif answer == "B":
        choice = "texture"
    else:
        choice = "unclear"
    results.append({
        **res,
        "stim_id": stimulus["stim_id"],
        "word": word,
        "word_type": word_type,
        "word_length": word_length,
        "ordering": "shape_first",
        "a_is": "shape",
        "b_is": "texture",
        "choice": choice,
    })

    # Ordering 2: A=texture, B=shape
    res = run_with_retry(run_fn, [ref, texture, shape], prompt)
    answer = res.get("parsed_answer")
    if answer == "A":
        choice = "texture"
    elif answer == "B":
        choice = "shape"
    else:
        choice = "unclear"
    results.append({
        **res,
        "stim_id": stimulus["stim_id"],
        "word": word,
        "word_type": word_type,
        "word_length": word_length,
        "ordering": "texture_first",
        "a_is": "texture",
        "b_is": "shape",
        "choice": choice,
    })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run shape-bias evaluation")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names to evaluate. Use 'all-remote' or 'all-local'.")
    parser.add_argument("--stim-set", default=DEFAULT_STIM_SET,
                        help=f"Stimulus set directory name (default: {DEFAULT_STIM_SET})")
    parser.add_argument("--num-stimuli", type=int, default=None,
                        help="Number of stimuli to sample (default: all)")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help=f"Device for local models (default: {DEFAULT_DEVICE})")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: results/<timestamp>.csv)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve model list
    model_names = []
    for m in args.models:
        if m == "all-remote":
            model_names.extend(REMOTE_MODELS.keys())
        elif m == "all-local":
            from evaluation_pipe.models import list_models
            model_names.extend(list_models())
        else:
            model_names.append(m)

    # Classify models
    remote_names = [m for m in model_names if m in REMOTE_MODELS]
    local_names = [m for m in model_names if m not in REMOTE_MODELS]

    print(f"Remote models: {remote_names or '(none)'}")
    print(f"Local models:  {local_names or '(none)'}")

    # Load stimuli and words
    words = load_words()
    stimuli = load_stimuli(args.stim_set, args.num_stimuli)
    print(f"Stimuli: {len(stimuli)} from {args.stim_set}")
    print(f"Words: {len(words)} ({len(words)//2} sudo + {len(words)//2} random, length-matched)")
    print(f"Trials per model: {len(stimuli)} stimuli x {len(words)} words x 2 orderings = {len(stimuli) * len(words) * 2}")
    print()

    # Prepare output
    output_path = args.output
    if output_path is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"eval_{timestamp}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # --- Run remote models ---
    for model_key in remote_names:
        print(f"{'='*60}")
        print(f"Remote model: {model_key}")
        print(f"{'='*60}")

        def run_fn(images, prompt, _mk=model_key):
            return run_remote(_mk, images, prompt)

        for stim in stimuli:
            for w in words:
                word, word_type, word_length = w["name"], w["type"], w["length"]
                print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                trial_results = run_trial(run_fn, stim, word, word_type, word_length)
                for r in trial_results:
                    r["model"] = model_key
                    print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                    all_results.append(r)

    # --- Run local models ---
    for model_key in local_names:
        print(f"\n{'='*60}")
        print(f"Local model: {model_key}")
        print(f"{'='*60}")

        from evaluation_pipe.models import create_model
        model = create_model(model_key, device=args.device)
        print(f"  Loaded: {model.name}")

        def run_fn(images, prompt, _m=model):
            return run_local(_m, images, prompt)

        for stim in stimuli:
            for w in words:
                word, word_type, word_length = w["name"], w["type"], w["length"]
                print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                trial_results = run_trial(run_fn, stim, word, word_type, word_length)
                for r in trial_results:
                    r["model"] = model_key
                    print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                    all_results.append(r)

        model.unload()
        print(f"  Unloaded {model_key}")

    # --- Write CSV ---
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {output_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"  {'Model':25s}  {'Shape':>6s}  {'Texture':>8s}  {'Unclear':>8s}  {'Shape %':>8s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")

    for model_key in model_names:
        model_res = [r for r in all_results if r["model"] == model_key]
        s = sum(1 for r in model_res if r["choice"] == "shape")
        t = sum(1 for r in model_res if r["choice"] == "texture")
        u = sum(1 for r in model_res if r["choice"] == "unclear")
        pct = f"{s / (s + t) * 100:.0f}%" if (s + t) > 0 else "N/A"
        print(f"  {model_key:25s}  {s:>6d}  {t:>8d}  {u:>8d}  {pct:>8s}")

    # Breakdown by ordering
    print(f"\n  {'Model':25s}  {'Ordering':15s}  {'Shape':>6s}  {'Texture':>8s}  {'Shape %':>8s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*8}")
    for model_key in model_names:
        for ordering in ["shape_first", "texture_first"]:
            subset = [r for r in all_results
                      if r["model"] == model_key and r["ordering"] == ordering]
            s = sum(1 for r in subset if r["choice"] == "shape")
            t = sum(1 for r in subset if r["choice"] == "texture")
            pct = f"{s / (s + t) * 100:.0f}%" if (s + t) > 0 else "N/A"
            print(f"  {model_key:25s}  {ordering:15s}  {s:>6d}  {t:>8d}  {pct:>8s}")


if __name__ == "__main__":
    main()
