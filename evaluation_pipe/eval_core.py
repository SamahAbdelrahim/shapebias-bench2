"""Shared evaluation logic for shape-bias benchmarking.

This module contains all shared constants, helpers, and trial-running logic
used by both `scripts/run_remote.py` and `scripts/run_local.py`.
"""

from __future__ import annotations

import csv
import os
import random
import re
import time
from pathlib import Path

from PIL import Image

# ===========================================================================
# PATHS
# ===========================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
STIMULI_DIR = REPO_ROOT / "stimuli_pipe" / "stimuli_per_stl_packages"
RESULTS_DIR = REPO_ROOT / "results"
# When `run_remote.py` runs `--eval-mode human_matched` without `-o`, CSVs default here
# (under RESULTS_DIR or `$RESULTS_DIR`), next to other `model.results` trees for analysis_pipe.
HUMAN_MATCHED_REMOTE_CSV_SUBDIR = Path("model.results") / "human_matched"
ENV_PATH = REPO_ROOT / ".env"


def default_session_results_dir(kind: str, *, date: str | None = None, host: str | None = None) -> Path:
    """Dated session folder under ``results/{kind}.results/``.

    ``kind`` is typically ``playground`` or ``probe``. Host defaults to
    ``$RESULTS_SESSION_HOST`` or ``farmshare``.
    """
    import time as _time

    day = date or _time.strftime("%Y-%m-%d")
    label = host or os.environ.get("RESULTS_SESSION_HOST", "farmshare")
    return RESULTS_DIR / f"{kind}.results" / f"session_{day}_{label}"


# Human-friendly stimulus packages (must match human-experiment/server.js).
HUMAN_STIM_PACKAGES = frozenset(
    {"stimuli_unique_texture_per_stl_v1", "stimuli_unique_texture_per_stl_v2"}
)
BENCHMARK_STIM_PACKAGE = "stimuli_per_stl_packages"

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
DEFAULT_HUMAN_TRIAL_LIMIT = 30

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

# Frequent English bigrams — mirrors human-experiment/public/experiment.js
ENGLISH_BIGRAM_WEIGHTS: dict[str, float] = {
    "th": 1.0, "he": 0.98, "in": 0.96, "er": 0.94, "an": 0.93, "re": 0.92, "on": 0.91, "at": 0.9,
    "en": 0.9, "nd": 0.89, "ti": 0.88, "es": 0.87, "or": 0.86, "te": 0.86, "of": 0.85, "ed": 0.85,
    "is": 0.84, "it": 0.84, "al": 0.83, "ar": 0.82, "st": 0.82, "to": 0.82, "nt": 0.81, "ng": 0.81,
    "se": 0.8, "ha": 0.8, "as": 0.79, "ou": 0.79, "io": 0.78, "le": 0.78, "ve": 0.77, "co": 0.77,
    "me": 0.76, "de": 0.76, "hi": 0.75, "ri": 0.75, "ro": 0.74, "ic": 0.74, "ne": 0.74, "ea": 0.73,
    "ra": 0.73, "ce": 0.72, "li": 0.72, "ch": 0.72, "ll": 0.71, "be": 0.71, "ma": 0.7, "si": 0.7,
    "om": 0.69, "ur": 0.69, "ca": 0.68, "el": 0.68, "ta": 0.68, "la": 0.67, "ns": 0.67, "di": 0.67,
    "fo": 0.66, "ho": 0.66, "pe": 0.65, "ec": 0.65, "pr": 0.65, "no": 0.64, "wa": 0.64, "wi": 0.64,
    "us": 0.63, "tr": 0.63, "wh": 0.63, "ge": 0.62, "po": 0.62, "lo": 0.62, "im": 0.61, "il": 0.61,
    "mo": 0.61, "un": 0.6, "ai": 0.6, "ie": 0.59, "oo": 0.59, "ee": 0.58, "ss": 0.57, "tt": 0.57,
}

# ===========================================================================
# PROMPT TEMPLATES
# ===========================================================================
PROMPT_TEMPLATES = {
    "noun_label": (
        "The first image is a {word}. "
        "Which of the following two images (1 or 2) is also a {word}? "
        "Your response must be exactly one character: 1 or 2. No other text."
    ),
    "noun_label_AB": (
        "The first image is a {word}. "
        "Which of the following two images (A or B) is also a {word}? "
        "Your response must be exactly one character: A or B. No other text."
    ),
    "no_word_category": (
        "See this object in the first image. "
        "Can you find another one of the two (1 or 2)? "
        "Your response must be exactly one character: 1 or 2. No other text."
    ),
    "no_word_category_AB": (
        "See this object in the first image. "
        "Can you find another one of the two (A or B)? "
        "Your response must be exactly one character: A or B. No other text."
    ),
    "binary_yes_no": (
        "The first image is a {word}. "
        "The second image is one candidate object. "
        "Is the second image the same object category as the first image? "
        "Your response must be exactly one word: YES or NO. No other text."
    ),
    "binary_yes_no_conservative": (
        "You see 2 images: Image 1 is the reference object and Image 2 is one candidate object. "
        "Question: Is Image 2 the same object category as Image 1? "
        "Rules: answer with exactly one token YES or NO. If uncertain, answer NO."
    ),
    "binary_score": (
        "The first image is a {word}. "
        "The second image is one candidate object. "
        "Rate how likely the second image is the same object category as the first image, "
        "from 0 to 100. "
        "Your response must be exactly: SCORE=<integer 0-100>. No other text."
    ),
    "binary_score_0_3": (
        "Image 1 is reference and Image 2 is one candidate. "
        "Rate match strength from 0 to 3, where 0 means no match and 3 means very strong match. "
        "Your response must be exactly: SCORE=<integer 0-3>. No other text."
    ),
    "rank_forced": (
        "Image 1 is the reference object. Image 2 and Image 3 are candidates. "
        "Rank the candidates by similarity to Image 1. "
        "Your response must be exactly: BETTER=<1 or 2>; WORSE=<1 or 2>. "
        "BETTER and WORSE must be different."
    ),
}

# Backward compat for scripts that import a single noun_label template string.
PROMPT_TEMPLATE = PROMPT_TEMPLATES["noun_label"]

# ===========================================================================
# Vision turn layout (shared: local Transformers + remote OpenAI-compatible API)
# ===========================================================================
VISION_USER_IMAGE_LABELS_3 = ("Reference image:", "Image 1:", "Image 2:")
VISION_USER_IMAGE_LABELS_2 = ("Reference image:", "Candidate image:")

# Shared system line for **local** Transformers VLM wrappers (generate + score_choices).
# Keep generate and logit scoring on the same messages so the two paths stay consistent.
LOCAL_VLM_SYSTEM_PROMPT = "Answer concisely. Do not explain your reasoning."
# Backward-compatible alias (older Qwen3.5-only name).
QWEN35_VLM_SYSTEM_PROMPT = LOCAL_VLM_SYSTEM_PROMPT

# **Remote** (`run_remote.py`): one uniform system line for every `REMOTE_MODELS` entry.
# See `interpret/remote_eval_prompt_policy.md`.
REMOTE_UNIFORM_SYSTEM_PROMPT = (
    "Follow the user's instructions exactly. "
    "When they ask for a single character (1 or 2), reply with only that character and no other text."
)


def build_transformers_vision_user_content(
    images: list[Image.Image],
    prompt: str,
) -> list[dict]:
    """Content block for a single user message (Transformers chat templates)."""
    if len(images) == 3:
        labels = VISION_USER_IMAGE_LABELS_3
    elif len(images) == 2:
        labels = VISION_USER_IMAGE_LABELS_2
    else:
        raise ValueError("expected 2 or 3 images")
    content: list[dict] = []
    for label, img in zip(labels, images):
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": prompt})
    return content


def build_openai_compatible_vision_messages(
    images: list[Image.Image],
    prompt: str,
    *,
    image_to_url,
    system_prompt: str | None = None,
) -> list[dict]:
    """Chat messages for OpenAI-style vision APIs (e.g. HF router).

    *image_to_url* maps each PIL image to a URL string (e.g. data:image/jpeg;base64,...).
    """
    if len(images) == 3:
        labels = VISION_USER_IMAGE_LABELS_3
    elif len(images) == 2:
        labels = VISION_USER_IMAGE_LABELS_2
    else:
        raise ValueError("expected 2 or 3 images")
    content: list[dict] = []
    for label, img in zip(labels, images):
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_url(img)},
        })
    content.append({"type": "text", "text": prompt})
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


# ===========================================================================
# CSV OUTPUT FIELDS
# ===========================================================================
CSV_FIELDS = [
    "model", "model_name", "stim_id", "word", "word_type", "word_length",
    "prompt_condition",
    "decision_mode", "swap_correct",
    "ordering", "order_method", "a_is", "b_is", "raw_text", "parsed_answer", "choice",
    "prob_1_abs", "prob_2_abs", "swap_prob_1_abs", "swap_prob_2_abs", "swap_corrected_a_abs", "swap_corrected_b_abs",
    "generation_time_s", "num_tokens_generated", "attempts",
    "repeat", "temperature",
    "eval_mode", "stim_pkg", "stim_set", "human_word_seed", "stimulus_shuffle_condition",
    "word_mode", "word_min_len", "word_max_len", "sudo_threshold", "trial_limit",
]

# ---------------------------------------------------------------------------
# Human-eval: hash / RNG (match experiment.js)
# ---------------------------------------------------------------------------
def _u32(x: int) -> int:
    return x & 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return _u32(a * b)


def hash_string(s: str) -> int:
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = _imul(h, 16777619)
    return _u32(h)


def mulberry32(seed: int):
    t = _u32(seed)

    def rand() -> float:
        nonlocal t
        t = _u32(t + 0x6D2B79F5)
        x = _imul(t ^ (t >> 15), t | 1)
        x = _u32(x ^ (x + _imul(x ^ (x >> 7), x | 61)))
        return ((x ^ (x >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return rand


def random_int(rand_fn, min_inclusive: int, max_inclusive: int) -> int:
    return int(rand_fn() * (max_inclusive - min_inclusive + 1)) + min_inclusive


def english_transition_score(word: str) -> float:
    w = "".join(c for c in word.lower() if "a" <= c <= "z")
    if len(w) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(w) - 1):
        bg = w[i : i + 2]
        total += ENGLISH_BIGRAM_WEIGHTS.get(bg, 0.02)
        n += 1
    return total / n


def make_pseudo_word(rand_fn, length: int) -> str:
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    out = []
    for i in range(length):
        bank = consonants if i % 2 == 0 else vowels
        out.append(bank[random_int(rand_fn, 0, len(bank) - 1)])
    return "".join(out)


def make_random_word(rand_fn, length: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(letters[random_int(rand_fn, 0, 25)] for _ in range(length))


def build_unique_human_words(
    count: int,
    seed_text: str,
    *,
    mode: str = "sudo_only",
    word_min_len: int = 4,
    word_max_len: int = 8,
    sudo_threshold: float = 0.62,
) -> list[dict]:
    """Match human-experiment/public/experiment.js buildUniqueHumanWords."""
    rand_fn = mulberry32(hash_string(seed_text))
    seen: set[str] = set()
    out: list[dict] = []
    lengths = list(range(word_min_len, word_max_len + 1))

    sudo_count = (count + 1) // 2 if mode == "mixed" else count
    random_count = count // 2 if mode == "mixed" else 0
    type_pool = ["sudo"] * sudo_count + ["random"] * random_count

    for i in range(len(type_pool) - 1, 0, -1):
        j = random_int(rand_fn, 0, i)
        type_pool[i], type_pool[j] = type_pool[j], type_pool[i]

    for idx in range(count):
        word_type = type_pool[idx]
        length = lengths[idx % len(lengths)]
        candidate = ""
        if word_type == "sudo":
            best_candidate = ""
            best_score = -1.0
            accepted = False
            for _ in range(400):
                maybe = make_pseudo_word(rand_fn, length)
                if maybe in seen:
                    continue
                sc = english_transition_score(maybe)
                if sc > best_score:
                    best_score = sc
                    best_candidate = maybe
                if sc >= sudo_threshold:
                    candidate = maybe
                    accepted = True
                    break
            if not accepted:
                candidate = best_candidate or make_pseudo_word(rand_fn, length)
        else:
            while True:
                candidate = make_random_word(rand_fn, length)
                if candidate not in seen:
                    break
        seen.add(candidate)
        out.append({"name": candidate, "type": word_type, "length": length})
    return out


def maybe_sample_stimuli_human(stimuli: list[dict], trial_limit: int, seed_text: str) -> list[dict]:
    """Match experiment.js maybeSampleStimuli (Fisher–Yates with mulberry32)."""
    if trial_limit <= 0 or trial_limit >= len(stimuli):
        return list(stimuli)
    rand_fn = mulberry32(hash_string(seed_text))
    idxs = list(range(len(stimuli)))
    for i in range(len(idxs) - 1, 0, -1):
        j = random_int(rand_fn, 0, i)
        idxs[i], idxs[j] = idxs[j], idxs[i]
    return [stimuli[i] for i in idxs[:trial_limit]]


def resolve_stim_set_name(stim_set: str | None) -> str:
    if stim_set is None:
        env_dataset = os.environ.get("IMAGE_DATASET")
        return Path(env_dataset).name if env_dataset else DEFAULT_STIM_SET
    return stim_set


def human_eval_seed_text(
    human_eval_seed: str,
    stim_set: str,
    stim_pkg: str,
    condition: str,
    word_mode: str,
) -> str:
    """Base seed string aligned with human-friendly builder (participant ids → one token)."""
    return f"{human_eval_seed}|{stim_set}|{stim_pkg}|{condition}|{word_mode}"


def benchmark_csv_meta(stim_set: str) -> dict[str, str]:
    return {
        "eval_mode": "benchmark",
        "stim_pkg": BENCHMARK_STIM_PACKAGE,
        "stim_set": stim_set,
        "human_word_seed": "",
        "stimulus_shuffle_condition": "",
        "word_mode": "",
        "word_min_len": "",
        "word_max_len": "",
        "sudo_threshold": "",
        "trial_limit": "",
    }


def human_matched_csv_meta(
    *,
    stim_pkg: str,
    stim_set: str,
    human_word_seed: str,
    stimulus_shuffle_condition: str,
    word_mode: str,
    word_min_len: int,
    word_max_len: int,
    sudo_threshold: float,
    trial_limit: int,
) -> dict[str, str]:
    return {
        "eval_mode": "human_matched",
        "stim_pkg": stim_pkg,
        "stim_set": stim_set,
        "human_word_seed": human_word_seed,
        "stimulus_shuffle_condition": stimulus_shuffle_condition,
        "word_mode": word_mode,
        "word_min_len": str(word_min_len),
        "word_max_len": str(word_max_len),
        "sudo_threshold": str(sudo_threshold),
        "trial_limit": str(trial_limit),
    }


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


def make_prompt(word: str | None = None, prompt_condition: str = "noun_label") -> str:
    if prompt_condition not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt_condition '{prompt_condition}'")
    template = PROMPT_TEMPLATES[prompt_condition]
    if "{word}" in template:
        return template.format(word=word or "object")
    return template


# ---------------------------------------------------------------------------
# Stimuli loading
# ---------------------------------------------------------------------------
def load_stimuli(stim_set: str | None = None,
                 num_stimuli: int | None = None) -> list[dict]:
    stim_set = resolve_stim_set_name(stim_set)
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


def load_stimuli_human_package(stim_pkg: str, stim_set: str | None = None) -> list[dict]:
    """Load stimuli from v1/v2 human packages in manifest.csv order."""
    if stim_pkg not in HUMAN_STIM_PACKAGES:
        raise ValueError(
            f"stim_pkg must be one of {sorted(HUMAN_STIM_PACKAGES)}, got {stim_pkg!r}"
        )
    stim_set = resolve_stim_set_name(stim_set)
    manifest_path = REPO_ROOT / "stimuli_pipe" / stim_pkg / stim_set / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stimuli: list[dict] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("stl_id", "")).strip()
            if not sid:
                continue
            trial_dir = REPO_ROOT / "stimuli_pipe" / stim_pkg / stim_set / sid
            stimuli.append({
                "stim_id": sid,
                "reference": Image.open(trial_dir / "reference.png").convert("RGB"),
                "shape_match": Image.open(trial_dir / "shape_match.png").convert("RGB"),
                "texture_match": Image.open(trial_dir / "texture_match.png").convert("RGB"),
            })
    return stimuli


# ---------------------------------------------------------------------------
# Answer parsing with retry
# ---------------------------------------------------------------------------
def parse_answer(raw_text: str) -> str | None:
    has_1 = "1" in raw_text
    has_2 = "2" in raw_text
    if has_1 and has_2:
        return None
    if has_1:
        return "1"
    if has_2:
        return "2"
    return None


def parse_yes_no(raw_text: str) -> str | None:
    txt = (raw_text or "").strip().lower()
    has_yes = "yes" in txt
    has_no = "no" in txt
    if has_yes and has_no:
        return None
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return None


def parse_score_0_100(raw_text: str) -> int | None:
    txt = (raw_text or "").strip().lower()
    m = re.search(r"score\s*=\s*(\d{1,3})\b", txt)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    # Fallback: first standalone integer in range
    m2 = re.search(r"\b(\d{1,3})\b", txt)
    if m2:
        v = int(m2.group(1))
        if 0 <= v <= 100:
            return v
    return None


def parse_score_0_3(raw_text: str) -> int | None:
    txt = (raw_text or "").strip().lower()
    m = re.search(r"score\s*=\s*([0-3])\b", txt)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b([0-3])\b", txt)
    if m2:
        return int(m2.group(1))
    return None


def parse_rank_forced(raw_text: str) -> str | None:
    """Return BETTER label ('1' or '2') from strict rank output."""
    txt = (raw_text or "").strip().lower()
    m = re.search(r"better\s*=\s*([12])\s*;\s*worse\s*=\s*([12])", txt)
    if not m:
        m = re.search(r"worse\s*=\s*([12])\s*;\s*better\s*=\s*([12])", txt)
        if m:
            worse, better = m.group(1), m.group(2)
            if better != worse:
                return better
            return None
        return None
    better, worse = m.group(1), m.group(2)
    if better == worse:
        return None
    return better


def run_with_retry(run_fn, images: list[Image.Image], prompt: str) -> dict:
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = run_fn(images, prompt)
        except Exception as e:
            print(f"    [retry {attempt}/{MAX_RETRIES}] error: {e}")
            continue
        answer = parse_answer(result["raw_text"])
        if answer is not None:
            result["parsed_answer"] = answer
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty: {result['raw_text']!r}")

    if result is None:
        result = {"raw_text": "", "generation_time_s": 0, "model_name": "", "num_tokens_generated": 0}
    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


def run_with_retry_yes_no(run_fn, images: list[Image.Image], prompt: str) -> dict:
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = run_fn(images, prompt)
        except Exception as e:
            print(f"    [retry {attempt}/{MAX_RETRIES}] error: {e}")
            continue
        answer = parse_yes_no(result["raw_text"])
        if answer is not None:
            result["parsed_answer"] = answer
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty YES/NO: {result['raw_text']!r}")

    if result is None:
        result = {"raw_text": "", "generation_time_s": 0, "model_name": "", "num_tokens_generated": 0}
    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


def run_with_retry_score(run_fn, images: list[Image.Image], prompt: str) -> dict:
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = run_fn(images, prompt)
        except Exception as e:
            print(f"    [retry {attempt}/{MAX_RETRIES}] error: {e}")
            continue
        score = parse_score_0_100(result["raw_text"])
        if score is not None:
            result["parsed_answer"] = str(score)
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty SCORE: {result['raw_text']!r}")

    if result is None:
        result = {"raw_text": "", "generation_time_s": 0, "model_name": "", "num_tokens_generated": 0}
    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


def run_with_retry_score_0_3(run_fn, images: list[Image.Image], prompt: str) -> dict:
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = run_fn(images, prompt)
        except Exception as e:
            print(f"    [retry {attempt}/{MAX_RETRIES}] error: {e}")
            continue
        score = parse_score_0_3(result["raw_text"])
        if score is not None:
            result["parsed_answer"] = str(score)
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/empty SCORE(0-3): {result['raw_text']!r}")

    if result is None:
        result = {"raw_text": "", "generation_time_s": 0, "model_name": "", "num_tokens_generated": 0}
    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


def run_with_retry_rank_forced(run_fn, images: list[Image.Image], prompt: str) -> dict:
    result = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = run_fn(images, prompt)
        except Exception as e:
            print(f"    [retry {attempt}/{MAX_RETRIES}] error: {e}")
            continue
        better = parse_rank_forced(result["raw_text"])
        if better is not None:
            result["parsed_answer"] = better
            result["attempts"] = attempt
            return result
        print(f"    [retry {attempt}/{MAX_RETRIES}] ambiguous/invalid rank output: {result['raw_text']!r}")

    if result is None:
        result = {"raw_text": "", "generation_time_s": 0, "model_name": "", "num_tokens_generated": 0}
    result["parsed_answer"] = None
    result["attempts"] = MAX_RETRIES
    return result


# ---------------------------------------------------------------------------
# Single trial (both orderings)
# ---------------------------------------------------------------------------
def run_trial(run_fn, stimulus: dict, word: str, word_type: str,
              word_length: int = 0, ordering: str = "both",
              prompt_condition: str = "noun_label") -> list[dict]:
    """Run one stimulus in specified ordering(s). Returns list of result dicts.

    ordering: "shape_first", "texture_first", "random", or "both" (default).
    """
    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word, prompt_condition=prompt_condition)

    # Determine which orderings to run
    orderings_config = {
        "shape_first":   [("shape_first",   shape, texture, "shape", "texture")],
        "texture_first": [("texture_first", texture, shape, "texture", "shape")],
        "both": [
            ("shape_first",   shape, texture, "shape", "texture"),
            ("texture_first", texture, shape, "texture", "shape"),
        ],
    }
    if ordering == "random":
        configs = list(orderings_config["both"])
        random.shuffle(configs)
        configs = configs[:1]  # pick one at random
        order_method = "random"
    else:
        configs = orderings_config[ordering]
        order_method = "deterministic"

    results = []
    for ord_name, img_a, img_b, a_is, b_is in configs:
        res = run_with_retry(run_fn, [ref, img_a, img_b], prompt)
        answer = res.get("parsed_answer")
        if answer == "1":
            choice = a_is
        elif answer == "2":
            choice = b_is
        else:
            choice = "unclear"
        results.append({
            **res,
            "stim_id": stimulus["stim_id"],
            "word": word,
            "word_type": word_type,
            "word_length": word_length,
            "prompt_condition": prompt_condition,
            "ordering": ord_name,
            "order_method": order_method,
            "a_is": a_is,
            "b_is": b_is,
            "choice": choice,
        })

    return results


def run_trial_binary_pair(
    run_fn,
    stimulus: dict,
    word: str,
    word_type: str,
    word_length: int = 0,
    prompt_condition: str = "binary_yes_no_conservative",
) -> list[dict]:
    """Run two independent binary decisions (shape candidate, texture candidate)."""
    if prompt_condition not in {"binary_yes_no", "binary_yes_no_conservative", "binary_score"}:
        raise ValueError(
            "run_trial_binary_pair expects prompt_condition in "
            "{'binary_yes_no','binary_yes_no_conservative','binary_score'}"
        )

    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word, prompt_condition=prompt_condition)

    if prompt_condition == "binary_score":
        shape_res = run_with_retry_score(run_fn, [ref, shape], prompt)
        texture_res = run_with_retry_score(run_fn, [ref, texture], prompt)
        s_shape = int(shape_res["parsed_answer"]) if shape_res.get("parsed_answer") is not None else None
        s_texture = int(texture_res["parsed_answer"]) if texture_res.get("parsed_answer") is not None else None
        if s_shape is not None and s_texture is not None:
            if s_shape > s_texture:
                choice = "shape"
            elif s_texture > s_shape:
                choice = "texture"
            else:
                choice = "unclear"
        else:
            choice = "unclear"
    else:
        shape_res = run_with_retry_yes_no(run_fn, [ref, shape], prompt)
        texture_res = run_with_retry_yes_no(run_fn, [ref, texture], prompt)
        shape_ans = shape_res.get("parsed_answer")
        texture_ans = texture_res.get("parsed_answer")
        if shape_ans == "yes" and texture_ans == "no":
            choice = "shape"
        elif shape_ans == "no" and texture_ans == "yes":
            choice = "texture"
        elif (
            prompt_condition == "binary_yes_no_conservative"
            and shape_ans == "yes"
            and texture_ans == "yes"
        ):
            # Tie-break only when both candidates are positive.
            tie_prompt = make_prompt(word, prompt_condition="binary_score_0_3")
            shape_tie = run_with_retry_score_0_3(run_fn, [ref, shape], tie_prompt)
            texture_tie = run_with_retry_score_0_3(run_fn, [ref, texture], tie_prompt)
            s_shape = int(shape_tie["parsed_answer"]) if shape_tie.get("parsed_answer") is not None else None
            s_texture = int(texture_tie["parsed_answer"]) if texture_tie.get("parsed_answer") is not None else None
            if s_shape is not None and s_texture is not None:
                if s_shape > s_texture:
                    choice = "shape"
                elif s_texture > s_shape:
                    choice = "texture"
                else:
                    choice = "unclear"
            else:
                choice = "unclear"
            # Attach tie-break detail to raw/parsed payload.
            shape_res["raw_text"] = (
                f"{shape_res.get('raw_text','')} | tie_score={shape_tie.get('raw_text','')}"
            )
            texture_res["raw_text"] = (
                f"{texture_res.get('raw_text','')} | tie_score={texture_tie.get('raw_text','')}"
            )
            shape_res["parsed_answer"] = (
                f"{shape_ans};tie={shape_tie.get('parsed_answer') or 'none'}"
            )
            texture_res["parsed_answer"] = (
                f"{texture_ans};tie={texture_tie.get('parsed_answer') or 'none'}"
            )
        else:
            choice = "unclear"

    raw_text = (
        f"shape_candidate={shape_res.get('raw_text', '')!r}; "
        f"texture_candidate={texture_res.get('raw_text', '')!r}"
    )
    parsed = (
        f"shape={shape_res.get('parsed_answer') or 'none'};"
        f"texture={texture_res.get('parsed_answer') or 'none'}"
    )

    return [
        {
            "raw_text": raw_text,
            "generation_time_s": round(
                float(shape_res.get("generation_time_s", 0.0))
                + float(texture_res.get("generation_time_s", 0.0)),
                2,
            ),
            "model_name": shape_res.get("model_name") or texture_res.get("model_name", ""),
            "num_tokens_generated": (
                int(shape_res.get("num_tokens_generated") or 0)
                + int(texture_res.get("num_tokens_generated") or 0)
            ),
            "parsed_answer": parsed,
            "attempts": int(shape_res.get("attempts", MAX_RETRIES))
            + int(texture_res.get("attempts", MAX_RETRIES)),
            "stim_id": stimulus["stim_id"],
            "word": word,
            "word_type": word_type,
            "word_length": word_length,
            "prompt_condition": prompt_condition,
            "ordering": "binary_pair",
            "order_method": "independent_binary",
            "a_is": "shape",
            "b_is": "texture",
            "choice": choice,
        }
    ]


def run_trial_rank_forced(
    run_fn,
    stimulus: dict,
    word: str,
    word_type: str,
    word_length: int = 0,
    ordering: str = "both",
    prompt_condition: str = "rank_forced",
) -> list[dict]:
    """Run 3-image forced ranking with strict BETTER/WORSE parse."""
    if prompt_condition != "rank_forced":
        raise ValueError("run_trial_rank_forced expects prompt_condition='rank_forced'")

    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word, prompt_condition=prompt_condition)

    orderings_config = {
        "shape_first": [("shape_first", shape, texture, "shape", "texture")],
        "texture_first": [("texture_first", texture, shape, "texture", "shape")],
        "both": [
            ("shape_first", shape, texture, "shape", "texture"),
            ("texture_first", texture, shape, "texture", "shape"),
        ],
    }
    if ordering == "random":
        configs = list(orderings_config["both"])
        random.shuffle(configs)
        configs = configs[:1]
        order_method = "random"
    else:
        configs = orderings_config[ordering]
        order_method = "deterministic"

    results = []
    for ord_name, img_a, img_b, a_is, b_is in configs:
        res = run_with_retry_rank_forced(run_fn, [ref, img_a, img_b], prompt)
        answer = res.get("parsed_answer")
        if answer == "1":
            choice = a_is
        elif answer == "2":
            choice = b_is
        else:
            choice = "unclear"
        results.append(
            {
                **res,
                "stim_id": stimulus["stim_id"],
                "word": word,
                "word_type": word_type,
                "word_length": word_length,
                "prompt_condition": prompt_condition,
                "ordering": ord_name,
                "order_method": order_method,
                "a_is": a_is,
                "b_is": b_is,
                "choice": choice,
            }
        )
    return results

def run_trial_logit_scoring(
    run_score_fn,
    stimulus: dict,
    word: str,
    word_type: str,
    word_length: int = 0,
    *,
    choice_texts=("1", "2"),
    ordering: str = "both",
    prompt_condition: str = "noun_label",
    swap_correct: bool = False,
) -> list[dict]:
    ref = stimulus["reference"]
    shape = stimulus["shape_match"]
    texture = stimulus["texture_match"]
    prompt = make_prompt(word, prompt_condition=prompt_condition)

    orderings_config = {
        "shape_first": [("shape_first", shape, texture, "shape", "texture")],
        "texture_first": [("texture_first", texture, shape, "texture", "shape")],
        "both": [
            ("shape_first", shape, texture, "shape", "texture"),
            ("texture_first", texture, shape, "texture", "shape"),
        ],
    }
    if ordering == "random":
        configs = list(orderings_config["both"])
        random.shuffle(configs)
        configs = configs[:1]
        order_method = "random"
    else:
        configs = orderings_config[ordering]
        order_method = "deterministic"

    results = []
    for ord_name, img_a, img_b, a_is, b_is in configs:
        base = run_score_fn([ref, img_a, img_b], prompt)
        p1_abs, p2_abs = base["choice_probs_absolute"]
        l1, l2 = base["choice_logits"]
        total_t = float(base["generation_time_s"])
        decision_a_abs = p1_abs
        decision_b_abs = p2_abs

        sp1_abs, sp2_abs = None, None
        swap_corrected_a_abs, swap_corrected_b_abs = None, None

        if swap_correct:
            sw = run_score_fn([ref, img_b, img_a], prompt)
            sp1_abs, sp2_abs = sw["choice_probs_absolute"]  # 1->b, 2->a (relative to base semantics)
            sl1, sl2 = sw["choice_logits"]
            decision_a_abs = 0.5 * (p1_abs + sp2_abs)
            decision_b_abs = 0.5 * (p2_abs + sp1_abs)
            swap_corrected_a_abs = decision_a_abs
            swap_corrected_b_abs = decision_b_abs
            total_t += float(sw["generation_time_s"])
            raw_text = (
                f"base[p1_abs={p1_abs:.4f},p2_abs={p2_abs:.4f},l1={l1:.3f},l2={l2:.3f}] "
                f"swap[p1_abs={sp1_abs:.4f},p2_abs={sp2_abs:.4f},l1={sl1:.3f},l2={sl2:.3f}] "
                f"corr[a={swap_corrected_a_abs:.4f},b={swap_corrected_b_abs:.4f}]"
            )
        else:
            raw_text = f"p1_abs={p1_abs:.4f},p2_abs={p2_abs:.4f},l1={l1:.3f},l2={l2:.3f}"

        # under swapped condition, 'parsed' is relative to the original prompt
        if decision_a_abs > decision_b_abs:
            parsed = choice_texts[0]
            choice = a_is
        elif decision_b_abs > decision_a_abs:
            parsed = choice_texts[1]
            choice = b_is
        else:
            parsed = None
            choice = "unclear"

        results.append(
            {
                "raw_text": raw_text,
                "generation_time_s": round(total_t, 2),
                "model_name": base.get("model_name", ""),
                "num_tokens_generated": 0,
                "parsed_answer": parsed,
                "attempts": 1,
                "stim_id": stimulus["stim_id"],
                "word": word,
                "word_type": word_type,
                "word_length": word_length,
                "prompt_condition": prompt_condition,
                "ordering": ord_name,
                "order_method": "logit_forced_swap_corrected" if swap_correct else "logit_forced",
                "a_is": a_is,
                "b_is": b_is,
                "choice": choice,

                "prob_1_abs": p1_abs,
                "prob_2_abs": p2_abs,
                "swap_prob_1_abs": sp1_abs,
                "swap_prob_2_abs": sp2_abs,
                "swap_corrected_a_abs": swap_corrected_a_abs,
                "swap_corrected_b_abs": swap_corrected_b_abs,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Shared CLI argument setup
# ---------------------------------------------------------------------------
def add_common_args(parser) -> None:
    """Add arguments shared by both local and remote scripts."""
    parser.add_argument("--stim-set", default=None,
                        help=f"Stimulus set directory name (default: IMAGE_DATASET env var or {DEFAULT_STIM_SET})")
    parser.add_argument("--num-stimuli", type=int, default=None,
                        help="Number of stimuli to sample (default: all)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: results/<timestamp>.csv)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")


# ---------------------------------------------------------------------------
# CSV writing and summary
# ---------------------------------------------------------------------------
def _csv_row(row: dict) -> dict:
    return {k: row.get(k, "") for k in CSV_FIELDS}


def write_results(all_results: list[dict], output_path: Path,
                   append: bool = False, quiet: bool = False) -> None:
    """Write results to CSV. When append=True, adds rows without overwriting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    mode = "a" if append else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_FIELDS, extrasaction="ignore", restval="",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(_csv_row(r) for r in all_results)
    if not quiet:
        print(f"\nResults {'appended to' if append else 'saved to'} {output_path}")


def _normalize_csv_row_keys(row: dict[str, str | None]) -> dict[str, str]:
    """Strip whitespace and BOM from header-derived keys (Excel / utf-8-sig quirks)."""
    out: dict[str, str] = {}
    for k, v in row.items():
        nk = (k or "").strip().lstrip("\ufeff")
        if not nk:
            continue
        out[nk] = (v or "").strip() if isinstance(v, str) else (str(v) if v is not None else "")
    return out


def load_completed_trial_keys(csv_path: Path) -> set[tuple[str, str, str, str, str]]:
    """Keys for completed benchmark rows: (model, stim_id, word, ordering, repeat).

    Matches skip/resume logic in ``run_remote_benchmark_standardized`` / local rerun.
    Rows missing any of those fields are skipped (legacy or non-benchmark CSVs).
    """
    done: set[tuple[str, str, str, str, str]] = set()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return done
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            hdr = {(n or "").strip().lstrip("\ufeff") for n in reader.fieldnames if n}
            if "model" not in hdr:
                print(
                    "Warning: resume CSV has no 'model' column; "
                    "no trials will be skipped (check file is a benchmark results CSV)."
                )
        for raw in reader:
            row = _normalize_csv_row_keys(raw)
            model = row.get("model")
            stim_id = row.get("stim_id")
            word = row.get("word")
            ordering = row.get("ordering")
            if not model or not stim_id or not word or not ordering:
                continue
            rep = row.get("repeat", "1")
            if rep == "":
                rep = "1"
            done.add((model, stim_id, word, ordering, str(rep)))
    return done


def _shape_pct_decisive(s: int, t: int) -> str:
    return f"{s / (s + t) * 100:.0f}%" if (s + t) > 0 else "N/A"


def _shape_pct_all(s: int, t: int, u: int) -> str:
    tot = s + t + u
    return f"{s / tot * 100:.0f}%" if tot > 0 else "N/A"


def print_summary(all_results: list[dict], model_names: list[str]) -> None:
    """Print shape-bias summary tables."""
    normalized_results = [_normalize_csv_row_keys(r) for r in all_results]

    print(f"\n{'='*90}")
    print(f"{'SUMMARY':^90}")
    print(f"{'='*90}")
    hdr = (
        f"  {'Model':25s}  {'Shape':>6s}  {'Texture':>8s}  {'Unclear':>8s}  "
        f"{'Shape%(dec)':>12s}  {'Shape%(all)':>12s}"
    )
    print(hdr)
    print(f"  {'-'*25}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")

    for model_key in model_names:
        model_res = [r for r in normalized_results if r.get("model") == model_key]
        s = sum(1 for r in model_res if r.get("choice") == "shape")
        t = sum(1 for r in model_res if r.get("choice") == "texture")
        u = sum(1 for r in model_res if r.get("choice") == "unclear")
        print(
            f"  {model_key:25s}  {s:>6d}  {t:>8d}  {u:>8d}  "
            f"{_shape_pct_decisive(s, t):>12s}  {_shape_pct_all(s, t, u):>12s}"
        )

    # Breakdown by ordering
    print(
        f"\n  {'Model':25s}  {'Ordering':15s}  {'Shape':>6s}  {'Texture':>8s}  "
        f"{'Shape%(dec)':>12s}  {'Shape%(all)':>12s}"
    )
    print(f"  {'-'*25}  {'-'*15}  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}")
    for model_key in model_names:
        for ordering in ["shape_first", "texture_first"]:
            subset = [
                r for r in normalized_results
                if r.get("model") == model_key and r.get("ordering") == ordering
            ]
            s = sum(1 for r in subset if r.get("choice") == "shape")
            t = sum(1 for r in subset if r.get("choice") == "texture")
            u = sum(1 for r in subset if r.get("choice") == "unclear")
            print(
                f"  {model_key:25s}  {ordering:15s}  {s:>6d}  {t:>8d}  "
                f"{_shape_pct_decisive(s, t):>12s}  {_shape_pct_all(s, t, u):>12s}"
            )


def resolve_output_path(
    output_arg: str | None,
    prefix: str = "eval",
    *,
    default_subdir: Path | str | None = None,
) -> Path:
    """Resolve output CSV path from CLI arg or generate timestamped default.

    If ``output_arg`` is None, the file is written under ``results_dir`` (from
    ``$RESULTS_DIR`` or ``RESULTS_DIR``). When ``default_subdir`` is set, it is
    appended first (e.g. ``model.results/human_matched`` for organized remote runs).
    """
    if output_arg is not None:
        return Path(output_arg)
    results_dir = Path(os.environ["RESULTS_DIR"]) if os.environ.get("RESULTS_DIR") else RESULTS_DIR
    results_dir.mkdir(exist_ok=True)
    base = results_dir
    if default_subdir is not None:
        base = results_dir / Path(default_subdir)
        base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return base / f"{prefix}_{timestamp}.csv"
