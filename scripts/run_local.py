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

    # Use levante_bench runtime wrapper (model selected by id)
    python scripts/run_local.py --models levante-runtime --ordering shape_first --levante-model-name qwen35

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
    load_completed_trial_keys,
    make_prompt,
    print_summary,
    resolve_output_path,
    resolve_stim_set_name,
    run_trial,
    run_trial_binary_pair,
    run_trial_rank_forced,
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


def run_local_logit_forced(model, images: list[Image.Image], prompt: str) -> dict:
    if not hasattr(model, "score_choices"):
        raise RuntimeError(
            f"Model {getattr(model, 'name', '<unknown>')} does not support logit scoring."
        )
    return model.score_choices(images=images, prompt=prompt, choice_texts=("1", "2"))


def run_trial_logit_forced_12(
    run_score_fn,
    stimulus: dict,
    word: str,
    word_type: str,
    word_length: int = 0,
    *,
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
        p1, p2 = base["choice_probs"]
        l1, l2 = base["choice_logits"]
        total_t = float(base["generation_time_s"])

        if swap_correct:
            sw = run_score_fn([ref, img_b, img_a], prompt)
            sp1, sp2 = sw["choice_probs"]  # 1->b, 2->a (relative to base semantics)
            sl1, sl2 = sw["choice_logits"]
            p_a = 0.5 * (p1 + sp2)
            p_b = 0.5 * (p2 + sp1)
            total_t += float(sw["generation_time_s"])
            raw_text = (
                f"base[p1={p1:.4f},p2={p2:.4f},l1={l1:.3f},l2={l2:.3f}] "
                f"swap[p1={sp1:.4f},p2={sp2:.4f},l1={sl1:.3f},l2={sl2:.3f}] "
                f"corr[p_a={p_a:.4f},p_b={p_b:.4f}]"
            )
        else:
            p_a, p_b = p1, p2
            raw_text = f"p1={p1:.4f},p2={p2:.4f},l1={l1:.3f},l2={l2:.3f}"

        if p_a > p_b:
            parsed = "1"
            choice = a_is
        elif p_b > p_a:
            parsed = "2"
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
            }
        )
    return results


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
    parser.add_argument("--prompt-condition", default="noun_label",
                        choices=[
                            "noun_label",
                            "no_word_category",
                            "binary_yes_no",
                            "binary_yes_no_conservative",
                            "binary_score",
                            "binary_score_0_3",
                            "rank_forced",
                        ],
                        help="Prompt variant to use (default: noun_label)")
    parser.add_argument("--decision-mode", default="2afc",
                        choices=["2afc", "binary_pair", "binary_pair_conservative", "binary_rank_forced", "logit_forced_12"],
                        help="Decision policy: standard 2AFC, binary_pair, binary_pair_conservative, binary_rank_forced, or logit_forced_12.")
    parser.add_argument("--swap-correct", action="store_true",
                        help="For logit_forced_12, average with swapped candidate order to reduce position bias.")
    parser.add_argument("--resume", default=None, metavar="CSV",
                        help="Resume from a partial CSV — skip already-completed trials and append new results")
    parser.add_argument(
        "--levante-model-name",
        default="qwen35",
        help=(
            "Model ID passed to levante_bench.runtime.load_model when using "
            "--models levante-runtime (default: qwen35)."
        ),
    )
    parser.add_argument(
        "--levante-model-config-path",
        default=None,
        help=(
            "Optional model config YAML path passed to levante runtime when using "
            "--models levante-runtime."
        ),
    )
    parser.add_argument(
        "--levante-configs-root",
        default=None,
        help=(
            "Optional configs root containing models/*.yaml for levante runtime "
            "lookups."
        ),
    )
    add_common_args(parser)
    args = parser.parse_args()

    if args.decision_mode == "binary_pair" and args.prompt_condition not in {
        "binary_yes_no",
        "binary_score",
    }:
        print("Info: --decision-mode binary_pair requires a binary prompt; using --prompt-condition binary_score.")
        args.prompt_condition = "binary_score"
    if args.decision_mode == "binary_pair_conservative":
        if args.prompt_condition != "binary_yes_no_conservative":
            print(
                "Info: --decision-mode binary_pair_conservative uses "
                "--prompt-condition binary_yes_no_conservative."
            )
        args.prompt_condition = "binary_yes_no_conservative"
    if args.decision_mode == "binary_rank_forced":
        if args.prompt_condition != "rank_forced":
            print(
                "Info: --decision-mode binary_rank_forced uses "
                "--prompt-condition rank_forced."
            )
        args.prompt_condition = "rank_forced"
    if args.decision_mode == "logit_forced_12" and args.prompt_condition not in {"noun_label", "no_word_category"}:
        print(
            "Info: --decision-mode logit_forced_12 expects a 2AFC prompt; "
            "using --prompt-condition noun_label."
        )
        args.prompt_condition = "noun_label"

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
    print(f"Prompt cond: {args.prompt_condition}")
    print(f"Decision:    {args.decision_mode}")
    if args.decision_mode == "logit_forced_12":
        print(f"Swap corr:   {args.swap_correct}")
    print(f"Stimuli:     {len(stimuli)} from {BENCHMARK_STIM_PACKAGE}/{stim_set_label}")
    print(f"Words:       {len(words)} ({len(words)//2} sudo + {len(words)//2} random)")
    ord_mult = 1 if args.decision_mode in {"binary_pair", "binary_pair_conservative"} else (2 if args.ordering == "both" else 1)
    trials_per = len(stimuli) * len(words) * args.repeats * ord_mult
    print(f"Trials per model: {len(stimuli)} x {len(words)} x {args.repeats} repeats x {ord_mult} orderings = {trials_per}")
    print()

    done_keys: set[tuple[str, str, str, str, str]] = set()
    if args.resume:
        output_path = Path(args.resume)
        done_keys = load_completed_trial_keys(output_path)
        print(f"Resume file: {output_path}")
        print(f"Completed trial rows detected: {len(done_keys)}")
    else:
        output_path = resolve_output_path(args.output, prefix="local")
    all_results = []

    for model_key in model_names:
        print(f"{'='*60}")
        print(f"Local model: {model_key}")
        print(f"{'='*60}")

        create_kwargs = {"device": args.device}
        if model_key == "levante-runtime":
            create_kwargs.update(
                {
                    "model_id": args.levante_model_name,
                    "model_config_path": args.levante_model_config_path,
                    "configs_root": args.levante_configs_root,
                }
            )
        model = create_model(model_key, **create_kwargs)
        print(f"  Loaded: {model.name}")

        def run_fn(images, prompt, _m=model):
            return run_local(_m, images, prompt, temperature=args.temperature)

        for repeat in range(1, args.repeats + 1):
            if args.repeats > 1:
                print(f"\n  --- Repeat {repeat}/{args.repeats} ---")
            for stim in stimuli:
                for w in words:
                    word, word_type, word_length = w["name"], w["type"], w["length"]

                    if args.decision_mode in {"binary_pair", "binary_pair_conservative"}:
                        expected_orderings = ["binary_pair"]
                    elif args.ordering == "both":
                        expected_orderings = ["shape_first", "texture_first"]
                    elif args.ordering == "random":
                        expected_orderings = ["shape_first", "texture_first"]
                    else:
                        expected_orderings = [args.ordering]
                    trial_key_prefix = (model_key, stim["stim_id"], word, str(repeat))
                    all_done = all(
                        (trial_key_prefix[0], trial_key_prefix[1], trial_key_prefix[2], ord_name, trial_key_prefix[3])
                        in done_keys
                        for ord_name in expected_orderings
                    )
                    if all_done:
                        continue

                    print(f"  Stimulus {stim['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
                    if args.decision_mode in {"binary_pair", "binary_pair_conservative"}:
                        trial_results = run_trial_binary_pair(
                            run_fn,
                            stim,
                            word,
                            word_type,
                            word_length,
                            prompt_condition=args.prompt_condition,
                        )
                    elif args.decision_mode == "binary_rank_forced":
                        trial_results = run_trial_rank_forced(
                            run_fn,
                            stim,
                            word,
                            word_type,
                            word_length,
                            ordering=args.ordering,
                            prompt_condition=args.prompt_condition,
                        )
                    elif args.decision_mode == "logit_forced_12":
                        trial_results = run_trial_logit_forced_12(
                            lambda images, p, _m=model: run_local_logit_forced(_m, images, p),
                            stim,
                            word,
                            word_type,
                            word_length,
                            ordering=args.ordering,
                            prompt_condition=args.prompt_condition,
                            swap_correct=args.swap_correct,
                        )
                    else:
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
                        r["decision_mode"] = args.decision_mode
                        r["swap_correct"] = "true" if args.swap_correct else "false"
                        r.update(csv_meta)
                        print(f"    {r['ordering']:15s} -> {r['raw_text']!r:10s}  choice={r['choice']}")
                        all_results.append(r)
                    # Save incrementally after each stimulus+word trial
                    write_results(trial_results, output_path, append=True, quiet=True)
                    for r in trial_results:
                        done_keys.add(
                            (
                                model_key,
                                str(r.get("stim_id", "")),
                                str(r.get("word", "")),
                                str(r.get("ordering", "")),
                                str(repeat),
                            )
                        )

        model.unload()
        print(f"  Unloaded {model_key}")

    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
