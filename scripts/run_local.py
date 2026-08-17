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

    # Logit-forced scoring with absolute next-token probabilities
    python scripts/run_local.py --models smolvlm --ordering shape_first \
        --decision-mode logit_forced --choice-texts 1 2

    # Append results to existing CSV
    python scripts/run_local.py --models smolvlm --ordering shape_first -o results/model.results/run.csv
    python scripts/run_local.py --models smolvlm --ordering texture_first -o results/model.results/run.csv
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

from playgrounds.embedding_robust import (
    build_geirhos_unaltered_triplets,
    build_smith_probe_triplets
)

from evaluation_pipe.eval_core import (
    BENCHMARK_STIM_PACKAGE,
    DEFAULT_DEVICE,
    ENV_PATH,
    MAX_TOKENS_LOCAL,
    add_common_args,
    benchmark_csv_meta,
    load_stimuli,
    load_stimuli_grid,
    load_stimuli_triad_dir,
    load_words,
    load_completed_trial_keys,
    make_prompt,
    materialize_stimulus,
    print_summary,
    resolve_output_path,
    resolve_stim_set_name,
    run_trial,
    run_trial_binary_pair,
    run_trial_rank_forced,
    run_trial_logit_scoring,
    write_results,
)

load_dotenv(ENV_PATH)

_LOGIT_PROMPT_CONDITIONS = {
    "noun_label",
    "noun_label_AB",
    "no_word_category",
    "no_word_category_AB",
    "no_word_category_similar",
    "no_word_category_similar_AB",
    "no_word_similarity",
    "no_word_similarity_AB",
}


# ---------------------------------------------------------------------------
# Local inference
# ---------------------------------------------------------------------------
def run_local(model, images: list[Image.Image], prompt: str,
              temperature: float = 0.0, choice_texts: tuple[str, str] | None = None) -> dict:
    from evaluation_pipe.models.base import ModelResponse
    resp: ModelResponse = model.generate(
        images=images, prompt=prompt,
        max_new_tokens=MAX_TOKENS_LOCAL, temperature=temperature,
        choice_texts=choice_texts
    )
    return {
        "raw_text": resp.raw_text,
        "generation_time_s": round(resp.generation_time_s, 2),
        "model_name": resp.model_name,
        "num_tokens_generated": resp.num_tokens_generated,
        "choice_logits": resp.choice_logits,
        "choice_probs": resp.choice_probs,
    }


def run_local_logit_forced(
    model, images: list[Image.Image], prompt: str, choice_texts: tuple[str, str]
) -> dict:
    if not hasattr(model, "score_choices"):
        raise RuntimeError(
            f"Model {getattr(model, 'name', '<unknown>')} does not support logit scoring."
        )
    return model.score_choices(images=images, prompt=prompt, choice_texts=choice_texts)

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
                            "noun_label_AB",
                            "no_word_category",
                            "no_word_category_AB",
                            "no_word_category_similar",
                            "no_word_category_similar_AB",
                            "no_word_similarity",
                            "no_word_similarity_AB",
                            "binary_yes_no",
                            "binary_yes_no_conservative",
                            "binary_score",
                            "binary_score_0_3",
                            "rank_forced",
                        ],
                        help="Prompt variant to use (default: noun_label)")
    parser.add_argument("--decision-mode", default="2afc",
                        choices=["2afc", "binary_pair", "binary_pair_conservative", "binary_rank_forced", "logit_forced"],
                        help="Decision policy: standard 2AFC, binary_pair, binary_pair_conservative, binary_rank_forced, or logit_forced.")
    parser.add_argument("--choice-texts", nargs=2, default=None,
                        help="Two choice strings for 2AFC / logit scoring and matching image-slot labels (e.g. A B or 1 2). "
                             "Default: A B when --prompt-condition ends with _AB, else 1 2.")
    parser.add_argument("--swap-correct", action="store_true",
                        help="For logit_forced, average with swapped candidate order to reduce position bias.")
    parser.add_argument("--resume", default=None, metavar="CSV",
                        help="Resume from a partial CSV — skip already-completed trials and append new results")
    parser.add_argument("--word", default=None,
                        help="Restrict word-bearing prompts to one curated word (e.g. shiple). "
                             "Default: all 10 words. Ignored by no-word prompt conditions.")
    parser.add_argument("--grid-pkg", default=None,
                        help="Full texture-grid package under stimuli_pipe/ (e.g. stimuli_texture_grid_v1). "
                             "Trials are read from its manifest.csv and images opened one at a time.")
    parser.add_argument("--triad-dir", default=None,
                        help="Path to a flat directory of named triad folders, each holding "
                             "reference.png / shape_match.png / texture_match.png "
                             "(e.g. the cue-conflict cc_triads and decomposition_triads sets). "
                             "Folder names of the form <shape_id>-<texture_id> populate the "
                             "stl_id and texture_set columns.")
    parser.add_argument("--smith-probe", default=None, help="Path to Linda Smith probe-shapematch-colormatch dataset")
    parser.add_argument("--geirhos-unaltered", default=None, help="Path to Geirhos unaltered dataset (cue_conflict/original/texture).")
    parser.add_argument("--image-mode", default="normal",
                        choices=["normal", "blank", "phase_scramble"],
                        help="Blind baseline: replace every stimulus image with "
                             "a mid-grey blank or an amplitude-preserving phase "
                             "scramble. Grid packages only (--grid-pkg).")
    parser.add_argument("--intervene-bank", default=None,
                        help="Eraser bank NPZ from playgrounds/intervene.py; "
                             "attaches a projector-output hook that erases or "
                             "amplifies the fitted concept subspace.")
    parser.add_argument("--intervene-arm", default="target",
                        help="'target' (the bank's fitted concept, fold-aware) "
                             "or 'random<seed>' for the rank-matched control.")
    parser.add_argument("--intervene-alpha", type=float, default=0.0,
                        help="0.0 erases the subspace; values > 1 amplify it.")
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
    if args.decision_mode == "logit_forced" and args.prompt_condition not in _LOGIT_PROMPT_CONDITIONS:
        print(
            "Info: --decision-mode logit_forced expects a 2AFC prompt; "
            "using --prompt-condition noun_label."
        )
        args.prompt_condition = "noun_label"
    if "--choice-texts" in sys.argv and args.decision_mode not in {"logit_forced", "2afc"}:
        print("Info: --choice-texts has no effect unless using --decision-mode 2afc or logit_forced.")

    if args.choice_texts is None:
        args.choice_texts = (
            ["A", "B"] if args.prompt_condition.endswith("_AB") else ["1", "2"]
        )

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

    if args.prompt_condition in {
        "no_word_category",
        "no_word_category_AB",
        "no_word_similarity",
        "no_word_similarity_AB",
    }:
        words = [{
            "name": "",
            "type": "none",
            "length": 0,
        }]
    elif args.word:
        words = [w for w in words if w["name"] == args.word]
        if not words:
            available = [w["name"] for w in load_words()]
            print(f"Error: unknown word '{args.word}'. Available: {available}")
            sys.exit(1)

    if args.geirhos_unaltered:
        geirhos_triplets = build_geirhos_unaltered_triplets(
            Path(args.geirhos_unaltered),
            args.num_stimuli,
            args.seed,
        )

        stimuli = []
        for sid, ref, shape, texture in geirhos_triplets:
            stimuli.append(
                {
                    "stim_id": sid,
                    "reference": ref,
                    "shape_match": shape,
                    "texture_match": texture,
                }
            )

        stim_set_label = "geirhos_unaltered"
        csv_meta = benchmark_csv_meta(stim_set_label)

        print(f"Using Geirhos unaltered dataset: {len(stimuli)} stimuli")
    elif args.smith_probe:
        smith_triplets = build_smith_probe_triplets(
            Path(args.smith_probe),
            args.num_stimuli,
            args.seed,
        )

        stimuli = []
        for sid, ref, shape, texture in smith_triplets:
            stimuli.append(
                {
                    "stim_id": sid,
                    "reference": ref,
                    "shape_match": shape,
                    "texture_match": texture,
                }
            )

        stim_set_label = "smith_probe"
        csv_meta = benchmark_csv_meta(stim_set_label)

        print(f"Using Smith probe dataset: {len(stimuli)} stimuli")
    elif args.triad_dir:
        stimuli = load_stimuli_triad_dir(args.triad_dir, args.num_stimuli)
        stim_set_label = Path(args.triad_dir).name
        csv_meta = benchmark_csv_meta(stim_set_label, stim_pkg="triad_dir")

        n_shapes = len({s["stl_id"] for s in stimuli})
        n_textures = len({s["texture_set"] for s in stimuli})
        print(f"Using triad directory {args.triad_dir}: "
              f"{len(stimuli)} triads ({n_shapes} shape ids x {n_textures} texture ids)")
    elif args.grid_pkg:
        stimuli = load_stimuli_grid(args.grid_pkg, args.stim_set, args.num_stimuli)
        stim_set_label = resolve_stim_set_name(args.stim_set)
        csv_meta = benchmark_csv_meta(stim_set_label, stim_pkg=args.grid_pkg)

        n_shapes = len({s["stl_id"] for s in stimuli})
        n_textures = len({s["texture_set"] for s in stimuli})
        print(f"Using texture grid {args.grid_pkg}/{stim_set_label}: "
              f"{len(stimuli)} stimuli ({n_shapes} shapes x {n_textures} textures)")
    else:
        stimuli = load_stimuli(args.stim_set, args.num_stimuli)
        stim_set_label = resolve_stim_set_name(args.stim_set)
        csv_meta = benchmark_csv_meta(stim_set_label)

    # Both loaders return path records with stl_id / texture_set, so the triad
    # sets ride the same lazy-open and CSV-column path as the texture grid.
    grid_mode = bool(args.grid_pkg or args.triad_dir)
    if args.image_mode != "normal" and not grid_mode:
        parser.error("--image-mode blank/phase_scramble requires --grid-pkg "
                     "or --triad-dir (path-based stimuli)")
    print(f"Models:      {model_names}")
    print(f"Device:      {args.device}")
    print(f"Ordering:    {args.ordering}")
    print(f"Repeats:     {args.repeats}")
    print(f"Temperature: {args.temperature}")
    print(f"Prompt cond: {args.prompt_condition}")
    print(f"Decision:    {args.decision_mode}")
    if args.decision_mode in {"logit_forced", "2afc"}:
        print(f"Choice txt:  {args.choice_texts[0]!r} / {args.choice_texts[1]!r}")
    if args.decision_mode == "logit_forced":
        print(f"Swap corr:   {args.swap_correct}")
    if args.smith_probe:
        print(f"Stimuli:     {len(stimuli)} from {args.smith_probe}")
    elif args.triad_dir:
        print(f"Stimuli:     {len(stimuli)} from {args.triad_dir}")
    elif grid_mode:
        print(f"Stimuli:     {len(stimuli)} from {args.grid_pkg}/{stim_set_label}")
    else:
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
        output_path = resolve_output_path(
            args.output, prefix="local", default_subdir="model.results"
        )
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

        itv = None
        if args.intervene_bank:
            from playgrounds.intervene import Intervention, attach

            itv = Intervention(Path(args.intervene_bank),
                               arm=args.intervene_arm,
                               alpha=args.intervene_alpha)
            attach(model._model, itv)

        def run_fn(images, prompt, _m=model, choice_texts=None):
            return run_local(_m, images, prompt, temperature=args.temperature, choice_texts=choice_texts)

        for repeat in range(1, args.repeats + 1):
            if args.repeats > 1:
                print(f"\n  --- Repeat {repeat}/{args.repeats} ---")
            for stim_rec in stimuli:
                # Grid records hold paths; open the triad once, and only if some
                # word still needs running (resume skips must not pay the decode).
                stim = None if grid_mode else stim_rec
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
                    trial_key_prefix = (model_key, stim_rec["stim_id"], word, str(repeat))
                    all_done = all(
                        (trial_key_prefix[0], trial_key_prefix[1], trial_key_prefix[2], ord_name, trial_key_prefix[3])
                        in done_keys
                        for ord_name in expected_orderings
                    )
                    if all_done:
                        continue

                    if stim is None:
                        stim = materialize_stimulus(stim_rec,
                                                    image_mode=args.image_mode)
                    if itv is not None:
                        itv.set_stimulus(stl_id=stim_rec.get("stl_id"),
                                         texture_set=stim_rec.get("texture_set"))

                    print(f"  Stimulus {stim_rec['stim_id']:>3s} (word={word}, type={word_type}, len={word_length})")
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
                    elif args.decision_mode == "logit_forced":
                        trial_results = run_trial_logit_scoring(
                            lambda images, p, _m=model, ct=args.choice_texts: run_local_logit_forced(_m, images, p, ct),
                            stim,
                            word,
                            word_type,
                            word_length,
                            choice_texts=tuple(args.choice_texts),
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
                            choice_texts=tuple(args.choice_texts),
                        )
                    for r in trial_results:
                        r["model"] = model_key
                        r["repeat"] = repeat
                        r["temperature"] = args.temperature
                        r["decision_mode"] = args.decision_mode
                        r["swap_correct"] = "true" if args.swap_correct else "false"
                        r["image_mode"] = args.image_mode
                        r.update(csv_meta)
                        if grid_mode:
                            r["stl_id"] = stim_rec["stl_id"]
                            r["texture_set"] = stim_rec["texture_set"]

                        logit_info = ""

                        if r.get("choice_logits") is not None:
                            logit_info = (
                                f"\n      logits={r['choice_logits']}"
                                f"\n      probs={r['choice_probs']}"
                            )
                        elif r.get("shape_choice_logits") is not None:
                            logit_info = (
                                f"\n      shape YES/NO:"
                                f"\n        logits={r['shape_choice_logits']}"
                                f"\n        probs={r['shape_choice_probs']}"
                                f"\n      texture YES/NO:"
                                f"\n        logits={r['texture_choice_logits']}"
                                f"\n        probs={r['texture_choice_probs']}"
                            )

                        print(
                            f"    {r['ordering']:15s} -> {r['raw_text']!r:10s} "
                            f"choice={r['choice']}"
                            f"{logit_info}"
                        )

                    # Save incrementally after each stimulus+word trial
                    write_results(trial_results, output_path, append=True, quiet=True)
                    all_results.extend(trial_results)
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

        if itv is not None:
            print(f"  Intervention hook fired {itv.n_calls} times")
            if itv.n_calls == 0:
                print("  ERROR: intervention hook never fired; results invalid")
                sys.exit(1)
        model.unload()
        print(f"  Unloaded {model_key}")

    print_summary(all_results, model_names)


if __name__ == "__main__":
    main()
