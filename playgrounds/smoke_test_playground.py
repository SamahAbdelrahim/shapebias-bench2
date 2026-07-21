#!/usr/bin/env python3
"""Playground smoke test: generate + logit scoring across multiple trials.

By default runs **both** scoring paths so collaborators can compare:
  - two_pass: ``generate()`` then ``score_choices()`` (July 2026 FarmShare method)
  - one_pass: ``generate(..., choice_texts=labels)`` (Adam's merged path)

Outputs land under ``results/playground.results/session_YYYY-MM-DD_farmshare/``.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from evaluation_pipe.data import load_trials
from evaluation_pipe.eval_core import (
    LOCAL_VLM_SYSTEM_PROMPT,
    PROMPT_TEMPLATES,
    default_session_results_dir,
    make_prompt,
)
from evaluation_pipe.models import create_model, list_models

# Default matches eval_core / notebook. Override with --prompt-condition.
DEFAULT_PROMPT_CONDITION = "no_word_category_AB"

# Models present in local_model_playground.ipynb
NOTEBOOK_MODELS = [
    "smolvlm",
    "internvl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
]


def _parse_choice(text: str, choice_labels: tuple[str, str]) -> str | None:
    alternatives = "|".join(re.escape(label) for label in choice_labels)
    matches = re.findall(rf"\b({alternatives})\b", text or "", re.IGNORECASE)
    if not matches:
        return None
    candidate = matches[-1]
    for label in choice_labels:
        if candidate.casefold() == label.casefold():
            return label
    return None


def _relative_probs(logits: list[float]) -> tuple[float, float]:
    m = max(logits)
    ea, eb = math.exp(logits[0] - m), math.exp(logits[1] - m)
    z = ea + eb
    return ea / z, eb / z


def _fmt_abs(p: float) -> str:
    return f"{p:.1e}"


def _match_label(pick: str | None, ground_truth: str) -> str:
    if pick is None:
        return "UNPARSED"
    if pick == ground_truth:
        return f"MATCH vs ground truth {ground_truth}"
    return f"MISMATCH vs ground truth {ground_truth}"


def _choice_to_feature(pick: str, ground_truth: str) -> str:
    """Map a label pick to shape/texture using the shape-match label."""
    return "shape" if pick == ground_truth else "texture"


def _logit_pick_from(logits, choice_labels: tuple[str, str]) -> str:
    return choice_labels[0] if logits[0] >= logits[1] else choice_labels[1]


def _print_score_block(
    label: str,
    score: dict | None,
    score_error: str | None,
    ground_truth: str,
    gen_pick: str | None,
    choice_labels: tuple[str, str],
) -> str | None:
    print(f"Logit scoring [{label}] ({choice_labels[0]} / {choice_labels[1]})")
    print()
    if score_error:
        print(f"Logit scoring failed: {score_error}")
        return None

    assert score is not None
    logits = score["choice_logits"]
    abs_probs = score["choice_probs_absolute"]
    rel_a, rel_b = _relative_probs([float(logits[0]), float(logits[1])])
    logit_pick = _logit_pick_from(logits, choice_labels)

    print(f"Logits: [{float(logits[0]):.4f}, {float(logits[1]):.4f}]")
    print(
        f"Absolute probs: ~{_fmt_abs(float(abs_probs[0]))} / {_fmt_abs(float(abs_probs[1]))} "
        "(full-vocab softmax — tiny, as expected)"
    )
    print(f"Relative probs: {rel_a:.3f} / {rel_b:.3f}")
    print(f"Logit pick: {logit_pick} ({_match_label(logit_pick, ground_truth)})")

    if gen_pick and gen_pick != logit_pick:
        gen_side = _choice_to_feature(gen_pick, ground_truth)
        logit_side = _choice_to_feature(logit_pick, ground_truth)
        print(
            f"So generation chose {gen_side}, logits preferred {logit_side} on this trial."
        )
    elif gen_pick and gen_pick == logit_pick:
        side = _choice_to_feature(gen_pick, ground_truth)
        print(f"Generation and logits both preferred {side} on this trial.")
    print()
    return logit_pick


def _print_trial_block(
    *,
    model_key: str,
    trial_id,
    order: str,
    ground_truth: str,
    gen_text: str,
    gen_time: float,
    gen_tokens: int,
    two_pass_score: dict | None,
    two_pass_error: str | None,
    one_pass_score: dict | None,
    one_pass_error: str | None,
    paths: list[str],
    choice_labels: tuple[str, str],
) -> dict:
    print(
        f"\nResults for trial {trial_id} ({order}, "
        f"ground truth {ground_truth} = shape match):"
    )
    print(f"Model: {model_key}")
    print()
    print("Generated answer")
    print()
    print(f"Answer: {gen_text!r}")
    print(f"Time: {gen_time:.2f}s · Tokens: {gen_tokens}")
    gen_pick = _parse_choice(gen_text, choice_labels)
    if gen_pick:
        print(f"Generation pick: {gen_pick} ({_match_label(gen_pick, ground_truth)})")
    print()

    picks: dict[str, str | None] = {"gen": gen_pick}
    if "two_pass" in paths:
        picks["two_pass"] = _print_score_block(
            "two_pass",
            two_pass_score,
            two_pass_error,
            ground_truth,
            gen_pick,
            choice_labels,
        )
    if "one_pass" in paths:
        picks["one_pass"] = _print_score_block(
            "one_pass",
            one_pass_score,
            one_pass_error,
            ground_truth,
            gen_pick,
            choice_labels,
        )

    if "two_pass" in paths and "one_pass" in paths:
        tp, op = picks.get("two_pass"), picks.get("one_pass")
        if tp is not None and op is not None:
            if tp == op:
                print(f"Path agreement: two_pass and one_pass both picked {tp}.")
            else:
                print(
                    f"Path disagreement: two_pass={tp}, one_pass={op}."
                )
            print()
    return picks


def _completed_models_in_log(path: Path, n_trials: int) -> set[str]:
    """Models with a full block in an existing log: n_trials results + unload marker."""
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    done: set[str] = set()
    for block in re.split(r"\n={72}\nMODEL: ", text)[1:]:
        name = block.split("\n", 1)[0].strip()
        n_results = len(re.findall(r"^Results for trial ", block, re.M))
        if n_results >= n_trials and f"\nUnloaded {name}" in block and "MODEL LOAD/RUN FAILED" not in block:
            done.add(name)
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=NOTEBOOK_MODELS)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--order", default="shape_first")
    parser.add_argument(
        "--prompt-condition",
        default=DEFAULT_PROMPT_CONDITION,
        choices=sorted(PROMPT_TEMPLATES.keys()),
        help="User prompt key from eval_core.PROMPT_TEMPLATES.",
    )
    parser.add_argument(
        "--word",
        default=None,
        help="Fixed novel word for prompts containing {word}, e.g. shiple.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=("two_pass", "one_pass"),
        default=["two_pass", "one_pass"],
        help="Scoring paths to run (default: both).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output log path (default: results/playground.results/session_*/playground_smoke_...).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models already complete in the existing log; append the rest. "
        "A model is complete when its block has all trials plus the unload marker.",
    )
    args = parser.parse_args()
    prompt_condition = args.prompt_condition
    template = PROMPT_TEMPLATES[prompt_condition]
    if "{word}" in template and not args.word:
        parser.error(f"--word is required for prompt condition {prompt_condition}")
    prompt = make_prompt(args.word, prompt_condition=prompt_condition)
    choice_labels = ("A", "B") if prompt_condition.endswith("_AB") else ("1", "2")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run inside an srun GPU allocation.")
        return 1

    if args.out is None:
        session = default_session_results_dir("playground")
        word_suffix = f"_{args.word}" if args.word else ""
        args.out = (
            session
            / (
                f"playground_smoke_{args.n_trials}trials_{args.order}_"
                f"{prompt_condition}{word_suffix}.txt"
            )
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)

    run_models = list(args.models)
    resume_append = False
    if args.resume:
        done = _completed_models_in_log(args.out, args.n_trials)
        skipped = [m for m in run_models if m in done]
        run_models = [m for m in run_models if m not in done]
        if skipped and not run_models:
            print(f"RESUME: all {len(skipped)} requested models already complete in {args.out}; nothing to do.")
            return 0
        if skipped:
            resume_append = True
            print(f"RESUME: skipping completed models {skipped}; running {run_models}")

    # Tee stdout to file
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    out_f = open(args.out, "a" if resume_append else "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out_f)
    if resume_append:
        print(f"\n===== RESUMED {time.strftime('%Y-%m-%d %H:%M:%S')}: running {run_models} =====")

    device = "cuda"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Available models: {list_models()}")
    print(f"Testing models: {run_models}")
    print(f"Trials: {args.n_trials} (order={args.order})")
    print(f"Paths: {args.paths}")
    print(f"Prompt condition: {prompt_condition}")
    print(f"Word: {args.word or '(none)'}")
    print(f"Choice labels: {choice_labels[0]} / {choice_labels[1]}")
    print(f"User prompt: {prompt}")
    print(f"System prompt (LOCAL_VLM_SYSTEM_PROMPT): {LOCAL_VLM_SYSTEM_PROMPT}")
    print(f"Writing results to: {args.out}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    dataset = Path(os.environ["IMAGE_DATASET"])
    if not dataset.is_absolute():
        dataset = REPO_ROOT / dataset

    trials = load_trials(dataset, order=args.order)
    selected = trials[: args.n_trials]
    if len(selected) < args.n_trials:
        print(f"WARNING: only {len(selected)} trials available")

    # Preload images once
    loaded = []
    for trial in selected:
        ref, a, b = trial.load_images()
        loaded.append((trial, ref, a, b))
        print(f"Loaded trial {trial.trial_id} (ground_truth={trial.ground_truth})")

    failures: list[str] = []
    # model -> list of per-trial pick dicts
    agreement_rows: list[tuple[str, object, dict]] = []

    for name in run_models:
        print("\n" + "=" * 72)
        print(f"MODEL: {name}")
        print("=" * 72)
        model = None
        try:
            model = create_model(name, device=device)
            print(f"Loaded: {model.name} on {next(model._model.parameters()).device}")

            for trial, ref, a, b in loaded:
                images = [ref, a, b]
                try:
                    two_pass_score = None
                    two_pass_error = None
                    one_pass_score = None
                    one_pass_error = None

                    if "one_pass" in args.paths and "two_pass" not in args.paths:
                        # One generate call carries text + logits.
                        gen = model.generate(
                            images=images,
                            prompt=prompt,
                            choice_texts=choice_labels,
                        )
                        if gen.choice_logits is None or gen.choice_probs is None:
                            one_pass_error = "generate returned no choice_logits/choice_probs"
                        else:
                            one_pass_score = {
                                "choice_logits": list(gen.choice_logits),
                                "choice_probs_absolute": list(gen.choice_probs),
                            }
                    else:
                        # Always get a free-text generation (matches July 10 smoke).
                        gen = model.generate(images=images, prompt=prompt)
                        if "two_pass" in args.paths:
                            try:
                                two_pass_score = model.score_choices(
                                    images=images,
                                    prompt=prompt,
                                    choice_texts=choice_labels,
                                )
                            except Exception as exc:
                                two_pass_error = f"{type(exc).__name__}: {exc}"
                        if "one_pass" in args.paths:
                            try:
                                one_gen = model.generate(
                                    images=images,
                                    prompt=prompt,
                                    choice_texts=choice_labels,
                                )
                                if one_gen.choice_logits is None or one_gen.choice_probs is None:
                                    one_pass_error = (
                                        "generate returned no choice_logits/choice_probs"
                                    )
                                else:
                                    one_pass_score = {
                                        "choice_logits": list(one_gen.choice_logits),
                                        "choice_probs_absolute": list(one_gen.choice_probs),
                                    }
                            except Exception as exc:
                                one_pass_error = f"{type(exc).__name__}: {exc}"

                    picks = _print_trial_block(
                        model_key=name,
                        trial_id=trial.trial_id,
                        order=trial.order,
                        ground_truth=(
                            choice_labels[0]
                            if trial.ground_truth == "A"
                            else choice_labels[1]
                        ),
                        gen_text=gen.raw_text,
                        gen_time=gen.generation_time_s,
                        gen_tokens=gen.num_tokens_generated,
                        two_pass_score=two_pass_score,
                        two_pass_error=two_pass_error,
                        one_pass_score=one_pass_score,
                        one_pass_error=one_pass_error,
                        paths=list(args.paths),
                        choice_labels=choice_labels,
                    )
                    agreement_rows.append((name, trial.trial_id, picks))
                    if two_pass_error:
                        failures.append(f"{name}/trial{trial.trial_id}/two_pass")
                    if one_pass_error:
                        failures.append(f"{name}/trial{trial.trial_id}/one_pass")
                except Exception as exc:
                    failures.append(f"{name}/trial{trial.trial_id}/generate")
                    print(
                        f"\nResults for trial {trial.trial_id}: GENERATE FAILED — "
                        f"{type(exc).__name__}: {exc}"
                    )
        except Exception as exc:
            failures.append(name)
            print(f"MODEL LOAD/RUN FAILED: {type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                try:
                    model.unload()
                except Exception:
                    pass
                del model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"\nUnloaded {name}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if "two_pass" in args.paths and "one_pass" in args.paths:
        n_agree = sum(
            1
            for _, _, p in agreement_rows
            if p.get("two_pass") is not None
            and p.get("one_pass") is not None
            and p["two_pass"] == p["one_pass"]
        )
        n_both = sum(
            1
            for _, _, p in agreement_rows
            if p.get("two_pass") is not None and p.get("one_pass") is not None
        )
        n_gen_two = sum(
            1
            for _, _, p in agreement_rows
            if p.get("gen") is not None
            and p.get("two_pass") is not None
            and p["gen"] == p["two_pass"]
        )
        n_gen_one = sum(
            1
            for _, _, p in agreement_rows
            if p.get("gen") is not None
            and p.get("one_pass") is not None
            and p["gen"] == p["one_pass"]
        )
        print(f"two_pass vs one_pass agreement: {n_agree}/{n_both}")
        print(f"gen vs two_pass agreement: {n_gen_two}/{n_both}")
        print(f"gen vs one_pass agreement: {n_gen_one}/{n_both}")

        # Per-model gen vs two_pass (fair compare to July 10 baseline)
        by_model: dict[str, list[dict]] = {}
        for m, _, p in agreement_rows:
            by_model.setdefault(m, []).append(p)
        print("\nPer-model gen vs two_pass (July-10-comparable):")
        for m, rows in by_model.items():
            n = len(rows)
            agree = sum(
                1
                for p in rows
                if p.get("gen") is not None
                and p.get("two_pass") is not None
                and p["gen"] == p["two_pass"]
            )
            print(f"  {m}: {agree}/{n} gen==two_pass")

    if failures:
        print(f"Failures ({len(failures)}): {failures}")
        rc = 1
    else:
        print(
            f"All {len(run_models)} models × {len(loaded)} trials completed "
            f"(paths={args.paths})."
        )
        rc = 0
    print(f"Full log: {args.out}")

    sys.stdout = sys.__stdout__
    out_f.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
