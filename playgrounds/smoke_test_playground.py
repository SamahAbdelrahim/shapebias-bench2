#!/usr/bin/env python3
"""Playground smoke test: generate + logit scoring across multiple trials."""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from evaluation_pipe.data import load_trials
from evaluation_pipe.models import create_model, list_models

PROMPT = (
    "You are given three images. The first image is the reference. "
    "Which of the other two images (A or B) is more similar to the reference? "
    "Answer with just 'A' or 'B'."
)

# Models present in local_model_playground.ipynb
NOTEBOOK_MODELS = [
    "smolvlm",
    "internvl",
    "qwen3-vl-2b",
    "qwen3-vl-4b",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
]


def _parse_ab(text: str) -> str | None:
    matches = re.findall(r"\b([ABab])\b", text or "")
    if not matches:
        return None
    return matches[-1].upper()


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


def _ab_to_feature(pick: str, ground_truth: str) -> str:
    """Map A/B pick to shape/texture using GT position of shape_match."""
    return "shape" if pick == ground_truth else "texture"


def _print_trial_block(
    *,
    model_key: str,
    trial_id,
    order: str,
    ground_truth: str,
    gen_text: str,
    gen_time: float,
    gen_tokens: int,
    score: dict | None,
    score_error: str | None,
) -> None:
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
    gen_pick = _parse_ab(gen_text)
    if gen_pick:
        print(f"Generation pick: {gen_pick} ({_match_label(gen_pick, ground_truth)})")
    print()
    print("Logit scoring (A / B)")
    print()
    if score_error:
        print(f"Logit scoring failed: {score_error}")
        return

    assert score is not None
    logits = score["choice_logits"]
    abs_probs = score["choice_probs_absolute"]
    rel_a, rel_b = _relative_probs(logits)
    logit_pick = "A" if logits[0] >= logits[1] else "B"

    print(f"Logits: [{logits[0]:.4f}, {logits[1]:.4f}]")
    print(
        f"Absolute probs: ~{_fmt_abs(abs_probs[0])} / {_fmt_abs(abs_probs[1])} "
        "(full-vocab softmax — tiny, as expected)"
    )
    print(f"Relative probs: {rel_a:.3f} / {rel_b:.3f}")
    print(f"Logit pick: {logit_pick} ({_match_label(logit_pick, ground_truth)})")

    if gen_pick and gen_pick != logit_pick:
        gen_side = _ab_to_feature(gen_pick, ground_truth)
        logit_side = _ab_to_feature(logit_pick, ground_truth)
        print(
            f"So generation chose {gen_side}, logits preferred {logit_side} on this trial."
        )
    elif gen_pick and gen_pick == logit_pick:
        side = _ab_to_feature(gen_pick, ground_truth)
        print(f"Generation and logits both preferred {side} on this trial.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=NOTEBOOK_MODELS)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--order", default="shape_first")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "playground_smoke_5trials.txt",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run inside an srun GPU allocation.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

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

    out_f = open(args.out, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, out_f)

    device = "cuda"
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Available models: {list_models()}")
    print(f"Testing models: {args.models}")
    print(f"Trials: {args.n_trials} (order={args.order})")
    print(f"Writing results to: {args.out}")

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

    for name in args.models:
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
                    gen = model.generate(images=images, prompt=PROMPT)
                    score = None
                    score_error = None
                    try:
                        score = model.score_choices(
                            images=images, prompt=PROMPT, choice_texts=("A", "B")
                        )
                    except Exception as exc:
                        score_error = f"{type(exc).__name__}: {exc}"

                    _print_trial_block(
                        model_key=name,
                        trial_id=trial.trial_id,
                        order=trial.order,
                        ground_truth=trial.ground_truth,
                        gen_text=gen.raw_text,
                        gen_time=gen.generation_time_s,
                        gen_tokens=gen.num_tokens_generated,
                        score=score,
                        score_error=score_error,
                    )
                    if score_error:
                        failures.append(f"{name}/trial{trial.trial_id}/logits")
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
    if failures:
        print(f"Failures ({len(failures)}): {failures}")
        rc = 1
    else:
        print(
            f"All {len(args.models)} models × {len(loaded)} trials completed "
            "(generate + logit scoring)."
        )
        rc = 0
    print(f"Full log: {args.out}")

    sys.stdout = sys.__stdout__
    out_f.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
