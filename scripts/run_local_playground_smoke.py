#!/usr/bin/env python3
"""Run local_model_playground SmolVLM cells and print model outputs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from evaluation_pipe.data import load_trials
from evaluation_pipe.models import create_model, list_models

IMAGE_DATASET = REPO_ROOT / os.environ["IMAGE_DATASET"]

if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

PROMPT = (
    "You are given three images. The first image is the reference. "
    "Which of the other two images (A or B) is more similar to the reference? "
    "Answer with just 'A' or 'B'."
)

MODEL_KEY = os.environ.get("PLAYGROUND_MODEL", "smolvlm")
NUM_TRIALS = int(os.environ.get("PLAYGROUND_NUM_TRIALS", "3"))


def _relative_probs(abs_probs):
    if abs_probs is None:
        return None
    total = float(abs_probs[0]) + float(abs_probs[1])
    if total <= 0:
        return [0.5, 0.5]
    return [float(abs_probs[0]) / total, float(abs_probs[1]) / total]


def main() -> None:
    print(f"torch={torch.__version__} device={DEVICE} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"IMAGE_DATASET={IMAGE_DATASET}")
    print(f"Available models: {list_models()}")
    print(f"Prompt: {PROMPT}")

    trials = load_trials(IMAGE_DATASET, order="shape_first")
    selected = trials[:NUM_TRIALS]
    print(
        f"Running {len(selected)} trials "
        f"(A=shape_match, B=texture_match under shape_first)"
    )
    print("=" * 60)

    print(f"Loading model: {MODEL_KEY}")
    model = create_model(MODEL_KEY, device=DEVICE)
    print(f"Loaded: {model.name}")
    try:
        param_device = next(model._model.parameters()).device
        print(f"Parameter device: {param_device}")
    except Exception as exc:  # noqa: BLE001
        print(f"(could not read param device: {exc})")

    summary = []
    for trial in selected:
        reference, image_a, image_b = trial.load_images()
        print("=" * 60)
        print(
            f"TRIAL {trial.trial_id} | ground_truth={trial.ground_truth}"
        )

        print("-" * 60)
        print("GENERATED ANSWER")
        response = model.generate(
            images=[reference, image_a, image_b],
            prompt=PROMPT,
        )
        gen_ans = (response.raw_text or "").strip().upper()[:1]
        print(f"Answer: {response.raw_text!r}")
        print(f"Time:   {response.generation_time_s:.2f}s")
        print(f"Tokens: {response.num_tokens_generated}")
        gen_ok = gen_ans == trial.ground_truth
        print(
            f"Generate pick: {gen_ans or '?'} "
            f"({'MATCH' if gen_ok else 'MISMATCH'})"
        )

        print("-" * 60)
        print("LOGIT SCORING (choice_texts=A/B)")
        scored = model.score_choices(
            images=[reference, image_a, image_b],
            prompt=PROMPT,
            choice_texts=("A", "B"),
        )
        abs_probs = scored.get("choice_probs_absolute")
        rel_probs = scored.get("choice_probs_relative") or _relative_probs(abs_probs)

        print(f"Choices:      {scored.get('choice_texts')}")
        print(f"Logits:       {scored.get('choice_logits')}")
        print(f"Absolute Probabilities: {abs_probs}")
        print(f"Relative Probabilities: {rel_probs}")
        print(f"Time:         {scored.get('generation_time_s', 0):.2f}s")

        logit_pick = None
        logit_ok = None
        if abs_probs is not None:
            logit_pick = "A" if abs_probs[0] >= abs_probs[1] else "B"
            logit_ok = logit_pick == trial.ground_truth
            print(
                f"Logit pick: {logit_pick} "
                f"({'MATCH' if logit_ok else 'MISMATCH'})"
            )

        summary.append(
            {
                "trial": trial.trial_id,
                "gt": trial.ground_truth,
                "gen": gen_ans or "?",
                "gen_ok": gen_ok,
                "logit": logit_pick or "?",
                "logit_ok": logit_ok,
                "rel": rel_probs,
            }
        )

    model.unload()
    print("=" * 60)
    print("SUMMARY")
    for row in summary:
        rel = row["rel"]
        rel_s = (
            f"relA={rel[0]:.3f} relB={rel[1]:.3f}" if rel is not None else "rel=?"
        )
        print(
            f"  trial {row['trial']}: gt={row['gt']} | "
            f"gen={row['gen']} ({'ok' if row['gen_ok'] else 'miss'}) | "
            f"logit={row['logit']} "
            f"({'ok' if row['logit_ok'] else 'miss'}) | {rel_s}"
        )
    gen_acc = sum(1 for r in summary if r["gen_ok"]) / len(summary)
    logit_acc = sum(1 for r in summary if r["logit_ok"]) / len(summary)
    print(f"Generate accuracy: {gen_acc:.0%} ({sum(1 for r in summary if r['gen_ok'])}/{len(summary)})")
    print(f"Logit accuracy:    {logit_acc:.0%} ({sum(1 for r in summary if r['logit_ok'])}/{len(summary)})")
    print("Done.")


if __name__ == "__main__":
    main()
