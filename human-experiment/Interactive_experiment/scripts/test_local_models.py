#!/usr/bin/env python3
"""Smoke test: load each local VLM, run a 3-image 2AFC-style prompt, unload."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# Load .env from repo root (sets HF_API_TOKEN for gated model downloads)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from evaluation_pipe.models import create_model, list_models  # noqa: E402


def make_dummy_images() -> list[Image.Image]:
    """Return 3 solid-colour 64x64 images (reference, A, B)."""
    return [
        Image.new("RGB", (64, 64), color="red"),    # reference
        Image.new("RGB", (64, 64), color="green"),   # image A
        Image.new("RGB", (64, 64), color="blue"),    # image B
    ]


PROMPT = (
    "The first image is a shiple. "
    "Which of the following two images (Image 1 or Image 2) is also a shiple? "
    "Answer with just 'Image 1' or 'Image 2'."
)

MODELS_TO_TEST = ["smolvlm", "internvl", "tinyllava"]


def test_model(name: str) -> bool:
    """Load *name*, generate from 3 dummy images, print result, unload."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    try:
        model = create_model(name)
        images = make_dummy_images()
        response = model.generate(images=images, prompt=PROMPT)

        print(f"  model_name         : {response.model_name}")
        print(f"  raw_text           : {response.raw_text!r}")
        print(f"  generation_time_s  : {response.generation_time_s:.2f}")
        print(f"  num_tokens_generated: {response.num_tokens_generated}")

        assert response.raw_text, "raw_text is empty!"
        print(f"  PASSED")

        model.unload()
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    print(f"Registered models: {list_models()}")

    # Allow filtering via CLI args, e.g.: python test_local_models.py smolvlm
    names = sys.argv[1:] if len(sys.argv) > 1 else MODELS_TO_TEST

    results: dict[str, bool] = {}
    for name in names:
        results[name] = test_model(name)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
