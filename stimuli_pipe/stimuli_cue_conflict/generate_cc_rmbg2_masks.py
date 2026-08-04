#!/usr/bin/env python3
"""Generate object alpha masks for Geirhos original images using RMBG-2.0.

Output masks are black (0) where the object is, white (255) everywhere else
(i.e. the alpha matte's foreground/background polarity is inverted relative
to a typical alpha channel, where foreground is usually high-valued).

Directory structure:
    --input/
        original/<shape>/<shape><id>.png
    --output/
        <shape>/<shape><id>.png   (mask: 0 = object, 255 = background)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "briaai/RMBG-2.0"
IMAGE_SIZE = (1024, 1024)  # RMBG-2.0's native training resolution

# Fraction of matte pixels landing in the "ambiguous" mid-gray band before
# we flag an image for manual review. Pixels near 0 or 255 are confident
# foreground/background decisions; a lot of mass in between suggests the
# model struggled (e.g. glossy reflections, fine fur, low contrast edges).
MID_GRAY_LOW, MID_GRAY_HIGH = 60, 195
AMBIGUOUS_FRACTION_THRESHOLD = 0.08


def load_model(device: str):
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ]
    )


def compute_matte(model, transform, image: Image.Image, device: str) -> Image.Image:
    """Run RMBG-2.0 and return a single-channel alpha matte at original size.

    In this raw matte, high values (~255) correspond to the foreground
    object and low values (~0) correspond to background — the opposite of
    the mask convention this script writes to disk.
    """
    orig_size = image.size  # (W, H)
    rgb = image.convert("RGB")

    input_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    matte = transforms.ToPILImage()(pred).resize(orig_size, Image.BILINEAR)
    return matte  # mode "L", values 0-255


def convert_matte_to_mask(matte: Image.Image) -> Image.Image:
    """Convert RMBG matte to binary mask:
    object = black (0), background = white (255).
    """
    arr = np.array(matte)

    # RMBG: foreground high, background low
    mask = np.where(arr >= 128, 0, 255)

    return Image.fromarray(mask.astype(np.uint8), mode="L")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.input.is_absolute():
        args.input = REPO_ROOT / args.input
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    print(f"Loading {MODEL_ID} on {args.device}...")
    model = load_model(args.device)
    transform = build_transform()

    original_root = args.input / "original"

    flagged = []
    count = 0

    for shape_dir in sorted(p for p in original_root.iterdir() if p.is_dir()):
        for original_path in sorted(shape_dir.glob("*.png")):
            key = original_path.stem  # e.g. "cat4"
            shape = shape_dir.name

            original_img = Image.open(original_path)
            matte = compute_matte(model, transform, original_img, args.device)

            arr = np.array(matte)
            ambiguous = np.logical_and(arr >= MID_GRAY_LOW, arr <= MID_GRAY_HIGH)
            frac = ambiguous.mean()
            if frac >= AMBIGUOUS_FRACTION_THRESHOLD:
                print(f"  [FLAG for review] {key} ambiguous matte: {frac:.1%}")
                flagged.append((key, frac))

            mask = convert_matte_to_mask(matte)

            output_path = args.output / shape / f"{key}.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mask.save(output_path)

            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Created {count} masks.")
    if flagged:
        print(f"\n{len(flagged)} original image(s) flagged for manual review:")
        for key, frac in flagged:
            print(f"  {key}  ({frac:.1%} ambiguous)")
    else:
        print("No original images flagged for review.")


if __name__ == "__main__":
    raise SystemExit(main())