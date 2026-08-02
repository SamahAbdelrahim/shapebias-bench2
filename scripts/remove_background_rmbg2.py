#!/usr/bin/env python3
"""Replace Geirhos cue-conflict backgrounds with white, using RMBG-2.0 alpha
mattes computed from the matching original-image silhouettes (instead of a
white-pixel distance threshold).

Directory structure:
    --input/
        original/<shape>/<shape><id>.png
        cue_conflict/<shape>/<shape><id>-<texture><id>.png
    --output/
        <shape>/<shape><id>-<texture><id>.png   (background replaced with white)

Mattes are computed once per original image and reused for every cue-conflict
image referencing that shape id, since segmenting the cue-conflict image
directly is unreliable (the texture confuses the segmentation model).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

REPO_ROOT = Path(__file__).resolve().parent.parent

_GEIRHOS_PAT = re.compile(r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$")

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
    """Run RMBG-2.0 and return a single-channel alpha matte at original size."""
    orig_size = image.size  # (W, H)
    rgb = image.convert("RGB")

    input_tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    matte = transforms.ToPILImage()(pred).resize(orig_size, Image.BILINEAR)
    return matte  # mode "L", values 0-255


def composite_on_white(cue_image: Image.Image, matte: Image.Image) -> Image.Image:
    """Composite cue_image onto a white background using matte as alpha."""
    cue = np.array(cue_image.convert("RGB"))

    if matte.size != cue_image.size:
        matte = matte.resize(cue_image.size, Image.BILINEAR)

    alpha = np.array(matte).astype(float) / 255.0
    alpha = alpha[..., None]

    white = np.ones_like(cue) * 255
    result = (cue * alpha + white * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(result)


def save_debug(orig_arr, matte: Image.Image, new_img: np.ndarray, output_path: Path):
    mask_img = np.array(matte)
    mask_img = np.stack([mask_img] * 3, axis=2)  # grayscale -> RGB

    debug = np.concatenate([orig_arr, mask_img, new_img], axis=1)

    debug_path = output_path.parent / "debug" / output_path.name
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(debug).save(debug_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--debug", action="store_true", help="Save side-by-side debug images.")
    ap.add_argument(
        "--save-mattes", type=Path, default=None,
        help="Also save standalone RGBA alpha-matte PNGs here (e.g. for mask_review.py).",
    )
    args = ap.parse_args()

    if args.save_mattes is not None and not args.save_mattes.is_absolute():
        args.save_mattes = REPO_ROOT / args.save_mattes

    if not args.input.is_absolute():
        args.input = REPO_ROOT / args.input
    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    print(f"Loading {MODEL_ID} on {args.device}...")
    model = load_model(args.device)
    transform = build_transform()

    cue_root = args.input / "cue_conflict"

    mattes_by_shape_id: dict[str, Image.Image] = {}
    flagged = []
    count = 0

    for shape_dir in sorted(p for p in cue_root.iterdir() if p.is_dir()):
        for ref_path in sorted(shape_dir.iterdir()):
            m = _GEIRHOS_PAT.match(ref_path.name)
            if not m:
                continue

            shape, shape_id, texture, texture_id = m.groups()
            key = f"{shape}{shape_id}"

            original_path = args.input / "original" / shape / f"{key}.png"
            if not original_path.exists():
                print(f"Missing: {original_path}")
                continue

            # Compute (and cache) the matte for this shape id once.
            if key not in mattes_by_shape_id:
                original_img = Image.open(original_path)
                matte = compute_matte(model, transform, original_img, args.device)
                mattes_by_shape_id[key] = matte

                arr = np.array(matte)
                ambiguous = np.logical_and(arr >= MID_GRAY_LOW, arr <= MID_GRAY_HIGH)
                frac = ambiguous.mean()
                if frac >= AMBIGUOUS_FRACTION_THRESHOLD:
                    print(f"  [FLAG for review] {key} ambiguous matte: {frac:.1%}")
                    flagged.append((key, frac))

                if args.save_mattes is not None:
                    rgba = original_img.convert("RGBA")
                    rgba.putalpha(matte)
                    matte_out_path = args.save_mattes / shape / f"{key}.png"
                    matte_out_path.parent.mkdir(parents=True, exist_ok=True)
                    rgba.save(matte_out_path)

            matte = mattes_by_shape_id[key]

            cue_img = Image.open(ref_path).convert("RGB")
            new_img_pil = composite_on_white(cue_img, matte)
            new_img = np.array(new_img_pil)

            output_path = args.output / shape / ref_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.debug:
                orig_arr = np.array(Image.open(original_path).convert("RGB"))
                save_debug(orig_arr, matte, new_img, output_path)

            new_img_pil.save(output_path)

            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Created {count} masked cue-conflict images.")
    if flagged:
        print(f"\n{len(flagged)} original image(s) flagged for manual review:")
        for key, frac in flagged:
            print(f"  {key}  ({frac:.1%} ambiguous)")
    else:
        print("No original images flagged for review.")


if __name__ == "__main__":
    raise SystemExit(main())