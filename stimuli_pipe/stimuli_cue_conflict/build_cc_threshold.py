#!/usr/bin/env python3
"""Replace Geirhos cue-conflict backgrounds with white using original-image silhouettes."""

from __future__ import annotations

import argparse
import re
import numpy as np
import cv2
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

_GEIRHOS_PAT = re.compile(r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$")

def remove_background(cue_conflict_path, original_path, output_path, threshold=245):
    cue_img = Image.open(cue_conflict_path).convert("RGB")
    original = Image.open(original_path).convert("RGB")

    orig = np.array(original)

    # # object pixels = True, background pixels = False
    # mask = np.any(orig < threshold, axis=2)

    dist = np.linalg.norm(orig.astype(float) - 255, axis=2)
    mask = dist > threshold

    new_img = np.array(cue_img).copy()

    # replace background with white
    new_img[~mask] = [255, 255, 255]

    # debug: original | mask | result

    mask_img = (mask * 255).astype(np.uint8)  # False -> 0, True -> 255
    mask_img = np.stack([mask_img] * 3, axis=2)  # grayscale -> RGB

    debug = np.concatenate([orig, mask_img, new_img], axis=1)

    debug_path = output_path.parent / "debug" / output_path.name
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(debug).save(debug_path)

    Image.fromarray(new_img).save(output_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=int, default=245)
    args = ap.parse_args()

    if not args.input.is_absolute():
        args.input = REPO_ROOT / args.input

    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    cue_root = args.input / "cue_conflict"

    for shape_dir in sorted(p for p in cue_root.iterdir() if p.is_dir()):
        for ref_path in sorted(shape_dir.iterdir()):
            m = _GEIRHOS_PAT.match(ref_path.name)
            if not m:
                continue

            shape, shape_id, texture, texture_id = m.groups()

            original_path = (args.input / "original" / shape / f"{shape}{shape_id}.png")

            if not original_path.exists():
                print(f"Missing: {original_path}")
                continue

            output_path = (args.output / shape / ref_path.name)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            remove_background(
                cue_conflict_path=ref_path,
                original_path=original_path,
                output_path=output_path,
                threshold=args.threshold,
            )

            print(f"Saved: {output_path}")

    
    ...


if __name__ == "__main__":
    raise SystemExit(main())