#!/usr/bin/env python3
"""Apply binary silhouette masks to Geirhos cue-conflict images.

The mask should have:

    black = foreground/object
    white = background

The background of each cue-conflict image is replaced with white while the
foreground is preserved.

Directory structure:

    --cue-conflict/
        <shape>/<shape><id>-<texture><id>.png

    --masks/
        <shape>/<shape><id>.png

Outputs:

    --output/
        <shape>/<shape><id>-<texture><id>.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

_GEIRHOS_PAT = re.compile(r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$")


def apply_mask(
    cue_conflict_path: Path,
    mask_path: Path,
    output_path: Path,
):
    cue = Image.open(cue_conflict_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    cue_arr = np.array(cue)
    mask_arr = np.array(mask)

    # black = foreground
    foreground = mask_arr < 128

    result = cue_arr.copy()
    result[~foreground] = [255, 255, 255]

    Image.fromarray(result).save(output_path)

    return cue_arr, mask_arr, result


def save_debug(
    cue_arr: np.ndarray,
    mask_arr: np.ndarray,
    result_arr: np.ndarray,
    output_path: Path,
):
    mask_rgb = np.stack([mask_arr] * 3, axis=2)

    debug = np.concatenate(
        [
            cue_arr,
            mask_rgb,
            result_arr,
        ],
        axis=1,
    )

    debug_dir = output_path.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(debug).save(
        debug_dir / output_path.name
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cue-conflict",
        type=Path,
        required=True,
        help="Directory containing cue-conflict images.",
    )

    ap.add_argument(
        "--masks",
        type=Path,
        required=True,
        help="Directory containing binary masks.",
    )

    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory.",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Save cue | mask | result debug strips.",
    )

    args = ap.parse_args()

    if not args.cue_conflict.is_absolute():
        args.cue_conflict = REPO_ROOT / args.cue_conflict

    if not args.masks.is_absolute():
        args.masks = REPO_ROOT / args.masks

    if not args.output.is_absolute():
        args.output = REPO_ROOT / args.output

    count = 0

    for shape_dir in sorted(
        p for p in args.cue_conflict.iterdir()
        if p.is_dir()
    ):

        shape = shape_dir.name

        for cue_path in sorted(shape_dir.glob("*.png")):

            m = _GEIRHOS_PAT.match(cue_path.name)

            if m is None:
                continue

            shape_name, shape_id, texture, texture_id = m.groups()

            key = f"{shape_name}{shape_id}"

            mask_path = (
                args.masks
                / shape
                / f"{key}.png"
            )

            if not mask_path.exists():
                print(f"Missing mask: {mask_path}")
                continue

            output_path = (
                args.output
                / shape
                / cue_path.name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            cue_arr, mask_arr, result_arr = apply_mask(
                cue_path,
                mask_path,
                output_path,
            )

            if args.debug:
                save_debug(
                    cue_arr,
                    mask_arr,
                    result_arr,
                    output_path,
                )

            print(f"Saved: {output_path}")
            count += 1

    print(f"\nDone. Created {count} masked cue-conflict images.")


if __name__ == "__main__":
    raise SystemExit(main())