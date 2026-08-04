#!/usr/bin/env python3
"""Build Geirhos cue-conflict triads.

For each cue-conflict image:

    reference:
        shape A + texture 1

    shape_match:
        shape A (original object)

    texture_match:
        texture 1 (different shape, same texture)

Directory expectations:

cue-conflict:
    <shape>/<shape><id>-<texture><id>.png

shape directory:
    <shape>/<shape><id>.png

texture directory:
    <texture>/<texture><id>.png

Output:

<output>/<reference_name>/
    reference.png
    shape_match.png
    texture_match.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def save_triad(
    reference: Image.Image,
    shape_match: Image.Image,
    texture_match: Image.Image,
    out_dir: Path,
):
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference.save(out_dir / "reference.png")
    shape_match.save(out_dir / "shape_match.png")
    texture_match.save(out_dir / "texture_match.png")


def build_triads(
    shape_dir: Path,
    cue_conflict_dir: Path,
    texture_dir: Path,
    output_dir: Path,
):
    count = 0

    for cue_path in cue_conflict_dir.rglob("*.png"):

        if "debug" in cue_path.parts:
            continue

        # Example:
        # airplane1-car3.png

        parts = cue_path.stem.split("-")

        if len(parts) != 2:
            continue

        shape_id = parts[0]
        texture_id = parts[1]

        shape = "".join(
            c for c in shape_id if c.isalpha()
        )

        shape_num = "".join(
            c for c in shape_id if c.isdigit()
        )

        texture = "".join(
            c for c in texture_id if c.isalpha()
        )

        texture_num = "".join(
            c for c in texture_id if c.isdigit()
        )


        # ------------------
        # Shape match
        # ------------------

        shape_path = (
            shape_dir /
            shape /
            f"{shape}{shape_num}.png"
        )


        # ------------------
        # Texture match
        # ------------------

        texture_path = (
            texture_dir /
            texture /
            f"{texture}{texture_num}.png"
        )


        if not shape_path.exists():
            print("Missing shape:", shape_path)
            continue

        if not texture_path.exists():
            print("Missing texture:", texture_path)
            continue


        trial_name = cue_path.stem

        out_dir = output_dir / trial_name


        save_triad(
            Image.open(cue_path).convert("RGB"),
            Image.open(shape_path).convert("RGBA"),
            Image.open(texture_path).convert("RGB"),
            out_dir,
        )

        print(f"Saved: {out_dir}")
        count += 1

    print(f"\nDone. Created {count} triads.")


def main():

    parser = argparse.ArgumentParser(
        description="Build cue-conflict triads."
    )

    parser.add_argument(
        "--shape-dir",
        type=Path,
        required=True,
        help="Directory containing original shape images.",
    )

    parser.add_argument(
        "--cue-conflict-dir",
        type=Path,
        required=True,
        help="Directory containing cue-conflict images.",
    )

    parser.add_argument(
        "--texture-dir",
        type=Path,
        required=True,
        help="Directory containing texture exemplars.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write triads.",
    )

    args = parser.parse_args()

    build_triads(
        shape_dir=args.shape_dir,
        cue_conflict_dir=args.cue_conflict_dir,
        texture_dir=args.texture_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()