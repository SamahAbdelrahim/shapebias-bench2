#!/usr/bin/env python3
"""Build shape-bias triads from Geirhos cue-conflict images.

Each cue-conflict image is:

    shape_instance-texture_instance.png

Example:
    airplane10-bicycle2.png

A triad is:

    reference.png
        airplane10 + bicycle2

    shape_match.png
        airplane10 + some other texture (e.g. car5)

    texture_match.png
        some other shape instance + bicycle2

where:
    - reference and shape_match share the same shape instance
    - reference and texture_match share the same texture instance
    - shape_match and texture_match are otherwise unconstrained relative
      to each other (this is the classic shape-bias triplet setup: the
      shape match and texture match are not required to share a texture)

One triad is created per valid reference image.

Random choices are controlled by --seed.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from PIL import Image


_GEIRHOS_PAT = re.compile(
    r"^([a-z]+)(\d+)-([a-z]+)(\d+)\.png$"
)


def parse_filename(path: Path):
    """
    Returns:
        shape,
        shape_id,
        texture,
        texture_id
    """

    m = _GEIRHOS_PAT.match(path.name)

    if m is None:
        return None

    shape, shape_id, texture, texture_id = m.groups()

    return (
        shape,
        int(shape_id),
        texture,
        int(texture_id),
    )


def save_triad(
    reference_path,
    shape_match_path,
    texture_match_path,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.open(reference_path).convert("RGB").save(
        output_dir / "reference.png"
    )

    Image.open(shape_match_path).convert("RGB").save(
        output_dir / "shape_match.png"
    )

    Image.open(texture_match_path).convert("RGB").save(
        output_dir / "texture_match.png"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="cue_conflict directory",
    )

    ap.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = ap.parse_args()

    rng = random.Random(args.seed)


    # --------------------------------------------------
    # Load all cue-conflict images
    # --------------------------------------------------

    images = []

    for category_dir in sorted(args.input.iterdir()):

        if not category_dir.is_dir():
            continue

        for img_path in category_dir.glob("*.png"):

            parsed = parse_filename(img_path)

            if parsed is None:
                continue

            shape, shape_id, texture, texture_id = parsed

            images.append(
                {
                    "path": img_path,
                    "shape": shape,
                    "shape_id": shape_id,
                    "texture": texture,
                    "texture_id": texture_id,
                }
            )


    print(f"Found {len(images)} cue-conflict images.")


    # --------------------------------------------------
    # Build lookup:
    #
    # (shape, shape_id, texture) -> image
    #
    # and:
    #
    # (texture, texture_id) -> all images with that exact texture instance
    #
    # --------------------------------------------------

    lookup = {}

    texture_lookup = {}

    for item in images:

        key = (
            item["shape"],
            item["shape_id"],
            item["texture"],
        )

        lookup[key] = item["path"]

        texture_lookup.setdefault(
            (item["texture"], item["texture_id"]),
            []
        ).append(item)

    # Distinct texture *names* (not (name, id) instances). Used below when
    # searching for a same-shape / different-texture shape match: we only
    # care whether some OTHER texture name was ever paired with this shape
    # instance, not which particular texture instance it was.
    texture_names = {
        texture_name
        for (texture_name, texture_id) in texture_lookup.keys()
    }


    # --------------------------------------------------
    # Build triads
    # --------------------------------------------------

    created = 0
    skipped = 0


    for reference in images:

        shape = reference["shape"]
        shape_id = reference["shape_id"]
        texture1 = reference["texture"]


        # candidates for shape match:
        # same shape instance, different texture

        shape_candidates = []

        for texture_name in texture_names:

            if texture_name == texture1:
                continue

            key = (
                shape,
                shape_id,
                texture_name,
            )

            if key in lookup:

                shape_candidates.append(
                    lookup[key]
                )


        if len(shape_candidates) == 0:
            skipped += 1
            continue


        shape_match = rng.choice(
            shape_candidates
        )


        # candidates for texture match:
        # same texture instance as reference
        # different shape instance

        texture_candidates = []

        for item in texture_lookup[(texture1, reference["texture_id"])]:

            if (
                item["shape"] == shape
                and item["shape_id"] == shape_id
            ):
                continue

            texture_candidates.append(
                item
            )


        if len(texture_candidates) == 0:
            skipped += 1
            continue


        texture_match = rng.choice(
            texture_candidates
        )


        # folder named after reference
        triad_dir = (
            args.output /
            reference["path"].stem
        )


        save_triad(
            reference["path"],
            shape_match,
            texture_match["path"],
            triad_dir,
        )


        created += 1


    print()
    print(f"Created {created} triads.")
    print(f"Skipped {skipped} references.")


if __name__ == "__main__":
    raise SystemExit(main())