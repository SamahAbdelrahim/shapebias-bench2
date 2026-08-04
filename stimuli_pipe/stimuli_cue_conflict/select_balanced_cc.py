#!/usr/bin/env python3
"""Balance Geirhos cue-conflict images.

Selects one cue-conflict exemplar per shape identity while balancing:
1. texture categories
2. texture exemplars

Removes same-category shape/texture conflicts.

Input:
    <input>/<shape>/<shape><id>-<texture><id>.png

Output:
    <output>/<shape>/<shape><id>-<texture><id>.png
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict, Counter
from pathlib import Path


def parse_filename(path: Path):
    """
    Example:
        airplane1-car2.png

    Returns:
        shape_id: airplane1
        shape: airplane
        texture_id: car2
        texture: car
    """

    shape_part, texture_part = path.stem.split("-")

    shape = "".join(
        c for c in shape_part if c.isalpha()
    )

    texture = "".join(
        c for c in texture_part if c.isalpha()
    )

    return (
        shape_part,
        shape,
        texture_part,
        texture,
    )


def build_balanced_selection(input_dir: Path):

    candidates_by_shape = defaultdict(list)

    for path in sorted(input_dir.rglob("*.png")):

        if "debug" in path.parts:
            continue

        shape_id, shape, texture_id, texture = parse_filename(path)

        # Remove same-category texture swaps
        if shape == texture:
            continue

        candidates_by_shape[shape_id].append(
            {
                "path": path,
                "texture_id": texture_id,
                "texture": texture,
                "shape": shape,
            }
        )

    print(
        "Found shape exemplars:",
        len(candidates_by_shape)
    )

    texture_exemplar_counts = Counter()
    texture_category_counts = Counter()
    used_texture_categories = defaultdict(set)

    selected = {}

    for shape_id in sorted(candidates_by_shape):

        options = candidates_by_shape[shape_id]

        if len(options) == 0:
            print("No valid options for:", shape_id)
            continue

        shape_category = options[0]["shape"]

        diverse_options = [
            x for x in options
            if x["texture"]
            not in used_texture_categories[shape_category]
        ]

        if len(diverse_options) == 0:

            print(
                "WARNING:",
                shape_id,
                "ran out of unique texture categories. Relaxing."
            )

            diverse_options = options


        minimum_category_count = min(
            texture_category_counts[x["texture"]]
            for x in diverse_options
        )

        category_balanced_options = [
            x for x in diverse_options
            if texture_category_counts[x["texture"]]
            == minimum_category_count
        ]


        minimum_exemplar_count = min(
            texture_exemplar_counts[x["texture_id"]]
            for x in category_balanced_options
        )

        exemplar_balanced_options = [
            x for x in category_balanced_options
            if texture_exemplar_counts[x["texture_id"]]
            == minimum_exemplar_count
        ]


        choice = sorted(
            exemplar_balanced_options,
            key=lambda x: x["texture_id"]
        )[0]


        selected[shape_id] = choice

        texture_category_counts[
            choice["texture"]
        ] += 1

        texture_exemplar_counts[
            choice["texture_id"]
        ] += 1

        used_texture_categories[
            shape_category
        ].add(
            choice["texture"]
        )

        print(
            "Selected:",
            shape_id,
            "<-",
            choice["texture_id"]
        )

    return (
        selected,
        texture_category_counts,
        texture_exemplar_counts,
        used_texture_categories,
    )


def write_output(selected, output_dir: Path):

    if output_dir.exists():

        print("\nRemoving old output...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for _, item in selected.items():

        category = item["path"].parent.name

        out_dir = output_dir / category

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            item["path"],
            out_dir / item["path"].name,
        )


def main():

    parser = argparse.ArgumentParser(
        description="Balance Geirhos cue-conflict images."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input cue-conflict directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output balanced directory.",
    )

    args = parser.parse_args()

    selected, texture_category_counts, texture_exemplar_counts, used_texture_categories = (
        build_balanced_selection(args.input)
    )

    write_output(
        selected,
        args.output,
    )

    print()

    print(
        "Done. Created",
        len(selected),
        "balanced cue-conflict images."
    )

    print()

    print("Texture category counts:")
    for texture, count in sorted(texture_category_counts.items()):
        print(f"{texture:15s}", count)

    print()

    print("Texture exemplar counts:")
    for texture_id, count in sorted(texture_exemplar_counts.items()):
        print(f"{texture_id:20s}", count)

    print()

    print("Texture category usage per shape:")

    for shape, textures in sorted(
        used_texture_categories.items()
    ):
        print(
            f"{shape:15s}: {sorted(textures)}"
        )


if __name__ == "__main__":
    main()