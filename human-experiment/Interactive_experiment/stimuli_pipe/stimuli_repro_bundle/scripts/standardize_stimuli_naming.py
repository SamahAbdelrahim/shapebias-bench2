"""
Standardize ALICE stimuli naming and manifests.

What this script does:
1) Packaged folders (stimuli_per_stl_packages/*/<id>/):
   - Ensure filenames are:
       example_image.png, reference.png, shape_match.png, texture_match.png
   - Rewrite manifest.csv columns to:
       mode,stl_id,example_image,reference,shape_match,texture_match

2) Non-packaged folders (stimuli_A_auto_contrast/*/<id>/ and
   stimuli_B_controlled_simple/*/<id>/):
   - Rename version_1.png -> reference.png
   - Rename version_2.png -> shape_match.png
   - Rewrite manifest.csv columns to:
       mode,stl_id,reference,shape_match

Run from repo root:
  python3 scripts/standardize_stimuli_naming.py
"""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ALICE = PROJECT / "data" / "ALICE_stl_(Xu & Sandhofer, 2024)"

PACKAGES = ALICE / "stimuli_per_stl_packages"
NON_PACKAGED = {
    "stimuli_A_auto_contrast": ALICE / "stimuli_A_auto_contrast",
    "stimuli_B_controlled_simple": ALICE / "stimuli_B_controlled_simple",
}


def _sort_key(path: Path):
    name = path.name
    return (0, int(name)) if name.isdigit() else (1, name)


def _rename_if_exists(folder: Path, old_name: str, new_name: str) -> None:
    src = folder / old_name
    dst = folder / new_name
    if not src.exists():
        return
    if dst.exists():
        dst.unlink()
    src.rename(dst)


def _standardize_packaged_mode(mode_folder: str) -> int:
    mode_dir = PACKAGES / mode_folder
    if not mode_dir.exists():
        return 0

    stems = sorted([p for p in mode_dir.iterdir() if p.is_dir()], key=_sort_key)
    for stem_dir in stems:
        _rename_if_exists(stem_dir, "reference_image.png", "example_image.png")
        _rename_if_exists(stem_dir, "test_object_1.png", "reference.png")
        _rename_if_exists(stem_dir, "test_object_2.png", "shape_match.png")
        _rename_if_exists(stem_dir, "test_object_3.png", "texture_match.png")

    manifest = mode_dir / "manifest.csv"
    rows_out = []
    for stem_dir in stems:
        stem = stem_dir.name
        rows_out.append(
            {
                "mode": mode_folder,
                "stl_id": stem,
                "example_image": f"stimuli_per_stl_packages/{mode_folder}/{stem}/example_image.png",
                "reference": f"stimuli_per_stl_packages/{mode_folder}/{stem}/reference.png",
                "shape_match": f"stimuli_per_stl_packages/{mode_folder}/{stem}/shape_match.png",
                "texture_match": f"stimuli_per_stl_packages/{mode_folder}/{stem}/texture_match.png",
            }
        )

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "stl_id",
                "example_image",
                "reference",
                "shape_match",
                "texture_match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return len(stems)


def _standardize_non_packaged_mode(mode_folder: str, mode_dir: Path) -> int:
    if not mode_dir.exists():
        return 0

    stems = sorted([p for p in mode_dir.iterdir() if p.is_dir()], key=_sort_key)
    for stem_dir in stems:
        _rename_if_exists(stem_dir, "version_1.png", "reference.png")
        _rename_if_exists(stem_dir, "version_2.png", "shape_match.png")

    manifest = mode_dir / "manifest.csv"
    rows_out = []
    for stem_dir in stems:
        stem = stem_dir.name
        rows_out.append(
            {
                "mode": mode_folder,
                "stl_id": stem,
                "reference": f"{mode_folder}/{stem}/reference.png",
                "shape_match": f"{mode_folder}/{stem}/shape_match.png",
            }
        )

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "stl_id", "reference", "shape_match"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    return len(stems)


def main() -> None:
    packaged_counts = {
        mode: _standardize_packaged_mode(mode)
        for mode in ("stimuli_A_auto_contrast", "stimuli_B_controlled_simple")
    }
    non_packaged_counts = {
        mode: _standardize_non_packaged_mode(mode, path) for mode, path in NON_PACKAGED.items()
    }

    print("Standardization complete.")
    print(f"Packaged: {packaged_counts}")
    print(f"Non-packaged: {non_packaged_counts}")


if __name__ == "__main__":
    main()
