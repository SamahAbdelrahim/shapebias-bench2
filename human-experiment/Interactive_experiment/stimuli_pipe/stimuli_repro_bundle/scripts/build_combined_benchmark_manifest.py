"""
Build a combined benchmark-ready manifest from packaged ALICE stimuli.

Inputs (auto-detected):
  - stimuli_per_stl_packages/stimuli_A_auto_contrast/manifest.csv
  - stimuli_per_stl_packages/stimuli_B_controlled_simple/manifest.csv

Output:
  - stimuli_per_stl_packages/combined_benchmark_manifest.csv

Columns:
  trial_id,mode,stl_id,reference,image_a,image_b,correct_label,shape_match,texture_match,example_image,target,distractor
"""

from __future__ import annotations

import csv
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = BUNDLE_ROOT.parent / "stimuli_per_stl_packages"
ALICE_PACKAGES = (
    BUNDLE_ROOT / "data" / "ALICE_stl_(Xu & Sandhofer, 2024)" / "stimuli_per_stl_packages"
)
PACKAGES = LOCAL_PACKAGES if LOCAL_PACKAGES.exists() else ALICE_PACKAGES

MANIFESTS = [
    PACKAGES / "stimuli_A_auto_contrast" / "manifest.csv",
    PACKAGES / "stimuli_B_controlled_simple" / "manifest.csv",
]
OUT = PACKAGES / "combined_benchmark_manifest.csv"


def _mode_tag(mode_value: str) -> str:
    if "A_auto_contrast" in mode_value:
        return "A"
    if "B_controlled_simple" in mode_value:
        return "B"
    return "U"


def main() -> None:
    rows_out = []
    for manifest in MANIFESTS:
        if not manifest.exists():
            continue
        with manifest.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                mode = str(row.get("mode", "")).strip()
                stl_id = str(row.get("stl_id", "")).strip()
                if not mode or not stl_id:
                    continue
                rows_out.append(
                    {
                        "trial_id": f"{_mode_tag(mode)}_{int(stl_id):03d}",
                        "mode": mode,
                        "stl_id": stl_id,
                        "reference": row.get("reference", ""),
                        # Canonical 2AFC mapping for evaluation loaders:
                        # image_a = shape_match, image_b = texture_match.
                        "image_a": row.get("shape_match", ""),
                        "image_b": row.get("texture_match", ""),
                        "correct_label": "A",
                        "shape_match": row.get("shape_match", ""),
                        "texture_match": row.get("texture_match", ""),
                        "example_image": row.get("example_image", ""),
                        # Backward-compatible alias used in older drafts.
                        "target": row.get("reference", ""),
                        # Placeholder for future negative-control assignment.
                        "distractor": "",
                    }
                )

    rows_out.sort(key=lambda r: (r["mode"], int(r["stl_id"])))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "mode",
                "stl_id",
                "reference",
                "image_a",
                "image_b",
                "correct_label",
                "shape_match",
                "texture_match",
                "example_image",
                "target",
                "distractor",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {OUT}")


if __name__ == "__main__":
    main()
