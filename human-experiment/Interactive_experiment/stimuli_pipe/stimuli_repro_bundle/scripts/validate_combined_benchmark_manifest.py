"""
Validate combined benchmark manifest schema and file references.

Default behavior:
1) Prefer validating:
   stimuli_pipe/stimuli_per_stl_packages/combined_benchmark_manifest.csv
2) Fallback to:
   stimuli_pipe/stimuli_repro_bundle/manifests/combined_benchmark_manifest.csv

Checks:
- Required columns exist (and in expected order).
- Required fields are non-empty for each row.
- Referenced files exist on disk.
- `correct_label` is in {"A", "B"}.

Run:
  python3 scripts/validate_combined_benchmark_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
STIMULI_PIPE_ROOT = BUNDLE_ROOT.parent

PREFERRED_MANIFEST = STIMULI_PIPE_ROOT / "stimuli_per_stl_packages" / "combined_benchmark_manifest.csv"
FALLBACK_MANIFEST = BUNDLE_ROOT / "manifests" / "combined_benchmark_manifest.csv"

EXPECTED_COLUMNS = [
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
]

REQUIRED_NON_EMPTY = [
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
]

PATH_COLUMNS_REQUIRED = [
    "reference",
    "image_a",
    "image_b",
    "shape_match",
    "texture_match",
    "example_image",
]

PATH_COLUMNS_OPTIONAL = [
    "target",
    "distractor",
]


def _default_manifest() -> Path:
    if PREFERRED_MANIFEST.exists():
        return PREFERRED_MANIFEST
    return FALLBACK_MANIFEST


def _resolve_path(stimuli_root: Path, raw: str) -> Path:
    value = raw.strip()
    p = Path(value)
    return p if p.is_absolute() else (stimuli_root / p)


def validate(manifest_path: Path, stimuli_root: Path) -> int:
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != EXPECTED_COLUMNS:
            errors.append(
                "Header mismatch.\n"
                f"  expected: {EXPECTED_COLUMNS}\n"
                f"  found   : {reader.fieldnames}"
            )

        row_count = 0
        for i, row in enumerate(reader, start=2):
            row_count += 1

            for key in REQUIRED_NON_EMPTY:
                if not str(row.get(key, "")).strip():
                    errors.append(f"Row {i}: missing required value for `{key}`")

            label = str(row.get("correct_label", "")).strip()
            if label and label not in {"A", "B"}:
                errors.append(f"Row {i}: invalid `correct_label`={label!r} (expected 'A' or 'B')")

            target = str(row.get("target", "")).strip()
            reference = str(row.get("reference", "")).strip()
            if target and reference and target != reference:
                warnings.append(
                    f"Row {i}: `target` differs from `reference` "
                    "(allowed, but check if intended)"
                )

            for key in PATH_COLUMNS_REQUIRED:
                raw = str(row.get(key, "")).strip()
                if not raw:
                    continue
                path = _resolve_path(stimuli_root, raw)
                if not path.exists():
                    errors.append(f"Row {i}: missing file for `{key}` -> {raw}")

            for key in PATH_COLUMNS_OPTIONAL:
                raw = str(row.get(key, "")).strip()
                if not raw:
                    continue
                path = _resolve_path(stimuli_root, raw)
                if not path.exists():
                    warnings.append(f"Row {i}: optional file for `{key}` not found -> {raw}")

    if row_count == 0:
        errors.append("Manifest has no data rows.")

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for msg in warnings[:20]:
            print(f"  - {msg}")
        if len(warnings) > 20:
            print(f"  - ... {len(warnings) - 20} more warnings")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for msg in errors[:50]:
            print(f"  - {msg}")
        if len(errors) > 50:
            print(f"  - ... {len(errors) - 50} more errors")
        return 1

    print(f"Validation passed: {manifest_path} ({row_count} rows)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate combined benchmark manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_default_manifest(),
        help="Path to combined_benchmark_manifest.csv",
    )
    parser.add_argument(
        "--stimuli-root",
        type=Path,
        default=STIMULI_PIPE_ROOT,
        help="Root path used to resolve relative image paths in the manifest.",
    )
    args = parser.parse_args()

    exit_code = validate(manifest_path=args.manifest, stimuli_root=args.stimuli_root)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
