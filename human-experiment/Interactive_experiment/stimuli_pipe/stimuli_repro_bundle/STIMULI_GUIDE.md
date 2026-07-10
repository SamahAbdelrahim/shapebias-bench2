# ALICE Stimuli Guide

This folder contains two stimulus modes and two organization styles.

## Modes

- `stimuli_B_controlled_simple`
- `stimuli_A_auto_contrast`

## Non-packaged folders

Each mode has:

- `<mode>/<stl_id>/reference.png`
- `<mode>/<stl_id>/shape_match.png`
- `<mode>/manifest.csv` with columns:
  - `mode,stl_id,reference,shape_match`

## Packaged folders (benchmark-ready)

Each mode has:

- `stimuli_per_stl_packages/<mode>/<stl_id>/example_image.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/reference.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/shape_match.png`
- `stimuli_per_stl_packages/<mode>/<stl_id>/texture_match.png`
- `stimuli_per_stl_packages/<mode>/manifest.csv` with columns:
  - `mode,stl_id,example_image,reference,shape_match,texture_match`

Naming semantics:

- `example_image`: original image reference
- `reference`: same-shape rendered object
- `shape_match`: same-shape with different material
- `texture_match`: different-shape with same material as `reference`

## Repro scripts

From repo root:

```bash
# Generate non-packaged 2-image stimuli for B
ALICE_STIMULUS_MODE=B_controlled_simple bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py

# Generate non-packaged 2-image stimuli for A
ALICE_STIMULUS_MODE=A_auto_contrast bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py

# Add packaged texture_match (third test image)
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py

# Strictly sync reference + texture_match materials for B only
ALICE_ONLY_MODES=stimuli_B_controlled_simple \
ALICE_REPAIR_REFERENCE_TEXTURES=1 \
ALICE_STIMULUS_USE_IMAGE_TEXTURES=1 \
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py

# Strictly sync reference + texture_match materials for A only
ALICE_ONLY_MODES=stimuli_A_auto_contrast \
ALICE_REPAIR_REFERENCE_TEXTURES=1 \
ALICE_STIMULUS_USE_IMAGE_TEXTURES=1 \
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py

# Normalize names/manifests after generation
python3 scripts/standardize_stimuli_naming.py

# Rebuild combined benchmark manifest
python3 scripts/build_combined_benchmark_manifest.py
```

## Make textures look realistic

The stimulus shader now supports image-based PBR texture maps.

1. Put extracted texture folders in:
   - `data/texture_library/` (or set `ALICE_TEXTURE_LIBRARY` to another path)
2. Enable image textures during render:

```bash
ALICE_STIMULUS_USE_IMAGE_TEXTURES=1 \
ALICE_STIMULUS_MODE=A_auto_contrast \
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py
```

## Material consistency guarantee (packaged)

For each STL package:
- `reference.png`: same shape, reference material (variant 1)
- `texture_match.png`: different shape, same material as `reference.png`

`add_test_object_3_different_shape.py` supports deterministic texture locking per STL
for both A and B modes, and can optionally re-render `reference.png` so both images
are regenerated in one pass with the same chosen texture set.
