# Stimuli Repro Bundle

This folder contains the files needed to reproduce the ALICE stimuli generation pipeline in another repository (for example, `shapebias-bench/stimuli_pipe`).

## Included

- `fixed_blender_centering_alice_texture.py`
- `add_test_object_3_different_shape.py`
- `run_blender.sh`
- `scripts/stl_spin_render.py`
- `scripts/stl_material_overlay_render.py`
- `scripts/standardize_stimuli_naming.py`
- `scripts/build_combined_benchmark_manifest.py`
- `colab_render.ipynb`
- `colab_render_drive.ipynb`
- `STIMULI_GUIDE.md`
- `manifests/stimuli_B_manifest.csv`
- `manifests/stimuli_A_manifest.csv`
- `manifests/packages_B_manifest.csv`
- `manifests/packages_A_manifest.csv`
- `manifests/combined_benchmark_manifest.csv`

## Expected data layout

These scripts expect ALICE data at:

- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stl/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/images/`

and write outputs under:

- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_B_controlled_simple/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_A_auto_contrast/`
- `data/ALICE_stl_(Xu & Sandhofer, 2024)/stimuli_per_stl_packages/`

## How to use in shapebias-bench

1. Copy this folder into `shapebias-bench/stimuli_pipe/`.
2. Ensure Blender and ffmpeg are installed in the target environment.
3. From the copied bundle folder, run one mode at a time:

```bash
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py
```

With environment variables:

```bash
ALICE_STIMULUS_MODE=B_controlled_simple \
ALICE_STIMULUS_RES=1024 \
ALICE_STIMULUS_SAMPLES=128 \
ALICE_STIMULUS_MATCH_REFERENCE=1 \
ALICE_STIMULUS_MATCH_STEP_DEG=2 \
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py
```

Then for A:

```bash
ALICE_STIMULUS_MODE=A_auto_contrast \
ALICE_STIMULUS_RES=1024 \
ALICE_STIMULUS_SAMPLES=128 \
ALICE_STIMULUS_MATCH_REFERENCE=1 \
ALICE_STIMULUS_MATCH_STEP_DEG=2 \
bash ./run_blender.sh -b -P fixed_blender_centering_alice_texture.py
```

## Notes

- The package layout and manifests are deterministic across runs (same input -> same naming).
- The generated files per shape package are:
  - `example_image.png`
  - `reference.png`
  - `shape_match.png`
  - `texture_match.png` (different shape from `data/random_stl`, same material recipe as `reference`)

## Add the third test object + standardize names

After `stimuli_per_stl_packages` exists, run:

```bash
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py
python3 scripts/standardize_stimuli_naming.py
```

For strict `reference`/`texture_match` texture sync (deterministic, per STL):

```bash
ALICE_ONLY_MODES=stimuli_A_auto_contrast \
ALICE_REPAIR_REFERENCE_TEXTURES=1 \
ALICE_STIMULUS_USE_IMAGE_TEXTURES=1 \
bash ./run_blender.sh -b -P add_test_object_3_different_shape.py
```

Use `ALICE_ONLY_MODES=stimuli_B_controlled_simple` for B, or omit
`ALICE_ONLY_MODES` to process both modes.

This updates:

- `stimuli_per_stl_packages/stimuli_B_controlled_simple/*/texture_match.png`
- `stimuli_per_stl_packages/stimuli_A_auto_contrast/*/texture_match.png`
- package manifests to:
  - `mode,stl_id,example_image,reference,shape_match,texture_match`
- non-packaged manifests to:
  - `mode,stl_id,reference,shape_match`

To build one benchmark-ingestion manifest:

```bash
python3 scripts/build_combined_benchmark_manifest.py
```

## Realistic Texture Option (Image PBR Maps)

The stimulus material generator can use image texture sets (fabric/steel/etc.)
instead of procedural-only textures.

- Enable: `ALICE_STIMULUS_USE_IMAGE_TEXTURES=1` (default on)
- Optional texture root override: `ALICE_TEXTURE_LIBRARY=/absolute/path/to/texture_sets`
- Default expected location in this project:
  - `data/texture_library/<set_name>/...maps...`

Map filename keywords supported:
- base color: `basecolor`, `albedo`, `color`, `diffuse`
- roughness: `roughness`, `rough`
- normal: `normalgl`, `normal`, `nor`
- height/displacement: `height`, `displacement`, `disp`
