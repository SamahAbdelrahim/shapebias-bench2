# Stimuli Quick Start Guide

This guide is the top-level entrypoint for working with stimuli in this repository.

## Start Here

Read these two documents first (in this order):

1. `stimuli_pipe/stimuli_repro_bundle/README.md`
2. `stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md`

They explain the generation scripts, naming conventions, and expected data layout.

## Which Stimuli Set To Use

Two benchmark inputs, split by purpose:

- `stimuli_pipe/stimuli_per_stl_packages` — the 30-set package. Everything in
  `playgrounds/` uses this, via `IMAGE_DATASET`. Keep testing and smoke runs here.
- `stimuli_pipe/stimuli_texture_grid_v1` — the full texture grid, 30 shapes x 38
  textures = 1,140 trials per mode. Production scale-up only, and only through
  `evaluation_pipe` (`scripts/run_local.py --grid-pkg`). See
  [the grid section](#full-texture-grid-1140-trials) below.

Do not use other stimuli folders directly as benchmark inputs. In particular
`stimuli_unique_texture_per_stl_v1` / `_v2` are the older flat 30-folder
human-matched packages, and despite the similar name they are unrelated to
`stimuli_texture_grid_v1`.

Each per-shape package in `stimuli_per_stl_packages` is benchmark-ready and contains:

- `example_image.png`
- `reference.png`
- `shape_match.png`
- `texture_match.png`

## Canonical Modes

The packaged benchmark stimuli are organized by:

- `stimuli_A_auto_contrast`
- `stimuli_B_controlled_simple`

Both modes live under `stimuli_pipe/stimuli_per_stl_packages/`.

## Full Texture Grid (1,140 trials)

`stimuli_texture_grid_v1` is a symlink to the `triad-stimuli-pipeline` output at
`data/generated_stimuli/stimuli_unique_texture_per_stl_v1/`. It adds a texture
level under each shape, so a trial is `<stl_id>/<texture_set>/`, not `<stl_id>/`:

```
stimuli_texture_grid_v1/stimuli_A_auto_contrast/1/Carpet008_1K-JPG/
    example_image.png  reference.png  shape_match.png  texture_match.png
```

Because of that extra level, the 30-set loaders (`load_trials`, `load_stimuli`)
do not read it: they scan for numbered directories holding images directly.
Grid trials come from `manifest.csv` instead, via `load_stimuli_grid()`, which
returns paths rather than open images (decoding all 1,140 triads at once would
hold roughly 10 GB). Trial identity is `stim_id` = `1/Carpet008_1K-JPG`; result
CSVs also carry `stl_id` and `texture_set` as separate columns.

Run it with:

```bash
sbatch scripts/run_full_grid_v1a.sbatch          # 9 models x 6 cells
python analysis_pipe/full_grid_summary.py        # readout
```

Details, including how to rebuild the scratch stage, are in
`results/model.results/session_full_grid_v1a/README.md`.

## Suggested Workflow

1. Read the two repro-bundle docs listed above.
2. If you need regeneration, run scripts from `stimuli_pipe/stimuli_repro_bundle`.
3. For playground and smoke testing, point data loading to
   `stimuli_pipe/stimuli_per_stl_packages` only.
4. For full-scale evaluation, use `--grid-pkg stimuli_texture_grid_v1` through
   `evaluation_pipe`.
