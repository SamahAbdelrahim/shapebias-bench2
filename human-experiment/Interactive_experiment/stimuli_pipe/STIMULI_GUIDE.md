# Stimuli Quick Start Guide

This guide is the top-level entrypoint for working with stimuli in this repository.

## Start Here

Read these two documents first (in this order):

1. `stimuli_pipe/stimuli_repro_bundle/README.md`
2. `stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md`

They explain the generation scripts, naming conventions, and expected data layout.

## Which Stimuli Set To Use

For benchmark usage, use only:

- `stimuli_pipe/stimuli_per_stl_packages`

Do not use other stimuli folders directly as benchmark inputs.

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

## Suggested Workflow

1. Read the two repro-bundle docs listed above.
2. If you need regeneration, run scripts from `stimuli_pipe/stimuli_repro_bundle`.
3. For experiments/evaluation, point data loading to `stimuli_pipe/stimuli_per_stl_packages` only.
