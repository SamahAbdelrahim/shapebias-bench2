# Shape Bias Human Experiment (jsPsych)

This folder contains a Prolific-ready human experiment that mirrors the benchmark
task used for model evaluation: 3-image 2AFC with one reference image and two
candidates (shape-match vs texture-match), counterbalanced.

## What is included

- `server.js`: Express server + MongoDB logging endpoint
- `models/shapebias-human-logger.js`: Mongoose schema for trial-level rows
- `public/index.html`: jsPsych experiment page
- `public/experiment.js`: trial generation, prompts, response coding, logging
- `public/experiment.css`: basic layout/styling

## Install and run

From this folder:

```bash
npm install
npm start
```

The app runs on `http://localhost:3041` by default.

## Encapsulation guarantee

Runtime dependencies are contained in this repository, with experiment runtime rooted in `human-experiment/`:

- Server code, model schema, frontend assets, and logging endpoint are all local.
- jsPsych core/plugins are loaded from local `node_modules` via `/vendor/*`.
- Mongo credentials are read only from `human-experiment/mongo_auth.json` (or `MONGO_URI` env var).
- Stimuli are read from the canonical repo dataset at `stimuli_pipe/*` (repo root), matching model evaluation inputs.

No runtime file reads from other repositories (for example `perceiving_complexity`) or any path outside this repo.

## MongoDB setup

Server connection priority:

1. `MONGO_URI` env var (full URI), else
2. `human-experiment/mongo_auth.json`

Credential JSON should include `username` and `password`.

Optional env vars:

- `MONGO_DB` (default: `samah`)
- `MONGO_HOST` (default: `127.0.0.1`)
- `MONGO_PORT` (default: `27017`)
- `PORT` (default: `3041`)
- `DEFAULT_STIM_SET` (default: `stimuli_A_auto_contrast`)
- `PROLIFIC_COMPLETION_CODE` (default: `TESTCODE`)

## Experiment URL parameters

- `PROLIFIC_PID`, `STUDY_ID`, `SESSION_ID` (standard Prolific params)
- `design`: `human_friendly` (default) or `benchmark`
- `condition`: `noun_label` (default) or `no_word_category`
- `stim_set`: e.g. `stimuli_A_auto_contrast`
- `ordering`: in `human_friendly` default is `random`; in `benchmark` default is `both`
- `trial_limit`: in `human_friendly` default is `30`; in `benchmark`, `0` means full set
- `shuffle`: `1` (default) or `0`
- `cc`: completion code override

## Trial construction

The experiment reads:

`stimuli_pipe/stimuli_per_stl_packages/<stim_set>/manifest.csv` (repo root)

Required structure under repo-root `stimuli_pipe/`:

- `stimuli_per_stl_packages/stimuli_A_auto_contrast/<stl_id>/reference.png`
- `stimuli_per_stl_packages/stimuli_A_auto_contrast/<stl_id>/shape_match.png`
- `stimuli_per_stl_packages/stimuli_A_auto_contrast/<stl_id>/texture_match.png`
- same pattern for `stimuli_B_controlled_simple/`

For each stimulus, design behavior depends on mode:

- `human_friendly` (default):
  - one exposure per selected object (no model-style exhaustive Cartesian repetition),
  - one unique generated word per trial for noun-label condition (no word reused across different objects),
  - random left/right mapping by default, logged via `ordering`, `a_is`, `b_is`.

- `benchmark`:
  - model-style full expansion over 10 fixed benchmark words and counterbalanced orderings.

Default benchmark full condition with `ordering=both` is:

`30 stimuli x 10 words x 2 orderings = 600 trials`.

Use `trial_limit` for pilots.
