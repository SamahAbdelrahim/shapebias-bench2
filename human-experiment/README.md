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
- `stim_pkg`: optional explicit stimulus package override; if omitted, the app auto-selects by `design`
- `ordering`: in `human_friendly` default is `random`; in `benchmark` default is `both`
- `trial_limit`: in `human_friendly` default is `30`; in `benchmark`, `0` means full set
- `shuffle`: `1` (default) or `0`
- `sudo_threshold`: pseudo-word English-transition threshold in `[0,1]` (default `0.62`); applies only to sudo words in `human_friendly`
- `cc`: completion code override

## Trial construction

The experiment reads:

`stimuli_pipe/<stim_pkg>/<stim_set>/manifest.csv` (repo root)

Required structure under repo-root `stimuli_pipe/`:

- `stimuli_per_stl_packages/<stim_set>/<stl_id>/{reference,shape_match,texture_match}.png` (benchmark)
- `stimuli_unique_texture_per_stl_v1/<stim_set>/<stl_id>/{reference,shape_match,texture_match}.png` (human)
- `stimuli_unique_texture_per_stl_v2/<stim_set>/<stl_id>/{reference,shape_match,texture_match}.png` (human)

For each stimulus, design behavior depends on mode:

- `human_friendly` (default):
  - participant is assigned one texture-unique package at random (deterministically from `PROLIFIC_PID|STUDY_ID|SESSION_ID`): `stimuli_unique_texture_per_stl_v1` or `stimuli_unique_texture_per_stl_v2`,
  - one exposure per selected object (no model-style exhaustive Cartesian repetition),
  - one unique generated word per trial for noun-label condition (no word reused across different objects),
  - random left/right mapping by default, logged via `ordering`, `a_is`, `b_is`.

- `benchmark`:
  - uses `stimuli_per_stl_packages` (model-evaluation package),
  - model-style full expansion over 10 fixed benchmark words and counterbalanced orderings.

Default benchmark full condition with `ordering=both` is:

`30 stimuli x 10 words x 2 orderings = 600 trials`.

Use `trial_limit` for pilots.

## Word-list analysis utility

You can reproduce the generated human-friendly word list for a participant and
estimate how English-like each word is (character transition probabilities).

From `human-experiment/`:

```bash
npm run analyze:words -- --prolific_pid P1 --study_id S1 --session_id T1
```

Optional flags:

- `--stim_set` (default: `stimuli_A_auto_contrast`)
- `--stim_pkg` (default: deterministic v1/v2 package choice)
- `--condition` (default: `noun_label`)
- `--trial_limit` (default: `30`)

Outputs are written to `human-experiment/reports/`:

- `human_words_analysis_<...>.csv` (per-word scores/classification)
- `human_words_analysis_<...>.json` (summary counts and thresholds)
