# Complexity Comparison Survey

Prolific-ready jsPsych experiment where participants compare pairs of 3D objects
(STL or GLB) and choose which looks more complex.

This lives under `human-experiment/Interactive_experiment/` so it does **not**
replace the parent shape-bias 2AFC image experiment in `human-experiment/`.

## Quick start (local preview)

No MongoDB required. Uses parent `human-experiment/node_modules` and
`general_assets` when present; otherwise preview mode falls back to CDN jsPsych.

```bash
cd human-experiment
npm install          # once, for shared deps
cd Interactive_experiment
npm run preview
```

Open **http://localhost:3042** (port 3042 by default so it can run beside the
parent experiment on 3041).

## Production run (MongoDB logging)

```bash
cd human-experiment/Interactive_experiment
npm start
```

Mongo credentials (first match wins):

1. `MONGO_URI` env var
2. `Interactive_experiment/mongo_auth.json`
3. `human-experiment/mongo_auth.json`

Optional env vars: `MONGO_DB`, `MONGO_HOST`, `MONGO_PORT`, `PORT`,
`PREVIEW_PORT`, `PROLIFIC_COMPLETION_CODE`

## 3D model files

Place models inside this folder:

| Folder | Format | Notes |
|--------|--------|-------|
| `stl/` | `.stl` | Untextured meshes (gray material); 22 sample files included |
| `glb/` | `.glb`, `.gltf` | Textured models (optional) |

## Configuring comparisons

Edit `configs/comparison_survey.json`, then restart the server (or refresh).

See `configs/COMPARISON_SURVEY.md` for pair modes and options.

## Layout

```
Interactive_experiment/
  comparison-survey.js      # trial builder
  server.js                 # production (Mongo)
  server.local-preview.js   # UI preview (no Mongo)
  models/                   # mongoose schema (complexity fields)
  public/                   # experiment UI + Three.js model viewer
  configs/                  # comparison_survey.json
  stl/                      # 3D meshes
  glb/                      # optional textured models
```
