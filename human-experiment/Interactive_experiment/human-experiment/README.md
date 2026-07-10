# Complexity Comparison Survey

A Prolific-ready jsPsych experiment where participants compare pairs of 3D objects (STL or GLB) and choose which looks more complex.

Part of **[shapebias-bench](../README.md)** — see the root README for the full repo (model evaluation, `stimuli_pipe`, analysis, and the original shape-bias 2AFC task).

## Quick start (local preview)

No MongoDB required.

```bash
cd human-experiment
npm install
npm run preview
```

Open **http://localhost:3041**

## Production run (with MongoDB logging)

```bash
cd human-experiment
npm install
npm start
```

The app runs on **http://localhost:3041** by default.

### MongoDB setup

Server connection priority:

1. `MONGO_URI` env var (full URI), else
2. `human-experiment/mongo_auth.json` with `username` and `password`

Optional env vars: `MONGO_DB`, `MONGO_HOST`, `MONGO_PORT`, `PORT`, `PROLIFIC_COMPLETION_CODE`

Copy `.env.example` from the repo root if you use environment variables.

## 3D model files

Place models in the repo root:

| Folder | Format | Notes |
|--------|--------|-------|
| `stl/` | `.stl` | Untextured meshes (gray material) |
| `glb/` | `.glb`, `.gltf` | Textured models (when you add them) |

The server serves both folders and scans them automatically when building trials.

## Configuring comparisons

Edit **`configs/comparison_survey.json`** at the repo root. Restart the server (or refresh the page) after changes.

### Random pairs (default)

```json
{
  "pair_mode": "random",
  "trial_count": 15,
  "shuffle_trials": true,
  "allow_repeat_pairs": false,
  "model_sources": ["stl", "glb"],
  "prompt": "Which object looks more complex?"
}
```

- Builds pairs from all files in `stl/` and `glb/`
- `trial_count` = how many comparisons per participant
- Pair order is **deterministic per participant** (seeded from `PROLIFIC_PID|STUDY_ID|SESSION_ID`)
- Left/right placement is counterbalanced each trial

To switch back to random after using fixed pairs, set `"pair_mode": "random"`.

### Fixed pairs (specific matchups)

```json
{
  "pair_mode": "fixed",
  "shuffle_trials": true,
  "fixed_pairs": [
    { "left": "22 1.stl", "right": "22 10.stl" },
    { "left": "chair.glb", "right": "22 5.stl", "left_source": "glb", "right_source": "stl" }
  ]
}
```

- Each entry is one trial
- Use filenames as they appear in `stl/` or `glb/`
- Add `"left_source"` / `"right_source"` (`"stl"` or `"glb"`) when the same filename could exist in both folders

### Other options

| Field | Purpose |
|-------|---------|
| `model_sources` | Which folders to scan: `["stl"]`, `["glb"]`, or both |
| `shuffle_trials` | Randomize trial order per participant |
| `allow_repeat_pairs` | Allow the same pair twice in random mode |
| `prompt` | Question shown above each pair |

See also: `configs/COMPARISON_SURVEY.md`

## Project layout

```
human-experiment/
  public/
    index.html          # Experiment page
    experiment.js       # jsPsych timeline, interaction gating, logging
    model-viewer.js     # Three.js STL/GLB renderer
    experiment.css      # Styling
  comparison-survey.js  # Trial generation from config + model folders
  server.js             # Production server (MongoDB)
  server.local-preview.js  # Local preview (no MongoDB)
stl/                    # STL files (repo root)
glb/                    # GLB files (repo root)
configs/
  comparison_survey.json
```

## URL parameters

Standard Prolific params: `PROLIFIC_PID`, `STUDY_ID`, `SESSION_ID`

Debug without Prolific:

```
http://localhost:3041/?PROLIFIC_PID=debug&STUDY_ID=debug&SESSION_ID=debug
```

## How a trial works

1. Two 3D objects appear side by side (Object A / Object B)
2. Participant drags each object to explore it
3. Choice buttons unlock after both objects have been dragged (or after a short interaction fallback)
4. Participant clicks **Object A is more complex** or **Object B is more complex**
5. Response is logged to MongoDB (or console in preview mode)

## Related repo docs

| Topic | Location |
|-------|----------|
| Root overview (all pipelines) | [`../README.md`](../README.md) |
| Comparison pair config reference | [`../configs/COMPARISON_SURVEY.md`](../configs/COMPARISON_SURVEY.md) |
| Rendered benchmark stimuli (PNG) | [`../stimuli_pipe/STIMULI_GUIDE.md`](../stimuli_pipe/STIMULI_GUIDE.md) |
| Shape-bias human protocol rationale | [`HUMAN_PROTOCOL_RATIONALE.md`](HUMAN_PROTOCOL_RATIONALE.md) |
| Model evaluation pipeline | [`../evaluation_pipe/README.md`](../evaluation_pipe/README.md) |

**Note:** Root `stl/` and `glb/` folders are for this live 3D survey. The benchmark image packages under `stimuli_pipe/stimuli_per_stl_packages/` are a separate render pipeline used for model evaluation and the legacy 3-image human task.
