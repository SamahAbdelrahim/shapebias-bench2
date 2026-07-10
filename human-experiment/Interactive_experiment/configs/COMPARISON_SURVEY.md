# Comparison survey configuration

Edit `configs/comparison_survey.json` to control which 3D models are shown in the human complexity comparison task.

## Model folders

- `stl/` — untextured STL meshes (currently populated)
- `glb/` — textured GLB/GLTF models (supported when files are added)

Both folders are served by the human experiment server and scanned automatically.

## Pair modes

### Random (default)

```json
{
  "pair_mode": "random",
  "trial_count": 15,
  "shuffle_trials": true,
  "allow_repeat_pairs": false,
  "model_sources": ["stl", "glb"]
}
```

Random mode builds unique pairs from all available models. Pair order is deterministic per participant (`PROLIFIC_PID|STUDY_ID|SESSION_ID`), and left/right placement is counterbalanced per trial.

### Fixed pairs

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

Use fixed mode when you want exact matchups. Filenames can be bare names (matched across `stl/` and `glb/`) or include an explicit source.

## Prompt text

Change the trial question with:

```json
{
  "prompt": "Which object looks more complex?"
}
```

## Run locally

From `human-experiment/`:

```bash
npm run preview
```

Open `http://localhost:3041`.
