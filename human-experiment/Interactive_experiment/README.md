# shapebias-bench

A benchmark for evaluating shape bias in CNNs and VLMs, with pipelines for stimulus generation, model evaluation, human data collection, and analysis.

## Repository layout


| Folder / path       | Purpose                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `stimuli_pipe/`     | Create rendered stimulus packages (PNG manifests from STL sources)                       |
| `evaluation_pipe/`  | Model evaluation pipeline (local + remote VLMs)                                          |
| `analysis_pipe/`    | R analysis, plots, validity gates                                                        |
| `human-experiment/` | Prolific-ready human studies (jsPsych + Express + MongoDB)                               |
| `configs/`          | Experiment configs (`experiment_default.yaml`, `prompts.yaml`, `comparison_survey.json`) |
| `stl/`              | Raw STL meshes for the **complexity comparison survey**                                  |
| `glb/`              | Raw GLB/GLTF meshes for the **complexity comparison survey** (textured)                  |
| `scripts/`          | Batch runners, debugging, export utilities                                               |


```
shapebias-bench/
├── README.md
├── configs/
│   ├── experiment_default.yaml
│   ├── prompts.yaml
│   ├── comparison_survey.json      # human complexity survey pairings
│   └── COMPARISON_SURVEY.md
├── stl/                            # survey 3D objects (STL)
├── glb/                            # survey 3D objects (GLB, when added)
├── stimuli_pipe/                   # rendered benchmark stimuli
├── evaluation_pipe/
│   ├── models/
│   │   ├── base.py
│   │   ├── local_models/           # smolvlm, internvl, tinyllava, qwen, …
│   │   └── provider_models/
│   └── README.md
├── human-experiment/               # human studies (see below)
├── analysis_pipe/
└── scripts/
```



## Human experiments

This repo supports two human-facing tasks:

### 1. Complexity comparison survey (3D objects)

Participants see pairs of interactive 3D models (STL/GLB) and choose which looks more **complex**.

**Quick start (no MongoDB):**

```bash
cd human-experiment
npm install
npm run preview
```

Open **[http://localhost:3041](http://localhost:3041)**

**Configure pairings:** edit `configs/comparison_survey.json`

- `"pair_mode": "random"` — auto-pair files from `stl/` and `glb/`
- `"pair_mode": "fixed"` — specify exact matchups in `fixed_pairs`

Full docs: `[human-experiment/README.md](human-experiment/README.md)` and `[configs/COMPARISON_SURVEY.md](configs/COMPARISON_SURVEY.md)`

### 2. Shape-bias 3-image 2AFC (rendered PNG stimuli)

The original shape-bias task uses pre-rendered reference / shape-match / texture-match images from `stimuli_pipe/`. The server still exposes `/api/stimuli` for manifest-based trials; see `[human-experiment/HUMAN_PROTOCOL_RATIONALE.md](human-experiment/HUMAN_PROTOCOL_RATIONALE.md)` for protocol details.

**Production run (MongoDB logging):**

```bash
cd human-experiment
npm install
npm start
```



## Model evaluation

Remote evaluation (benchmark vs human-matched stimuli/words) is documented in `[evaluation_pipe/README.md](evaluation_pipe/README.md)`.

Model validity, selection rationale, and local vs remote prompt parity:

- `interpret/model_choice_decision_log.md` (if present in your checkout)
- `interpret/models_validity.md`



## Stimuli onboarding (benchmark PNG packages)

For the **rendered image benchmark** (not the live 3D survey):

1. Start with `[stimuli_pipe/STIMULI_GUIDE.md](stimuli_pipe/STIMULI_GUIDE.md)`
2. Then read:
  - `[stimuli_pipe/stimuli_repro_bundle/README.md](stimuli_pipe/stimuli_repro_bundle/README.md)`
  - `[stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md](stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md)`
3. For benchmark use, point to `stimuli_pipe/stimuli_per_stl_packages` only.

The root `stl/` and `glb/` folders are separate: they feed the **interactive complexity survey** directly, without going through the Blender render pipeline.

## No-word diagnostic trio

If you need the finalized no-word vs noun-label comparison for the 3 diagnostic models:

- Summary report: `interpret/no_word_trio_interim_report.md`
- Model-level contrast table: `results/model.results/no_word_trio_interim_contrast.csv`
- Full deduplicated trio no-word run: `results/model.results/no_word_full_remote_trio_dedup.csv`

Rationale for why these 3 models were selected:

- `interpret/discussion_draft.md` (see "Staged evidence strategy and diagnostic trio rationale")
- `interpret/results_narrative_draft.md` (diagnostic trio selection under budget constraints)



## Model validity (all local + remote)

For the latest validity-gate rerun and narrative interpretation across all current models:

- `interpret/models_validity.md` (after updating the CSV, run `python scripts/update_models_validity_md.py` to refresh the auto-generated failure table)
- `results/data/model_validity_summary.csv`



## Environment

- **Node.js** — `human-experiment/` (see `human-experiment/package.json`)
- **Python** — `evaluation_pipe/`, `stimuli_pipe/`, `scripts/` (see `requirements.txt`)
- **R** — `analysis_pipe/` (see `analysis_pipe/renv.lock`)

Copy `[.env.example](.env.example)` for optional env vars (`MONGO_URI`, `PORT`, etc.).