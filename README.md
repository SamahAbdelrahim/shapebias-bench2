# shapebias-bench
A benchmark for evaluating the shape bias in CNN and VLM 

Three main folders:
stimuli_pipe to create stimuli + evaluation and model pipeline + analysis_pipe

Stimuli onboarding:
- Start with `STIMULI_GUIDE.md` (repo root/stimuli_pipe)
- Then read:
  - `stimuli_pipe/stimuli_repro_bundle/README.md`
  - `stimuli_pipe/stimuli_repro_bundle/STIMULI_GUIDE.md`
- For benchmark use, point to `stimuli_pipe/stimuli_per_stl_packages` only.

## No-word diagnostic trio: where to find it

If you need the finalized no-word vs noun-label comparison for the 3 diagnostic models:

- Summary report: `interpret/no_word_trio_interim_report.md`
- Model-level contrast table: `results/model.results/no_word_trio_interim_contrast.csv`
- Full deduplicated trio no-word run: `results/model.results/no_word_full_remote_trio_dedup.csv`

Rationale for why these 3 models were selected is documented in:

- `interpret/discussion_draft.md` (see "Staged evidence strategy and diagnostic trio rationale")
- `interpret/results_narrative_draft.md` (diagnostic trio selection under budget constraints)

## Model validity (all local + remote)

For the latest validity-gate rerun and narrative interpretation across all current
models, see:

- `interpret/models_validity.md`
- `results/data/model_validity_summary.csv`


```
shapebias-bench/                                                                                                                                      
  ├── README.md                                                                                                                                                                      
  ├── .gitignore                                                                                                                                      
  ├── configs/
  │   ├── experiment_default.yaml
  │   └── prompts.yaml
  ├── evaluation_pipe/
  │   ├── models/
  │   │   ├── __init__.py
  │   │   ├── base.py
  │   │   ├── local_models/
  │   │   │   ├── __init__.py
  │   │   │   ├── smolvlm.py
  │   │   │   ├── internvl.py
  │   │   │   └── tinyllava.py
  │   │   └── provider_models/
  │   │       ├── __init__.py
  ├── stimuli_pipe/
```