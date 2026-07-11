# Archive

Files that are no longer part of the active pipeline. Nothing here is deleted; it is kept for provenance. This folder is gitignored (local only).

| Item | What it is | Why it was archived |
|------|-----------|---------------------|
| `results_copy/` | Duplicate of three CSVs from `results/model.results/` (formerly `results/results copy/`) | Exact duplicates; the canonical copies live in `results/model.results/`. Archiving this was an open item in `PROJECT_CHECKLIST.md`. |
| `temp_screenshots/` | Two pasted screenshots (formerly `temp/`) | Untitled working screenshots with no reference from any doc or script. |
| `human_experiment_debug/` | `output.log` and one debug run of `analyze_human_words.js` (formerly in `human-experiment/` and `human-experiment/reports/`) | Debug artifacts from local testing; `human-experiment/reports/` remains the live output folder for the word-analysis utility and is now gitignored. |
| `Onboarding.html` | Rendered HTML copy of the RA onboarding doc | The markdown source `interpret/RA_ONBOARDING.md` is the maintained version. |
| `cursor_shape_bias_model_behavior_discus.md` | Exported Cursor chat log (April 2026) about SmolVLM/InternVL side bias | Historically important: this is where the logit-forced (`logit_forced_12`) + swap-correction method was worked out. The method is now documented in `interpret/RA_ONBOARDING.md` and the levante-bench `integrate-shapebias` branch, so the raw chat log is archival. |

Archived on 2026-07-10 during the repo reorganization (see `MEMORY.md` at repo root).
