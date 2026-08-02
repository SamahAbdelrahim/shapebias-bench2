# Archived scripts

Moved here 2026-07-31 during repo cleanup. Nothing in the active pipeline imports these.

- `run_evaluation.py` — old unified runner, superseded by `run_local.py` + `run_remote.py`.
- `run_local_playground_smoke.py`, `run_smolvlm_3trials.sbatch`, `run_smoke_dual_path.sbatch` — early smoke tests, superseded by `playgrounds/smoke_test_playground.py`.
- `run_local.sh` — wrapper pointing at the retired `hackathon` conda env.
- `debug_bias.py`, `verify_visual.py`, `run_bias_decomposition.py`, `analyze_side_bias.py` — one-off March/April diagnostics.
- `compute_order_bias_validity.py`, `update_models_validity_md.py` — write to `interpret/` paths that now live under `interpret/archive/`.
- `build_vision_vs_language_report.py` — one-shot builder for `vision_vs_language_2026-07-17.html` (output kept in results).
- `run_*_30.sbatch` (qwen ladders, no_word_category reruns, noun_label_ab_shiple, sudo_word_generality, prompt_compare, numeric_and_qwen8, smith_ladder, embedding_fill) — historical FarmShare jobs pinned to July 17-25 sessions. AB outputs from these were archived in `results/playground.results/_archived_ab_label_mismatch_pre_2026-07-28/`; the current jobs are `run_ab_label_fix_*.sbatch`.
