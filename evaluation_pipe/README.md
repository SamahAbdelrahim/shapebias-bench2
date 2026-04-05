# Evaluation pipe

Shared logic lives in `eval_core.py`. Entry points:

- `scripts/run_remote.py` — API / Hugging Face router models
- `scripts/run_local.py` — local GPU models (`stimuli_per_stl_packages` benchmark only)

## Benchmark mode (default)

Uses `stimuli_pipe/stimuli_per_stl_packages/<stim_set>/` and fixed `WORD_PAIRS` (same as model-style benchmark in the human app’s `design=benchmark`).

```bash
python scripts/run_remote.py --models qwen3.5-9b --ordering both
```

CSV rows include `eval_mode=benchmark` and `stim_pkg=stimuli_per_stl_packages`.

## Human-matched mode (`human_friendly` images + words)

Uses the **same** stimulus packages as the human experiment (`stimuli_unique_texture_per_stl_v1` or `v2`), manifest row order, and the same pseudo-word rules as `human-experiment/public/experiment.js` (including `sudo_threshold`, `word_mode`, length band). Stimulus subsampling and word generation use deterministic seeds derived from `--human-eval-seed`, `stim_set`, `stim_pkg`, `condition`, and `word_mode`, with `|stimuli` / `|words` suffixes matching the browser.

**Not replicated from the browser (by design):** unseeded per-trial ordering and trial shuffle. Choose `--ordering` explicitly (e.g. `both` for two API calls per stimulus–word pair).

```bash
python scripts/run_remote.py \
  --eval-mode human_matched \
  --stim-pkg stimuli_unique_texture_per_stl_v1 \
  --stim-set stimuli_A_auto_contrast \
  --ordering both \
  --models qwen3.5-9b
```

Options:

| Flag | Default | Role |
|------|---------|------|
| `--trial-limit` | `30` | After deterministic shuffle, keep first *N* stimuli; `0` = all |
| `--human-eval-seed` | `model_eval` | Stand-in for `PROLIFIC_PID\|STUDY_ID\|SESSION_ID` in seedText |
| `--word-mode` | `sudo_only` | `sudo_only` or `mixed` |
| `--word-min-len` / `--word-max-len` | 4 / 8 | Generated word lengths |
| `--sudo-threshold` | `0.62` | English bigram score floor for sudo words |

Default output when `-o` is omitted: `results/model.results/human_matched/remote_human_<timestamp>.csv` (under `$RESULTS_DIR` if set). You can still pass `-o` to any path, including the same folder or a custom name.

To run **v1 and v2** back-to-back for all remote registry models with named CSVs under `results/model.results/human_matched/`, use [`scripts/run_human_matched_remote_batch.sh`](../scripts/run_human_matched_remote_batch.sh) from the repo root (requires `pip install -r requirements.txt` and a Hugging Face token in `.env`).

See also `human-experiment/HUMAN_PROTOCOL_RATIONALE.md` for how human vs benchmark designs differ.

## Summary table (`print_summary`)

After a run, stdout includes **Shape%(dec)** = shape / (shape + texture) and **Shape%(all)** = shape / (shape + texture + unclear).

## Vision prompt parity (local vs remote)

Local VLMs and `run_remote.py` share the same **image-slot labels** (`Reference image:`, `Image 1:`, `Image 2:`) and the same task string from `eval_core.make_prompt`.

**System message:** **Remote** runs use one uniform **`REMOTE_UNIFORM_SYSTEM_PROMPT`** for every `REMOTE_MODELS` entry (task-aligned 1/2 output); **local** Qwen3.5 still uses **`QWEN35_VLM_SYSTEM_PROMPT`**. Rationale and exact wording: **`interpret/remote_eval_prompt_policy.md`**.

Remote calls still **resize/JPEG-encode** images (max side 768) for upload; local runs use raw PIL inputs — so numerical parity is not guaranteed even for the same Hugging Face model ID. See **`interpret/model_choice_decision_log.md`** for the full decision trail and residual differences.

## Model selection rationale

See **`interpret/model_choice_decision_log.md`** for validity-gate interpretation, which checkpoints were valid/borderline, and how that maps to choosing remote models.
