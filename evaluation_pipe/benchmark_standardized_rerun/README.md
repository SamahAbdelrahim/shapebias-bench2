# Benchmark rerun — standardized prompts (isolated)

This folder holds **copies** of the benchmark runners with prompts aligned to **human-matched remote**: the same **system** line (`REMOTE_UNIFORM_SYSTEM_PROMPT`) and **user** templates (`make_prompt` / `PROMPT_TEMPLATES` noun-label) as in [`evaluation_pipe/eval_core.py`](../eval_core.py). Legacy [`scripts/run_local.py`](../../scripts/run_local.py) and [`scripts/run_remote.py`](../../scripts/run_remote.py) are **unchanged**.

## Why

Early word-benchmark **local** runs used family-specific system strings (e.g. Qwen3.5 `"Answer concisely…"`). Remote runs were later unified (see [`interpret/remote_eval_prompt_policy.md`](../../interpret/remote_eval_prompt_policy.md)). This package lets you regenerate **benchmark** CSVs under one prompt policy without overwriting historical outputs.

## Outputs (new directory only)

Defaults write to:

`results/model.results/benchmark_standardized_rerun/`

- `local_eval_standardized.csv` — local GPU path  
- `remote_all_fixed_standardized.csv` — HF router / API path  

Override with `-o` / `--output` if you want timestamped or alternate paths.

## Scripts

From **repository root**:

```bash
# Local (needs CUDA GPU + HF cache for weights)
python evaluation_pipe/benchmark_standardized_rerun/run_local_benchmark_standardized.py \
  --models all --ordering both --repeats 1

# Remote (needs HF_TOKEN / HF_API_TOKEN / etc.)
python evaluation_pipe/benchmark_standardized_rerun/run_remote_benchmark_standardized.py \
  --models all --ordering both --workers 8
```

See [`MODELS_RUNBOOK.md`](MODELS_RUNBOOK.md) for per-model IDs and local vs remote notes.

## Eleven models (full word-benchmark matrix)

These are the models that appear in the merged word-benchmark validity table (`results/data/model_validity_summary_word.csv`):

| # | Registry key | Typical path |
|---|--------------|--------------|
| 1 | `internvl` | Local only (HF router often unsupported for this VLM) |
| 2 | `llama4-scout` | Remote (Groq via HF router) |
| 3 | `qwen3-vl-2b` | Local + remote |
| 4 | `qwen3-vl-4b` | Local + remote (optional dedicated endpoint env) |
| 5 | `qwen3.5-0.8b` | Local + remote |
| 6 | `qwen3.5-4b` | Local + remote |
| 7 | `smolvlm` | Local + remote |
| 8 | `qwen3.5-9b` | Remote |
| 9 | `qwen3.5-27b` | Remote |
| 10 | `qwen3.5-35b-a3b` | Remote |
| 11 | `qwen3.5-122b-a10b` | Remote |

- **`--models all`** on the **local** script runs every key in this list that is registered for local inference (six models; no Llama / large Qwen3.5 here).  
- **`--models all`** on the **remote** script runs all **ten** entries in the copied remote registry (InternVL is omitted until you add a served ID).

## Human-matched

Human-matched stimuli and validity live elsewhere: [`interpret/human_matched_validity.md`](../../interpret/human_matched_validity.md). These scripts use **`--eval-mode`-style benchmark** data only (default path in original `run_remote`); the remote copy here is **benchmark-only** for a smaller file.
