# Using levante-bench exports in shapebias-bench2

This guide documents how to run ShapeBias local evaluation through the exported runtime APIs from `levante-bench` (branch `vlm-eval`).

## What this integration does

- Adds a local model key: `levante-runtime`
- Delegates model loading and trial execution to:
  - `levante_bench.runtime.load_model`
  - `levante_bench.runtime.run_trials`
- Lets ShapeBias keep its own prompts/stimulus flow while reusing LEVANTE model adapters/configs.

## Prerequisites

- Local repos:
  - `/home/david/levante/levante-bench`
  - `/home/david/digital-pro/shapebias-bench2`
- `levante-bench` checked out to branch `vlm-eval`
- A Python environment used for running ShapeBias scripts

## One-time setup

```bash
# 1) Ensure levante runtime branch is active
cd /home/david/levante/levante-bench
git checkout vlm-eval

# 2) Activate the environment you run shapebias from
source /home/david/levante/levante-bench/.venv/bin/activate

# 3) Install shapebias deps
cd /home/david/digital-pro/shapebias-bench2
pip install -r requirements.txt

# 4) Install levante-bench as editable package
pip install -e /home/david/levante/levante-bench
```

## Quick smoke run

```bash
cd /home/david/digital-pro/shapebias-bench2
PYTHONPATH=. python scripts/run_local.py \
  --models levante-runtime \
  --levante-model-name qwen35 \
  --ordering shape_first \
  --num-stimuli 1 \
  --repeats 1
```

## Main usage

```bash
cd /home/david/digital-pro/shapebias-bench2
PYTHONPATH=. python scripts/run_local.py \
  --models levante-runtime \
  --levante-model-name qwen35 \
  --ordering both \
  --repeats 1
```

## Using remote models from levante-bench

You can route ShapeBias trials through remote model configs defined in `levante-bench`, for example:

- `gemini_pro`
- `gpt53` (or `gpt52`)

Example:

```bash
export GEMINI_API_KEY="..."
PYTHONPATH=. python scripts/run_local.py \
  --models levante-runtime \
  --levante-model-name gemini_pro \
  --ordering shape_first \
  --num-stimuli 1
```

```bash
export OPENAI_API_KEY="..."
PYTHONPATH=. python scripts/run_local.py \
  --models levante-runtime \
  --levante-model-name gpt53 \
  --ordering shape_first \
  --num-stimuli 1
```

## Runtime-specific flags

`scripts/run_local.py` accepts:

- `--levante-model-name`
  - Registered LEVANTE model id (`qwen35`, `smolvlm2`, `internvl35`, `qwen25vl_qlora`, etc.)
- `--levante-model-config-path`
  - Optional YAML path with model config (`name`, `hf_name`, and optional overrides)
- `--levante-configs-root`
  - Optional custom configs root containing `models/*.yaml`

## Choosing models

To see available LEVANTE model IDs:

```bash
cd /home/david/levante/levante-bench
source .venv/bin/activate
levante-bench list-models
```

Then pass one in `--levante-model-name`.

## How image data is passed

ShapeBias currently passes image triplets (`reference`, `image1`, `image2`) as PIL images.
The wrapper writes temporary PNGs and sends those paths to `levante_bench.runtime`.
No permanent image copies are created.

## Troubleshooting

- `ModuleNotFoundError: dotenv`
  - Run `pip install -r requirements.txt` in `shapebias-bench2`.
- `ModuleNotFoundError: levante_bench`
  - Run `pip install -e /home/david/levante/levante-bench` in the active env.
- Model key not found
  - Ensure you used `--models levante-runtime` (exact key).
- Wrong model set loaded
  - Confirm `levante-bench` is on branch `vlm-eval` before editable install.
- Import errors when running ad-hoc Python snippets
  - Run from repo root with `PYTHONPATH=.` so `evaluation_pipe` is importable.
