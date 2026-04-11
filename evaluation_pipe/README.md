# Evaluation pipe

Shared logic: [`eval_core.py`](eval_core.py). Runners: **`scripts/run_remote.py`** (API), **`scripts/run_local.py`** (GPU, benchmark stimuli only).

- `scripts/run_remote.py` — API / Hugging Face router models (registry includes former **local-only** VLMs where the router serves them: `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3.5-0.8b`, `qwen3.5-4b`, `smolvlm`, plus larger Qwen3.5 / Llama4 Scout). **`internvl` is not registered remotely** by default — `OpenGVLab/InternVL3-1B-hf` often returns `model_not_supported` on the HF router; add a `REMOTE_MODELS` entry when your account exposes a compatible ID.
- `scripts/run_local.py` — local GPU models (`stimuli_per_stl_packages` benchmark only)

### Using levante-bench runtime in local runs

You can route local ShapeBias inference through the sibling `levante-bench` runtime:

```bash
pip install -e <path-to-levante-bench>
python scripts/run_local.py \
  --models levante-runtime \
  --levante-model-name qwen35 \
  --ordering shape_first
```

Optional flags:
- `--levante-model-config-path` to pass a custom model YAML
- `--levante-configs-root` to resolve `models/*.yaml` from a custom config tree

For complete setup/troubleshooting, see `../LEVANTE_RUNTIME_INTEGRATION.md`.

## Hugging Face Inference Providers (why some `model_id`s work and others don’t)

`run_remote.py` talks to the **OpenAI-compatible router** (`router.huggingface.co/v1` for most keys, and `router.huggingface.co/groq/openai/v1` for `llama4-scout`). The router does **not** run every Hugging Face checkpoint itself: it **dispatches** each request to **Inference Providers** you have set up (API keys and/or HF-billed routing).

What your **Providers routing mode** page means in practice:

- **Custom provider API key:** If you add a key for Groq, Together, Featherless, etc., calls that the router assigns to that provider use **your** key (on the website, widgets, and the same router API).
- **No custom key:** Requests can still go **through HF’s routing** and be **billed to your HF account** (credit card + spend limit), **if** that provider/model pair is available that way.
- **`model_not_supported` (400):** No provider in **your** enabled + billed setup currently exposes that exact model on the router. It is not a bug in the repo; the checkpoint is simply not routable for your account yet.

**How to unblock a vision model (e.g. `Qwen/Qwen3-VL-4B-Instruct`, SmolVLM):**

1. Open the **model card** on Hugging Face → **Inference** / **Deploy** and check which **Inference Providers** are listed for that model.
2. Enable or add billing/API access for **one of those providers** (often **HF Inference API** for first-party HF hosting, or a partner that lists that model).
3. Probe from the repo root. **Prefer the full Conda interpreter path** (works even when `python` is missing or the shell is `sh`):

```bash
~/miniconda3/envs/r-env/bin/python scripts/run_remote.py --eval-mode human_matched \
  --stim-pkg stimuli_unique_texture_per_stl_v1 \
  --stim-set stimuli_A_auto_contrast \
  --ordering shape_first \
  --models qwen3-vl-4b \
  --trial-limit 1 \
  -o /tmp/probe_qwen3vl.csv
```

If the command prints **`model_not_supported` (400)** but otherwise runs, the script is fine — enable an Inference Provider for that model on Hugging Face (see above). If you see **`source: not found`**, the session is **`/bin/sh`**: `source` is **bash-only**. Open a **bash** terminal, or avoid `conda activate` and keep using the **`.../envs/r-env/bin/python`** path above.

Your **Last used** timestamps (Groq, Novita, Together, Featherless, etc.) line up with why **some** Qwen3.5 and **Groq Llama** runs already work: those providers are active. **Qwen3-VL** and **SmolVLM** typically need a **different** provider row (or **HF Inference API**) to show as available for that specific model.

**Dedicated Inference Endpoint:** If you deploy your own replica, set **`HF_ENDPOINT_QWEN3_VL_4B_BASE_URL`** in `.env` to the host from the endpoint page (example in [`.env.example`](../.env.example)). That sends `qwen3-vl-4b` calls to your URL instead of the public router. If the API returns **`Your endpoint is in error, check its status on endpoints.huggingface.co`**, the replica failed to start or crashed — open that endpoint in the HF dashboard, read **Logs**, fix the image/config (custom non-catalog models often need extra env or a larger instance), then **Restart**; our script is only surfacing HF’s status.

## Python environment for `run_remote.py`

**Why an automated run can “fail”:** `run_remote.py` only needs a small set of packages (`Pillow`, `python-dotenv`, `openai`). Common issues: (1) **`python: not found`** — on many Linux images only **`python3`** exists, or use **`~/miniconda3/envs/r-env/bin/python`**. (2) **`source: not found`** — **POSIX `sh`** (sometimes Cursor’s default) has no `source`; use **bash** or skip activation with the full **`envs/r-env/bin/python`** path. (3) **Wrong env** — `/usr/bin/python3` may lack deps; use Conda `r-env` or a venv with `requirements.txt`.

**Ways to run reliably (pick one):**

| Approach | When to use |
|----------|----------------|
| **Conda env** (e.g. `conda activate r-env`, then `python`) | You already use Miniconda; same env as before for diagnostic runs. Ensure `pip install python-dotenv openai` (or full `requirements.txt`) in that env. |
| **apt + venv** | `sudo apt install python3-pip python3-venv`, then `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Good for a clean, project-local env. |
| **`uv`** | [astral.sh/uv](https://github.com/astral-sh/uv): install without system `pip`; `uv venv && uv pip install -r requirements.txt`. |
| **`get-pip.py`** | Bootstrap pip if you cannot use apt (less ideal but works on bare Python). |
| **Explicit path** | Call the interpreter you know has deps, e.g. `~/miniconda3/envs/r-env/bin/python scripts/run_remote.py ...`. |
| **Docker / other machine** | Reproducible CI or laptop with a known image. |

[`scripts/run_human_matched_remote_batch.sh`](../scripts/run_human_matched_remote_batch.sh) uses **`$PYTHON` if set**, else **`$CONDA_PREFIX/bin/python`** when Conda is activated, else **`~/miniconda3/envs/r-env/bin/python`** if present, else `python3`.

## Benchmark mode (default)

Uses `stimuli_pipe/stimuli_per_stl_packages/<stim_set>/` and fixed `WORD_PAIRS` (same as model-style benchmark in the human app’s `design=benchmark`).

```bash
python3 scripts/run_remote.py --models qwen3.5-9b --ordering both
```

CSV rows include `eval_mode=benchmark` and `stim_pkg=stimuli_per_stl_packages`.

## Human-matched mode (`human_friendly` images + words)

Uses the **same** stimulus packages as the human experiment (`stimuli_unique_texture_per_stl_v1` or `v2`), manifest row order, and the same pseudo-word rules as `human-experiment/public/experiment.js` (including `sudo_threshold`, `word_mode`, length band). Stimulus subsampling and word generation use deterministic seeds derived from `--human-eval-seed`, `stim_set`, `stim_pkg`, `condition`, and `word_mode`, with `|stimuli` / `|words` suffixes matching the browser.

**Not replicated from the browser (by design):** unseeded per-trial ordering and trial shuffle. Choose `--ordering` explicitly (e.g. `both` for two API calls per stimulus–word pair).

```bash
python3 scripts/run_remote.py \
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

**Validity (human-matched):** [`scripts/compute_validity_human_matched.py`](../scripts/compute_validity_human_matched.py) writes `results/data/model_validity_summary_human_matched.csv` (v1+v2 combined, with protocol-specific gates). Refresh [`interpret/human_matched_validity.md`](../interpret/human_matched_validity.md) via `scripts/update_models_validity_md.py --skip-word-sensitivity-gate` (see that file for the exact command).

See also `human-experiment/HUMAN_PROTOCOL_RATIONALE.md` for how human vs benchmark designs differ.

## Summary table (`print_summary`)

After a run, stdout includes **Shape%(dec)** = shape / (shape + texture) and **Shape%(all)** = shape / (shape + texture + unclear).

## Vision prompt parity (local vs remote)

Local VLMs and `run_remote.py` share the same **image-slot labels** (`Reference image:`, `Image 1:`, `Image 2:`) and the same task string from `eval_core.make_prompt`.

**System message:** **Remote** runs use one uniform **`REMOTE_UNIFORM_SYSTEM_PROMPT`** for every `REMOTE_MODELS` entry (task-aligned 1/2 output); **local** Qwen3.5 still uses **`QWEN35_VLM_SYSTEM_PROMPT`**. Rationale and exact wording: **`interpret/remote_eval_prompt_policy.md`**.

Remote calls still **resize/JPEG-encode** images (max side 768) for upload; local runs use raw PIL inputs — so numerical parity is not guaranteed even for the same Hugging Face model ID. See **`interpret/model_choice_decision_log.md`** for the full decision trail and residual differences.

## Model selection rationale

See **`interpret/model_choice_decision_log.md`** for validity-gate interpretation, which checkpoints were valid/borderline, and how that maps to choosing remote models.
