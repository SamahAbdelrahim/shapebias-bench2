# Model choice and local vs remote evaluation — decision log

This note records the reasoning used when choosing which models to trust for shape-bias interpretation, how that relates to local vs remote runners, and what was changed so **local and remote use the same vision prompt layout**. It is meant for traceability (why we gated, what we run next, what still differs between pipelines).

## 1. Where the validity labels come from

- **Source tables:** `results/data/model_validity_summary.csv` and the narrative in `interpret/models_validity.md` (failure-analysis table and counts are regenerated from the CSV via `python scripts/update_models_validity_md.py`).
- **Inputs:** Combined, de-duplicated model evaluation rows from `results/model.results/` (local + remote CSVs), as described in `models_validity.md`.
- **Rules (from `analysis_pipe/src/validity_gates.R`):**
  - **Valid:** image-tracking rate ≥ 0.70, word-sensitivity rate ≥ 0.20, parse quality ≥ 0.97.
  - **Borderline:** image-tracking ≥ 0.50 but not all valid thresholds met.
  - **Invalid:** otherwise.

**Interpretation:** Shape-bias metrics are only straightforward to interpret when the model (1) follows which image is which when left/right order swaps, (2) is not oblivious to the word manipulation, and (3) usually returns a parseable “1” or “2”. The gates encode that minimum behavioral sanity.

## 2. Outcome of the last gate (11 models)

As of the update summarized in `interpret/models_validity.md`:

| Label        | Model(s)        | Role in planning                                      |
|-------------|-----------------|--------------------------------------------------------|
| **Valid**   | `qwen3-vl-4b`   | Primary candidate for interpretation and new runs      |
| **Borderline** | `qwen3.5-4b` | Useful but treat with caution; re-check after prompt/API parity |
| **Invalid** | Remaining 9     | Do not use as primary evidence until fixed or re-gated |

**Observation:** The two best-scoring registry names are **local** keys (`scripts/run_local.py` / `evaluation_pipe/models/local_models/`). The **remote** registry (`scripts/run_remote.py` → `REMOTE_MODELS`) listed different checkpoints (e.g. larger Qwen3.5 text-oriented IDs, Llama 4 Scout), which **are not the same weights** as `qwen3-vl-4b` or `qwen3.5-4b`.

## 3. Decision: what to run remotely next

1. **Prefer the same Hugging Face IDs as the valid/borderline local models** if the goal is “same model, different serving path”:
   - Local **valid:** `Qwen/Qwen3-VL-4B-Instruct` (`qwen3-vl-4b`).
   - Local **borderline:** `Qwen/Qwen3.5-4B` (`qwen3.5-4b`).
2. **Add those as new keys** in `REMOTE_MODELS` (not done automatically in this doc; model IDs must match what the Hugging Face router actually serves for your account).
3. **Re-run the validity pipeline** on the new remote CSVs. Passing the gate remotely is not guaranteed even for the same weights (see §6).

**Broader coverage (families):** After Tier A (parity with valid/borderline IDs), it is useful to add **one strong model per family** for generalization: e.g. Qwen2.5-VL, InternVL, Llama 3.2 vision (if available on the same API), etc. That is a scientific choice (diversity vs cost), not a validity requirement.

### 3.1 Former local models on the Hugging Face router

[`REMOTE_MODELS`](../scripts/run_remote.py) includes the **same Hugging Face IDs** as the local wrappers for `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3.5-0.8b`, `qwen3.5-4b`, and `smolvlm`, so you can run human-matched (or benchmark) **remotely** when your account can serve them. If the router returns **`model_not_supported` (400)**, enable the right **Inference / partner provider** for that checkpoint in Hugging Face settings, or keep using [`run_local.py`](../scripts/run_local.py). **InternVL** is not in the default registry — `OpenGVLab/InternVL3-1B-hf` is commonly unavailable on the default router; add an entry when you have a served ID.

## 4. Remote system prompt (uniform, task-aligned)

For **remote** API runs, all models share one **`system`** line so cross-model comparison is not confounded by family-specific priors. Options considered, reasoning, and the **exact frozen string** are documented in **`interpret/remote_eval_prompt_policy.md`**.

**Local** Qwen3.5 still uses `QWEN35_VLM_SYSTEM_PROMPT` in `qwen35.py` unless you deliberately align local with remote.

## 5. Prompt parity fix (local vs remote)

### 5.1 What was wrong before

- **Local VLMs** (`qwen`, `qwen35`, `internvl`, `smolvlm`) built a user turn with explicit text labels **before each image** (`Reference image:`, `Image 1:`, `Image 2:`) and appended the task prompt from `eval_core.make_prompt` at the end.
- **Remote** (`run_remote.build_messages`) sent **three images with no labels**, then the same task prompt.

So for the **same** HF weights, the **tokenized input could differ**, changing answers and validity.

### 5.2 What we standardized (single source of truth)

In `evaluation_pipe/eval_core.py`:

- `VISION_USER_IMAGE_LABELS` — the three caption lines (must stay aligned with `[ref, option_a, option_b]` from `run_trial`).
- `build_transformers_vision_user_content` — used by local wrappers.
- `build_openai_compatible_vision_messages` — used by remote `run_remote` (with `image_to_base64_url`).
- **`QWEN35_VLM_SYSTEM_PROMPT`** — used by **local** `qwen35.py` only (historical line).
- **`REMOTE_UNIFORM_SYSTEM_PROMPT`** — one **task-aligned** system line for **every** `REMOTE_MODELS` entry (remote only); see `interpret/remote_eval_prompt_policy.md`. This decouples remote cross-model comparability from local Qwen3.5’s historical system text.

After this change, **task text** still comes only from `eval_core.make_prompt` (noun vs no-word conditions); only the **framing around the three images** is aligned.

## 6. Why remote vs local can still differ (even after prompt parity)

Even with the same model ID and aligned prompts, expect **residual non-equivalence**:

| Factor | Local (`run_local`) | Remote (`run_remote`) |
|--------|---------------------|------------------------|
| **Pixels** | Full-resolution PIL tensors into the processor | Images JPEG-compressed, resized to max side 768 before base64 (HTTP limits) |
| **Stack** | Your `transformers` + local `generate` | Provider’s server build, possibly different `transformers`/CUDA |
| **Chat template** | `apply_chat_template` / processor on your machine | Router’s OpenAI-compatible path may apply templates slightly differently |
| **Decoding** | `do_sample=False`, greedy local | API `temperature=0` (not always bitwise-identical to local greedy) |
| **Thinking / extras** | Qwen3.5: `enable_thinking=False` in local template | Remote: `extra_body` for Qwen when supported; provider may ignore |
| **Token cap** | `max_new_tokens=128` | `max_tokens=128` (analogous but not always identical accounting) |

**Implication:** Treat “local valid” as evidence about **your local stack**. After adding the same ID remotely and aligning prompts, **recompute validity on remote outputs** before trusting remote shape-bias numbers for publication.

## 7. Files touched for traceability

| Area | File(s) |
|------|---------|
| Validity narrative | `interpret/models_validity.md`, `results/data/model_validity_summary.csv` |
| Shared vision layout + Qwen3.5 system string | `evaluation_pipe/eval_core.py` |
| Remote messages | `scripts/run_remote.py` (`REMOTE_MODELS` + `build_messages`) |
| Local VLMs | `evaluation_pipe/models/local_models/qwen.py`, `qwen35.py`, `internvl.py`, `smolvlm.py` |
| Runbook | `evaluation_pipe/README.md` |

## 8. Related docs

- Human vs benchmark stimuli: `human-experiment/HUMAN_PROTOCOL_RATIONALE.md`
- Human-matched model eval mode: `evaluation_pipe/README.md`
