# Model runbook — standardized benchmark rerun (11 models)

Registry keys match CSV columns and [`scripts/run_remote.py`](../../scripts/run_remote.py) / local wrappers under [`evaluation_pipe/models/local_models/`](../models/local_models/).

**Shared prompts (this package):** system = `REMOTE_UNIFORM_SYSTEM_PROMPT`; user = noun-label `PROMPT_TEMPLATES["noun_label"]` via `run_trial` / `make_prompt` (same as human-matched remote).

---

## 1. `internvl`

| Field | Value |
|--------|--------|
| **HF weights (local default)** | `OpenGVLab/InternVL3-1B-hf` |
| **Local** | Yes — [`internvl.py`](../models/local_models/internvl.py), `AutoModelForImageTextToText`, bf16, `device_map` |
| **VRAM (rule of thumb)** | ~4–8 GB+ for 1B VLM; scale up for larger checkpoints if you change `model_id` |
| **Remote** | Not in default `REMOTE_MODELS`; HF Inference router often returns `model_not_supported` for InternVL. Use a **dedicated Inference Endpoint** or skip. |

---

## 2. `llama4-scout`

| Field | Value |
|--------|--------|
| **HF API / router model id** | `meta-llama/llama-4-scout-17b-16e-instruct` |
| **Provider** | `huggingface-groq` → `https://router.huggingface.co/groq/openai/v1` |
| **Local** | No wrapper in repo |
| **Remote** | Yes — token via `HF_TOKEN` / `HUGGING_FACE` / etc. |

---

## 3. `qwen3-vl-2b`

| Field | Value |
|--------|--------|
| **HF weights** | `Qwen/Qwen3-VL-2B-Instruct` |
| **Local** | Yes — [`qwen.py`](../models/local_models/qwen.py) |
| **VRAM** | ~6–10 GB typical with bf16 |
| **Remote** | Yes — same `model_id` on HF router |

---

## 4. `qwen3-vl-4b`

| Field | Value |
|--------|--------|
| **HF weights** | `Qwen/Qwen3-VL-4B-Instruct` |
| **Local** | Yes — [`qwen.py`](../models/local_models/qwen.py) |
| **VRAM** | ~10–16 GB+ |
| **Remote** | Yes — router default, or **dedicated endpoint** via `HF_ENDPOINT_QWEN3_VL_4B_BASE_URL` (+ optional `HF_ENDPOINT_QWEN3_VL_4B_MODEL`) in `.env` |

---

## 5. `qwen3.5-0.8b`

| Field | Value |
|--------|--------|
| **HF weights** | `Qwen/Qwen3.5-0.8B` |
| **Local** | Yes — [`qwen35.py`](../models/local_models/qwen35.py), `Qwen3_5ForConditionalGeneration` |
| **VRAM** | ~4–8 GB |
| **Remote** | Yes — HF router |

---

## 6. `qwen3.5-4b`

| Field | Value |
|--------|--------|
| **HF weights** | `Qwen/Qwen3.5-4B` |
| **Local** | Yes — [`qwen35.py`](../models/local_models/qwen35.py) |
| **VRAM** | ~8–12 GB |
| **Remote** | Yes — HF router |

---

## 7. `smolvlm`

| Field | Value |
|--------|--------|
| **HF weights** | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| **Local** | Yes — [`smolvlm.py`](../models/local_models/smolvlm.py) |
| **VRAM** | ~6–10 GB |
| **Remote** | Yes — HF router |

---

## 8. `qwen3.5-9b`

| Field | Value |
|--------|--------|
| **HF API model id** | `Qwen/Qwen3.5-9B` |
| **Local** | No registered wrapper in repo |
| **Remote** | Yes — HF router; `enable_thinking=False` passed via `extra_body` on router (not on dedicated endpoints) |

---

## 9. `qwen3.5-27b`

| Field | Value |
|--------|--------|
| **HF API model id** | `Qwen/Qwen3.5-27B:featherless-ai` (router-hosted variant) |
| **Local** | No |
| **Remote** | Yes — verify ID remains valid on your HF account |

---

## 10. `qwen3.5-35b-a3b`

| Field | Value |
|--------|--------|
| **HF API model id** | `Qwen/Qwen3.5-35B-A3B` |
| **Local** | No |
| **Remote** | Yes — HF router |

---

## 11. `qwen3.5-122b-a10b`

| Field | Value |
|--------|--------|
| **HF API model id** | `Qwen/Qwen3.5-122B-A10B` |
| **Local** | No |
| **Remote** | Yes — HF router; large / slower |

---

## Environment

- **Local:** CUDA, PyTorch, Transformers, `HF_TOKEN` or cached weights.  
- **Remote:** `HUGGING_FACE`, `HF_API_TOKEN`, `HF_TOKEN`, or `OPENAI_API_KEY` (as implemented in original `run_remote.py`).

## Edits when HF changes routing

Update **`run_remote_benchmark_standardized.py`** `REMOTE_MODELS` dict in lockstep with [`scripts/run_remote.py`](../../scripts/run_remote.py) if IDs or providers change.
