# Remote evaluation: system prompt policy

This document records **options considered**, **reasoning**, and the **final decision** for the `system` message in `scripts/run_remote.py` (OpenAI-compatible vision API). The **user** turn (image labels + task text from `eval_core.make_prompt`) is unchanged and identical across models; only the optional **system** message was under discussion.

## Context

- Remote runs use [`build_openai_compatible_vision_messages`](projects/shapebias-bench-2/evaluation_pipe/eval_core.py): optional `system`, then user content with `VISION_USER_IMAGE_LABELS` + task string from [`PROMPT_TEMPLATES`](projects/shapebias-bench-2/evaluation_pipe/eval_core.py).
- **Local** Qwen3.5 wrappers still use a **different** historical system string in [`qwen35.py`](projects/shapebias-bench-2/evaluation_pipe/models/local_models/qwen35.py) (`QWEN35_VLM_SYSTEM_PROMPT`) for continuity with earlier local runs. Remote policy is **decoupled** so cross-model remote comparisons share one system line.

## Options considered

### A. Uniform absent (`system_prompt: None` for every model)

- **Pros:** Strongest claim that only the user message defines the task; no extra prior.
- **Cons:** Chatty models may produce rationales or multi-token answers, hurting parse quality and validity gates.

### B. Uniform minimal (one shared string for all models)

Several candidate strings were discussed:

| Variant | Example content | Pros | Cons |
|--------|-------------------|------|------|
| **B1 — Historical Qwen3.5** | `Answer concisely. Do not explain your reasoning.` | Matches legacy Qwen3.5 tuning; short. | Does not mention the **1/2** format; weaker alignment with the scorer (`parse_answer` in `eval_core`). |
| **B2 — Task-aligned (chosen)** | See **Final decision** below. | Matches the user instruction (“exactly one character: 1 or 2”); easy to justify for this benchmark. | Slightly longer; must be fixed verbatim in methods. |
| **B3 — Hybrid** | Concise + format, e.g. brevity plus “follow output format exactly.” | Middle ground. | More moving parts; redundant with B2 for this task. |

### C. Family-specific system prompts

- **Pros:** Mirrors each stack’s local defaults (“implementation-faithful” per model family).
- **Cons:** **Confound for cross-family comparison:** differences mix weights and instruction priors. Requires explicit disclosure in methods.

## Reasoning

1. **Comparability:** For ranking or comparing remote VLMs on the same human-matched or benchmark protocol, **holding the system message constant** avoids an extra free variable.
2. **Alignment with scoring:** The pipeline rewards a single **`1` or `2`** in the model output. A system line that **reinforces that same constraint** is more **task-faithful** than a generic “be concise” line.
3. **Uniform absent (A)** remains a valid ablation if parse rates are acceptable; it was not selected as the default because prior runs suggested many models benefit from a light format nudge.

## Final decision (frozen)

**All entries in `REMOTE_MODELS` use one shared system message** (task-aligned, uniform minimal — option **B2**):

```text
Follow the user's instructions exactly. When they ask for a single character (1 or 2), reply with only that character and no other text.
```

**Canonical definition in code:** constant `REMOTE_UNIFORM_SYSTEM_PROMPT` in [`evaluation_pipe/eval_core.py`](projects/shapebias-bench-2/evaluation_pipe/eval_core.py); every `REMOTE_MODELS` entry in [`scripts/run_remote.py`](projects/shapebias-bench-2/scripts/run_remote.py) sets `system_prompt` to that constant.

**Historical reference only:** `QWEN35_VLM_SYSTEM_PROMPT` (`Answer concisely. Do not explain your reasoning.`) remains for **local** Qwen3.5 wrappers unless you later choose to align local with remote.

### Comparability with older remote CSVs

Remote runs **before** this policy used **mixed** system messages: Qwen3.5 keys saw `QWEN35_VLM_SYSTEM_PROMPT`, while Llama had **no** system line (`None`). The **user** turn was already shared; the **system** turn was not. Changing the system message can shift model behavior (verbosity, adherence to “1/2 only,” parse rates, and downstream shape-bias metrics). So **new** remote CSVs are not the same experimental condition as those legacy files. For a paper or internal analysis you should either **re-run** models you want to compare under the new policy, **re-gate** validity on the new outputs, or add a **short methods caveat** when mixing old and new batches (e.g. “earlier remote Llama runs omitted a system message; later runs used a uniform task-aligned system line for all models”).

## Related docs

- [`interpret/model_choice_decision_log.md`](projects/shapebias-bench-2/interpret/model_choice_decision_log.md) — local vs remote parity, pixels/API caveats.
- [`evaluation_pipe/README.md`](projects/shapebias-bench-2/evaluation_pipe/README.md) — vision user-turn layout.

## Changelog

- **Adopted** task-aligned uniform remote `system_prompt` as above (`REMOTE_UNIFORM_SYSTEM_PROMPT` + `REMOTE_MODELS` wiring).
