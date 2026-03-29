# Results (Draft Narrative)

## Overview

We evaluated visual shape bias using a 3-image 2AFC paradigm. On each trial, the model saw a reference image and two candidates: one matching shape and one matching texture. Candidate order was counterbalanced (`shape_first`, `texture_first`). Trials were repeated across 30 stimulus sets and 10 novel words (5 pseudo-words and 5 random length-matched strings).

The central question was not only whether models prefer shape over texture, but whether they do so in a way that is behaviorally valid and cognitively interpretable.

## 1. Task-validity checks before interpretation

Before interpreting shape-bias rates, we applied three validity checks:

- **Image tracking across order swap**: whether choices follow image identity when positions are swapped.
- **Word sensitivity**: whether responses vary across words for the same stimulus/ordering context.
- **Parse quality**: low ambiguous/unclear responses and low retry dependence.

These checks separate genuine feature-based behavior from response heuristics (for example, always choosing position 1 or 2).

Use these output files to report the exact numbers:

- `results/data/model_validity_summary.csv`
- `results/data/table_model_validity_gate.csv`

## 2. Model behavior falls into distinct regimes

The benchmark shows at least three model regimes:

1. **Validity-passing shape-preferring models**: high image-tracking with stable shape preference.
2. **Mixed models**: above-chance shape preference but weaker consistency.
3. **Validity-failing models**: apparent shape/texture rates that are largely explained by order/position effects.

This distinction is critical because pooled shape-choice rate alone can be misleading in counterbalanced tasks when order sensitivity is high.

## 3. Shape bias among validity-passing models

After filtering to validity-passing models, shape-bias estimates remain heterogeneous across model families and scales. This indicates that shape preference is not a uniform property of modern VLMs, even under a shared prompt and stimulus protocol.

Report model-level point estimates and uncertainty from:

- `results/data/table_model_shape_ci.csv`
- Figure: `A_model_bias_valid_only.png`

## 4. Ordering effects reveal latent heuristics

Ordering-effect estimates (shape-first minus texture-first) expose decision instability that is not visible in pooled metrics. Models with large ordering deltas are unlikely to represent stable object-level matching behavior under this task format.

Report these from:

- `results/data/table_model_ordering_effect_ci.csv`
- `results/data/table_robustness_diagnostics.csv`

## 5. Stimulus-level covariates and mechanistic signal

Linking trial data to stimulus-generation metadata (mode, texture family, texture consistency, plus optional human-rated dimensions such as realism/complexity/novelty) supports mechanistic analysis of when models shift toward texture capture versus shape matching.

Report these analyses from:

- `results/data/table_stimulus_covariate_summary.csv`
- `results/data/table_stimulus_covariate_model_coefficients.csv`

## 6. Human-comparison readiness

A frozen subset and protocol specification were generated to align human behavioral testing with the exact model paradigm:

- `results/data/human_frozen_stimulus_subset.csv`
- `results/data/human_protocol_spec.csv`

This supports direct model-vs-human comparison under matched task structure and response coding.

## 7. Condition contrast: noun label vs no-word

To test the labeling hypothesis directly, we added a no-word category-style control prompt:

- `See this object in the first image. Can you find another one of the two (1 or 2)?`

Contrast outputs are now integrated into the analysis pipeline:

- `results/data/table_condition_contrast.csv`
- Figure: `E_condition_delta_shape.png`

Pilot run artifacts:

- `results/no_word_pilot_remote_dedup.csv`
- `results/no_word_pilot_diagnostics.csv`
- `results/no_word_pilot_summary.md`

The reduced pilot (60 trials/model) already showed meaningful model-dependent divergence:

- strong shape decrease for `qwen3.5-9b` and `qwen3.5-27b`,
- slight decrease for `llama4-scout`,
- increase for `qwen3.5-35b-a3b`,
- near-stable trend for `qwen3.5-122b-a10b`.

To confirm this efficiently under budget constraints, we ran a full matched follow-up on a diagnostic trio chosen to span distinct pilot regimes:

- `qwen3.5-9b` (large decrease),
- `qwen3.5-27b` (large decrease with quality changes),
- `qwen3.5-122b-a10b` (near-stable high-shape trend).

Final trio evidence is reported in:

- `results/no_word_full_remote_trio_dedup.csv`
- `results/no_word_trio_interim_contrast.csv`
- `interpret/no_word_trio_interim_report.md`

## Suggested concise takeaway paragraph

Across models, shape-bias behavior is best understood as a combination of feature preference and task-validity constraints. Some models exhibit stable shape-preferring behavior consistent with content tracking, while others are dominated by response heuristics revealed by order counterbalancing. Validity-gated analysis therefore changes interpretation from "which model has the highest shape rate" to "which models implement stable, cognitively meaningful shape-based decisions under controlled shape-texture conflict."
