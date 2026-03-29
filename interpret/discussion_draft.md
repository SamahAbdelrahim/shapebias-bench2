# Discussion (Draft)

## Main interpretation

This benchmark extends standard shape-bias evaluation by combining a controlled 3-image 2AFC design with reproducible, engineered stimuli. The data indicate that shape-bias estimates are only interpretable when models first pass behavioral validity checks. In other words, high shape-choice proportion is not sufficient evidence of shape-based cognition if choices do not track image identity across counterbalanced orderings.

## What this implies for cognitive modeling

Our results suggest that current multimodal models vary along at least two partially independent axes:

- **Perceptual preference axis** (shape vs texture tendency)
- **Task-control axis** (susceptibility to position bias, parsing fragility, and prompt-format dependence)

Models that score highly on both axes are the strongest candidates for cognitive comparison with humans. Models that fail the task-control axis may still be useful engineering systems, but they are weaker cognitive process models in this paradigm.

## Methodological contributions

This project contributes a reusable research workflow, not only model scores:

- Reproducible stimulus generation with provenance and auditability
- Controlled shape-texture conflict using pose-aligned renderings
- Counterbalanced evaluation that can diagnose position heuristics
- Shared infrastructure for model and future human experiments

Because stimuli are reusable and protocol-compatible, this benchmark can serve as a bridge between machine evaluation and developmental/behavioral human studies.

## Substantive implications

Three implications follow from the current findings:

1. **Counterbalancing is non-negotiable** in model behavior work; pooled metrics can mask response heuristics.
2. **Model comparisons should be validity-gated**; otherwise rankings confound feature preference with task artifacts.
3. **Stimulus properties likely modulate behavior**; integrating realism/complexity/novelty dimensions is essential for mechanism-level claims.

## Labeling hypothesis test (new control condition)

We implemented a no-word category-style control condition to test whether shape bias depends on linguistic labeling versus general visual categorization. The contrast is designed to isolate prompt semantics while keeping stimulus, ordering, and response mapping constant.

If models show reduced shape preference or degraded image tracking in the no-word condition, that supports a label-linked account. If behavior remains stable, that supports a more general visual-bias account. This direct noun-label versus no-word contrast is now included in the report pipeline.

## Staged evidence strategy and diagnostic trio rationale

We intentionally used a two-stage design:

1. **Reduced pilot across all five remote models** to detect directional effects quickly and cheaply.
2. **Full matched follow-up on a diagnostic trio** selected from pilot outcomes to maximize scientific resolution per additional cost.

The trio was selected because each model represented a distinct pilot regime:

- `qwen3.5-9b`: large no-word shape drop with improved tracking.
- `qwen3.5-27b`: large no-word shape drop with reduced tracking and higher unclear rate.
- `qwen3.5-122b-a10b`: near-stable high-shape behavior in pilot.

This design preserves broad screening while enabling confirmatory depth where it is most informative.

## Finalized trio evidence (full matched no-word)

With full 600/600 no-word matched trials for each diagnostic model:

- `qwen3.5-9b`: strong shape reduction without labels (`delta_shape ~ -0.326`), with improved tracking.
- `qwen3.5-27b`: strong shape reduction (`delta_shape ~ -0.334`), accompanied by quality degradation (`unclear` increase, tracking decrease).
- `qwen3.5-122b-a10b`: near-stable/slight shape decrease (`delta_shape ~ -0.020`) but notable increase in no-word unclear responses.

Taken together, this supports a **model-dependent labeling effect** rather than a single universal mechanism.

## Limitations

Key limitations to acknowledge clearly:

- Current runs may not yet include all 11 target models in one finalized canonical report output.
- Some stimulus-level scalar annotations (for example, realism or artifact-like ratings) may still be incomplete and rely on template placeholders.
- Statistical inference depends on available package environment (`lme4` mixed-effects model vs fallback generalized linear model).
- Prompt/response-format robustness has diagnostics in place, but full perturbation sweeps should be completed before strong generalization claims.
- Full no-word matched confirmation currently covers the diagnostic trio; extending matched confirmation to the remaining remote models would improve generalizability.

## Next experiments (immediate)

- Run human pilot on the frozen stimulus subset using matched instructions and coding.
- Complete covariate annotation file and refit the covariate model.
- Execute prompt-format perturbation runs (strict numeric, A/B format, no-word control).
- Replicate key effects on held-out texture families and additional model checkpoints.

## Reviewer-safe claim set

Use claims at this confidence level:

- **Strong**: Counterbalanced 2AFC reveals that some models exhibit order/position-driven behavior; validity gating changes interpretation.
- **Moderate**: A subset of models appears to show stable shape-preferring behavior under this benchmark.
- **Exploratory**: Stimulus metadata dimensions may explain model-specific shifts in shape vs texture decisions.

## Suggested closing paragraph

Overall, the benchmark supports a shift from score-centric evaluation toward process-centric evaluation. By integrating controlled stimulus engineering, behavioral validity tests, and direct human-study compatibility, this framework enables more defensible claims about when model behavior approximates human-like shape-based categorization and when it reflects brittle task heuristics.
