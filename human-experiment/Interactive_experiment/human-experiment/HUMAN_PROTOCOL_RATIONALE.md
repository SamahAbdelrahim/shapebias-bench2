# Human-Friendly Protocol Rationale

## Purpose

This note documents why the human experiment uses `design=human_friendly` by default, how that design differs from the model benchmark, and how results from the two protocols should be interpreted.

The goal is reproducibility and review clarity: the divergence from the benchmark protocol is deliberate, theoretically motivated, and implemented in code rather than introduced ad hoc during data collection.

## Core Design Decision

The model benchmark and the human experiment answer related but not identical methodological needs.

- The model benchmark is optimized for exhaustive, tightly controlled measurement across repeated word, stimulus, and ordering combinations.
- The human experiment is optimized for ecological and psychological validity in participants who remember prior trials.

In practice, this means the benchmark protocol may reuse the same object, texture, or word many times because models do not accumulate memory across trials in the way people do. Human participants do remember repeated content, so repeated exposure can change strategy, induce carry-over effects, reduce task naturalness, and weaken interpretability.

## Why `human_friendly` Exists

The default experiment behavior is implemented in [public/experiment.js](public/experiment.js).

The key human-friendly assumptions are:

- One trial per selected object, rather than exhaustive repetition over the same object.
- One unique generated label per trial in the noun-label condition, rather than reusing the same benchmark word across many objects.
- Randomized left/right assignment by default, without requiring both orderings for every object.
- Deterministic assignment of one unique-texture stimulus package per participant, reducing repeated exposure while preserving consistent participant-level stimulus selection.

This is reflected directly in the current implementation:

- `design` defaults to `human_friendly`
- `ordering` defaults to `random` under `human_friendly`
- `trial_limit` defaults to `30` under `human_friendly`
- the trial builder comment states: `one trial per object with unique labels to avoid memory carry-over`

## Why Human Participants Need A Different Protocol

Human participants differ from models in several ways relevant to this task:

1. Humans remember previous words, objects, and textures.
2. Humans may form hypotheses about the experiment when repetition is obvious.
3. Repeated exposure can shift attention toward task artifacts rather than the intended categorization judgment.
4. Long repeated protocols can create fatigue, boredom, and demand-characteristic effects.

For these reasons, the human protocol prioritizes:

- reduced memory carry-over
- lower repetition burden
- more natural trial-to-trial judgments
- cleaner interpretation of each response as a relatively fresh categorization decision

## What Is Held Constant Across Human And Model Settings

Despite the protocol divergence, several core elements remain aligned across human and model experiments:

- the same general 3-image 2AFC task format
- the same shape-versus-texture decision structure
- the same stimulus identifiers (`stim_id`)
- the same response coding logic from option choice to `shape` / `texture`
- the same logging of `ordering`, `a_is`, and `b_is`

These shared fields support cautious cross-system comparison at a coarse level.

## What Differs From The Benchmark Protocol

The benchmark-oriented protocol described in [README.md](README.md) and prepared in [../analysis_pipe/analysis.qmd](../analysis_pipe/analysis.qmd) differs from `human_friendly` in several important ways.

### Benchmark-oriented structure

- fixed benchmark word list
- repeated trials across many stimulus-word combinations
- both orderings included systematically
- exhaustive or near-exhaustive Cartesian expansion

### Human-friendly structure

- unique generated labels for human participants
- one exposure per selected object
- one ordering per trial by default
- participant-level deterministic sampling of one unique-texture stimulus package

## Interpretation Trade-Off

This design is not a failed replication of the benchmark protocol. It is a trade-off.

- Benefit: stronger ecological and psychological validity for human participants
- Cost: weaker one-to-one equivalence with the benchmark model pipeline

Accordingly, `human_friendly` data should be used for:

- pilot quality control
- descriptive summaries of human behavior
- exploratory comparisons to model-level tendencies
- stimulus-level pattern checks with explicit caveats

It should not be used for:

- strict benchmark-equivalence claims
- direct word-level inference against benchmark model runs
- ordering-counterbalance inference that assumes both orderings were shown to each participant for each object

## Recommended Reporting Language

Suggested language for reproducibility, ethics/clearance, or methods notes:

> Human participants were tested with a memory-sensitive variant of the benchmark task. Unlike model evaluations, the human protocol avoided repeated exposure to the same word-object-texture combinations in order to reduce memory carry-over, demand characteristics, and fatigue. The resulting human data preserve the core shape-versus-texture choice structure but are not strictly trial-equivalent to the exhaustive model benchmark. Human-versus-model comparisons should therefore be treated as exploratory and approximate rather than as exact protocol-matched estimates.

## Reproducibility Notes

Relevant implementation files:

- [public/experiment.js](public/experiment.js): frontend task design, condition handling, trial generation, and logging
- [server.js](server.js): storage endpoint for trial rows
- [models/shapebias-human-logger.js](models/shapebias-human-logger.js): stored human trial schema
- [README.md](README.md): experiment parameters and protocol summary
- [../analysis_pipe/analysis.qmd](../analysis_pipe/analysis.qmd): model analysis pipeline and human-comparison preparation outputs

## Analysis Consequence

The appropriate next step is not to force human data into a benchmark-equivalent claim, but to:

1. export the logged human trial rows into analysis-ready format
2. summarize the current human-friendly participant data descriptively
3. compare human summaries to model summaries only on overlapping high-level metrics
4. label those outputs as approximate and exploratory
