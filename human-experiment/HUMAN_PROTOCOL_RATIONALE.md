# Human Protocol Rationale

## Current protocol: `matched_v2`

The protocol described from "Human-Friendly Protocol Rationale" onward is the March 2026 pilot (`design=human_friendly`). It is superseded. That pilot reduced to one number, 95.2% shape choices from 28 adults over 30 stimuli in a single condition, which could not be compared cell-for-cell against the model results because it ran a different stimulus package, had no no-word baseline, and showed each triad in one option order.

`matched_v2` keeps the reasons the pilot diverged from the benchmark, since they are about memory and not about stimulus coverage, and fixes what blocked comparison.

What carries over from the pilot: one exposure per triad per participant, a unique pseudo-word per trial in the noun condition, and no exhaustive Cartesian expansion. Human participants remember words and objects; models do not.

What changed:

- **Stimulus set.** 114 triads from the novel texture grid, the same triads behind the embedding panel, since both use the shape-stratified round-robin at `seed=0`. The pilot used a different 30-shape package.
- **Framing.** Noun-label and no-word-category, between participants. The no-word wording now matches the model template: "This first image is an object. Which of the following two images is another one?" The pilot's wording said the first image was a "label", which asserted a label in the condition designed to withhold one.
- **Option order.** Counterbalanced between participants. A triad's order is fixed by the parity of its index in the pool and flipped for ordering group B, so every triad appears in both orders across people while no one sees a triad twice. Within a participant orders alternate, so a side preference cannot masquerade as a shape preference. This supports an item-level analogue of the model tracking gate: an item answered on content has the same shape rate under both orders, while one answered on position has roughly complementary rates.
- **Attention checks.** Four per session, each offering an exact duplicate of the reference against an unrelated object. More than one error excludes the participant.
- **Session.** 27 grid trials plus 4 checks, 31 trials, close to the pilot's 30.
- **One shape per participant per session.** Each of the 27 trials uses a different shape, so no object is ever seen twice and no object carries two pseudo-words. Which texture a shape appears with rotates across participants, so all 114 triads are still used.

Sample size scales as `114 triads x 2 conditions x k / 27 test trials`. At k = 16 observations per triad per condition that is about 135 participants. Simulating 135 gives 16.3 observations per triad in the noun condition and 15.6 in the no-word condition, every triad covered in both orders in both conditions. Re-simulate with `node human-experiment/verify_assignment.js N` before committing to a number.

One caveat worth stating in any writeup: adults were at 95.2% in the pilot, and more participants at ceiling buy precision, not information. What buys information here is the no-word contrast.

## Why cue-conflict is not in the human experiment

An earlier version of this design gave each participant 18 grid triads and 18 Geirhos cue-conflict triads. It was dropped for humans, though the models still run both.

The cue-conflict images are photographs of familiar named categories: airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant, keyboard, knife, oven, truck. In the noun condition the participant is told "this first image is a *rilas*" while looking at an obvious elephant. That asks an adult to accept a novel name for something already named, and the usual response is to look for something else the word could pick out. The most available candidate in the display is the texture, so the pseudo-word would push responses toward texture for a lexical reason rather than a perceptual one, and only in that set. The novel grid has no competing name, so the manipulation is not comparable across the two sets.

Two further problems were specific to that set. The cue-conflict pool drew 160 triads from only 108 distinct shape exemplars, so in about a fifth of sessions the same photograph appeared twice wearing different textures and received two different pseudo-words. And the set contrast confounded more than novelty: the grid pairs a novel shape against a *material* (carpet, leather, metal), while cue-conflict pairs a familiar object against *another familiar object's* surface, at a different resolution and in a different rendering style.

The selection code is kept behind `--include-cc` in `scripts/build_human_trial_pool.py`, and nothing in the frontend assumes a single set, so the comparison can be restored as a separate study rather than a second block inside this one.

The pilot is not pooled with `matched_v2`. It is kept separate by `design` and `pool_version`.

---

# Human-Friendly Protocol Rationale (March 2026 pilot, superseded)

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
