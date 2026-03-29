# Slides Talk Track (Draft)

## Slide 1 — Title and motivation

**Title:** Shape Bias Under Controlled Shape-Texture Conflict  
**Message:** We are not just benchmarking scores; we are testing whether model behavior is cognitively interpretable.

**Say:**
- We built a controlled 3-image 2AFC benchmark for shape-vs-texture decisions.
- The goal is to evaluate models as candidate cognitive models, not just engineering systems.

## Slide 2 — Task design

**Visual:** Trial diagram (reference + candidate 1 + candidate 2).  
**Message:** For each novel word and stimulus, one candidate matches shape and the other matches texture; order is counterbalanced.

**Say:**
- Prompt format: "The first image is a {word}. Which of the following two images (1 or 2) is also a {word}?"
- Counterbalancing lets us separate content tracking from position heuristics.

## Slide 3 — Stimulus engineering contribution

**Visual:** Pipeline from STL object -> render variants -> packaged benchmark set.  
**Message:** Stimuli were engineered and audited for reproducibility and reuse.

**Say:**
- 3D printable STL-based object sets, pose-aligned renderings, and texture consistency audits.
- Designed for reuse in future human behavioral studies with matching protocol.

## Slide 4 — Why validity checks are required

**Visual:** Concept figure showing same pooled shape% from two different underlying strategies.  
**Message:** Pooled shape-choice rate can be misleading.

**Say:**
- A model that always answers "1" can appear shape-biased depending on ordering distribution.
- We therefore apply validity gates before interpreting shape bias.

## Slide 5 — Validity gate definition

**Visual:** Three-gate checklist.  
**Message:** Interpret only validity-passing models.

**Say:**
- Gate 1: image tracking across order swaps.
- Gate 2: non-trivial word sensitivity.
- Gate 3: parse quality (low unclear/retry pathology).
- Output labels: valid, borderline, invalid.

## Slide 6 — Main findings by behavioral regime

**Visual:** Model grouping (valid shape-preferring / mixed / position-biased).  
**Message:** Models split into qualitatively different behavioral regimes.

**Say:**
- Some models show stable shape-preferring behavior.
- Some are mixed.
- Some are dominated by ordering/position effects, so high-level shape claims are not warranted.

## Slide 7 — Inference and uncertainty

**Visual:** Coefficient/CI table snapshot and model-level CI plot.  
**Message:** We report uncertainty, not only point estimates.

**Say:**
- Trial-level logistic inference quantifies ordering, word-type, and model-size/family effects.
- Model-level confidence intervals show which differences are robust.

## Slide 8 — Stimulus covariates and mechanisms

**Visual:** Covariate summary plot/table.  
**Message:** We can test when models switch toward shape or texture.

**Say:**
- Join trial outcomes to stimulus metadata (mode, texture family, consistency).
- Optional annotated dimensions (realism, complexity, novelty, artifact-like) enable mechanism-level hypotheses.

## Slide 9 — Human comparison readiness

**Visual:** Frozen subset table + protocol checklist.  
**Message:** The benchmark now supports direct model-human comparison.

**Say:**
- We generate a balanced frozen subset for human testing.
- Same task structure and response mapping support one-to-one comparability.

## Slide 10 — Robustness and next steps

**Visual:** Robustness diagnostics table + roadmap bullets.  
**Message:** Strong claims require replication under perturbations.

**Say:**
- Run prompt-format perturbations and held-out texture replications.
- Complete human pilot on frozen subset.
- Compare model and human signatures under identical counterbalanced conditions.

## Backup slide A — Reviewer-safe claims

**Use this wording:**
- Strong: Counterbalancing reveals order-driven heuristics in some models.
- Moderate: A subset shows stable shape-preferring behavior under validity constraints.
- Exploratory: Stimulus-level covariates may explain model shifts between shape and texture responses.

## Backup slide B — One-sentence contribution

**Sentence:**  
We provide a reproducible, validity-gated shape-bias benchmark that bridges model evaluation and human behavioral experimentation under shared controlled stimuli and task structure.
