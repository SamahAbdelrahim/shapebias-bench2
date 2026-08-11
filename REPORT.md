# Shape Bias Benchmark — Living Project Report

Local-only (gitignored). Update this file whenever results, model runs, or project phase change. Last updated: **2026-08-10**.

---

## 1. Theoretical question

Do vision-language models show a shape bias in novel word extension, and if so, is it naming-linked the way it is in children?

The shape bias is usually described as a fact about how children treat objects. The stronger reading (see `interpret/theoretical_motivation_vision_language_shape_bias.md`) is that it is a fact about how naming interacts with perception: the bias appears, or appears more strongly, when a label is in play. If that is right, VLMs are a natural experiment on the question developmental theory actually cares about: what kind of information gives rise to a word-learning inductive bias. VLMs have web-scale image-text statistics but none of the social, ostensive learning a child has. If a model reproduces the bias and its naming-linked signature, the statistical route may be sufficient; if it shows only a label-free visual preference, that points the other way.

As of the full-grid ladder the answer to the second half has moved. The models that can do the task at all prefer shape at rates close to the adult ceiling, and they do so with or without the novel word. That is a dissociation rather than a confirmation: the statistical route reproduces the output but not the naming signature, which means output match is not evidence of mechanism match.

A second argument has grown large enough to stand on its own, and it now carries more of the project than the naming contrast does. Asking whether a system "has a shape bias" underdetermines a measurement. Three things have to be fixed that the literature routinely leaves free: whether the system is doing the task at all (validity gating), how the representation is read out (the read-out ladder), and which stimuli are used (the four-set comparison). Vary any one of the three and the answer inverts. Shape bias is locus-bound, and the developmental question picks out one locus, a positioned forced choice over novel objects with a novel label, which is the only locus that carries position artifacts and the only one that matches what children are asked to do.

The project also carries a methodological claim (the centerpiece of the grant proposal): running developmental paradigms on AI models produces valid data only with behavioral validity checks, the same logic developmental researchers apply to children. A model that always picks the left option looks like data but is not doing the task.

## 2. Hypotheses and predictions

1. **Label dependence (the core prediction).** If a model's shape preference is naming-linked (child-like), removing the novel word from the prompt should reduce shape choosing.
   *Current status: not supported at full scale. Across the ten cells where both the label and the no-label condition clear the gate, the mean shift is +0.017 (range -0.045 to +0.131). The earlier "no-word trio" drops of about -0.33 came from 600 remote trials whose no-word runs had degraded tracking, and they do not survive gating on 1,140 triads.*
2. **Validity gating changes conclusions.** Aggregate shape-choice rates conflate visual preference with response heuristics.
   *Current status: supported, and more strongly than before. On the novel grid 20 of 84 cells pass the 0.70 image-tracking gate; 8 of 14 models never pass a single cell. Every failure is a position lock, and a locked model returns a shape rate of exactly 0.500, which reads as a clean null.*
3. **Measurement mode matters.** Forced-choice language responses and internal probabilities should dissociate for models with strong response habits.
   *Current status: supported. Swap correction and PriDe agree to within 0.029 across all 84 grid cells (mean 0.0035), so the choice of estimator is not what matters; whether you correct at all is. Raw generation and the corrected estimate differ by more than 0.10 in 20 of 84 cells.*
4. **Read-out power (new, and now the strongest technical result).** If cosine is a fixed unweighted read-out, a learned read-out on the same vectors should be able to recover shape where cosine cannot.
   *Current status: supported on the full ladder. Mesh identity decodes at 0.989 to 1.000 in every model and every layer against a chance of 0.033, while centred cosine on the same vectors returns 0.09 to 0.62 shape and a held-out learned metric returns 0.67 to 0.96.*
5. **Planned (grant 2x2).** The label-driven shift toward shape should be larger in systems trained with language supervision than in vision-only systems. The vision-only arm does not exist yet, so this is untested in our own data.
6. **Planned (AI norming).** VLM free-form descriptions embedded in a semantic space should recover the perceptual dimensions of the stimulus set and predict human word-extension choices.

## 3. Task and measures

3-image 2AFC: reference object + shape match + texture match; the model or participant is asked which candidate the novel pseudo-word names. Six prompt cells cross three framings (similarity, category, noun+`shiple`) with two option-label formats (numeric 1/2, letters A/B), and both orderings are run for every trial.

Prompt texts live in `PROMPT_TEMPLATES` in `evaluation_pipe/eval_core.py`. Note for interpretation: the no-label conditions are not label-free in the developmental sense. `no_word_category` reads "This first image is an object. Which of the following two is another one?", which is still a kind question; `no_word_similarity` is the perceptual one. Both give the same naming null, which helps, but the manipulation removes the novel word and not the categorical framing.

**Response measures:**
- **Forced-choice language response**: the model generates "1"/"2" or "A"/"B"; parsed into shape/texture/unclear.
- **Logit probing**: probability of the two option tokens at the decision position, with **swap correction** averaging each trial with its position-flipped twin.
- **PriDe debiasing**: prior-estimate correction run as an independent check on swap correction.
- **Embedding read-outs**: raw and centred cosine over four pooling layers (`proj_mean`, `vit_last_mean`, `vit_penult_mean`, `vit_pooler`).
- **Read-out ladder and linear probes**: `playgrounds/linear_probe.py`, six rungs from raw cosine to a leaky ceiling, plus mesh-identity and texture-identity classifiers with mesh-pair and texture-pair aware splits.

**Validity gates** (`analysis_pipe/src/validity_gates.R`, and the grid path in `analysis_pipe/full_grid_summary.py`):
- Image tracking >= 0.70 (choices follow image identity under position swaps)
- Word sensitivity >= 0.20 (N/A in the human-matched protocol and in the single-word grid design)
- Parse quality >= 0.97
- Threshold sensitivity was checked from 0.50 to 0.80; the gate is a power dial, not a conclusion dial (`playgrounds/threshold_sensitivity.py`).

## 4. Models: the 14-model ladder

All runs since 2026-08-07 use one shared list in `scripts/model_ladder.sh`, family-grouped and ascending within family. Before that date every runner carried its own hardcoded 9-entry array, which is why `internvl-2b`, `internvl-8b`, `internvl-14b`, `smolvlm-256m`, and `qwen3.5-2b` were registered but never run. `internvl` alone means InternVL3-**1B**.

Grid gate counts are out of six cells per model.

| Model | Family / spec | Grid gate | Shape rate where passing | Notes |
|-------|---------------|-----------|--------------------------|-------|
| `qwen3.5-4b` | Qwen3.5 4B | **6/6** | 0.797–0.913 | The only model that passes every cell. Highest tracking on the ladder (0.92). |
| `qwen3.5-9b` | Qwen3.5 9B | **5/6** | 0.808–0.939 | Highest shape rates. See the AB discrepancy in section 7. |
| `qwen3-vl-4b` | Qwen3-VL 4B Instruct | **3/6** | 0.811–0.823 | |
| `qwen3-vl-8b` | Qwen3-VL 8B Instruct | 2/6 | 0.727–0.757 | The n=30 "crosses the gate at 8B" reading does not survive the full grid. |
| `qwen3.5-27b` | Qwen3.5 27B (4-bit) | 2/6 | 0.827–0.895 | 4-bit caveat; passes fewer cells than 4B and 9B. |
| `internvl-14b` | InternVL3 14B | 2/6 | 0.586–0.602 | Passes only on AB labels. Much lower shape rate than the Qwen3.5 passers. |
| `qwen3-vl-2b` | Qwen3-VL 2B | 0/6 | — | Max tracking 0.63. |
| `internvl-2b` | InternVL3 2B | 0/6 | — | Max tracking 0.69. |
| `internvl-8b` | InternVL3 8B | 0/6 | — | Max tracking 0.51. |
| `internvl` | InternVL3 1B | 0/6 | — | Max tracking 0.04. Position-locked in every cell. |
| `qwen3.5-2b` | Qwen3.5 2B | 0/6 | — | Max tracking 0.60. |
| `qwen3.5-0.8b` | Qwen3.5 0.8B | 0/6 | — | Max tracking 0.47. |
| `smolvlm` | SmolVLM2 ~2.2B | 0/6 | — | Max tracking 0.45. See the logit-path flag in section 7. |
| `smolvlm-256m` | SmolVLM 256M | 0/6 | — | Max tracking 0.54. |

**Deliberate exclusions**, documented in `scripts/model_ladder.sh`: `qwen3-vl-32b` (no bf16 fit on one 48 GB L40S), `tinyllava` (incompatible with current transformers), `levante-runtime` (a wrapper, not a rung).

**Coverage limit worth stating in any talk or paper:** all fourteen rungs are instruct-tuned VLMs from three families. No CLIP-style model and no vision-only model is in the set, so "family matters more than scale" rests on three families.

**Earlier eras, superseded but on disk.** The 11-model remote/generation gate (4 valid, 1 borderline, 6 invalid) and the human-matched protocol validity notes are in `interpret/human_matched_validity.md` and `results/model.results/_archive_pilots`. They remain the audit trail for the first pass and should not be quoted as current numbers.

## 5. Results snapshot

Numbers below were recomputed from the tidy CSVs and probe JSONs on 2026-08-10, not carried over from earlier prose.

**Completed scale.**

| Stimulus set | Triads | Cells | Trials |
|---|---|---|---|
| Novel texture grid (30 ALICE shapes x 38 textures) | 1,140 | 84 | ~191,500 |
| `cc_triads` (edited Geirhos, all three images isolated on white) | 320 | 84 | ~53,800 |
| `decomposition_triads` (options are the source photographs) | 320 | 84 | ~53,800 |
| Smith probe | 30 | 84 | ~5,000 |
| Geirhos original cue-conflict | 30 | embeddings only | — |

Each cell runs both orderings, generation plus forced logit, plus PriDe. Embeddings are complete 14/14 on all five sets. Linear probes: 36 JSONs (14 models x 3–4 read-out layers).

**1. Most models cannot do the task.** 20 of 84 grid cells pass; 8 of 14 models pass nothing. Every failure is a position lock with `pos_first` near 0 or 1, and the resulting shape rate is exactly 0.500.

**2. Scale is necessary but not sufficient, and family matters more than size.** Within Qwen3.5 the crossing sits between 2B (0/6) and 4B (6/6), and 27B falls back to 2/6. InternVL crosses only at 14B and only on AB labels. There is no monotonic scaling curve.

**3. The naming effect is absent in the models that pass.** Ten double-gated pairs, mean shift +0.017, range -0.045 to +0.131. Examples: `qwen3.5-9b` noun 0.914 against category 0.917; `qwen3.5-4b` noun 0.797 against category 0.825; `qwen3.5-27b` noun 0.895 against category 0.827. What these models have is a label-free visual preference for shape.

**4. Read-out power decides everything.** Same 1,140 held-out triads, `proj_mean`, means across 14 models: raw cosine 0.41, centred cosine 0.42 (range 0.09–0.62, and this is the metric the published literature uses), ZCA whitened 0.73, held-out learned metric 0.81 (range 0.67–0.96), leaky ceiling 0.94. Mesh identity decodes at 0.989–1.000 in every model and layer against 0.033 chance; texture identity at about 0.99; the pixel baseline decodes mesh at 0.998 but texture at only 0.14. McNemar centred against learned runs from p about 1e-70 to 1e-131. Shape is not merely present in the encoder, it is perfectly decodable, in `smolvlm-256m` as much as in `qwen3.5-27b`. The correct statement is that a representational shape bias is a property of a model crossed with a read-out, and every encoder-level shape-bias number in the literature is partly a statement about the metric its authors chose. This supersedes the earlier framing that the behavioral bias "is not inherited from the encoder."

**5. The same read-out on different stimuli gives opposite answers.** Centred cosine shape rate, `proj_mean`, mean across 14 models: Geirhos original cue-conflict 0.23 (the canonical texture result), novel grid 0.42, `cc_triads` 0.59, Smith 0.62, `decomposition_triads` 0.63. Holding the read-out fixed and varying only the stimulus set moves the answer from texture to shape. With result 4 this makes shape bias jointly determined by stimulus set and read-out, with model identity contributing least.

**6. Position-bias correction is estimator-invariant.** Swap and PriDe agree within 0.029 (mean 0.0035) across all 84 cells, but 20 of 84 cells shift by more than 0.10 between raw generation and the corrected estimate.

**7. Behavior on the edited cue-conflict sets.** `cc_triads` 35/84 pass with a mean shape rate of 0.589 among passers; `decomposition_triads` 40/84 with 0.665; Smith 20/84 with 0.727. More models clear the tracking gate on the cue-conflict triads than on the novel grid, but shape rates among passers are lower and closer to 0.5–0.7. `decomposition_triads` runs higher than `cc_triads` for several models, which is consistent with the format confound (isolated object against cluttered photo) and is why `cc_triads` is the primary set.

**8. Humans (pilot, adults only).** 28 participants, 840 trials, **95.2% shape**, 0% unclear, median RT 3,059 ms. The best models land at 0.80–0.94, so the aggregate rate is in range, but the item-level correlation with models across the 30 shared stimuli is **r = 0.08**, with a mean absolute per-item difference of 0.20. Matching the average while not matching which items are hard is itself a result.

Figures: `results/figures/full_grid/` (fig1–7b, regenerated 2026-08-10), `results/figures/readout_power/` (stale, see section 7), `results/figures/playground/` (n=30 era). Descriptions in `analysis_pipe/PLOT_DESCRIPTIONS.md` and `analysis_pipe/PLAYGROUND_FIGURES.md`.

## 6. Figures that carry the argument

For a talk, in narrative order.

**The problem.** `fig1_validity_gates.png`. The tracking heatmap reads from the back of a room and carries the whole methodological argument, and the dagger marks for position locks let you say that those red cells produced a shape rate of exactly 0.50 and that it means nothing.

**What survives.** `fig2_shape_bias.png`, filled for pass and faded for fail, with the adult 0.95 line already drawn. The 14-model legend is dense for a slide; a talk variant showing only the 20 passing cells with a stated exclusion count would read better.

**The naming null.** `fig3_naming_effect.png` does not currently make the point, because nearly all the dumbbells are faded failures and the ten double-gated pairs are lost among them. Rebuild as ten dumbbells ordered by effect size. The point is a cluster of near-zero segments and it should look like one.

**Read-out power.** `figA_readout_power_ladder_proj_mean.png` has the right design (flat at chance through centred cosine, then the jump at ZCA and learned metric, with each model's behavioral rate as a dotted line) and needs regenerating with all 14 models. Annotate the "centred cosine (published)" tick, since that is where the existing literature sits.

**Stimulus dependence.** `fig7b_sets_behavior.png` is informative but dense for a slide at three panels by fourteen models by four sets with CIs. A one-panel version showing the across-model distribution per stimulus set, embedding beside behavior, would carry the 0.23-to-0.63 shift without the model-level detail fighting it.

**Position bias.** `fig6_position_bias_correction.png` as a methods backup slide for the question about order control.

fig4 (label format), fig5 (by shape and texture), and fig7 are supplement material.

## 7. Open discrepancies to resolve before presenting or writing

These were found while recomputing on 2026-08-10 and are flagged rather than fixed.

1. **The read-out-power figures are stale.** `results/figures/readout_power/*.png` are dated 08-03 and show three models; the 14-model probe JSONs landed 08-09. Rerunning `analysis_pipe/readout_power_figure.py` turns the strongest new result from an n=3 demo into an n=14 universal.
2. **A numeric conflict on `qwen3.5-9b noun_label_AB`.** `full_grid_v1a_summary.csv` gives shape 0.939; `full_grid_v1a_pride.csv` gives `gen_shape` 0.632 for the same cell. These come from the generation session and the logit session respectively, and several AB cells disagree the same way. No AB number should go on a slide until this is reconciled.
3. **`smolvlm` and `smolvlm-256m` return `swap_shape` and `pride_shape` of exactly 0.000** in eight cells. Exact zeros usually indicate a degenerate logit path rather than a perfect texture bias, and this also contradicts the July SmolVLM result of 40.4% shape under logit forcing.
4. **Gate-pass counts disagree across documents.** Older figures cite 16 or 18 of 54 for the 9-model era and `manuscript/main.md` says "36 of 54 fail". The current 14-model number is 20 of 84. Pick one denominator per document.

## 8. Roadmap: phases and where we are

**Phase 0 — Stimuli (done).** Reproducible 3D STL rendering pipeline with texture audit trails; the 1,140-cell texture grid (30 shapes x 38 textures); benchmark package plus two human packages (v1/v2); two edited cue-conflict triad sets staged and wired in.

**Phase 1 — Forced-choice evaluation with validity gating (done at full scale).** 14 models, six prompt cells, four stimulus sets, generation and forced logit, PriDe, figures and reports regenerated 2026-08-10.

**Phase 2 — Label-dependence control (done, result reversed).** The no-label conditions ran on the full grid for all 14 models. The naming effect is absent among gate-passing models. The old `interpret/no_word_trio_interim_report.md` describes a superseded 600-trial remote result and should be marked as such.

**Phase 3 — Read-out power (analysis complete, figures pending).** Probes and ladder run for all 14 models across three to four layers. Remaining work is regenerating figA and figB and deciding which of the three pre-committed outcomes in `interpret/reviewer_response_readout_power.md` the numbers land on. From the current JSONs it looks like the third one, the outcome added to the reviewer's dichotomy: shape present and strongly weighted, yet the texture candidate still nearer in the directed comparison. If that holds, the claim relocates from what the encoder contains to how it arranges it, and the manuscript should say so rather than letting it read as a null.

**Phase 4 — Human experiment (in progress, critical path).** Static-image jsPsych experiment is Prolific-ready and has a 28-adult pilot at 95.2% shape. Summer 2026 rebuild (Adam and Andrew): interactive rotatable STL objects (`human-experiment/Interactive_experiment/`), drag-to-continue, mobile-friendly, fullscreen fix, children and adult versions, backend migration from Apache to Vercel with an admin panel. **Open and blocking:** children, and an adult no-label condition. The adult pilot sits at ceiling, which leaves almost no room to detect a naming effect in humans on these stimuli, so the human-model difference in result 3 cannot yet be shown to be a difference rather than an absence in both.

**Phase 5 — Human-model comparison (pending human data).** Hooks in `analysis_pipe/analysis.qmd` and `src/human_analysis.R`. The item-level r = 0.08 already asks its own question: `full_grid_v1a_by_shape_texture.csv` has the by-shape and by-texture breakdown needed to ask why human and model item difficulty diverge.

**Phase 6 — Extensions.** The vision-only arm of the 2x2 (not started, and needed before any claim that language supervision produces shape bias). Substance and mass-noun framing prompts, which the theory doc predicts should pull a competent model toward material and which the existing pipeline can run cheaply. A CLIP-style or vision-only rung to widen the family coverage. AI-based perceptual norming validated against human word-extension data. Transfer of the validity-gated workflow to a second forced-choice paradigm such as mutual exclusivity. Open-source R/Python packaging. Timeline and budget in `ai-impact-grant/Stanford impact labs grant.md`.

**Cross-cutting open items** (full list in `PROJECT_CHECKLIST.md`): the four discrepancies in section 7; stimulus covariate ratings (`realism`, review of `artifact_like`/`complexity`/`abstractness`); a methods note on logit-forced against generation-based measurement; copying human-matched CSVs into this checkout.

## 9. Contribution, as currently stated

The reusable methodological product is a validity-gated protocol for running developmental forced-choice paradigms on models: image tracking under position swap, word sensitivity, parse quality, plus swap or PriDe correction, with a demonstration that 8 of 14 current VLMs fail it and that 20 of 84 cells shift by more than 0.10 without correction. It transfers directly to mutual exclusivity, taxonomic against thematic, and any other 2AFC paradigm, and it is the piece the grant proposal centers.

The empirical contribution is that "shape bias in a model" underdetermines a measurement. The read-out ladder and the four-set comparison show the same vectors yielding 0.09 or 0.96 shape depending on the metric, and the same metric yielding 0.23 or 0.63 depending on the stimuli. That reframes a set of published disagreements as measurements of different quantities rather than conflicting results about one thing.

The developmental contribution is the negative one, and it needs the child and no-label human data to be complete: the models that can do the task show a shape preference that is not naming-linked. If that holds against a human no-label condition, it is evidence that web-scale word-to-image statistics are sufficient for the output and not for the signature, which is a constraint on the associative account rather than a benchmarking result.

There is a structural echo to `Lexical_Stats_project` worth using with a developmental audience. There the finding was that shape is a good cue rather than a common one. Here it is adjacent: shape is a decodable cue rather than a weighted one. It is fully present in every encoder tested, including a 256M model, and whether it shows up as a bias depends on how it is read out.

## 10. Anticipated merges (do not step on these)

- Adam: logit scoring in `evaluation_pipe/models/local_models/`. Keep `eval_core.py` and the wrappers stable until his PR lands; analysis loaders should tolerate extra columns.
- Andrew: `human-experiment/` backend (Vercel), UI changes in `public/`, and `Interactive_experiment/`. Possibly a new repo for the interactive experiment.
- Both RAs work on branches and merge via PR; avoid force-pushes to main.

## 11. Update log

- **2026-08-10** — Rewrote this report for the 14-model full-grid era. All section 5 numbers recomputed from `results/data/*.csv` and `results/probe.results/session_readout_power/probe_*.json` rather than carried from earlier prose. Substantive changes: the label-dependence prediction is now recorded as not supported (mean shift +0.017 across ten double-gated pairs, superseding the -0.33 no-word trio); read-out power added as prediction 4 and result 4, with mesh identity at 0.989–1.000 against centred cosine at 0.09–0.62 and a learned metric at 0.67–0.96; the four-set stimulus dependence added as result 5; the model table replaced with the 14-rung ladder and its gate counts; new section 6 (figures for a talk), section 7 (four open discrepancies), and section 9 (contribution). Sections 2 and 3 note that the no-label prompts remove the novel word but not the categorical framing.
- **2026-07-11** — Probe-era results integrated. New: `results/probe.results/` (FarmShare session data, threshold sensitivity, audit table), `results/README.md`, `farmshare/README.md` plus `farmshare/probe-experiment-results.html`, `manuscript/` (venue memo recommends CogSci 2027 then Open Mind; gitignored).
- **2026-07-10** — First version of this report, written during the repo reorganization. Reflected the 11-model remote gate (4 valid), the completed no-word trio, the SmolVLM logit-forced result, the human experiment pre-pilot, and the grant framing.
