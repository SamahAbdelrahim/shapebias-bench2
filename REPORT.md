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
   *Current status: supported. Genuine PriDe changes the raw logit decision rate by more than 0.10 in 23 of 84 cells. It stays close to order-swap once both are read as decision rates, mean |Δ| = 0.031 (max 0.111, 4 cells above 0.10). Two earlier readings of this comparison were wrong in opposite directions: an implementation that labeled a held-out swap average as `pride_shape` made the two look identical, and the correction that followed compared PriDe's decision rate against swap's mean probability, which made them look three times further apart than they are.*
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

**6. Correction matters; which corrector you use mostly does not.** Correcting at all moves the answer: PriDe changes the raw logit decision rate by more than 0.10 in 23 of 84 cells (9 numeric, 14 AB). Choosing between the two correctors moves it much less: PriDe against normalized order-swap, both as decision rates, gives mean |Δ| = 0.031 (max 0.111, 4 cells above 0.10). Two other things are separately necessary. Normalising by option mass: in the 12 cells with option mass below 0.5 the unnormalised estimate can read 0.000 where the normalized estimate reads 0.50 (section 7, item 3). And keeping decision rates apart from mean probabilities: the mean sits nearer 0.5 than the rate whenever the model is undecided, so subtracting one from the other reads a scale difference as an estimator difference. Four cells (qwen3-vl-4b and qwen3-vl-8b under both similarity framings) carry an option prior within 0.02 of 0 or 1, where PriDe divides by an almost-zero prior; they are flagged by `prior_degenerate` in the PriDe CSV.

**7. Behavior on the edited cue-conflict sets.** `cc_triads` 35/84 pass with a mean shape rate of 0.589 among passers; `decomposition_triads` 40/84 with 0.665; Smith 20/84 with 0.727. More models clear the tracking gate on the cue-conflict triads than on the novel grid, but shape rates among passers are lower and closer to 0.5–0.7. `decomposition_triads` runs higher than `cc_triads` for several models, which is consistent with the format confound (isolated object against cluttered photo) and is why `cc_triads` is the primary set.

**8. Humans (pilot, adults only).** 28 participants, 840 trials, **95.2% shape**, 0% unclear, median RT 3,059 ms. The best models land at 0.80–0.94, so the aggregate rate is in range, but the item-level correlation with models across the 30 shared stimuli is **r = 0.08**, with a mean absolute per-item difference of 0.20. Matching the average while not matching which items are hard is itself a result.

**9. Measurement mode dissociates, and only under letter labels.** Generated answers and option logits agree closely on the numeric cells (mean |Δ| = 0.009, 1 of 42 cells above 0.10) and diverge on the AB cells (mean |Δ| = 0.108, 18 of 42 above 0.10; Mann-Whitney p = 5.2e-7). The validity gate moves with the measure: 11 cells pass on both paths, 9 on generation only, 2 on logits only, 62 on neither, and every one of the 11 discordant cells is an AB cell. This is prediction 3 holding in a sharper form than expected. It also means a paper reporting "logit-based shape bias" and one reporting "generation-based shape bias" are not reporting the same quantity whenever letters label the options.

**10. The pooling layer changes the read-out conclusion.** Centred cosine barely moves across layers (mean paired difference from `proj_mean`: `vit_penult_mean` +0.008, p = 0.13; `vit_pooler` −0.045, p = 0.06), but the learned metric does: `vit_penult_mean` sits +0.117 above `proj_mean` (Wilcoxon p = 0.004, n = 10) and `vit_pooler` +0.202 (p = 0.06, n = 5). The correlation between embedding and behaviour also depends on which layer is read: at `proj_mean` it is flat (r = 0.05 to 0.19), while at `vit_penult_mean` it is strongly negative (category r = −0.83, noun r = −0.82, BH q = 0.007, n = 12), meaning the models whose penultimate-layer embeddings look most shape-biased under centred cosine are the ones that behave least shape-biased. With 12 models this is exploratory, and it is the single most interesting thing the layer breakdown turned up.

**11. Item-level human-model agreement is weak where it can be measured, and undefined where it cannot.** On the 114 shared grid triads, human item means against the mean of the gate-passing models give r = 0.22 (p = 0.017) in the no-word category framing and r = 0.21 (p = 0.023) in the noun framing, with mean absolute per-item differences of 0.16 and 0.13. Both estimates are attenuated: a model answers each triad twice, and human item n runs 6 to 23. Correcting for that gives r ≈ 0.50 in the category framing. In the noun framing the correction is not estimable, because human item reliability is 0.001: adults are at 0.963 with essentially no reliable variance between triads, so there is nothing at the item level for a model to track. Any item-level human-model claim in the noun condition needs stimuli that pull adults off ceiling.

Figures: `results/figures/full_grid/` (fig1–10, regenerated 2026-08-16), `results/figures/readout_power/` (figA/figB, regenerated 2026-08-16 at 14 models), `results/figures/playground/` (n=30 era). Descriptions in `analysis_pipe/PLOT_DESCRIPTIONS.md` and `analysis_pipe/PLAYGROUND_FIGURES.md`. Supporting numbers, including every test quoted above, are in `results/data/stats_summary.csv` from `analysis_pipe/stats_tables.py`.

## 6. Figures that carry the argument

For a talk, in narrative order.

**The problem.** `fig1_validity_gates.png`. The tracking heatmap reads from the back of a room and carries the whole methodological argument, and the dagger marks for position locks let you say that those red cells produced a shape rate of exactly 0.50 and that it means nothing.

**What survives.** `fig2_shape_bias.png`, filled for pass and faded for fail, with the adult 0.95 line already drawn, and now three framing columns rather than six cells since fig4 establishes that letter labels give the same answer. The 14-model legend is still dense for a slide; a talk variant showing only the passing cells with a stated exclusion count would read better.

**The naming null.** `fig3_naming_effect.png`, rebuilt 2026-08-16 as six numeric double-gated dumbbells ordered by effect size, with the pooled statistic and the smallest detectable shift as a footnote. It now looks like what it is, a cluster of near-zero segments, instead of hiding those pairs among 36 faded failures.

**Read-out power.** `figA_readout_power_ladder_proj_mean.png`, now at 14 models with the centred-cosine rung marked, since that is where the existing literature sits. Flat at chance through centred cosine, then the jump at ZCA and the learned metric, with each model's behavioural rate as a dotted line. `fig8_embedding_layers.png` is the companion for the question that always follows, which layer this is.

**Stimulus dependence.** `fig7b_sets_behavior.png` is informative but dense for a slide at three panels by fourteen models by four sets with CIs. A one-panel version showing the across-model distribution per stimulus set, embedding beside behavior, would carry the 0.23-to-0.63 shift without the model-level detail fighting it.

**Position bias.** `fig6_position_bias_correction.png` shows the generated answer, logit argmax and order-swap correction under numeric labels, all three as decision rates. `fig6b_position_bias_correction_ab.png` is the same layout for A/B. `fig6c_correction_by_label_format.png` contrasts how far PriDe moves each format off the raw argmax (A/B farther than numeric). `fig9b_logit_vs_generation_pride.png` separately shows genuine PriDe, which is the panel carrying how far correcting at all moves each cell.

**Measurement mode.** `fig9_logit_vs_generation.png`. Panel B is the one to show: the tracking gate computed on each path, with the AB cells falling off the diagonal. It makes the validity argument twice, once for the model and once for the measure.

fig4 (the label-format screen), fig5 (by shape and texture), fig7 and fig10 are supplement material. fig4 earns a mention in the methods even so, because it is what licenses every other panel to report numeric labels alone.

**One factor, one figure.** Any factor shown not to move the answer is tested once and then dropped, so the panels carry only what still discriminates. Numeric against A/B on the generated response: mean Δ = +0.022 on P(shape) (p = 0.19) and +0.036 on tracking (p = 0.52), gate decision agreeing in 36 of 42 pairs, so fig2 and fig3 report numeric. Raw against centred cosine: +0.019 (p = 0.33), so the figures show centred. `vit_last_mean` against `proj_mean`: exactly 0.000 in every model, so fig8 shows three loci. PriDe against order-swap, both as decision rates, is +0.031 (max 0.111), so the two correctors are close and fig6 carries swap alone; fig9b keeps PriDe because the quantity it shows is how far correcting moves a cell off the raw argmax, which is large. Two factors survive and keep their split: label format on the logit path (P(shape) Δ = −0.069, p = 0.0004) stays in fig9; framing stays throughout, since it changes which cells clear the gate (2, 5 and 4 passing under similarity, category and noun) even where it leaves the rate alone. The `factor_screen` rows of `stats_summary.csv` hold all of it.

## 7. Open discrepancies to resolve before presenting or writing

Items 1 to 3 were found on 2026-08-10 and closed on 2026-08-16; the resolutions are kept here because each one changes a number that earlier drafts quote.

1. **The read-out-power figures were stale.** *Closed.* `analysis_pipe/readout_power_figure.py` now orders models by the ladder, reads the behavioural reference lines from `full_grid_v1a_summary.csv` instead of a hardcoded five-model dict, marks the centred-cosine rung, and prints n. All figA/figB are regenerated at 14 models (10 for `vit_last_mean` and `vit_penult_mean`, 5 for `vit_pooler`).
2. **The `qwen3.5-9b noun_label_AB` conflict was not a conflict.** *Closed.* `full_grid_v1a_summary.csv` (0.939) reports the generated answer and `full_grid_v1a_pride.csv` `gen_shape` (0.632) reports the logit argmax of the same trials. They are two measurements, and fig6 previously labelled the second one "generation". The divergence is specific to letter labels: mean |logit − generation| is 0.108 across the 42 AB cells against 0.009 across the 42 numeric cells (Mann-Whitney p = 5.2e-7). In this cell the generated answers split A/B 1273/1007 while the logit argmax splits 1968/280 with 0.999 of the probability mass on the two option tokens, so the option logits carry a first-option bias that the generated answer does not. AB numbers can go on a slide provided the measure is named.
3. **The SmolVLM zeros were a normalisation artifact.** *Closed.* `full_grid_pride.py` averages the raw absolute option probabilities, so a model that puts almost no mass on the two option tokens gets a shape rate pulled toward 0 whatever its preference. Mean option mass for both SmolVLM models is 0.0000. Dividing by the option mass, which is what a forced choice asks, returns 0.500 to 0.506, that is indifference. `analysis_pipe/logit_validity.py` reports the normalised estimate and the `option_mass` column; 12 of 84 cells sit below 0.5 mass and fig6 marks them.
4. **Gate-pass counts disagree across documents.** *Open.* Older figures cite 16 or 18 of 54 for the 9-model era and `manuscript/main.md` says "36 of 54 fail". The current 14-model number is 20 of 84 on the generation path. Pick one denominator per document, and say which path: the logit path passes 13 of 84.
5. **`proj_mean` and `vit_last_mean` are the same vectors.** *Open, and it affects how the layer results are described.* In `embedding_grid.json` and in the probe JSONs the two are identical in dimension and in every shape rate for all 12 models that carry both, so `get_image_features()` is returning the vision tower's last hidden state rather than an LM-space projection. The ladder therefore has three distinct read-out loci, not four, and any sentence contrasting "the projected features" with "the tower output" is currently comparing a vector with itself.
6. **The embedding panels and the probe ladder use different samples.** *Open.* `embedding_grid.json` is n = 114 triads and the probe ladder is n = 1,140. Both are labelled "the novel grid". fig8 and figA now print their n; the prose should too.

## 8. Roadmap: phases and where we are

**Phase 0 — Stimuli (done).** Reproducible 3D STL rendering pipeline with texture audit trails; the 1,140-cell texture grid (30 shapes x 38 textures); benchmark package plus two human packages (v1/v2); two edited cue-conflict triad sets staged and wired in.

**Phase 1 — Forced-choice evaluation with validity gating (done at full scale).** 14 models, six prompt cells, four stimulus sets, generation and forced logit, PriDe, figures and reports regenerated 2026-08-10.

**Phase 2 — Label-dependence control (done, result reversed).** The no-label conditions ran on the full grid for all 14 models. The naming effect is absent among gate-passing models. The old `interpret/no_word_trio_interim_report.md` describes a superseded 600-trial remote result and should be marked as such.

**Phase 3 — Read-out power (analysis and figures complete).** Probes and ladder run for all 14 models across three to four layers, figA and figB regenerated 2026-08-16, layer breakdown in fig8 and in the `readout_layer_contrast` rows of `stats_summary.csv`. Remaining work is deciding which of the three pre-committed outcomes in `interpret/reviewer_response_readout_power.md` the numbers land on. From the current JSONs it looks like the third one, the outcome added to the reviewer's dichotomy: shape present and strongly weighted, yet the texture candidate still nearer in the directed comparison. If that holds, the claim relocates from what the encoder contains to how it arranges it, and the manuscript should say so rather than letting it read as a null.

**Phase 4 — Human experiment (in progress, critical path).** Static-image jsPsych experiment is Prolific-ready and has a 28-adult pilot at 95.2% shape. Summer 2026 rebuild (Adam and Andrew): interactive rotatable STL objects (`human-experiment/Interactive_experiment/`), drag-to-continue, mobile-friendly, fullscreen fix, children and adult versions, backend migration from Apache to Vercel with an admin panel. **Open and blocking:** children, and an adult no-label condition. The adult pilot sits at ceiling, which leaves almost no room to detect a naming effect in humans on these stimuli, so the human-model difference in result 3 cannot yet be shown to be a difference rather than an absence in both.

**Phase 5 — Human-model comparison (first pass done on the matched_v2 sample).** Hooks in `analysis_pipe/analysis.qmd` and `src/human_analysis.R`. `analysis_pipe/item_rates.py` writes per-triad model rates so the model side can join `human_item_means.csv`, and fig10 plus the `human_model_item` rows of `stats_summary.csv` carry result 11. The by-shape and by-texture breakdown in `full_grid_v1a_by_shape_texture.csv` is still the route to asking why item difficulty diverges. The blocking issue is now measurement rather than analysis: adults are at ceiling in the noun framing, so item-level variance is at the noise floor.

**Phase 6 — Extensions.** The vision-only arm of the 2x2 (not started, and needed before any claim that language supervision produces shape bias). Substance and mass-noun framing prompts, which the theory doc predicts should pull a competent model toward material and which the existing pipeline can run cheaply. A CLIP-style or vision-only rung to widen the family coverage. AI-based perceptual norming validated against human word-extension data. Transfer of the validity-gated workflow to a second forced-choice paradigm such as mutual exclusivity. Open-source R/Python packaging. Timeline and budget in `ai-impact-grant/Stanford impact labs grant.md`.

**Cross-cutting open items** (full list in `PROJECT_CHECKLIST.md`): the three open discrepancies in section 7, items 4 to 6; stimulus covariate ratings (`realism`, review of `artifact_like`/`complexity`/`abstractness`); a methods note on logit-forced against generation-based measurement; copying human-matched CSVs into this checkout.

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

- **2026-08-17 (A/B correction figures)** — The PriDe / swap CSV already covered all 84 cells; only fig6 was numeric-only. Added `fig6b` (A/B twin of the three-bar correction figure) and `fig6c` (mean |PriDe − argmax| and count of cells moved > 0.10 by label format). Three new `factor_screen` rows record that A/B cells move farther under correction than numeric ones.
- **2026-08-17** — Audited every code path that computes a prior-bias correction. Three defects fixed. (a) `swap_shape` was a mean probability while `pride_shape`, `logit_shape` and the generation shape rate are decision rates, and `full_grid_figures.py` and `stats_tables.py` subtracted one scale from the other; both quantities are now written separately (`*_shape_rate`, `*_mean_p_shape`), fig6 plots the rate, and the swap-versus-PriDe gap drops from 0.084 to 0.031. (b) The PriDe prior was fit on the lexicographically first 10% of `stim_id`, which is three shapes rather than a spread, since `stim_id` is `<stl_id>/<texture>`; it now uses a seeded random 10%. (c) `eval_core.py` averaged unnormalised absolute probabilities across the two passes, so whichever pass held more option mass dominated; it now conditions each pass on its own option mass first (latent only, `swap_correct` is false in all ten sessions). `smith_ladder_pride.csv` and both `cueconflict_*_pride.csv` were still holding pre-fix values from 8-9 August and were regenerated. A `prior_degenerate` flag marks the four cells whose option prior sits within 0.02 of 0 or 1.
- **2026-08-16 (later)** — Applied a screen-then-drop rule to the figures. `stats_tables.py` gained a `factor_screen` section (15 rows, `stats_summary.csv` now 180 rows) recording each factor that does not move the answer. fig4 became the two-panel label-format screen carrying tracking, P(shape) and the gate agreement count; fig2 and fig3 dropped the AB cells; fig3 was also rebuilt as double-gated pairs ordered by effect size; fig6 dropped the PriDe bar; fig8 dropped `vit_last_mean`. Label format is kept in fig1 and fig9 because it is not null on the logit path, and framing is kept everywhere because it changes which cells are valid.
- **2026-08-16** — Closed section 7 items 1 to 3 and added items 5 and 6. New analysis scripts: `analysis_pipe/logit_validity.py` (validity gating on the logit path, plus the `option_mass` column that explains the SmolVLM zeros), `analysis_pipe/item_rates.py` (per-triad model rates), `analysis_pipe/stats_tables.py` (`results/data/stats_summary.csv`, 165 rows). New figures fig8 (embedding layers), fig9 (logits against generation), fig10 (item-level human-model); fig6 now separates the generated answer from the logit argmax; fig7 and fig7b name `proj_mean` on every axis; figA and figB regenerated at 14 models. New results 9 to 11 in section 5. The pooled naming effect reproduces the ten double-gated pairs and +0.017 exactly, now with a Wilcoxon (p = 0.56) and the note that the smallest mean shift this design could detect at 80% power is 0.046, so the null is a bound rather than an absence.
- **2026-08-10** — Rewrote this report for the 14-model full-grid era. All section 5 numbers recomputed from `results/data/*.csv` and `results/probe.results/session_readout_power/probe_*.json` rather than carried from earlier prose. Substantive changes: the label-dependence prediction is now recorded as not supported (mean shift +0.017 across ten double-gated pairs, superseding the -0.33 no-word trio); read-out power added as prediction 4 and result 4, with mesh identity at 0.989–1.000 against centred cosine at 0.09–0.62 and a learned metric at 0.67–0.96; the four-set stimulus dependence added as result 5; the model table replaced with the 14-rung ladder and its gate counts; new section 6 (figures for a talk), section 7 (four open discrepancies), and section 9 (contribution). Sections 2 and 3 note that the no-label prompts remove the novel word but not the categorical framing.
- **2026-07-11** — Probe-era results integrated. New: `results/probe.results/` (FarmShare session data, threshold sensitivity, audit table), `results/README.md`, `farmshare/README.md` plus `farmshare/probe-experiment-results.html`, `manuscript/` (venue memo recommends CogSci 2027 then Open Mind; gitignored).
- **2026-07-10** — First version of this report, written during the repo reorganization. Reflected the 11-model remote gate (4 valid), the completed no-word trio, the SmolVLM logit-forced result, the human experiment pre-pilot, and the grant framing.
