# Plot Descriptions

## Full-grid figures (`results/figures/full_grid/`)

Written by `analysis_pipe/full_grid_figures.py`. Sections A–D below are the older
R pipeline and are kept for the archived runs.

A design factor is tested in one figure and then dropped from the rest. Label
format is screened in fig4 and, since numeric and A/B agree on the generated
answer, only numeric is reported downstream; the one place the two formats part
company is the logit path, which is why fig9 keeps the split and why fig6 has an
A/B twin (fig6b) plus a format-contrast panel (fig6c). Order-swap remains
in fig6 / fig6b; PriDe stays in fig9b, not because it disagrees with order-swap (it does
not, +0.031 once both are read as decision rates) but because fig9b measures how
far correcting at all moves a cell off the raw argmax. `vit_last_mean` is dropped
from fig8 as an alias of `proj_mean`, and the raw cosine is not plotted anywhere
because it tracks the centred cosine. Every one of these decisions has a
`factor_screen` row in `results/data/stats_summary.csv`. Framing is kept
throughout: it is the manipulation of interest, and it changes which cells clear
the validity gate even where it does not change the rate.

| File | What it plots |
|---|---|
| **fig1_validity_gates.png** | Tracking heatmap, 14 models x 6 prompt cells. Boxed at the 0.70 gate, dagger for a position lock. |
| **fig2_shape_bias.png** | P(shape) per framing, numeric labels only, filled with a Wilson CI where the cell passes the gate and faded where it fails. Human anchors sit only in the cells humans were run in. |
| **fig3_naming_effect.png** | Dumbbells from the no-word framing to noun+`shiple`, numeric labels, restricted to pairs where both cells clear the gate and ordered by effect size. Carries the pooled test and the smallest detectable shift. |
| **fig4_label_format.png** | The label-format screen: numeric against A/B on tracking (panel A) and on P(shape) (panel B), all three framings pooled, with the paired difference and the gate agreement count. This is the figure that licenses fig2 and fig3 to report numeric alone. |
| **fig5_by_shape_texture.png** | Mean shape rate by STL shape and by texture, gate-passing cells only. |
| **fig6_position_bias_correction.png** | Three estimates of the same cell under numeric labels: the generated answer (from the summary CSV), the logit argmax, and the normalised order-swap. All three are decision rates, so the swap bar reads `swap_shape_rate` and not the mean swap-corrected probability. Genuine PriDe is shown separately in fig9b. A red × marks cells where option mass is below 0.5. |
| **fig6b_position_bias_correction_ab.png** | Same three bars as fig6, A/B labels only. Label format is null on the generation path but not on the logit path, so the A/B cells keep their own correction figure. |
| **fig6c_correction_by_label_format.png** | How far PriDe moves a cell off the raw logit argmax, numeric against A/B. Panel A is the mean absolute move; panel B is the count of cells moved by more than 0.10. |
| **fig6d_generation_follows_logit_ab.png** | Follows fig6b: for each A/B cell, |generation − raw logit argmax| against |generation − swap-corrected|. Shorter bar = the logit measure generation follows more closely. |
| **fig7_vision_vs_behavior.png** | Panel A embedding shape rate across the four stimulus sets; panel B embedding against behavior per framing. Both name the pooling layer (`proj_mean`). |
| **fig7b_sets_behavior.png** | Behavior across the four stimulus sets per framing, with humans on the same axis. |
| **fig7b_sets_emb_vs_behavior.png** | Embedding against behavior, colour = stimulus set, marker = model. |
| **fig8_embedding_layers.png** | Panel A: centred-cosine shape rate for every model at each distinct pooling layer, with the number of models covered per layer. Panel B: paired difference from `proj_mean`. Answers which layer fig7 / fig7b are reporting and how much the answer moves if a different one is chosen. `vit_last_mean` is dropped and named in the caption because it returns the same vectors as `proj_mean` in every model. |
| **fig9_logit_vs_generation.png** | The same trials scored two ways. A: P(shape) from generated text against P(shape) from the logit argmax. B: the tracking gate computed on each path, with the 2x2 counts. C: first-option rate on each path. The three panels separate numeric from A/B cells, which is where the two measures part company. |
| **fig9b_logit_vs_generation_pride.png** | Twin of fig9 after genuine PriDe correction: the first-option prior is estimated on 10% of stimuli and applied to each held-out observation. A: generation P(shape) against PriDe-corrected logit P(shape). B: raw logit argmax against PriDe (how far correction moved each cell; more than 0.10 in 23 of 84). C: raw first-option rates (the bias being estimated). The prior is fit on a seeded random 10% rather than the head of the sorted stimulus ids, which would be a handful of shapes. Tracking is not redrawn because the corrected held-out decisions do not retain the original paired-gate definition. |
| **fig10_human_model_items.png** | Item-level agreement on the 114 shared triads: human P(shape) per triad against the mean of the gate-passing models, with a binned mean. Both axes are noisy, so read it with the reliability rows in `stats_summary.csv`. |

Numbers for these figures: `analysis_pipe/stats_tables.py` writes
`results/data/stats_summary.csv` (long format: analysis, group, statistic, value,
n, ci_lo, ci_hi, p_value, note).

## The same figures on the other stimulus sets

fig1 through fig6d and fig9 / fig9b read one stimulus set's own summary, PriDe
and logit-validity CSVs, so the same code writes them per set:

| Set | Directory | `--set` |
|---|---|---|
| novel grid, 1,140 triads | `results/figures/full_grid/` | `grid` |
| Smith probe ladder, 30 triads | `results/figures/smith/` | `smith` |
| cue-conflict cc_triads, 320 | `results/figures/cc_triads/` | `cc_triads` |
| cue-conflict decomposition, 320 | `results/figures/decomposition/` | `decomposition` |

`python analysis_pipe/full_grid_figures.py` builds all four; `--set smith`
builds one. Two things do not travel. fig5 needs a by-shape/texture export and
so is skipped for Smith, whose 30 triads do not have one. Human anchors in fig2
appear only for the grid and cc_triads, the two sets humans were run in.
fig7, fig7b, fig7c and fig8 are cross-set by construction and fig10 needs the
matched human items, so all five are built once, under `full_grid/`.

The per-set figures carry the set name in the title. Statements that were
estimated on the grid alone (the pooled naming test in fig3, the smallest
detectable shift) are printed on the grid figure only.

## Section A — Shape Bias by Model

| File | What it plots |
|---|---|
| **A_model_bias_main.png** | Dot plot of **proportion of "shape" choices per model** (all trials). Models on x-axis sorted by parameter count; y-axis 0–100%. Dashed line at 50% (chance). Each dot is labeled with the exact proportion. |
| **A_model_bias_supplement.png** | **Stacked bar chart** breaking down each model's responses into shape / texture / unclear proportions. Shows the full response distribution, not just shape rate. |
| **A_model_bias_deterministic.png** | Same as main but **filtered to deterministic trials only** (where both shape_first and texture_first orderings were run for each stimulus). |
| **A_model_bias_random_ordering.png** | Same as main but **filtered to random-ordering trials only** (one randomly chosen ordering per stimulus). |
| **A_model_bias_shape_first.png** | Same as main but **only trials where the shape image was presented first** (Image 1 = shape match). |
| **A_model_bias_texture_first.png** | Same as main but **only trials where the texture image was presented first** (Image 1 = texture match). |
| **A_model_bias_sudo.png** | Same as main but **only pseudo-word trials** (word-like nonsense strings). |
| **A_model_bias_random.png** | Same as main but **only random-string trials** (fully random character strings). |

## Section B — Shape Bias by Stimulus

| File | What it plots |
|---|---|
| **B_stimulus_bias_main.png** | Dot plot of **proportion of "shape" choices per stimulus** (averaged across all models). Each stimulus ID on x-axis; same y-axis and chance line as model plots. |
| **B_stimulus_bias_supplement.png** | **Stacked bar chart** breaking down each stimulus's responses into shape / texture / unclear. |
| **B_stimulus_bias_deterministic.png** | Stimulus bias, **deterministic trials only**. |
| **B_stimulus_bias_random_ordering.png** | Stimulus bias, **random-ordering trials only**. |
| **B_stimulus_bias_shape_first.png** | Stimulus bias, **shape_first ordering only**. |
| **B_stimulus_bias_texture_first.png** | Stimulus bias, **texture_first ordering only**. |
| **B_stimulus_bias_sudo.png** | Stimulus bias, **pseudo-words only**. |
| **B_stimulus_bias_random.png** | Stimulus bias, **random strings only**. |

## Section C — Position Bias Validation

| File | What it plots |
|---|---|
| **C_position_bias_validation.png** | **Stacked bar per model** showing the proportion of paired trials (same stimulus+word, both orderings) where the model **tracks the correct image** (green) vs. shows **position bias** (red, always picks same position regardless of content). A model below 50% tracking is not reliably doing the task. |
| **C_position_bias_by_stimulus.png** | **Dot plot per stimulus**, colored by model, showing the **image-tracking rate**. Reveals which specific stimuli cause position-biased behavior in which models. |

## Section D — Word Sensitivity Validation

| File | What it plots |
|---|---|
| **D_word_sensitivity_validation.png** | **Stacked bar per model** showing what fraction of (stimulus, ordering) groups are **word-sensitive** (orange, model gives different answers for different words) vs. **word-insensitive** (grey, same answer regardless of word). A word-insensitive model is ignoring the word entirely. |
| **D_word_sensitivity_by_stimulus.png** | **Dot plot per stimulus**, colored by model, showing the **word-sensitivity rate**. Shows which stimuli trigger word-sensitive behavior. |
| **D_word_type_effect.png** | **Grouped bar chart per model** comparing shape-bias rate between **pseudo-words** (blue) and **random strings** (purple), with SE error bars. Tests whether word-likeness affects visual judgments. |
