# Plot Descriptions

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
