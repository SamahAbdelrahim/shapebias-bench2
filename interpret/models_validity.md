## Model Validity Check (Remote + Local, updated run)

This summary reflects a rerun of the validity gates using the updated model outputs in:

- `results/model.results/local_eval.csv`
- `results/model.results/remote_all_fixed.csv`
- `results/model.results/remote_all.csv`
- `results/model.results/no_word_pilot_remote_dedup.csv`
- `results/model.results/no_word_pilot_remote.csv`
- `results/model.results/no_word_full_remote.csv`

Canonical combined rows after de-duplication: **9808**

### Validity criteria (same as `analysis_pipe/src/validity_gates.R`)

- **Valid** if all hold:
  - image tracking rate >= `0.70`
  - word sensitivity rate >= `0.20`
  - parse quality >= `0.97`
- **Borderline** if image tracking rate >= `0.50` but not fully valid
- **Invalid** otherwise

### Updated outcomes

- **Valid models:** 1 / 11
- **Borderline models:** 1 / 11
- **Invalid models:** 9 / 11

#### Per-model labels

- `qwen3-vl-4b` -> **valid**
- `qwen3.5-4b` -> **borderline**
- `internvl` -> invalid
- `smolvlm` -> invalid
- `qwen3-vl-2b` -> invalid
- `qwen3.5-0.8b` -> invalid
- `qwen3.5-9b` -> invalid
- `qwen3.5-27b` -> invalid
- `qwen3.5-35b-a3b` -> invalid
- `qwen3.5-122b-a10b` -> invalid
- `llama4-scout` -> invalid

### Notes

- This update is written to `results/data/model_validity_summary.csv`.
- Shape-bias interpretation should rely on the validity-gated subset first, then compare against full-sample behavior as a robustness check.
