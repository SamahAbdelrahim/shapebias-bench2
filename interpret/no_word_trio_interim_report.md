# No-Word Contrast Report (Diagnostic Trio, Full Completion)

## Dataset status

- Baseline: `/home/samah/projects/shapebias-bench-2/results/remote_all_fixed.csv` (noun-label, full remote benchmark)
- No-word full-match: `/home/samah/projects/shapebias-bench-2/results/no_word_full_remote_trio_dedup.csv`
- Computed contrast table: `/home/samah/projects/shapebias-bench-2/results/no_word_trio_interim_contrast.csv`

- `qwen3.5-9b` coverage: 600/600 (100.0%)
- `qwen3.5-27b` coverage: 600/600 (100.0%)
- `qwen3.5-122b-a10b` coverage: 600/600 (100.0%)

## Key contrasts (no-word minus noun-label)

| Model | delta shape | delta tracking | delta ordering effect | no-word unclear |
|---|---:|---:|---:|---:|
| qwen3.5-9b | -0.326 | +0.645 | -1.013 | 0.000 |
| qwen3.5-27b | -0.334 | -0.339 | +0.411 | 0.133 |
| qwen3.5-122b-a10b | -0.020 | -0.040 | +0.083 | 0.297 |

## Interpretation

- `qwen3.5-9b`: strong decrease in shape bias under no-word condition, with improved image tracking and reduced ordering bias.
- `qwen3.5-27b`: strong decrease in shape bias under no-word condition, with lower tracking and increased unclear responses.
- `qwen3.5-122b-a10b`: near-stable/slightly decreased shape bias overall, but with elevated unclear responses in no-word trials; interpret this contrast with caution.

These completed results support a model-dependent account: removing labels changes behavior for some models but not all.