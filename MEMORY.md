# MEMORY.md — shapebias-bench-2 session log

Local-only (gitignored). Read at the start of every session; add an entry after any significant decision. Newest entries first.

## 2026-07-24, qwen3.5-9b results + 27b 2-GPU sharding failed

**9b result (job 1655673, COMPLETED 18 min, clean):** qwen3.5-9b extends the passing family and is more robust than 4b. Generation-path tracking / mean shape (gate = trk ≥ 0.70): numeric similarity 0.83/0.92 PASS; numeric category 0.83/0.88 PASS (4b failed this at 0.13); numeric noun+shiple 0.87/0.93 PASS; A/B similarity 0.67/0.83 fail; A/B category 0.93/0.97 PASS (4b hard-locked at 0.03); A/B noun+shiple 0.60/0.80 fail. 9b passes 4 of 6 cells vs 4b's 3, and does not collapse under the "find another one" category framing that locks 4b. Where it tracks, shape rate is 0.88-0.97. Scaling the Qwen3.5 family up strengthens and stabilizes the shape preference. 9b is appended to the shared July-17 logs and shows in all rebuilt comparison HTMLs.

**27b 2-GPU bf16 FAILED (job 1655683, cancelled after 9 h).** device_map="auto" sharding of Qwen3.5's Gated-DeltaNet hybrid across 2 L40S produced nan logits, garbage generation, and ~625 s/trial (25 of 30 trials done in 9 h; would never finish). 9b on a single GPU with the same torch fallback (flash-linear-attention / causal-conv1d not installed) was clean and fast, so the cause is the sharding of the stateful linear-attention/conv layers, not the missing kernels. Cluster has only 48 GB L40S GPUs (4/node, nothing bigger), so 27b (~55 GB bf16) cannot run un-sharded on one GPU here. Cancelled the job and truncated the one broken partial 27b block out of `playground_smoke_30trials_shape_first_no_word_similarity.txt` (log now has the 7 originals + 9b, no 27b). Weights remain cached in scratch.

**Resolution (Samah chose 4-bit):** Run 27b in 4-bit (nf4, double-quant, bf16 compute) on a single L40S (~16 GB, no sharding). Added `bitsandbytes==0.49.2` to the uv venv + `requirements.txt`; added an opt-in `_quantization_4bit` flag on `_Qwen35Base` (builds a `BitsAndBytesConfig`); only `qwen3.5-27b` sets it (4B/9B stay bf16). Rewrote `scripts/run_qwen35_27b_ladder_30.sbatch` to `gres=gpu:1`, `mem=64G`, no dependency (9b/cancelled jobs are done), same 12-cell append to the shared July-17 logs. Submitted job 1656189 (oat-06, 1×L40S). 27b results will carry a quantization caveat vs the bf16 4B/9B rungs.

**27b 4-bit result (job 1656189, completed clean, 0 nan, ~2-3 s/trial, ~40 min):** gen tracking / mean shape (gate trk ≥ 0.70), PosFirst all 0.43-0.47 (no position lock): num similarity 0.87/0.83 PASS; num category 0.40/0.70 fail; num noun+shiple 0.87/0.93 PASS; A/B similarity 0.93/0.73 PASS; A/B category 0.50/0.72 fail; A/B noun+shiple 0.87/0.93 PASS. So 27b-4bit passes 4/6 cells (same count as 9b) but on different cells: it clears the A/B similarity and A/B noun+shiple cells 9b failed, yet regresses on both category ("find another one") cells that 9b passed, and its shape rates where passing are a bit lower (0.73-0.93). The 9b→27b step is not a clean monotonic strengthening; the differences are plausibly partly 4-bit quantization noise (logits much less peaked than bf16 9b), so 27b does not cleanly extend the trend beyond 9b. 9b remains the cleanest bf16 gain over 4b. All three rungs are in the rebuilt comparison HTMLs; 27b must be read with the quantization caveat.

## 2026-07-23 (later), add qwen3.5-27b on 2 GPUs (bf16)

**What was decided:** Add `qwen3.5-27b` (`Qwen/Qwen3.5-27B`, ~55.6 GB bf16, same `qwen3_5` type) as the next Qwen3.5 rung above 9B. It exceeds one 48 GB L40S, so it shards across 2 GPUs with `device_map="auto"` (bf16, no quantization). Added an optional `_device_map` class attribute to `_Qwen35Base` (`device_map = self._device_map or device`); only the 27B sets it to `"auto"`, so 4B/9B single-GPU behavior is unchanged. New script `scripts/run_qwen35_27b_ladder_30.sbatch` requests `gres=gpu:2`, `mem=120G`, `time=10:00:00`, and appends 27b into the same July-17 shared logs (same 12 cells) via `--resume`, then rebuilds the HTML. Submitted as job 1655683 with `--dependency=afterany:1655673` so it starts only after the 9B job releases the shared logs (avoids concurrent-append corruption). Downloaded the 27B weights into scratch. Verified: 27b registers with `_device_map=auto`; 9b/4b stay `None`.

**Why:** Samah asked to keep climbing the passing (Qwen3.5) family and take the 2-GPU bf16 route rather than quantize, keeping methodology identical to the existing bf16 ladder.

**What was rejected:** Running 27b concurrently with the 9B job (would race on the shared log files); quantization to fit one GPU (confounds scale with quantization).

## 2026-07-23, extend the Qwen ladder upward with qwen3.5-9b

**What was decided:** Add `qwen3.5-9b` (`Qwen/Qwen3.5-9B`, base native-multimodal, same `qwen3_5` model_type as the passing 4B) to the local ladder and run it under the same n=30 protocol as the existing rung. Registered it in `qwen35.py` reusing `_Qwen35Base` (no new class). Downloaded the full 18 GB (4 safetensors shards) into `/scratch/users/samahabd/hf_cache/huggingface` so the offline jobs can load it. New script `scripts/run_qwen35_9b_ladder_30.sbatch` appends only qwen3.5-9b into the existing July-17 shared session logs via `--resume` (numeric three-way: no_word_similarity / no_word_category / noun_label+shiple, plus the A/B three-way, both orders = 12 cells), then rebuilds the playground HTML. Submitted as job 1655673 (oat-03, 1×L40S, bf16). Resume-append verified on the login node: the shared numeric logs already have 7 complete models and the A/B logs 6, so `--resume` skips them and appends 9b instead of overwriting.

**Why:** The only playground gate-passer is `qwen3.5-4b` (numeric similarity trk 0.87 / shp 0.93). The direct test of whether the Qwen3.5 family keeps or strengthens the shape preference with scale is the next dense rung, 9B. Appending to the July-17 logs makes 9b appear automatically in the numeric-and-qwen8, prompt-compare, and label-set-effect HTML comparisons with no builder changes.

**What was rejected (for now):** Qwen3-VL-32B, Qwen3.5-27B, and the MoE variants (Qwen3-VL-30B-A3B, Qwen3.5-35B-A3B). Per Samah's choice they would run bf16 on 2 GPUs (device_map=auto) to keep methodology identical, but none fit bf16 on a single 48 GB L40S; 4/8-bit quantization was rejected because it confounds scale with quantization against the existing bf16 ladder. Only qwen3.5-9b fits one L40S in bf16 (~18 GB), so it is the first and only rung added this pass. There is no Qwen3-VL dense step between 8B and 32B.

## 2026-07-17 (late night), encoder fill + vision-vs-language report

**What was decided:** Ran embedding robust + Geirhos cue-conflict + simple readout for the three playground models missing July-10 encoder probes (`smolvlm`, `qwen3.5-0.8b`, `qwen3.5-4b`) as job 1645483 on oat-02 (`scripts/run_embedding_fill_qwen35.sbatch`). Outputs in `results/probe.results/session_2026-07-17_farmshare/embedding_*_fill.*`. Built `scripts/build_vision_vs_language_report.py` → `results/playground.results/vision_vs_language_2026-07-17.html`, which keeps vision-tower and language-side results in separate sections and evaluates the downstream claim.

**Encoder fill:** qwen3.5-4b proj_mean centred shape = 0.53 [0.37, 0.70], vit_penult = 0.50 with retrieval@1 = 1.00. Against today's gate-passing generation shape 0.82–0.95, gap ≈ 0.35. Matches the July-10 qwen3-vl-8b dissociation (behavior 0.83 vs embed 0.53). qwen3.5-0.8b leans texture (0.27); SmolVLM proj_mean = 0.10 and its ViT hidden-state path failed (`vit__err` unpack). Geirhos control for the fill recovers texture preference where retrieval is informative.

**Why:** Completes the encoder claim for the model that now carries every gate-passing playground result. Language-side naming and format effects cannot be attributed to vision-tower geometry.

## 2026-07-17 (late night), gated naming contrast

**What was decided:** Implemented `playgrounds/gated_naming_contrast.py`: pairs numeric `noun_label` (each of the five words) against numeric `no_word_similarity` (no word) per model, per stimulus, across both orders. Generation-level contrast interpreted only where both cells pass the tracking gate; latent contrast is per-stimulus swap-corrected P(shape) differences with 5000-resample bootstrap CIs and an exact sign test, reported for all 35 cells with gate status flagged. Outputs `gated_naming_contrast_2026-07-17.{csv,html}`. Reuses `playground_pride_debias.parse_log`.

**Findings:** Only qwen3.5-4b has a gate-passing no-word similarity baseline, so the strict comparison is qwen3.5-4b × its 4 passing words. Direction is the opposite of the category-baseline contrast: adding the noun LOWERS swap-corrected P(shape) from 0.88 to 0.68-0.80 (all four sign tests p < .001; e.g. shiple 1/29 stimuli favoring noun) and generation shape moves -0.12 to +0.02 (only shiple's CI excludes zero, at -0.117). So relative to the label-free similarity framing this model's shape preference is at or near its ceiling and the word subtracts slightly; the earlier "label raises latent shape" result holds only against the category ("find another one") baseline, whose own latent signal is depressed (0.56). In qwen3-vl-8b and 4b the latent noun-minus-similarity delta is positive (+0.09 to +0.17, bootstrap CIs mostly excluding zero) but per-stimulus sign tests are flat (13-16 of ~29), meaning a confidence shift on already-shape stimuli rather than flipped items; their similarity baselines also fail the generation gate, so these are latent-only observations.

**Why it matters:** The naming contrast is baseline-dependent. The word's effect looks facilitative against a degraded framing and null-to-negative against the best label-free framing. This is more consistent with the word re-engaging task compliance than with a child-like naming-specific shift toward shape.

## 2026-07-17 (late night), resume support + word-generality readout

**What was decided:** Added `--resume` to `playgrounds/smoke_test_playground.py` and to every playground sbatch script. On resume, the runner parses the existing log, treats a model as complete when its block has all `n_trials` "Results for trial" lines plus the `Unloaded <model>` marker and no load failure, skips those models, and appends the rest to the same log under a `===== RESUMED ... =====` marker. If every requested model is complete it exits 0 without touching the log. The HTML builder already keeps the last block per model, so a rerun of a half-finished model overrides the partial block. Verified the completeness parser against the finished 2026-07-17 session logs (all 7 models detected; n=31 probe returns empty).

**Why:** Job 1645405 ran 8+ hours; an interruption would previously have forced a full restart because logs were opened in write mode.

**What was rejected:** Per-trial resume within a model block (would need structured intermediate state; model loading, not trials, dominates runtime) and skipping via bash file-existence tests in sbatch (cannot distinguish complete from truncated logs).

**Word-generality readout (job 1645405 + PriDe rebuild, 98 cells):** The shiple pattern generalizes to all five sudo words. (1) A/B noun: 0/35 model-word cells pass the tracking gate; the letter-format collapse is word-general. (2) Numeric noun: passes concentrate in the largest Qwens (qwen3.5-4b 4/5 words, qwen3-vl-8b 4/5, qwen3-vl-4b 2/5; the four small models 0/5, qwen3.5-0.8b hard first-option locked at Pos1=1.00 for every word). (3) Among gate-passing cells, generated shape choice is uniformly high (0.82-0.95, mean 0.88) and does not vary meaningfully by word; word identity affects validity (tracking), not shape preference. (4) Debiased logits: qwen3-vl-8b keeps corrected P(shape) 0.74-0.97 across all words and both label sets even where generation is locked; qwen3.5-4b's latent signal is attenuated under letters (0.53-0.68 vs 0.59-0.83 numeric), so its letter collapse is not purely a response-format artifact; small models sit at ~0.50 corrected everywhere, so their failures reflect absent latent signal, not masking. (5) Naming contrast on matched framing (numeric noun vs numeric no-word category), swap-corrected mean P(shape): qwen3.5-4b 0.56 to 0.69 across words, qwen3-vl-8b 0.69 to 0.83; the novel label raises latent shape evidence relative to the same wording without a word, consistent across all five words.

## 2026-07-17 (late night), prompt PriDe + five-word generality

**What was decided:** Implemented `playgrounds/playground_pride_debias.py`, which parses saved one-pass probabilities from paired 30-trial logs and writes `prompt_pride_debias_2026-07-17.{csv,json,html}`. Method matches existing `pride_debias.py`: swap/full permutation on all 30; PriDe prior from first 10 and held-out estimates on 20. Added mean P(shape) as well as above-0.5 rates. Submitted job 1645405 (oat-02): remaining curated sudo words (`clapher`, `plailass`, `procation`, `adinefults`) × numeric/A-B noun prompts × 7 models × both orders. Shiple is reused. Builder will write `local_models_sudo_word_generality_30trials_2026-07-17.html`.

**Initial debias result:** All three gate-passing qwen3.5-4b cells retain shape evidence across estimators: numeric similarity swap/perm/PriDe mean P(shape)=.88/.89/.90(SF),.83(TF); AB similarity=.73/.74/.72,.73; numeric shiple=.69/.70/.71,.67. Qwen3-VL-8B has strong corrected latent shape probabilities despite generation gate failures (e.g. numeric noun swap .79, perm .83, PriDe .91/.79), but first-option priors are extreme (.01–.11), so order-specific PriDe disagreement and prior instability must be reported.

**Why:** Tests whether corrected latent choice evidence survives option bias and whether the noun-condition validity pattern is specific to `shiple`.

## 2026-07-17 (late night), numeric + qwen8 results

**What was decided/found:** Job 1645340 completed. Numeric labels change the picture in three ways. (1) qwen3.5-4b now passes the gate in two cells: numeric similarity (trk 0.87, shp 0.93) and numeric noun+shiple (trk 0.77, shp 0.82); the numeric no-word category cell still fails (trk 0.13, second-option lock). (2) Adding the noun raised tracking relative to the same-wording no-word category cell in 6 of 7 models (e.g. qwen3.5-4b 0.13→0.77, qwen3-vl-4b 0.07→0.63, qwen3-vl-8b 0.40→0.67); the label appears to re-engage the images. (3) qwen3-vl-8b passed no cell under playground prompts; best is numeric noun at trk 0.67, just under the 0.70 gate that the July 10 probe protocol had it passing (0.80). Numeric beats letters for qwen8 in all three framings. The category framing's second-option lock survives the label-set change (qwen3.5-4b catAB PosA 0.02, cat12 Pos1 0.07), so that lock is option-position, not letter identity. SmolVLM's gen-vs-logit dissociation persists numerically (sim 1/60, cat 0/60) but mostly resolves under the noun (43/60); all other models agree 60/60 everywhere.

**Why it matters:** The naming contrast is now partially interpretable in qwen3.5-4b, and the direction is unexpected relative to children: the word's clearest effect is on validity (image tracking), not on shape preference, and shape preference is slightly lower with the word (0.82) than under label-free similarity (0.93).

## 2026-07-17 (night), numeric labels + Qwen3-VL-8B follow-up

**What was decided:** Use the efficient design: run the three numeric conditions (`no_word_similarity`, `no_word_category`, `noun_label` + fixed `shiple`) on seven models, then fill only qwen3-vl-8b's three missing A/B cells. Job 1645340 runs 30 trials × both orders × both scoring paths on oat-02. The runner now infers A/B vs 1/2 from the prompt key, scores the actual labels, parses either set, and maps A/B stimulus ground truth to 1/2. Qwen8 A/B logs use distinct filenames so they cannot overwrite the completed six-model logs.

**Why:** This tests whether the locks are letter-specific and asks the naming contrast in qwen3-vl-8b, the Qwen-family gate-passer from the July 10 scaling ladder, without rerunning completed A/B cells.

**What was rejected:** Rerunning all six conditions on all seven models (duplicates 1,080 completed model-trials).

**Report:** `build_playground_results_html.py` will write `local_models_numeric_and_qwen8_30trials_2026-07-17.html` after all logs land: numeric condition tables, numeric wording/naming contrasts, and A/B-vs-1/2 comparisons for all three framings.

## 2026-07-17 (night), three-way prompt interpretation

**What was decided:** Wrote `results/playground.results/prompt_wording_interpretation_2026-07-17.html` interpreting the n=30 three-way (similarity / no_word_category_AB / noun_label_AB+shiple). Reading: only interpretable shape-bias cell is qwen3.5-4b under similarity (trk 0.73, shp 0.87, gen==logit 60/60, a genuine agreed pass unlike the July 10 dissociated near-pass). The noun condition collapses every model into letter locks (qwen3.5-4b trk 0.73 → 0.13, PosA 0.93), so the naming-linked-bias question is unanswerable at these scales; wording flips which letter models lock to (SmolVLM PosA 0.97 → 0.00 across conditions), supporting the language-side-artifact claim. SmolVLM gen-vs-logit agreement is itself prompt-dependent (60/60 shiple, 27/60 similarity, 2/60 no-word AB).

**Why:** Samah asked what the comparison means against the project's theoretical questions and what to do next.

**Next steps proposed:** numeric-label rerun of the three-way; add qwen3-vl-8b; swap/PriDe on saved one-pass logits; other four sudo words; fold wording manipulation into the manuscript audit section.

## 2026-07-17 (later), fixed-word noun_label_AB comparison

**What was decided:** Use option 1: one fixed curated sudo word, `shiple`, for all 30 stimuli. Job 1645318 runs the existing playground code with `noun_label_AB`, 30 trials, both orders, both scoring paths, and the same six local models. The smoke runner now accepts `--word`; word-bearing templates require it and include it in the result filename. The powered HTML report adds the shiple condition and pairwise comparisons when both logs finish.

**Why:** This isolates the effect of adding a novel category label from variation among words while matching the two completed n=30 prompt runs in every other respect.

**What was rejected:** Cycling the five sudo words (mixes prompt-word variation into the comparison); running all five (fivefold larger and unnecessary for this first test).

## 2026-07-17 (later), n=30 prompt compare results

**What was decided:** Job 1645312 completed (17 min, oat-05). Powered comparison in `local_models_prompt_compare_30trials_2026-07-17.html`. Finding: the wording change is not neutral. Under `no_word_category_AB`, every model fails the tracking gate; qwen3.5-4b drops from PASS (trk 0.73, shp 0.87) to trk 0.03 with PosA 0.02 (B-lock); smolvlm and qwen3-vl-2b move to near-total A-lock (PosA 0.97, 1.00). Similarity wording keeps qwen3.5-4b as the only gate pass, consistent with the n=5 smoke. SmolVLM gen-vs-logit dissociation worsens under AB (gen==two 2/60 vs 27/60 under similarity); two_pass==one_pass stays 60/60 everywhere. The follow-up HTML-rebuild job 1645313 failed (exit 127, `--wrap` env); rebuilt locally instead.

**Why:** n=5 hinted at gate flips; n=30 confirms the AB "find another one of the two" wording pushes small local models into letter/position locks rather than image-based choices.

## 2026-07-17 (late), n=30 dual no-word prompt compare

**What was decided:** Registered `no_word_similarity_AB` in `PROMPT_TEMPLATES` (exact July 17 similarity wording). Smoke accepts `--prompt-condition`. Submitted `scripts/run_prompt_compare_30.sbatch` (30 trials × 2 orders × similarity + category_AB × 6 models; scratch HF; exclude oat-01). HTML builder writes `local_models_prompt_compare_30trials_2026-07-17.html` when both n=30 log pairs exist.

**Why:** n=5 suggested prompt wording moves gate outcomes; need a powered comparison with a confirmed shared system prompt.

**Prompt contract confirmed:** local VLMs share `LOCAL_VLM_SYSTEM_PROMPT` in generate + score_choices; both user prompts are no-word A/B.

## 2026-07-17 (evening), AB smoke HTML + similarity comparison

**What was decided:** Built `local_models_smoke_no_word_category_AB_2026-07-17.html` from job 1645305 logs, with a side-by-side gen-path comparison to the similarity-prompt smoke. Updated similarity HTML section 2 to link there.

**Why:** Samah asked whether unifying to `no_word_category_AB` changed smoke validity / shape rates vs the earlier similarity wording.

## 2026-07-17 (late), AB smoke hang = oat-01 NFS + home I/O

**What was decided:** Cancelled 1645294 and 1645299. Keep scratch `HF_HOME` + offline Hub in `run_smoke_dual_path.sbatch`, and add `#SBATCH --exclude=oat-01`.

**Why:** 1645299 had scratch HF_HOME set correctly but still printed nothing for ~20 min: python stuck in `D` / `rpc_wait_bit_killable` importing home `.venv` (not model load yet). Both hung AB jobs landed on oat-01; completed similarity smoke was oat-02 (~16 min).

**What was rejected:** Waiting on oat-01; copying full `.venv` to scratch in this pass (larger follow-up).

## 2026-07-17 (evening), Playground HTML results report

**What was decided:** Added `scripts/build_playground_results_html.py` writing `results/playground.results/local_models_smoke_similarity_2026-07-17.html` (shape_first + texture_first picks, shape rates SF/TF/avg, tracking, PosA, gate) plus a copy of `probe-experiment-results.html`, embedding readout, and PriDe tables. AB-prompt companion HTML auto-builds when job 1644290 logs appear. Re-run: `python scripts/build_playground_results_html.py`.

**Why:** Samah asked for probe-style HTML covering smoke validity + probe-era readouts in one browseable folder under `playground.results`.

## 2026-07-17 (later), Playground prompt unified to no_word_category_AB

**What was decided:** All local playground / smoke entry points now use `make_prompt(prompt_condition="no_word_category_AB")` from `eval_core.PROMPT_TEMPLATES` (notebook, `smoke_test_playground.py`, `run_local_playground_smoke.py`, `test_local_models.py`). The July 17 dual-path smoke used the older hardcoded similarity prompt ("Which of the other two images ... more similar ..."), not this template.

**Why:** Notebook already had the AB no-word wording hardcoded; smoke and sibling scripts still used a different similarity string, so results were not comparable to the notebook.

**What was rejected:** Leaving separate hardcoded prompt strings per file (drifts again).

## 2026-07-17, Unify Adam's one-pass + system prompt across local VLMs; results layout 3C; dual-path smoke

**What was decided:**
- Shared `LOCAL_VLM_SYSTEM_PROMPT` (alias `QWEN35_VLM_SYSTEM_PROMPT`) on all four Transformers local wrappers (`qwen35`, `qwen`, `smolvlm`, `internvl`) for both `generate` and `score_choices`, so logit scoring cannot drift from generation the way qwen3.5-0.8b did before Adam's fix.
- One-pass `generate(..., choice_texts=...)` already present on all four; left in place. `score_choices` kept for debugging / `run_trial_logit_scoring`.
- Standardized local rerun no longer monkey-patches `generate`; it only sets `_system_prompt = REMOTE_UNIFORM_SYSTEM_PROMPT` on each class.
- Results layout (option 3C): defaults write under `results/model.results/`, `results/playground.results/session_YYYY-MM-DD_farmshare/`, `results/probe.results/session_*/`. Documented in `results/README.md`. Migrated July 10 scattered files into that layout.
- Smoke (options 1C + 2C): `playgrounds/smoke_test_playground.py` runs two_pass and one_pass, both orderings. Slurm job `scripts/run_smoke_dual_path.sbatch`.

**Why:** Adam fixed qwen3.5 by putting the system prompt on logit scoring and merging generate+score into one pass. Extending the same contract to every local Transformers VLM avoids the same inconsistency elsewhere. Unified gitignored results paths let collaborators reproduce without sharing data dumps.

**What was rejected:**
- Leaving SmolVLM / InternVL / Qwen3-VL without a system message (would keep generate vs score_choices asymmetric only on families that never had one).
- Keeping the standardized runner's full `generate` monkey-patches (would drop one-pass logits and fight the unified `_system_prompt` attribute).
- Dumping new smoke logs at `results/` root (conflicts with the July 11 probe.results decision).

**Open:** tinyllava (deprecated) and levante-runtime left without one-pass/score_choices. qwen3.5 `score_choices` tokenization aligned to generate (`enable_thinking=False`); re-smoke 1642264: qwen3.5-0.8b and 4b are 10/10 gen==two_pass==one_pass on both orderings.

## 2026-07-11, Probe-era results organized; audit + sensitivity analyses; manuscript started

**What was decided:**
- New results home for FarmShare probe-era runs: `results/probe.results/` with `session_2026-07-10_farmshare/` (extracted from `farmshare/sb_results.zip`) and `analysis/` (threshold sensitivity, audit table). Kept separate from `results/model.results/` because probe runs use playground scripts, not the benchmark pipeline, and are not loaded by `load_data.R`. Layout documented in new `results/README.md`.
- Gate-threshold sensitivity (`playgrounds/threshold_sensitivity.py`): swept 0.50-0.90 over 38 cells. Finding: all passing cells at any threshold 0.50-0.80 are noun+numeric; loosening admits only chance-level cells; qwen3-vl-8b bootstrap P(pass@0.70)=0.94, tracking CI [0.63,0.91]. The 0.70 gate affects power, not conclusions. The 24 probe cells are cell-level transcriptions from the canvas (per-trial JSON still on FarmShare).
- Literature audit table (`results/probe.results/analysis/audit_table.csv` + notes): papers classified by measurement locus (embedding / single-image classification / positioned choice) x which artifacts each locus can express. Framing rule adopted: do NOT claim prior work is debunked; embedding and single-image loci are structurally immune to position/selection bias; the claim is non-comparability across loci plus language-side origin of the artifacts.
- Canvas converted to `farmshare/probe-experiment-results.html` (self-contained, opens in any browser; the `.canvas.tsx` renders only inside Cursor). Corrections vs canvas: the "50% in the one pass" tile now shows gen 0.82 vs logit 0.50 as a dissociation; scaling/dissociation/PriDe/sensitivity/audit sections added; roadmap statuses updated.
- Manuscript started in `manuscript/` (added to `.gitignore`): `VENUES.md` (recommendation: CogSci 2027 for Parts 1-2, Open Mind for the full version; ICLR 2027 only if stimulus scale-up lands by September) and `main.md` (working abstract, intro/background, current studies + contribution, Part 1 methods/results, references with [verify] flags, [FS] flags on numbers transcribed from session logs).

**Why:** The July 10 FarmShare session produced the paper's core results (23/24 gate failures; Qwen crosses at 8B; behavior-embedding dissociation; estimator-invariant positive) but the record lived only in chat logs and a Cursor-only canvas. The audit table + sensitivity analysis were the two missing robustness pieces identified before manuscript writing could start.

**What was rejected:**
- Putting the probe runs inside `results/model.results/` (would mix pipeline-loaded CSVs with playground outputs).
- Publishing the HTML as a claude.ai artifact (private data; a local file serves the need).
- Writing the manuscript in LaTeX now (venue not locked; markdown ports to any template).
- Fabricating per-trial bootstrap for the 24 probe cells (per-trial JSON not local; cells marked cell-level only until synced).

**Open items:** sync `probe_experiment.json` + `pride_debias.csv` from FarmShare (scp in `results/README.md`); verification pass on flagged citations (Tartaglini venue, Pezeshkpour venue, Gershkoff-Stowe year, Gavrikov author order, Lu/Muttenthaler/Portelance author lists); read Kim & Lee 2026 (arXiv 2603.10834) before citing; manuscript Part 2 prose; audit-package items 2-3 (reproduce a published positioned-option protocol under correction; PriDe-prior-as-language-side-bias figure).

## 2026-07-11 (later), Expanded the audit table to the full shape-bias-in-models corpus

**What was decided:** Grew `results/probe.results/analysis/audit_table.csv` from 11 to 22 papers, driven by the reading list in `interpret/literature-review/` (paper.txt, papers list.rtf) plus a comprehensive search. Added the classic CNN cluster (Ritter 2017, Hosseini 2018, Hermann/Chen/Kornblith 2020, Geirhos 2021, Li/Wen/Li/Lee 2023 NeurIPS Oral, Subramanian et al. 2023 NeurIPS Oral), the emergent-language route (Portelance 2021 CoNLL), recent vision work (Heinert 2025 cue-decomposition, Golpayegani 2024, Lu et al. 2026 Nat Mach Intell, Muttenthaler et al. 2025 Nature), and the baseline-critique papers (Hermann & Firestone 2022 JOV, Kim & Lee 2026 unread). Organized by 5 measurement loci: embedding/representation, single-image cue-conflict classification, single-image VQA/captioning (VLM), emergent-communication referential game, positioned 2AFC — plus a methodological-critique group. Propagated into `audit_table_notes.md`, the HTML section 6 (grouped-by-locus table, 22 rows), and manuscript Part 1 audit paragraph + reference list.

**Why:** Samah asked for comprehensive coverage of computational-model shape-bias papers. The locus grouping is the load-bearing device: the classification/embedding papers are structurally immune to the position/selection critique, so the paper's claim must be non-comparability across loci, not debunking.

**What was rejected:** adding Vong/Lake CVCL, Islam 2021, Tuli 2021 for now (listed as candidates in the notes; add only if they earn a place in the argument). Reading the two large ACL/MPG PDFs inline (fetch tool size limits; used landing pages and search instead).

## 2026-07-10, Merged shapebias-bench-2 into shapebias-bench2 (bench2 as base)

**What was decided:**
- Copied files that existed only in `shapebias-bench-2/` into `shapebias-bench2/` via `rsync --ignore-existing` (~95 files: `interpret/`, `ai-impact-grant/`, `archive/`, `MEMORY.md`, `REPORT.md`, `PROJECT_CHECKLIST.md`, plus some `results/` and notes).
- On any path present in both trees, kept the `shapebias-bench2` version (12 content diffs left untouched, including `.gitignore`, `eval_core.py`, local model wrappers, playground notebooks, `scripts/run_local.py`).
- Did not stage or commit anything. Previously untracked files stay untracked. Left `shapebias-bench2/.gitignore` unchanged.

**Why:** One working tree for FarmShare smoke/probe work and the grant/interpret docs; `shapebias-bench2` already had the live GPU/playground state and should win conflicts.

**What was rejected:**
- Replacing `.gitignore` with the `shapebias-bench-2` version (would have ignored `MEMORY.md`/`REPORT.md`/`archive/` more cleanly, but user asked to keep bench2's `.gitignore`).
- Overwriting overlapping code/notebooks from bench-2.
- Any `git add` / commit.

**Side effect:** `MEMORY.md`, `REPORT.md`, and `archive/` now show as untracked under bench2's current `.gitignore` (bench-2's ignore listed them; bench2's does not). `interpret/`, `ai-impact-grant/`, and `PROJECT_CHECKLIST.md` remain ignored.

## 2026-07-10, Repo reorganization, living report, and gitignore hardening

**What was decided:**
- Created `archive/` (gitignored) and moved into it: `results/results copy/`, `temp/` screenshots, `human-experiment/output.log`, the debug outputs in `human-experiment/reports/`, `interpret/Onboarding.html` (rendered duplicate of RA_ONBOARDING.md), and `interpret/cursor_shape_bias_model_behavior_discus.md` (exported chat log where the logit-forced method was worked out). Item-by-item notes in `archive/README.md`. Nothing deleted.
- Untracked (git rm --cached, files kept on disk): all `.DS_Store` files, `temp/`, `human-experiment/output.log`, `human-experiment/reports/*`. These deletions are staged but NOT committed.
- Rewrote `.gitignore`: added `.DS_Store`, `*.Rhistory`, `.RData`, `archive/`, `MEMORY.md`, `REPORT.md`, `human-experiment/output.log`, `human-experiment/reports/`; removed dead entries (wrong `shapebias-bench-2/...` prefixes, absolute path). Private material (results, interpret, ai-impact-grant, checklists) stays off GitHub.
- Created `REPORT.md` (repo root, local-only): living report with the theoretical question, hypotheses, per-model status table, results snapshot, phase roadmap, and anticipated team merges. Update it as results land.
- Created `interpret/RA_mentoring/REPO_GUIDE.md`: walkthrough of every folder and file for new team members.
- Ticked the "Clean up results/results copy/" item in `PROJECT_CHECKLIST.md`.

**Why:** New RAs are joining (Adam: computational/logit track; Andrew: interactive human experiment, Vercel migration). The repo needed a single current entry point, private material verified off GitHub, and legacy files out of the way before team branches start merging.

**What was rejected:**
- Rewriting `README.md` (Samah chose to keep the old one; inconsistencies flagged instead: title says "CNN and VLM" but no CNNs are evaluated; it points to a repo-root `STIMULI_GUIDE.md` that only exists in `stimuli_pipe/`; the structure tree at the bottom is a stale fragment; it links `interpret/` files that are not on GitHub).
- Untracking `human-experiment/node_modules/` (1,716 tracked files). Deliberate vendoring per the encapsulation guarantee in `human-experiment/README.md`; revisit during the Vercel migration.
- Moving/archiving legacy results CSVs (`remote_all.csv`, `no_word_pilot_*`): `analysis_pipe/src/load_data.R` lists them as optional merge inputs; moving them changes `canonical_combined_eval.csv` row counts.
- Archiving `word_list/words.csv`: unreferenced by code, but it is the provenance list for `WORD_PAIRS` in `eval_core.py`.
- Touching `evaluation_pipe/models/local_models/`, `scripts/`, or `human-experiment/` code: Adam's logit PR and Andrew's backend work land there; restructuring now would create merge conflicts.

**Open questions for Samah:**
- `interpret/` is fully gitignored, but `RA_ONBOARDING.md`, `mentoring_plan.md`, and `REPO_GUIDE.md` are written FOR the RAs, who clone from GitHub. Either share those files directly, or whitelist them in `.gitignore` (e.g. `!interpret/RA_ONBOARDING.md`).
- The staged untrackings (`.DS_Store`, temp, logs) need a commit to take effect on GitHub.
- `results/model.results/human_matched/` still needs to be copied from the remote machine.
