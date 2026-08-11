# MEMORY.md — shapebias-bench-2 session log

Local-only (gitignored). Read at the start of every session; add an entry after any significant decision. Newest entries first.

## 2026-08-10, Cue-conflict dropped from the human experiment; grid-only, 31 trials

**What was decided:** Humans run the novel texture grid alone. `cc_triads` comes out of the human protocol (models keep it). A session is now 27 grid trials plus 4 attention checks, 31 trials, close to the March pilot's 30. Sample target drops from 244 to 135 for k = 16 observations per triad per condition. Condition and ordering group stay as they were; the block-order counterbalance falls away with only one set. Same `design=matched_v2`, `pool_version=v2`, since no data had been collected under the two-set version.

**Why:** The cue-conflict images are familiar named categories (airplane, cat, elephant). Telling an adult "this first image is a *rilas*" while showing an obvious elephant asks them to accept a second name for something already named, and the available referent for the new word is the texture. That would push responses toward texture for a lexical reason rather than a perceptual one, and only in that set, so the noun manipulation is not comparable across sets. Two further problems were specific to it: 160 triads drew on 108 distinct shape exemplars, so about a fifth of sessions showed one photograph twice under two different pseudo-words; and the set contrast confounded novelty with texture type (material versus another object's surface), resolution and rendering style.

**What was rejected:**
- Keeping cue-conflict without pseudo-words, i.e. running it in the no-word condition only. That drops the cell that makes the set interesting and leaves an unbalanced design.
- Deleting the cue-conflict selection code. It sits behind `--include-cc` in `scripts/build_human_trial_pool.py` (off), and `pickSetOrder` reads whatever sets the pool carries, so restoring it is a pool rebuild.
- Keeping the session at 40 trials by giving all 36 test trials to the grid (would have needed only 101 participants), and shortening it to 22. 27 was chosen to match the pilot length.

**Bug this surfaced:** the contiguous-window selection repeated a shape in 106 of 135 simulated sessions. 114 grid triads are emitted round-robin over 30 shapes, so the final round is partial (shapes 1 to 24) and a window wrapping past the end returns to shapes it already used. In the noun condition that means one object with two pseudo-words, the same defect found in cue-conflict. Selection now walks shapes rather than pool indices: 27 distinct shapes per session, each shape's own triads rotated per participant so all 114 are still covered. `verify_assignment.js` gained a repeated-shape check that fails the run. Verified at 135: 0 repeated shapes, all four condition-by-group cells filled, every triad covered in both orders in both conditions, 16.3 and 15.6 observations per triad.

## 2026-08-10, Human experiment rebuilt as matched_v2 (two stimulus sets, two framings)

**What was decided:** Replace the March 2026 human anchor with a new protocol, `design=matched_v2`, `pool_version=v2`. Participants run 18 novel-grid triads and 18 `cc_triads` (blocked, block order counterbalanced) plus 4 attention checks, 40 trials, about 7 to 8 minutes. Noun-label versus no-word-category between participants; option order counterbalanced between participants; unique pseudo-words retained per trial.

Trial selection moved offline into `scripts/build_human_trial_pool.py`: 114 grid triads reproducing `build_grid_triplets(seed=0)` exactly (verified identical, so the human items are the same ones behind the embedding panel), and 160 `cc_triads` at 10 per Geirhos class from the frozen n=320 subset, congruent pairs skipped. Images export to 384 px WebP under `human-experiment/public/stimuli/`, 870 files, 6.2 MB, so nothing at run time touches `/scratch`. Counterbalancing lives in a shared `public/assignment.js` so it can be checked in Node (`verify_assignment.js`) rather than only in a browser. Deployment is static plus two Vercel functions against Atlas; the Express servers stay for local testing.

**Why:** The old anchor was one scalar, 0.952, from 28 adults on 30 stimuli in one condition and one option order, drawn as a flat dotted line across every panel of fig2 including framings and AB variants no human ever saw. It could not be compared cell-for-cell with the model results: different stimulus package, no no-word baseline, no ordering swap, and an item-level human-model correlation of r = 0.083.

**Two real bugs found and fixed while building this:**
- `hash(seed|salt) % 2` made condition and ordering group perfectly correlated. FNV-1a ends in a multiply by an odd prime, so the digest's low bit is near-linear in the input bytes and two salts give the same coin. Every noun-label participant would have been ordering group A, so no triad would ever have appeared in both orders within a condition and the counterbalance would have been silently void. Factor assignment now goes through mulberry32 (`seededUnit`). Caught only because the simulation was rerun with realistic random Prolific IDs.
- The committed `human-experiment/node_modules/` had every package's `dist/` stripped, so `index.html` pointed at jsPsych files that did not exist. Only the local preview server worked, by redirecting to unpkg. `vendor_jspsych.sh` now pulls the published browser bundles into `public/vendor/`.

**What was rejected:**
- Pooling the March pilot with the new sample. Different stimulus package, no no-word condition, no ordering counterbalance. Kept separate by `design` and `pool_version`.
- Fixed option order per participant, which is what the models do. It would have let a side preference at the participant level masquerade as shape bias. Order now alternates within a participant and flips by group, so the item-level counterbalance is exact and the participant-level confound is gone.
- Reusing the models' fixed word `shiple`. Word identity moves validity, not shape rate, and unique words are what prevent carry-over.
- Congruent `cc_triads` (`dog1-dog2`): both options come from one class, so there is no shape-versus-texture answer. They stay in the model subset, labelled rather than dropped.
- Keeping the scalar human line in fig2. Anchors are now placed only in the cells humans were measured in.

**Not done, needs your accounts:** deployment and the Prolific launch. See `human-experiment/LAUNCH_RUNBOOK.md`. Also note R is not installed on this machine, so the new functions in `analysis_pipe/src/human_analysis.R` were written but never executed; the figure side was tested against a schema fixture, which was deleted afterwards so no fabricated human numbers remain in `results/data/`.

## 2026-08-10, fig7b split into two figures to fix overlap

**What was decided:** Replace the 2×3 `fig7b_stimulus_sets` with two 1×3 figures: `fig7b_sets_behavior` (behavior across the four sets per framing) and `fig7b_sets_emb_vs_behavior` (embedding vs behavior per framing). Deleted the old combined PNG/PDF and updated the report figure list. In the emb-vs-behavior figure the model legend now uses neutral grey markers, since colour encodes stimulus set and only marker shape encodes model.

**Why:** In the 2×3 version the rotated model names on the top row collided with the bottom row panel titles, and the set/model legends sat on top of the data.

**What was rejected:** Keeping one figure with a taller canvas and larger hspace; dropping model names from the top row.

## 2026-08-10, Fig6/7/7b faceted by numeric framing (similarity / category / noun)

**What was decided:** Option 1. Same figure layouts, three numeric framings side by side (AB left out). Fig6 is 1×3 PriDe bars; fig7 keeps one shared embedding panel A then three emb-vs-behavior panels; fig7b is 2×3 (behavior rates / emb-vs-behavior × framing).

**Why:** Embeddings are prompt-free so Panel A need not repeat; behavior and gate-pass do shift by framing (grid gate: similarity 2/14, category 5/14, noun 4/14).

**What was rejected:** Full 6-cell AB×framing grid; separate files per condition.

## 2026-08-10, Full-grid fig7/fig7b extended to four stimulus sets

**What was decided:** Option A. Rewrote `fig7_vision_vs_behavior` (Panel A: embedding shape rates for novel grid / Smith / cc_triads / decomposition; Panel B: grid emb vs grid behavior) and added `fig7b_stimulus_sets` (Panel A: behavioral similarity across the four sets; Panel B: emb vs behavior, color=set, marker=model). Wired both into `build_full_grid_report.py`. Old Geirhos class-folder cue-conflict is not in these panels.

**Why:** Playground had fig7 / fig7b for two-set comparisons; full_grid only had a three-panel scatter that still pointed at old Geirhos. Four-set behavioral + embedding data were already complete.

**What was rejected:** Keeping Geirhos as a fifth series; waiting for probe job 1679628 before regenerating (behavioral/embedding figs do not need NPZ probes).

## 2026-08-09, Remaining linear probes queued on CPU after emb-export

**What was decided:** Leave GPU tasks `1678529_10` (`qwen3.5-2b`) and `1678529_13` (`qwen3.5-27b`) running, then finish any still-missing probe JSONs on CPU via `scripts/run_remaining_probes_cpu.sbatch` with `--dependency=afterany:1678529` (job **1679628**). That job also rebuilds readout-power figures at the end.

**Why:** NPZ export for timed-out tasks (`qwen3-vl-2b`, `qwen3.5-0.8b`) already finished; only probes were incomplete. Probing does not need a GPU. Waiting avoids racing or canceling the two still-running array tasks.

**What was rejected:** Canceling 10/13 to move probes to CPU immediately; leaving the probe gap until a later ask.

## 2026-08-09, NPZ embedding export extended to the full 14-model ladder

**What was decided:** Run `scripts/run_grid_embedding_export.sbatch` on all 14 ladder models so the linear-probe / read-out-power analysis is not limited to the original four (qwen3.5-4b, qwen3.5-9b, qwen3-vl-8b, smolvlm). The array now sources `scripts/model_ladder.sh` (`--array=0-13%4`), skips GPU re-export when `${MODEL}_*.npz` already exists, and only runs `linear_probe.py` for missing probe JSONs.

**Why:** Cosine JSONs alone cannot settle the read-out-power objection; saved vectors are required for the probe. Filling NPZ for everyone keeps that test aligned with the behavioral ladder.

**Submitted:** emb-export array on FarmShare (queued behind the Smith ladder if GPUs are full). Outputs stay under `results/probe.results/session_readout_power/embeddings_grid/`.

## 2026-08-09, Smith behavioral+logit+PriDe backfill for the full 14-model ladder

**What was missing:** Smith had embeddings for all 14 models but generation only for the old 9 (playground `.txt`), no `run_local` CSVs, no logit session, and PriDe only from the 9-model playground path. Original Geirhos still cannot support 2AFC generation in its class-folder layout; that gap was already closed by the edited triad sets (`cc_triads` / `decomposition_triads`), which have the full readout stack, while original Geirhos embeddings remain 14/14 at n=30.

**What was decided:** Run Smith through `scripts/run_local.py --smith-probe` for all 14 models, matching the embedding sample (n=30, seed=0), six cells × both orders, generation then logit (`swap_correct` off). Finalize writes PriDe, summary, and `smith_ladder_report.html`. Submitted array **1678392** and finalize with `--dependency=afterany:1678392`.

**What was rejected:** Re-running original Geirhos as behavioral 2AFC without the triad rebuild (still the wrong layout); waiting on NPZ export / 30-set playground backfill in this pass (separate from the Smith/Geirhos readout gap the user asked to fill).

## 2026-08-08, Full ladder results in; reports regenerated for 14 models + cue-conflict triads

**What finished:** All submitted jobs completed with exit 0. Completeness for the agreed matrix is full: grid gen/logit 84/84 cells, cue-conflict gen/logit 168/168, embeddings 14/14 on grid+Smith+Geirhos and on both triad sets, PriDe 84 rows for grid and for each triad set.

**Reporting updates:** Extended `MODEL_ORDER` (and colors/markers) to 14 in `full_grid_figures.py` and `build_full_grid_report.py`. Regenerated fig1–7 and `full_grid_v1a_report.html`. Added `analysis_pipe/cueconflict_summary.py` and `scripts/build_cueconflict_report.py`; wrote per-set summaries and `cueconflict_triads_report.html`, linked from the playground index.

**Grid gate (generation):** 20/84 PASS. The old 9-model subset is still 18/54. Of the five new models only `internvl-14b` contributes (2/6); the other four are 0/6. `qwen3.5-4b` remains the only model with 6/6; 9b 5/6; 27b 2/6; Qwen3-VL-4B 3/6; 8B 2/6. Gate-passing noun_label shape rates stay high for the Qwen3.5 family (0.80–0.91).

**Cue-conflict triads (n=320):** cc_triads 35/84 PASS; decomposition 40/84. More models clear the tracking gate here than on the novel-object grid, but shape rates among passers are lower and closer to 0.5–0.7. Decomposition rates run higher than cc_triads for several models, consistent with the format confound (isolated object vs cluttered photo). `internvl-14b` passes all 6 on both triad sets but only 2/6 on the grid.

**Still deferred:** Smith/Geirhos behavioral+logit via `run_local`, NPZ export beyond 4 models, 30-set playground backfill for the 5 new models.

## 2026-08-07, Model ladder centralized; five registered models were never being run

**What was found:** `internvl-2b`, `internvl-8b`, `internvl-14b` and `smolvlm-256m` register correctly in `MODEL_REGISTRY` (16 keys) but appear nowhere in `results/model.results/`. Nothing was broken in the registry or the wrappers. Every runner hardcoded its own copy of the same 9-entry `MODELS=(...)` array, and the InternVL sizes were never added to it, so `--models` never received them. The three InternVL variants had already been downloaded to scratch and smoke-tested at n=30 on 2026-07-30 (`ret1` 0.7 / 0.9 / 0.8), then not promoted. `internvl` alone means InternVL3-**1B**, so the grid's InternVL rung was the 1B all along.

**What was decided:** One shared list in `scripts/model_ladder.sh`, sourced by `run_full_grid_v1a`, `run_full_grid_logit_v1a`, `run_full_grid_embedding_v1a`, `run_ab_label_fix_benchmark_30`, `run_ab_label_fix_smith_30`, and `run_embedding_bench_smith_31`. Family-grouped, ascending within family, now 14 models: added `smolvlm-256m`, `internvl-2b`, `internvl-8b`, `internvl-14b`, `qwen3.5-2b`. Downloaded Qwen3.5-2B (4.3 GB) so the Qwen3.5 rungs run 0.8B / 2B / 4B / 9B / 27B with no gap. All 14 verified to resolve from cache under `HF_HUB_OFFLINE=1`.

**Why sorted rather than appended:** an index that maps onto a position on the scale ladder is worth more than index stability, since resume keys off the output filename and not the array index, so existing CSVs are untouched by renumbering. Cost: `qwen3.5-9b` moved from array index 7 to 12. `bash scripts/model_ladder.sh` prints the mapping.

**Deliberate exclusions, now documented in the ladder file:** `qwen3-vl-32b` (no bf16 fit on one 48 GB L40S, 2026-07-23 entry), `tinyllava` (not imported, incompatible with current transformers), `levante-runtime` (wrapper, not a rung).

**What was rejected:** `--models all` as the fix, since it pulls in `levante-runtime` and any uncached model and crashes under `HF_HUB_OFFLINE=1`; centralizing `run_grid_embedding_export.sbatch` and `WORD_MODELS`, which are deliberate subsets rather than ladders.

## 2026-08-07, Edited cue-conflict triad sets reviewed, wired in, and run

**What arrived:** two zips, `cc_triads` and `decomposition_triads`, 1,280 triads each, three 224×224 PNGs per triad, nothing missing. The 1,280 `reference.png` files are byte-identical across the two sets, so the sets differ only in what the two answer options are. Staged on `/scratch/users/samahabd/cue_conflict_triads`, symlinked into `previous-lit-stimuli/cue-conflict/` (gitignored), following the grid staging pattern.

**Review, and why cc_triads is primary:** in `cc_triads` all three images are isolated objects on white. In `decomposition_triads` the options are the two source photographs, so reference and `shape_match` are both isolated objects on white while `texture_match` is a cluttered full-frame photo. A model preferring the matching image format will look shape-biased for reasons unrelated to shape. Its foils are also heavily reused: 160 unique shape options and 48 unique texture options across 1,280 triads, versus 850 and 830 in `cc_triads`, so foil identity should be a random effect. 80 of the 1,280 pairs are within-class (`dog1-dog2`), which are congruent rather than conflicting.

**What was built:** no existing loader could read these folders, because every entry point either filters on `d.name.isdigit()` or sorts by `int(d.name)`. Added `load_stimuli_triad_dir()` in `eval_core.py` returning lazy path records like the grid loader, plus `--triad-dir` on `run_local.py` and `embedding_robust.py`. The folder name is parsed into `stl_id` and `texture_set`, so shape and texture identity reach the CSV and congruent trials stay identifiable. The embedding builder returns `(triplets, meta)` like `build_grid_triplets` and labels by Geirhos *class*, which turns on `retrieval_mesh_at1` / `retrieval_texset_at1` against a 1/16 baseline.

**Scope agreed with Samah:** fixed 320-triad subset (20 per class, seed 0, manifest at `results/data/cueconflict_triad_subset_n320.csv`), identical ids in both sets so the two sets and all readouts pair trial for trial; congruent trials kept and labelled rather than excluded or oversampled (16 land in the subset, 5.0% against a 6.2% population rate, which is thin as a control); generation + logit + embeddings on all 14 models, plus the 5-model grid backfill. Rejected: the full 1,280 (~172 GPU-h versus ~43), and the Smith/Geirhos backfill and NPZ export, deferred.

**Two bugs found on the way.** `run_full_grid_embedding_v1a.sbatch` pointed at `previous-lit-stimuli/geirhos_cueconflict`, but that directory moved under `cue-conflict/`; since the step is guarded by `if [[ -d "$CUE" ]]`, the next run would have silently skipped every cue-conflict embedding and still exited 0. And `load_completed_trial_keys` required a non-empty `word`, so the four no-word conditions were unresumable and a requeued task appended a second copy of every row instead of skipping. Verified the finished grid sessions have zero duplicates (those tasks never needed to resume), then fixed the guard to require only model / stim_id / ordering.

## 2026-08-02 (latest), Read-out power objection: concede the scope, build the probe, fix the retrieval control

**What was decided:** Answer reviewer point 8 ("cosine is a fixed unweighted read-out") as an experiment rather than a caveat. Narrowed the encoder claim everywhere it was overstated, and built the linear-probe / read-out-power pipeline that decides it. Scope agreed with Samah: 4 models (qwen3.5-4b, qwen3.5-9b, qwen3-vl-8b, smolvlm), mode A only, code + claim revisions + response memo + the §4.1 numeric fixes.

**The concession:** cosine weights all 1,152 dims equally, so it cannot distinguish "shape absent from the encoder" from "shape present but low-weighted." What §4.1 licenses is only that shape is not a dominant axis of the encoder's *unweighted cosine geometry*. Previous phrasings ("not a property of the encoder's similarity structure", "made downstream of the vision encoder", "establish robustly") were narrowed in `manuscript/main.md` (abstract, §2, §4.1, §4.4), `REPORT.md:74`, `interpret/null-result-July10th-2026.md` (§1c, §1c-robustness), `scripts/build_master_interpretation_report.py`.

**The amendment we add to the reviewer's framing:** their dichotomy (absent vs. underweighted) misses a third outcome — shape present AND strongly weighted, yet the texture candidate still nearer in the *directed* comparison. Decoding one image and ranking three points are different questions. Our own n=30 retrieval control (shape-match retrieval@1 0.77-1.00 vs 1/30 chance, texture-match 0.10-0.23) already points there. H3 would relocate the claim from what the encoder contains to how it arranges it, which is the more interesting version, not a weaker one. Reading rules for all three outcomes are pre-committed in the memo before the numbers land.

**Defect found in our own sensitivity control (important):** `_retrieval_at1` demands the *exact* stimulus. At n=30 each mesh appears once so that equals mesh-identity retrieval; on the 114-triad grid `build_grid_triplets` round-robins across 30 meshes so each recurs ~4x, and a same-mesh/different-texture neighbour is scored as an **error**. The 0.05-0.15 we reported, and defended in the manuscript as "the probe is not dead," is largely a metric artifact, not lost sensitivity. Confirmed on synthetic data where mesh identity is perfectly decodable: exact@1 = 0.000 while mesh@1 = 0.797 on the same vectors. Added `_retrieval_by_label` (mesh-level and texture-set-level, chance 1/30 and 1/38); kept the old metric so prior numbers stay reproducible.

**Split design (the load-bearing technical piece):** the grid's foils are deterministic offsets, not random — texture-match foil = mesh s+15 (mod 30), shape-match foil = texture t+19 (mod 38). Both are involutions, so meshes form 15 pairs and textures 19. Splits must be built from whole pairs or a triad's foils leak across the boundary. Ladder uses 5 mesh-folds x 5 texture-folds = 25 blocks, covering all 1,140 cells exactly once, with test meshes and test textures both unseen. Note for the response: holding out *meshes* is undefined for a 30-way mesh classifier (cannot predict an untrained class); it is only coherent in the metric-learning formulation, which is why the design has both a probe and a ladder.

**Verified before any GPU time:** built the reviewer's hypothetical to spec (mesh in 20 of 1,152 dims, texture in the other 1,132). Centred cosine 0.00, learned metric 1.00, mesh probe 1.00 on identical triads. A second regime with shape dominant returns 1.00 at every rung, so the ladder does not manufacture a climb. `python playgrounds/linear_probe.py --self-test`.

**Two factual errors in §4.1 corrected** (independent of the objection, verified against the JSONs): "CIs include 0.5" is false for 12 of 32 cells, all below 0.5; the "0.20 to 0.55" range excludes SmolVLM proj_mean 0.08 [0.04,0.13]. The direction survives (no cell's CI is above 0.5) but the accurate summary is "no encoder is shape-biased, and those departing from chance depart toward texture," which is a *stronger* dissociation than a null. Left flagged, not fixed: gate-pass count disagrees across three places (16 in `full_grid_v1a_summary.csv`, 18 in MEMORY.md, "36 of 54 fail" in main.md).

**What was rejected:** running all 9 models (4 carry the argument; the clean within-model dissociations are only 4b/9b); both render modes (mode A keeps comparability to the published 114-triad sample); silently rewriting the overclaims without a record (flagged in place with dated scope notes per Samah's flag-don't-rewrite preference); asserting a result before the FarmShare export (§5 of the memo and §4.5 of the manuscript are explicitly PENDING).

**New/changed files:** `playgrounds/linear_probe.py`, `playgrounds/embedding_export.py`, `scripts/run_grid_embedding_export.sbatch`, `analysis_pipe/readout_power_figure.py`, `interpret/reviewer_response_readout_power.md` (all new); `playgrounds/embedding_robust.py`, `scripts/export_results_csv.py`, `requirements.txt` (+numpy/pandas/scipy/scikit-learn, previously undeclared). Local `.venv` created for the CPU analysis; the export itself needs FarmShare since grid images are not on this machine.

## 2026-08-02 (later), Manuscript Part 2 written from full-grid + three-set embedding results

**What was decided:** Wrote the Part 2 section of `manuscript/main.md` (§4) around the full-grid behavioral run and the three-stimulus-set embedding read-outs, and updated the abstract, §2 contribution, §3.2 stimuli note, §3.3 (new "full stimulus scale" replication paragraph), references (added Gatys, Ecker & Bethge 2016), and the Open-slots list. No code or run scripts touched, per Samah's instruction. Numbers taken from `results/data/full_grid_v1a_summary.csv` and `results/probe.results/session_full_grid_v1a/embedding_{grid,smith,cueconflict}_*.txt`.

**The load-bearing argument (§4):** hold the embedding read-out fixed and vary only the stimulus set. Novel grid (114 stratified triads): every encoder near chance (centred shape 0.20-0.55, CIs cover 0.5), retrieval above chance so the probe is live — a behavior-vs-encoder dissociation, since gate-passing Qwen3.5 models choose shape on 80-94% of behavioral trials. Smith (30 triads): same read-out returns shape (0.53-0.80, retrieval 0.30-0.67). Geirhos cue-conflict (30 triads): same read-out returns the canonical texture bias (0.03-0.13 at deep layers). Conclusion: representational "shape bias" is a property of the stimulus set, not the model; the familiarity ordering reproduces Tartaglini et al. 2022.

**Full-grid behavioral numbers used:** 54 cells, 36 fail the 0.70 gate; Qwen3.5-4B passes all 6 (shape 0.80-0.91), 9B 5/6, 27B 2/6 (4-bit caveat), Qwen3-VL-4B 3, 8B 2; four small models never pass (position locks). By-item structure from `full_grid_v1a_by_shape_texture.csv`.

**Geirhos method verified online** (arXiv 1811.12231 + rgeirhos/texture-vs-shape): 1,280 style-transfer composites (Gatys 2016), single-image 16-class classification, shape bias = correct-shape / (correct-shape + correct-texture), non-conflict images excluded. It is NOT a 2AFC. Wrote this into §4.3 plus the construction confound (composites cannot form a clean reference/shape-match/texture-match triad; candidates differ in silhouette/background), which is the methodological argument for using the rendered grid as the developmental substrate and Geirhos only as a representational positive control.

**Also written:** anticipated-reviewer subsection (§4.4: dead-probe, cross-set confound, power asymmetry, text-prior, 4-bit) and next-steps (§4.5: full-grid PriDe still running; full-scale embedding rerun + layer/pooling sweep; human anchor on the grid; behavioral Smith ladder; direct locus demonstration).

**Pending flags left in the manuscript:** full-grid PriDe/forced-logit numbers (job running; generation CSVs have empty prob_ columns); reconcile [FS] n=30 Part 1 numbers with grid where they overlap; re-check n=30 embedding points against merged JSONs after the finalize job.

**What was rejected:** deleting the n=30 Part 1 (kept as the audit/first pass, added a scale replication paragraph instead); correcting the existing intro sentence on Geirhos (already accurate as single-image classification); showing draft options before writing (task was fully specified and edits are additive/reversible).

## 2026-08-02, Merge local data from shapebias-bench2 into shapebias-bench-2 (option A)

**What was decided:** Keep `shapebias-bench-2` as the surviving checkout. Both folders already share the same git remote and HEAD (`0fbff6d`). Copy only paths that were missing here from `/Users/samahabdelrahim/git-repos/shapebias-bench2` (additive; never overwrite). Skip `.env` and `.schrodinger`. Do not delete the old folder until Samah confirms.

**What was copied:** lit stimuli (`smith_stimuli`, `geirhos_cueconflict`, `all_smith_stimuli_sets.zip`); full-grid / playground / probe result trees and tidy CSVs; `results/figures`; texture-grid packages under `stimuli_pipe/`; `interpret/archive`, `null-result-July10th-2026.md`, `RA_mentoring/`.

**What was rejected:** Overwriting conflicting files (including the larger `farmshare/stimuli.zip` already here); copying secrets (`.env`); deleting `shapebias-bench2` in the same step; taking newer docs (`CLAUDE.md`, `results/README.md`) from the other checkout.

## 2026-08-02, Full-grid reporting package + PriDe/embedding GPU jobs

**What was decided:** Build paper-facing figures and HTML from the completed 123,120-trial generation CSVs, and submit GPU follow-ups for logit/PriDe and vision embeddings (grid sample + Smith + Geirhos) so the full-scale report matches the 30-set analysis stack. Keep the 30-set playground figures/HTML untouched; full-grid outputs live under `results/figures/full_grid/` and `full_grid_v1a_report.html`.

**Offline (done):** `analysis_pipe/full_grid_figures.py` (fig1–5 + stubs for fig6–7), `scripts/build_full_grid_report.py` (patched into playground `index.html`), tidy CSV at `results/data/full_grid_tidy_behavioral.csv`. Gate readout unchanged: 18/54 PASS; qwen3.5-4b passes all 6.

**GPU submitted:** job **1673782** (`run_full_grid_logit_v1a.sbatch`, array 0–8%4) → `session_full_grid_v1a_logit/`; job **1673783** (`run_full_grid_embedding_v1a.sbatch`) → stratified n=114 grid + Smith n=30 + Geirhos n=30 per model under `probe.results/session_full_grid_v1a/`; job **1673784** finalize (dependency afterok both) merges embedding JSONs, runs `full_grid_pride.py`, rebuilds figures/HTML.

**Why a separate logit session:** generation CSVs have empty `prob_*` columns (`decision_mode=2afc`). PriDe needs `logit_forced` absolute probs; writing a second session avoids rewriting finished generation files.

**What was rejected:** Pointing `playground_figures.py` at grid CSVs in place (would break the 30-set paper path); embedding all 1,140 triads (use stratified 114); changing `load_data.R` to ingest 123k rows.

## 2026-08-01, Full texture grid (1,140 trials) evaluated through evaluation_pipe

**What was decided:** Scale model inference from 30 stimuli to the full texture grid produced by `triad-stimuli-pipeline` (30 ALICE shapes x 38 textures), scoped to **v1, mode A only** and the **same 6 cells** as the 30-set design (similarity / category / noun+`shiple`, each numeric and A/B, both orders). 1,140 x 2 x 6 = 13,680 trials per model, 123,120 across the 9-model ladder. The playground keeps using the 30-set package for testing; expansion runs only through `evaluation_pipe` / `scripts/run_local.py`.

**The blocking difference:** the grid inserts a texture level, so a trial is `<stl_id>/<texture_set>/` not `<stl_id>/`. Both existing loaders scan for numbered directories holding images directly (`d.name.isdigit()`, then `trial_dir / "reference.png"`), so neither reads it. `load_stimuli` also decoded every image up front, which at 1,140 triads is roughly 10 GB.

**What was built:**
- `stimuli_pipe/stimuli_texture_grid_v1` symlink into the pipeline output, plus `stimuli_texture_grid_v1_scratch` pointing at a 671 MB `/scratch` stage (hardlinks preserved). Both gitignored. Deliberately **not** named `stimuli_unique_texture_per_stl_v1`, which already exists in bench2 as the old flat 30-folder human-matched package.
- `load_stimuli_grid()` + `materialize_stimulus()` in `eval_core.py`: manifest-driven, returns paths, images opened one triad at a time. 1,140 records now cost 40 MB instead of ~10 GB.
- `--grid-pkg` and `--word` on `run_local.py`. The triad is decoded once per stimulus (not once per word) and only after the resume check, so a full resume does zero image I/O.
- `stl_id` and `texture_set` added to `CSV_FIELDS`, and `write_results` now reconciles against a CSV's existing header when appending. Without that, resuming any pre-2026-08-01 CSV would have silently shifted every column after `stim_id`.
- `scripts/run_full_grid_pilot.sbatch` (kept as the timing/schema reference), `scripts/run_full_grid_v1a.sbatch` (9-model array), `analysis_pipe/full_grid_summary.py`.

**Cluster limit found:** the `gpu` QOS caps this account at `gres/gpu=4` and 4 running jobs (32 submittable). So `%4` in the array is the hard limit, not a tuning choice; more concurrency is not available. Partition wall is 2 days.

**Pilot (job 1672131, qwen3-vl-2b, 40 stimuli, L40S):** 0.44 s/trial steady-state, 1.48 s/trial on the first cell including cold model load. Projects to ~1.7 h for that model's full 13,680 trials, and ~8-11 h for qwen3.5-27b-4bit at its recorded 2-3 s/trial. Pilot readout: `no_word_similarity` tracking 0.60 / shape 0.50 / pos1st 0.70, `noun_label+shiple` tracking 0.21 / shape 0.43 / pos1st 0.90 — both fail the 0.70 gate with a position lock, consistent with qwen3-vl-2b never passing at n=30.

**Why manifest-driven rather than a deeper directory scan:** the manifests already carry `stim_id`, `stl_id`, `texture_set` and the foil identities (`shape_match_texture_set`, `texture_match_stl_id`), so the join is authoritative and does not re-derive trial structure from paths.

**What was rejected:** changing `load_stimuli` / `load_trials` in place (six callers including `run_remote.py` and both standardized reruns; the new loader sits alongside instead); v2 and mode B in this pass (v2 is the same renders with different foils, a separate question, and reusable by changing `GRID_PKG`); the full 10-word benchmark (22,800 trials per model per condition, and word generality was already answered at n=30); a stratified subsample instead of the exhaustive grid; wiring these CSVs into `load_data.R` (its `get_candidate_result_paths()` is a fixed legacy list, and mixing 123k grid rows into `canonical_combined_eval.csv` would swamp it).

**Submitted:** job **1672132** (`scripts/run_full_grid_v1a.sbatch`, array 0-8%4, all 9 models). Queued behind other users' 2-day GPU jobs; Slurm estimated first start ~2026-08-02T05:42. Resume is per model x cell, so requeueing after any failure is safe. Check with `squeue -u $USER` and `results/model.results/jobs/grid_v1a_1672132_<task>.out`; read results with `python analysis_pipe/full_grid_summary.py`.

**Open:** `analysis_pipe/playground_figures.py` still only reads playground `.txt` logs, so grid results do not appear in the paper figures yet.

## 2026-07-31, Fig 7b + embedding fill job (benchmark vs Smith, full ladder)

**What was decided:** Add `fig7b_benchmark_vs_smith` (panel A: vision-tower shape rate for novel benchmark vs Linda Smith probe across all 9 ladder models; panel B: embedding vs behavioral similarity, filled=benchmark / open=Smith). Existing embedding JSONs only covered 7 models and had no Smith run, so submitted job **1670938** (`scripts/run_embedding_bench_smith_31.sbatch`): (1) qwen3.5-9b + 27b fill on IMAGE_DATASET → `session_2026-07-31_farmshare/embedding_robust_fill_large.json`; (2) full 9-model ladder on `previous-lit-stimuli/smith_stimuli` → `embedding_smith_probe.json`. Rerun `python analysis_pipe/playground_figures.py` after the job finishes to fill pending cells. Fig 7 (novel vs Geirhos) kept as-is.

**What was rejected:** Replacing fig 7; waiting to draw fig 7b until numbers land (skeleton with pending is useful now).

## 2026-07-31, Repo cleanup (archived legacy) + Python figure pipeline

**What was decided:** While the AB reruns finish (1670811/1670812 running, 1670813 finalize queued), archived legacy code and stale results, and added `analysis_pipe/playground_figures.py` (matplotlib, installed into `.venv` with pandas via uv) producing 8 paper-quality figures into `results/figures/playground/` plus a tidy CSV `results/data/playground_tidy_behavioral.csv`. Figures documented in `analysis_pipe/PLAYGROUND_FIGURES.md`; the script re-reads sessions on every run, so rerunning it after the Smith/finalize jobs land fills the pending cells (Smith noun AB, 2026-07-30 PriDe).

**What was archived:** `scripts/_archive/` (22 files: `run_evaluation.py`, old smoke wrappers, March/April diagnostics, all July 17-25 `*_30.sbatch` jobs, `build_vision_vs_language_report.py`, `update_models_validity_md.py`, `compute_order_bias_validity.py`); `playgrounds/_archive/` (`gated_naming_contrast.py`, `threshold_sensitivity.py`, 3 notebooks); `results/model.results/_archive_pilots/` (reasoning March dump, no_word smoke/pilot/interim CSVs not read by load_data.R); `results/playground.results/_archived_early_sessions/` (July-10 5-trial smoke session, one .partial file). Empty `configs/*.yaml` stubs deleted from git. Each archive folder has a README.

**What was rejected:** Archiving `pride_debias.py` (imported by `playground_pride_debias.py`), the `no_word_pilot_remote*.csv` / `no_word_full_remote*.csv` files (load_data.R reads them), the July 17-25 numeric session logs (still the current numeric baseline), and `compute_model_validity_split.py` / `compute_validity_human_matched.py` (human-matched track still planned).

## 2026-07-31, AB rerun jobs failed on 27b OOM; restored 4-bit + resume

**What happened:** Jobs 1670605 (benchmark) and 1670606 (Smith) FAILED; 1670607 finalize CANCELLED. Cause: each job completed 8 models on `shape_first` `no_word_similarity_AB`, then `qwen3.5-27b` CUDA OOM (~44 GB). Smoke exit 1 + `set -e` aborted the rest. Root cause: Adam's Jul-28 merge dropped `_quantization_4bit = True` / `_device_map = "auto"` from `Qwen35_27B`.

**What was decided:** Option 1 — restore 27b 4-bit, add `--resume` to both AB sbatch scripts, continue past per-cell smoke failures (exit non-zero only if any cell still failed), resubmit the three-job chain. Partial Jul-30 logs kept so resume skips the 8 finished models on the first cell and retries 27b.

## 2026-07-30 (later), AB label-fix full rerun + master interpretation report

**What was decided:** Submitted parallel Slurm reruns for all archived AB conditions with fixed Image A/B slots (jobs 1670605 benchmark, 1670606 Smith; finalize 1670607 after both). New sessions: `session_2026-07-30_farmshare` (benchmark AB three-way + sudo-word AB × 4 words × 7 models) and `session_2026-07-30_smith_farmshare` (Smith AB three-way). Finalize job runs PriDe on both sessions, probe_experiment AB-only rerun, `build_playground_results_html.py`, and new `build_master_interpretation_report.py`.

**Scripts added:** `run_ab_label_fix_benchmark_30.sbatch`, `run_ab_label_fix_smith_30.sbatch`, `run_ab_label_fix_finalize.sbatch`, `submit_ab_label_fix_rerun.sh`, `build_master_interpretation_report.py`. Master outputs: `master_interpretation_2026-07-30.html` + `.csv`. Smith AB HTML now prefers newest `session_*smith*` folder.

**Scope not rerun:** Numeric 1/2 logs (still valid). Embedding/probe numeric cells unchanged. Archived AB stays in `_archived_ab_label_mismatch_pre_2026-07-28/`.

## 2026-07-30, Archive mismatched AB runs + match image-slot labels everywhere

**What was decided:** Confirmed Adam's fixes are on main (`365d2f8` AB slot labels; `602c9b5` `--geirhos-unaltered`). Archived option A: AB-only raw logs + AB-derived reports into `results/playground.results/_archived_ab_label_mismatch_pre_2026-07-28/` (Jul 17/24/25 AB sessions, catAB job logs, pride/compare/generality HTML+CSV). Numeric logs left live. Mixed reports (`*_numeric_and_qwen8_*`, `vision_vs_language_*`) and probe_experiment mixed files left in place with a README note that AB cells there are invalid until rerun.

**Consistency fixes shipped:** (1) smoke dual-path free-text `generate` now passes `choice_texts`; (2) `run_local.py` defaults `choice_texts` to A/B when prompt ends with `_AB`, else 1/2; (3) remote `build_openai_compatible_vision_messages` takes `choice_texts` like local, threaded through `run_remote.py` and the standardized remote rerun; (4) `probe_experiment.py` free-text `generate` also passes `choice_texts`.

**Why:** Pre–Jul 28 AB runs asked for A/B while labeling slots Image 1/2. Dual-path free-text and remote still leaked that mismatch after the local fix. Reruns of archived AB conditions are required before interpreting letter-format results.

**What was rejected:** Archiving whole sessions (would bury usable numeric logs); moving mixed numeric/embedding HTML into the archive by default.

## 2026-07-25, Smith ladder resume into session_2026-07-25

**What happened:** Job 1656533 stalled overnight on qwen3-vl-8b during numeric noun texture_first (~3:48 AM, no further output) and was cancelled at 08:37. Midnight had also split logs: similarity+category under `session_2026-07-24_smith_farmshare/`, noun under `session_2026-07-25_smith_farmshare/`.

**What was decided:** Keep the July-25 session as canonical. Moved all Smith smoke logs (plus the pre-truncate backup) into `session_2026-07-25_smith_farmshare/` and removed the empty July-24 Smith folder. Incomplete qwen3-vl-8b block was already truncated so --resume sees 4 complete models on that log (smolvlm, internvl, qwen3-vl-2b, qwen3-vl-4b) and continues from qwen3-vl-8b. Fixed `run_smith_ladder_30.sbatch` to pin `RESULTS_SESSION_DATE=2026-07-25`, pass `--out` into that session for every cell, and write PriDe/HTML as `smith_*_2026-07-25.*`. Smoke and `default_session_results_dir` now honor `RESULTS_SESSION_DATE` so a midnight rollover cannot split one job. Builder `SESS_SMITH` points at the July-25 session.

**Resubmitted:** Job 1659833 (`scripts/run_smith_ladder_30.sbatch`, resume mode, oat-03). Remaining: finish numeric noun texture_first (5 models), then all three A/B conditions × 2 orders, then PriDe + HTML.

## 2026-07-24 (evening), playground.results tidy + Smith full ladder queued

**What was decided:** Flat layout with corrected Jul-24 names for the two composite reports rebuilt today (`local_models_numeric_and_qwen8_30trials_2026-07-24.html`, `local_models_prompt_compare_30trials_2026-07-24.html`). July-17-only reports keep `_2026-07-17` names. Sbatch stdout/stderr moved to `results/playground.results/jobs/`. Added `index.html` landing page. Smith stimuli extracted to `previous-lit-stimuli/smith_stimuli/` (150 JPGs). Added `load_smith_trials()` + `--smith-stimuli` on smoke playground; Smith logs go to `session_2026-07-24_smith_farmshare/`. Builder will write `smith_*_2026-07-24.html` + `smith_prompt_pride_debias_2026-07-24.*` when job completes.

**Smith run scope (user chose full ladder):** Same 6 cells as local numeric report — similarity / category / noun × numeric + A/B — 30 trials × 2 orders × 9 models, revised July-24 category wording, PriDe at end. Job 1656533 (`scripts/run_smith_ladder_30.sbatch`, 24 h, 1× L40S).

**What was rejected:** Dated subdirs or full `reports/`/`sessions/` restructure (kept flat + index to avoid breaking open paths and minimize builder churn).

## 2026-07-24 (later), no_word_category prompt revision rescues large-model validity

**What changed:** Samah rewrote both category prompts in `eval_core.py`. Old (July-17 runs): "See this object in the first image. Can you find another one of the two (1 or 2 / A or B)?" New: "You are given three images. This first image is an object. Which of the following two images (1 or 2 / A or B) is another one?" Only the wording changed. (Note: the file was mid-edit when first inspected; verified against the runtime string each job logged.)

**What was rerun:** Both `no_word_category` (numeric) and `no_word_category_AB` on the full ladder (smolvlm, internvl, qwen3-vl-2b/4b/8b, qwen3.5-0.8b/4b/9b, qwen3.5-27b-4bit), n=30 × both orders, into `session_2026-07-24_farmshare` (jobs 1656289, 1656290; single L40S each). July-17 old-prompt logs kept as baseline. Comparison saved to `results/playground.results/no_word_category_prompt_revision_2026-07-24.csv`.

**Result (gen trk / shp, gate 0.70):** The old category wording was depressing validity, not shape preference. Numeric: qwen3.5-4b 0.13/0.57 fail -> 0.93/0.97 PASS; qwen3.5-9b 0.83/0.88 -> 1.00/1.00; qwen3.5-27b(4bit) 0.40/0.70 -> 0.63/0.82 (still under gate); qwen3-vl-4b 0.07 -> 0.70 PASS; qwen3-vl-8b 0.40 -> 0.77 PASS; the four small models stay ~chance (absent signal, not prompt artifact). A/B: qwen3.5-4b 0.03 -> 0.60 (still under gate), 9b stays 0.83 PASS, 27b 0.50 -> 0.60, qwen3-vl-4b -> 0.70 PASS. So the revision flips the numeric category from a large-model failure to a large-model pass; the letter (A/B) format still adds difficulty that keeps 4b/27b just under the gate. 27b numbers carry the 4-bit quantization caveat.

**Not done (now done):** Regenerated canonical HTML from the July-24 revised category logs. `find_smoke_pair` now prefers the newest session; `try_build_numeric_qwen8_report` loads category (numeric + AB) from `session_2026-07-24_farmshare` while similarity/noun stay July-17. Rebuilt: `local_models_numeric_and_qwen8_30trials_2026-07-17.html`, `local_models_prompt_compare_30trials_2026-07-17.html` (and other builder outputs). Old July-17 category logs kept as baseline; CSV comparison still at `no_word_category_prompt_revision_2026-07-24.csv`. n=5 AB smoke page still shows the old wording (n=5 was not rerun).

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
