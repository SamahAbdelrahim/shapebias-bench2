# MEMORY.md — shapebias-bench-2 session log

Local-only (gitignored). Read at the start of every session; add an entry after any significant decision. Newest entries first.

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
