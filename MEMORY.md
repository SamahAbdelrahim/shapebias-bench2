# MEMORY.md — shapebias-bench-2 session log

Local-only (gitignored). Read at the start of every session; add an entry after any significant decision. Newest entries first.

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
