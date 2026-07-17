# Shape Bias Benchmark — Living Project Report

Local-only (gitignored). Update this file whenever results, model runs, or project phase change. Last updated: **2026-07-10**.

---

## 1. Theoretical question

Do vision-language models show a shape bias in novel word extension, and if so, is it naming-linked the way it is in children?

The shape bias is usually described as a fact about how children treat objects. The stronger reading (see `interpret/theoretical_motivation_vision_language_shape_bias.md`) is that it is a fact about how naming interacts with perception: the bias appears, or appears more strongly, when a label is in play. If that is right, VLMs are a natural experiment on the question developmental theory actually cares about: what kind of information gives rise to a word-learning inductive bias. VLMs have web-scale image-text statistics but none of the social, ostensive learning a child has. If a model reproduces the bias and its naming-linked signature, the statistical route may be sufficient; if it shows only a label-free visual preference, that points the other way.

The project also carries a methodological claim (now the centerpiece of the grant proposal): running developmental paradigms on AI models produces valid data only with behavioral validity checks, the same logic developmental researchers apply to children. A model that always picks the left option looks like data but is not doing the task.

## 2. Hypotheses and predictions

1. **Label dependence (the core prediction).** If a model's shape preference is naming-linked (child-like), removing the novel word from the prompt should reduce shape choosing. A model whose shape preference survives label removal has a label-free visual heuristic, not the developmental bias.
   *Current status: supported in a model-dependent way. Two of three trio models drop sharply without the label; the largest is near-stable.*
2. **Validity gating changes conclusions.** Aggregate shape-choice rates conflate visual preference with response heuristics (position bias, parse failures). Gated and ungated conclusions should differ.
   *Current status: supported. 6 of 11 models fail gates under generation-based prompting, almost always on image tracking.*
3. **Measurement mode matters.** Forced-choice language responses and internal probabilities (logits) should dissociate for models with strong response habits; logit probing with swap correction should recover a coherent preference where generation looks like noise.
   *Current status: supported for SmolVLM (invalid under generation; stable 40.4% shape rate, 0% order gap under logit-forced + swap correction). Extension to other models in progress.*
4. **Planned (grant 2x2).** The label-driven shift toward shape should be larger in systems trained with language supervision than in vision-only systems, tested by crossing training regime (vision-only vs vision+language) with test condition (label vs no label).
5. **Planned (AI norming).** VLM free-form descriptions embedded in a semantic space should recover the perceptual dimensions of the stimulus set and predict human word-extension choices, replacing the human rating study that precedes each new paradigm.

## 3. Task and measures

3-image 2AFC: reference object + shape match + texture match; the model/participant is asked which candidate the novel pseudo-word names ("1" or "2"). Conditions: noun-label vs no-word; orderings counterbalanced (shape_first / texture_first).

**Response measures:**
- **Forced-choice language response**: the model generates "1"/"2"; parsed into shape/texture/unclear.
- **Logit probing** (open local models only): probability of token "1" vs "2" at the decision position; **swap correction** averages each trial with its position-flipped twin to cancel side bias by design.

**Validity gates** (from `analysis_pipe/src/validity_gates.R`):
- Image tracking >= 0.70 (choices follow image identity under position swaps)
- Word sensitivity >= 0.20 (responses change when the word changes; N/A in human-matched protocol, one word per stimulus)
- Parse quality >= 0.97 (answers are parseable)
- Valid = all three; borderline = tracking >= 0.50 but not all; invalid = tracking < 0.50.
- A separate order-bias gate exists for the standardized local benchmark (`interpret/order_bias_validity.md`).

## 4. Models: who has been run, and where they stand

Registry names as used in the results CSVs. "Local" = run on our GPU from downloaded weights (open weights, logits accessible); "Remote" = Hugging Face router / provider APIs (text output only). Several small models ran both ways (same HF IDs registered in both runners; numerical parity not guaranteed, see `interpret/model_choice_decision_log.md`).

| Model | Family / spec | Local or remote | Word-benchmark validity (generation) | Shape-bias status and notes |
|-------|---------------|-----------------|--------------------------------------|------------------------------|
| `qwen3-vl-4b` | Qwen3-VL 4B Instruct (Alibaba) | Local (also registered remotely) | **Valid** | Interpretable under generation; primary small-model candidate. |
| `qwen3.5-27b` | Qwen3.5 27B | Remote | **Valid** | No-word trio member: shape drop -0.334 without label, but tracking fell and unclear rose (13%); no-word run itself invalid. |
| `qwen3.5-35b-a3b` | Qwen3.5 35B MoE (approx. 3B active) | Remote | **Valid** | Valid in human-matched runs too. No-word extension still to run. |
| `qwen3.5-122b-a10b` | Qwen3.5 122B MoE (approx. 10B active) | Remote | **Valid** | No-word trio: near-stable shape (-0.020) but 29.7% unclear in no-word; label-independent shape preference, interpret with caution. Valid in human-matched runs. |
| `qwen3.5-4b` | Qwen3.5 4B | Local (also remote) | Borderline (tracking 0.63) | Usable with caution. |
| `qwen3.5-9b` | Qwen3.5 9B | Remote | Invalid (tracking 0.10) | But: no-word trio member; without the label tracking improved (+0.645) and shape dropped -0.326. Valid in human-matched runs. The clearest naming-linked profile so far. |
| `smolvlm` | SmolVLM2 approx. 2.2B (Hugging Face) | Local (also remote registry) | Invalid (tracking 0.20) | **Logit-forced + swap correction: stable 40.4% shape (slight texture bias), 0% order gap.** The "invalid" label was partly a measurement artifact. |
| `internvl` | InternVL3 1B (OpenGVLab) | Local (not on default remote router) | Invalid (tracking 0.18) | Logit-scoring port in progress (Adam). |
| `qwen3-vl-2b` | Qwen3-VL 2B Instruct | Local (also remote) | Invalid (tracking 0.23) | Candidate for logit-forced re-run. |
| `qwen3.5-0.8b` | Qwen3.5 0.8B | Local (also remote) | Invalid (tracking 0.32) | Candidate for logit-forced re-run. |
| `llama4-scout` | Llama 4 Scout (Meta, served via Groq) | Remote | Invalid (tracking 0.18) | Different family; useful for generalization once re-gated. |
| `tinyllava` | TinyLLaVA (wrapper exists) | Local | Not in the gated 11 | Wrapper in `evaluation_pipe/models/local_models/tinyllava.py`; no gated results yet. |

**Summary of the current gate (noun-label benchmark, generation-based):** 4 valid, 1 borderline, 6 invalid out of 11. Every invalid label is driven by image tracking below 0.50, that is, by side/position bias.

**Human-matched protocol validity** (`interpret/human_matched_validity.md`): `qwen3.5-122b-a10b`, `qwen3.5-35b-a3b`, `qwen3.5-9b` valid (word-sensitivity gate structurally N/A). Note the human-matched CSVs live on the remote machine; copying them into this checkout is an open checklist item.

## 5. Results snapshot

- **Validity-gated shape bias is model-dependent, not universal.** Valid Qwen models show interpretable shape preference; most small models are dominated by position bias under generation prompting.
- **No-word trio (600 matched trials each; `interpret/no_word_trio_interim_report.md`):** qwen3.5-9b -0.326 shape without label (tracking improved), qwen3.5-27b -0.334 (quality degraded), qwen3.5-122b-a10b -0.020 (29.7% unclear). Consistent with a model-dependent labeling effect rather than a uniform mechanism.
- **Forced choice vs logits dissociate.** SmolVLM: generation looks like noise (habitual first-option answering); logits + swap correction reveal a stable slight texture preference. This is the motivation for peeking at internal probabilities wherever we can and checking whether they match the language response.
- Figures live in `results/model.results/figures/` (described in `analysis_pipe/PLOT_DESCRIPTIONS.md`).

**Probe era (FarmShare, July 2026; full record in `farmshare/`, data in `results/probe.results/`, dashboard `farmshare/probe-experiment-results.html`):**
- **Probe experiment (24 cells, 6 sub-4B models):** 23/24 cells fail the image-tracking gate; the one pass (qwen3.5-4b, noun, 1/2) is a generation-logit dissociation (gen 0.82, swap-corrected logit 0.50). Letter labels (A/B) are a confound: numeric tracks far better; never pool label sets.
- **Scaling (Point 1):** Qwen3-VL crosses the gate at 8B (tracking 0.80; shape 0.83 gen / 0.81 logit, agreeing); InternVL3 never passes by 14B. Scale necessary, family matters.
- **Embedding dissociation (Point 2):** all models at chance in embedding space on the same stimuli, all 4 read-outs, including the behaviorally-biased 8B; probe sensitivity proven by retrieval@1 (0.77-1.00 vs 0.03 chance) and by recovering texture preference on Geirhos cue-conflict stimuli. The behavioral shape bias is made downstream of the vision encoder.
- **Debiasing (Point 5):** swap, full permutation, and PriDe agree within 0.07; raw single-order reads are misleading in 16/38 cells; the 8B positive is estimator-invariant.
- **Threshold sensitivity (2026-07-11, this checkout):** at every gate threshold 0.50-0.80 the passing cells are noun+numeric only; looser thresholds admit only chance-level cells. The 0.70 gate is a power dial, not a conclusion dial. Analysis: `results/probe.results/analysis/`, script `playgrounds/threshold_sensitivity.py`.
- **Literature audit table** (measurement locus x bias exposure across Geirhos / Tartaglini / Gavrikov / selection-bias papers): `results/probe.results/analysis/audit_table.csv` + notes. Core claim: shape-bias numbers are locus-bound; only the positioned-choice locus carries option/position artifacts, and it is the only locus matching the developmental task.

## 6. Roadmap: phases and where we are

**Phase 0 — Stimuli (done).** Reproducible 3D STL rendering pipeline with texture audit trails; benchmark package + two human packages (v1/v2).

**Phase 1 — Benchmark forced-choice evaluation + validity gating (done).** 11 models, noun-label benchmark, gates applied, figures and validity reports generated.

**Phase 2 — No-word (label-dependence) control (partly done).** Pilot across 5 remote models, full matched trio complete. **Open:** no-word runs for `qwen3.5-35b-a3b` and any models newly validated via logits; rename the trio report to drop "interim".

**Phase 3 — Logit probing / side-bias correction (in progress, active team work).** SmolVLM done via levante-bench `integrate-shapebias`. David's logit branch merged; Adam is finishing logit output for local Qwen wrappers and InternVL in this repo. Goal: for each local model, compare the forced-choice language response with internal probabilities and see whether they match; re-gate the 6 invalid models under the logit mode. Local runs move to Stanford farmshare.

**Phase 4 — Human experiment (in progress, active team work).** Static-image jsPsych experiment is Prolific-ready. Summer 2026 rebuild (Adam and Andrew): interactive STL objects participants can rotate (`human-experiment/Interactive_experiment/`), drag-to-continue instead of a timer, mobile-friendly, fullscreen fix, design polish, children and adult versions, backend migration Apache to Vercel with an admin panel. **Open:** Prolific pilot (n=20-30), then full launch; export into `results/human.results/`.

**Phase 5 — Human-model comparison (pending human data).** Hooks exist in `analysis_pipe/analysis.qmd` and `src/human_analysis.R`. The two questions: does any model approximate the human shape rate, and does it show the human naming-linked signature (the no-word drop)?

**Phase 6 — Extensions (grant-scale, proposed).** AI-based perceptual norming probe validated against human word-extension data; the 2x2 crossing training regime with label condition; transfer of the validity-gated workflow to a second forced-choice paradigm (e.g., mutual exclusivity); open-source R/Python packaging. Timeline and budget in `ai-impact-grant/Stanford impact labs grant.md`.

**Cross-cutting open items** (full list in `PROJECT_CHECKLIST.md`): stimulus covariate ratings (`realism`, review of `artifact_like`/`complexity`/`abstractness`), prompt-format robustness sweeps (strict numeric, A/B), methods note on logit-forced vs generation-based measurement, copy human-matched CSVs into this checkout.

## 7. Anticipated merges (do not step on these)

- Adam: logit scoring in `evaluation_pipe/models/local_models/` (qwen, qwen35, internvl) and probably new probability columns in results CSVs. Keep `eval_core.py` and the wrappers stable until his PR lands; analysis loaders should tolerate extra columns.
- Andrew: `human-experiment/` backend (Vercel), UI changes in `public/`, and `Interactive_experiment/`. Possibly a new repo for the interactive experiment.
- Both RAs work on branches and merge via PR; avoid force-pushes to main.

## 8. Update log

- **2026-07-11** — Probe-era results integrated (section 5). New: `results/probe.results/` (FarmShare session data + threshold sensitivity + audit table), `results/README.md` (layout guide), `farmshare/README.md` + `farmshare/probe-experiment-results.html` (browser-readable replacement for the Cursor canvas), `manuscript/` (draft: intro/background, current studies, Part 1; venue memo recommends CogSci 2027 then Open Mind; gitignored). Still on FarmShare only: `probe_experiment.json`, `pride_debias.csv` (sync command in `results/README.md`).
- **2026-07-10** — First version of this report, written during the repo reorganization (see `MEMORY.md`). Reflects: 11-model gate (4 valid), completed no-word trio, SmolVLM logit-forced result, human experiment pre-pilot, Summer 2026 team plans from `interpret/RA_mentoring/discussions.txt`, and the updated grant framing (validity gating + AI norming as reusable tools).
