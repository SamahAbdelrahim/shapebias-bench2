# matched_v2 launch runbook

Everything in the repo is ready to collect. The remaining steps need credentials
and money, so they are written out here rather than executed.

## 1. Rebuild the pool if the stimuli changed

```bash
.venv/bin/python scripts/build_human_trial_pool.py
```

Writes `results/data/human_trial_pool_v2.csv`,
`human-experiment/public/trial_pool.json` and 390 WebP images (1.5 MB) under
`human-experiment/public/stimuli/`. The script reads the grid from `/scratch`,
so it has to run on a machine that can see it. Its outputs are self-contained;
nothing at run time touches `/scratch`.

Humans run the novel grid alone. Cue-conflict triads are behind `--include-cc`
and are off; see the rationale in `HUMAN_PROTOCOL_RATIONALE.md`.

If jsPsych is ever bumped in `package.json`, re-run
`bash human-experiment/vendor_jspsych.sh` so `public/vendor/` matches.

## 2. Check the design before paying anyone

```bash
node human-experiment/verify_assignment.js 135
```

Fails loudly if a session repeats a stimulus, repeats a shape (which would give
one object two pseudo-words), leaves a check without a correct answer, lets an
item take both orders within one group, or empties a condition-by-group cell.
It also prints observations per triad, which is the number to look at when
choosing a sample size.

## 3. Click through one session yourself

```bash
node human-experiment/server.local-preview.js
# http://localhost:3041/?PROLIFIC_PID=test1&STUDY_ID=test&SESSION_ID=test
```

No Mongo needed; trials print to the console. Add `&condition=no_word_category`
or `&ordering_group=B` to force a cell, and `&verbose_trials=1` for every row.
This is the step the automated checks cannot cover: they verify that every image
resolves and every row is accepted, not that the page reads correctly. Confirm
the wording, the example on the instructions page, and that the four checks feel
trivial.

Then drive the integration check against the same server:

```bash
node human-experiment/pilot_session.js 10 http://localhost:3041
```

## 4. Deploy

`vercel.json` serves `public/` at `/human-experiment/*` and runs `api/log.js`
and `api/config.js` as functions. Set in the Vercel project:

- `MONGO_URI` - the Atlas SRV string, with the database name in the path
- `PROLIFIC_COMPLETION_CODE` - the code Prolific issues for the study

`public/stimuli/` and `public/vendor/` must be committed, since a git-based
deploy only ships tracked files. Together they are about 2 MB.

Atlas needs network access from Vercel's egress; the serverless handler caches
its connection per warm container and caps the pool at 5.

After deploying, confirm:

```bash
curl -s https://<deployment>/api/config
curl -s -o /dev/null -w "%{http_code}\n" https://<deployment>/human-experiment/trial_pool.json
node human-experiment/pilot_session.js 2 https://<deployment>
```

Then delete those pilot rows from Mongo before real collection, or filter them
out later on `completion_code == "PILOT"`.

## 5. Prolific

One study, not two. Condition and ordering group are both assigned from the
`PROLIFIC_PID|STUDY_ID|SESSION_ID` seed, so a single study fills all four cells
at roughly equal rates and a participant who reloads returns to the same cell
with the same items. Splitting into separate studies would break that and risk
the same person appearing in two cells.

Study URL:

```
https://<deployment>/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

Sample size: 135 participants for k = 16 observations per triad per condition,
about 85 for k = 10. Verified spread at 135, from `verify_assignment.js`:

| condition | observations per triad |
| --- | --- |
| noun_label | mean 16.3, range 13 to 26 |
| no_word_category | mean 15.6, range 9 to 24 |

All 114 triads are covered in both option orders in both conditions.

Timing: 31 trials at the pilot's 3.06 s median response time is roughly 5 to 6
minutes including instructions. Set the reward from your own current rate; I
have no reliable figure for Prolific's present fee percentage, so check it in
your account rather than trusting a number from here.

Screen out anyone who took the March 2026 pilot: they saw the same task with
different stimuli.

## 6. Export and analyse

```bash
node human-experiment/export_human_results.js          # Mongo first, CSV fallback
```

Then render `analysis_pipe/analysis.qmd`, which writes:

- `results/data/human_catch_by_participant.csv` - who failed the checks
- `results/data/human_summary_by_set.csv` - shape rate by set and condition
- `results/data/human_item_means.csv` - item means and the position check
- `results/data/human_position_check.csv` - the position check rolled up

`analysis_pipe/full_grid_figures.py` picks up `human_summary_by_set.csv`
automatically and places per-cell human anchors in `fig2` and `fig7b`. Until
that file exists it falls back to the single pilot number, as it does today.

R is not installed on the machine this was built on, so the functions in
`analysis_pipe/src/human_analysis.R` have not been executed. Run the Quarto
document against the first real export and check the four CSVs before trusting
the figures.
