#!/usr/bin/env node
/**
 * Simulate participants against the built trial pool and check the design holds.
 *
 * The counterbalancing in public/assignment.js is the part of this experiment
 * that is easiest to get wrong and hardest to see in a browser, so it is
 * checked here over a simulated sample before anyone is paid to run it.
 *
 * Usage:
 *   node human-experiment/verify_assignment.js [n_participants]
 */

const fs = require("fs");
const path = require("path");

const SBAssignment = require("./public/assignment.js");

const POOL_PATH = path.join(__dirname, "public", "trial_pool.json");
const N_DEFAULT = 244;

function pct(x) {
  return `${(100 * x).toFixed(1)}%`;
}

function summarize(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(sorted.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  return {
    min: sorted[0],
    max: sorted[n - 1],
    mean,
    sd,
    p05: sorted[Math.floor(0.05 * n)],
    p95: sorted[Math.floor(0.95 * n)]
  };
}

function main() {
  if (!fs.existsSync(POOL_PATH)) {
    console.error(`Trial pool not found at ${POOL_PATH}.`);
    console.error("Run: .venv/bin/python scripts/build_human_trial_pool.py");
    process.exit(1);
  }
  const pool = JSON.parse(fs.readFileSync(POOL_PATH, "utf8"));
  const nParticipants = Number(process.argv[2] || N_DEFAULT);

  const conditionCounts = {};
  const groupCounts = {};
  const setOrderCounts = {};
  // coverage[condition][stim_id] = { shape_first, texture_first }
  const coverage = {};
  const perParticipantShapeFirstShare = [];
  const trialCounts = new Set();
  const catchCounts = new Set();
  let duplicateStimuli = 0;
  let catchWithoutMatch = 0;
  let orderingConflicts = 0;
  // orderingByItem[stim_id][group] = ordering, to confirm groups are opposites
  const orderingByItem = {};

  // Prolific ids are 24-char hex. Draw them from a fixed stream so the report
  // is reproducible while still exercising realistic, unstructured input: an
  // earlier version of this check used sequential ids and hid a real bug in
  // which condition and ordering group were perfectly correlated.
  const idRand = SBAssignment.mulberry32(20260810);
  const hex24 = () => {
    let out = "";
    for (let i = 0; i < 24; i += 1) out += "0123456789abcdef"[Math.floor(idRand() * 16)];
    return out;
  };
  const cellCounts = {};

  for (let i = 0; i < nParticipants; i += 1) {
    const seed = `${hex24()}|${hex24()}|${hex24()}`;
    const s = SBAssignment.assignParticipant(pool, seed, {});
    const cellKey = `${s.condition}/${s.orderingGroup}`;
    cellCounts[cellKey] = (cellCounts[cellKey] || 0) + 1;

    conditionCounts[s.condition] = (conditionCounts[s.condition] || 0) + 1;
    groupCounts[s.orderingGroup] = (groupCounts[s.orderingGroup] || 0) + 1;
    const soKey = s.setOrder.join(">");
    setOrderCounts[soKey] = (setOrderCounts[soKey] || 0) + 1;

    const tests = s.trials.filter((t) => !t.is_catch);
    const catches = s.trials.filter((t) => t.is_catch);
    trialCounts.add(s.trials.length);
    catchCounts.add(catches.length);

    const seenIds = new Set();
    for (const t of s.trials) {
      if (seenIds.has(t.stim_id)) duplicateStimuli += 1;
      seenIds.add(t.stim_id);
    }
    for (const t of catches) {
      if (t.a_is !== "match" && t.b_is !== "match") catchWithoutMatch += 1;
    }

    const cov = (coverage[s.condition] = coverage[s.condition] || {});
    let shapeFirst = 0;
    for (const t of tests) {
      const cell = (cov[t.stim_id] = cov[t.stim_id] || { shape_first: 0, texture_first: 0 });
      cell[t.ordering] += 1;
      if (t.ordering === "shape_first") shapeFirst += 1;

      const byGroup = (orderingByItem[t.stim_id] = orderingByItem[t.stim_id] || {});
      if (byGroup[s.orderingGroup] && byGroup[s.orderingGroup] !== t.ordering) {
        orderingConflicts += 1;
      }
      byGroup[s.orderingGroup] = t.ordering;
    }
    perParticipantShapeFirstShare.push(shapeFirst / (tests.length || 1));
  }

  console.log(`Simulated ${nParticipants} participants against pool ${pool.pool_version}`);
  console.log(`Pool counts: ${JSON.stringify(pool.counts)}`);
  console.log("");

  console.log("Session shape");
  console.log(`  trials per session: ${[...trialCounts].join(", ")}`);
  console.log(`  catch per session:  ${[...catchCounts].join(", ")}`);
  console.log(`  repeated stimuli within a session: ${duplicateStimuli}`);
  console.log(`  catch trials missing a correct option: ${catchWithoutMatch}`);
  console.log("");

  console.log("Between-participant cells");
  for (const [k, v] of Object.entries(conditionCounts)) {
    console.log(`  condition ${k}: ${v} (${pct(v / nParticipants)})`);
  }
  for (const [k, v] of Object.entries(groupCounts)) {
    console.log(`  ordering group ${k}: ${v} (${pct(v / nParticipants)})`);
  }
  for (const [k, v] of Object.entries(setOrderCounts)) {
    console.log(`  block order ${k}: ${v} (${pct(v / nParticipants)})`);
  }
  console.log("  condition x ordering group (all four cells must be filled)");
  const expectedCells = [];
  for (const c of SBAssignment.CONDITIONS) {
    for (const g of SBAssignment.ORDERING_GROUPS) expectedCells.push(`${c}/${g}`);
  }
  for (const key of expectedCells) {
    console.log(`    ${key.padEnd(24)} ${cellCounts[key] || 0}`);
  }
  const emptyCells = expectedCells.filter((k) => !cellCounts[k]);
  console.log("");

  const share = summarize(perParticipantShapeFirstShare);
  console.log("Within-participant ordering mix (share of trials shape-first)");
  console.log(
    `  mean ${share.mean.toFixed(3)}  sd ${share.sd.toFixed(3)}  ` +
      `range ${share.min.toFixed(2)}-${share.max.toFixed(2)}`
  );
  console.log(
    `  item-level ordering conflicts within a group (want 0): ${orderingConflicts}`
  );
  console.log("");

  console.log("Item coverage (observations per triad, by condition)");
  for (const setName of ["grid", "cc_triads"]) {
    const idsInSet = pool.test
      .filter((t) => t.stim_set_name === setName)
      .map((t) => t.stim_id);
    for (const cond of Object.keys(coverage)) {
      const counts = idsInSet.map((id) => {
        const c = coverage[cond][id];
        return c ? c.shape_first + c.texture_first : 0;
      });
      const st = summarize(counts);
      const uncovered = counts.filter((c) => c === 0).length;
      const bothOrderings = idsInSet.filter((id) => {
        const c = coverage[cond][id];
        return c && c.shape_first > 0 && c.texture_first > 0;
      }).length;
      console.log(
        `  ${setName.padEnd(10)} ${cond.padEnd(18)} ` +
          `mean ${st.mean.toFixed(1)}  sd ${st.sd.toFixed(1)}  ` +
          `min ${st.min}  max ${st.max}  ` +
          `uncovered ${uncovered}/${idsInSet.length}  ` +
          `both orderings ${bothOrderings}/${idsInSet.length}`
      );
    }
  }
  console.log("");

  const problems = [];
  if (duplicateStimuli > 0) problems.push("a session repeated a stimulus");
  if (catchWithoutMatch > 0) problems.push("a catch trial had no correct option");
  if (orderingConflicts > 0) problems.push("an item took both orderings within one group");
  if (trialCounts.size > 1) problems.push("session length varies across participants");
  if (emptyCells.length) {
    problems.push(`condition x ordering group cell empty: ${emptyCells.join(", ")}`);
  }
  for (const setName of ["grid", "cc_triads"]) {
    const idsInSet = pool.test.filter((t) => t.stim_set_name === setName).map((t) => t.stim_id);
    for (const cond of Object.keys(coverage)) {
      const missingBoth = idsInSet.filter((id) => {
        const c = coverage[cond][id];
        return !c || c.shape_first === 0 || c.texture_first === 0;
      });
      if (missingBoth.length) {
        problems.push(
          `${missingBoth.length}/${idsInSet.length} ${setName} items lack both ` +
            `orderings in ${cond}`
        );
      }
    }
  }
  if (problems.length) {
    console.log(`FAIL: ${problems.join("; ")}`);
    process.exit(1);
  }
  console.log("PASS: session shape and counterbalancing are consistent");
  return 0;
}

main();
