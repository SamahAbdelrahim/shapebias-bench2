#!/usr/bin/env node
/**
 * Drive simulated sessions against a running server.
 *
 * This is an integration check, not a rendering check: it resolves a real
 * participant assignment, fetches every image the session would show, and posts
 * trial rows through the live /api/log route. What it cannot verify is how the
 * page looks, so a human still has to click through one session in a browser
 * before launch.
 *
 * Usage:
 *   node human-experiment/server.local-preview.js &
 *   node human-experiment/pilot_session.js [n_sessions] [base_url]
 */

const SBAssignment = require("./public/assignment.js");

const N_DEFAULT = 10;
const BASE_DEFAULT = "http://localhost:3041";

function hex24(rand) {
  let out = "";
  for (let i = 0; i < 24; i += 1) out += "0123456789abcdef"[Math.floor(rand() * 16)];
  return out;
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

async function main() {
  const nSessions = Number(process.argv[2] || N_DEFAULT);
  const base = (process.argv[3] || BASE_DEFAULT).replace(/\/$/, "");

  const poolRes = await fetch(`${base}/human-experiment/trial_pool.json`);
  if (!poolRes.ok) {
    console.error(`Could not load the trial pool from ${base} (${poolRes.status}).`);
    console.error("Start a server first: node human-experiment/server.local-preview.js");
    process.exit(1);
  }
  const pool = await poolRes.json();
  console.log(`Pool ${pool.pool_version}: ${JSON.stringify(pool.counts)}`);
  console.log(`Driving ${nSessions} sessions against ${base}`);
  console.log("");

  const rand = SBAssignment.mulberry32(4242);
  const imageCache = new Map(); // url -> { status, bytes }
  const sessionSummaries = [];
  let logOk = 0;
  let logFail = 0;
  const failures = [];

  for (let s = 0; s < nSessions; s += 1) {
    const seed = `${hex24(rand)}|${hex24(rand)}|${hex24(rand)}`;
    const session = SBAssignment.assignParticipant(pool, seed, {});
    const [pid, studyId, sessionId] = seed.split("|");

    const urls = new Set();
    for (const t of session.trials) {
      urls.add(t.reference_url);
      urls.add(t.image_a_url);
      urls.add(t.image_b_url);
    }

    let sessionBytes = 0;
    const started = Date.now();
    for (const u of urls) {
      if (!imageCache.has(u)) {
        const res = await fetch(`${base}${u}`);
        const buf = res.ok ? await res.arrayBuffer() : new ArrayBuffer(0);
        imageCache.set(u, { status: res.status, bytes: buf.byteLength });
        if (!res.ok) failures.push(`image ${res.status}: ${u}`);
      }
      const rec = imageCache.get(u);
      if (rec.status !== 200 || rec.bytes === 0) {
        failures.push(`bad image (${rec.status}, ${rec.bytes}b): ${u}`);
      }
      sessionBytes += rec.bytes;
    }
    const fetchMs = Date.now() - started;

    // Post every trial the way the frontend would, with a plausible response.
    for (let i = 0; i < session.trials.length; i += 1) {
      const t = session.trials[i];
      // Catch trials are answered correctly; test trials lean shape, which is
      // what the March pilot showed, so the row shapes are realistic.
      let responseKey;
      if (t.is_catch) {
        responseKey = t.a_is === "match" ? "1" : "2";
      } else {
        const wantsShape = rand() < 0.85;
        responseKey = (t.a_is === "shape") === wantsShape ? "1" : "2";
      }
      const choice = responseKey === "1" ? t.a_is : t.b_is;
      const payload = {
        prolific_pid: pid,
        study_id: studyId,
        session_id: sessionId,
        completion_code: "PILOT",
        condition: t.condition,
        design: "matched_v2",
        pool_version: t.pool_version,
        stim_set_name: t.stim_set_name,
        trial_index: i,
        block_index: t.block_index,
        stim_id: t.stim_id,
        stl_id: t.stl_id,
        texture_set: t.texture_set,
        pool_index: t.pool_index,
        is_catch: Boolean(t.is_catch),
        catch_correct: t.is_catch ? choice === "match" : null,
        word: t.word,
        word_type: t.word_type,
        word_length: t.word_length,
        ordering: t.ordering,
        ordering_group: t.ordering_group,
        a_is: t.a_is,
        b_is: t.b_is,
        response_key: responseKey,
        choice,
        rt_ms: Math.round(2000 + rand() * 3000),
        reference_url: t.reference_url,
        image_a_url: t.image_a_url,
        image_b_url: t.image_b_url,
        shape_match_url: t.shape_match_url,
        texture_match_url: t.texture_match_url,
        browser_user_agent: "pilot-harness",
        timezone: "America/Los_Angeles"
      };
      const res = await fetch(`${base}/api/log`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        logOk += 1;
      } else {
        logFail += 1;
        if (failures.length < 20) failures.push(`log ${res.status} on trial ${i}`);
      }
    }

    const catchPositions = session.trials
      .map((t, i) => (t.is_catch ? i + 1 : null))
      .filter((x) => x !== null);
    sessionSummaries.push({
      condition: session.condition,
      group: session.orderingGroup,
      setOrder: session.setOrder.join(">"),
      trials: session.trials.length,
      uniqueImages: urls.size,
      bytes: sessionBytes,
      fetchMs,
      catchPositions
    });
  }

  const bytes = sessionSummaries.map((x) => x.bytes).sort((a, b) => a - b);
  const imgSizes = [...imageCache.values()].map((v) => v.bytes).sort((a, b) => a - b);

  console.log("Per session");
  console.log(`  trials:        ${[...new Set(sessionSummaries.map((x) => x.trials))].join(", ")}`);
  console.log(
    `  unique images: ${[...new Set(sessionSummaries.map((x) => x.uniqueImages))].join(", ")}`
  );
  console.log(
    `  payload:       median ${(quantile(bytes, 0.5) / 1e6).toFixed(2)} MB, ` +
      `max ${(bytes[bytes.length - 1] / 1e6).toFixed(2)} MB`
  );
  console.log(
    `  image size:    median ${(quantile(imgSizes, 0.5) / 1024).toFixed(1)} KB, ` +
      `p95 ${(quantile(imgSizes, 0.95) / 1024).toFixed(1)} KB`
  );
  console.log(`  catch at trials: ${sessionSummaries[0].catchPositions.join(", ")} (first session)`);
  console.log("");

  console.log("Assignment spread across these sessions");
  const tally = (key) => {
    const t = {};
    for (const s of sessionSummaries) t[s[key]] = (t[s[key]] || 0) + 1;
    return JSON.stringify(t);
  };
  console.log(`  condition:  ${tally("condition")}`);
  console.log(`  group:      ${tally("group")}`);
  console.log(`  block order:${tally("setOrder")}`);
  console.log("");

  console.log("Serving and logging");
  console.log(`  distinct images fetched: ${imageCache.size}`);
  console.log(`  trial rows accepted:     ${logOk}`);
  console.log(`  trial rows rejected:     ${logFail}`);
  console.log("");

  // A 40-trial session at the March pilot's ~4 s mean RT is about 2.7 minutes of
  // responding; the number worth watching here is how much image data a
  // participant on a slow connection has to pull.
  const medianMb = quantile(bytes, 0.5) / 1e6;
  console.log(
    `Estimated session: 40 trials, ~${medianMb.toFixed(2)} MB of images ` +
      `(~${(medianMb / 0.5).toFixed(1)} s to preload on a 4 Mbps link)`
  );
  console.log("");

  if (failures.length) {
    console.log(`FAIL (${failures.length} problems)`);
    for (const f of failures.slice(0, 20)) console.log(`  ${f}`);
    process.exit(1);
  }
  console.log("PASS: every image resolved and every trial row was accepted");
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
