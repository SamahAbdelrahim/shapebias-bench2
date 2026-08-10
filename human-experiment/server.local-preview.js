const path = require("path");
const fs = require("fs");
const express = require("express");
require("dotenv").config();

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = Number(process.env.PREVIEW_PORT || process.env.PORT || 3041);
const PUBLIC_DIR = path.join(__dirname, "public");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";
const POOL_PATH = path.join(PUBLIC_DIR, "trial_pool.json");

function loadPool() {
  if (!fs.existsSync(POOL_PATH)) {
    throw new Error(
      `Trial pool not found at ${POOL_PATH}. ` +
        "Run: .venv/bin/python scripts/build_human_trial_pool.py"
    );
  }
  return JSON.parse(fs.readFileSync(POOL_PATH, "utf8"));
}

// public/ also holds trial_pool.json and the stimuli/ image tree.
app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(__dirname, "general_assets")));
app.use("/vendor/jspsych", express.static(path.join(__dirname, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(__dirname, "node_modules", "@jspsych")));

// Local preview fallback: this repo copy does not include general_assets/jspsych/dist builds.
// Map the expected local file URLs to CDN assets only in preview mode.
const CDN = {
  jspsych: "https://unpkg.com/jspsych@8.2.3",
  preload: "https://unpkg.com/@jspsych/plugin-preload@2.1.0",
  instructions: "https://unpkg.com/@jspsych/plugin-instructions@2.1.0",
  htmlButtonResponse: "https://unpkg.com/@jspsych/plugin-html-button-response@2.1.0"
};

app.get("/general_assets/jspsych/dist/jspsych.css", (_req, res) => {
  res.redirect(302, `${CDN.jspsych}/css/jspsych.css`);
});
app.get("/general_assets/jspsych/dist/jspsych.js", (_req, res) => {
  res.redirect(302, CDN.jspsych);
});
app.get("/general_assets/jspsych/dist/plugin-preload.js", (_req, res) => {
  res.redirect(302, CDN.preload);
});
app.get("/general_assets/jspsych/dist/plugin-instructions.js", (_req, res) => {
  res.redirect(302, CDN.instructions);
});
app.get("/general_assets/jspsych/dist/plugin-html-button-response.js", (_req, res) => {
  res.redirect(302, CDN.htmlButtonResponse);
});

app.get("/", (_req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

app.get("/api/config", (_req, res) => {
  let poolVersion = null;
  let counts = null;
  try {
    const pool = loadPool();
    poolVersion = pool.pool_version || null;
    counts = pool.counts || null;
  } catch (_err) {
    // Reported by /api/pool; config should still answer.
  }
  res.json({
    completion_code: COMPLETION_CODE,
    design: "matched_v2",
    pool_version: poolVersion,
    pool_counts: counts
  });
});

app.get("/api/pool", (_req, res) => {
  try {
    res.json(loadPool());
  } catch (err) {
    console.warn("[local-preview] /api/pool failed:", String(err.message || err));
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/api/log", (req, res) => {
  const payload = req.body || {};
  console.log("[local-preview] trial log", {
    trial_index: payload.trial_index,
    stim_set_name: payload.stim_set_name,
    stim_id: payload.stim_id,
    is_catch: payload.is_catch,
    catch_correct: payload.catch_correct,
    ordering: payload.ordering,
    choice: payload.choice
  });
  res.json({ ok: true, preview_mode: true });
});

app.listen(PORT, () => {
  console.log(`Local preview server running at http://localhost:${PORT}`);
  console.log("This mode skips MongoDB and is safe for UI testing.");
});
