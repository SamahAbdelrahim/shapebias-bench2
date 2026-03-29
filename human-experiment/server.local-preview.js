const path = require("path");
const fs = require("fs");
const express = require("express");
require("dotenv").config();

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = Number(process.env.PREVIEW_PORT || process.env.PORT || 3041);
const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_ROOT = path.resolve(__dirname, "..");
const STIMULI_ROOT = path.join(REPO_ROOT, "stimuli_pipe");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";

function parseSimpleCsv(csvText) {
  const lines = csvText.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const header = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i += 1) {
    const cols = lines[i].split(",");
    const row = {};
    for (let j = 0; j < header.length; j += 1) {
      row[header[j]] = (cols[j] || "").trim();
    }
    rows.push(row);
  }
  return rows;
}

function loadStimuliFromManifest(stimSet) {
  const manifestPath = path.join(
    STIMULI_ROOT,
    "stimuli_per_stl_packages",
    stimSet,
    "manifest.csv"
  );
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest not found: ${manifestPath}`);
  }
  const rows = parseSimpleCsv(fs.readFileSync(manifestPath, "utf8"));
  return rows.map((r) => {
    const norm = (p) => `/${p.replace(/^\/+/, "")}`;
    return {
      stim_id: r.stl_id,
      mode: r.mode,
      reference_url: norm(path.posix.join("stimuli_pipe", r.reference)),
      shape_match_url: norm(path.posix.join("stimuli_pipe", r.shape_match)),
      texture_match_url: norm(path.posix.join("stimuli_pipe", r.texture_match))
    };
  });
}

function fallbackStimuli() {
  // Lightweight visual fallback so UI can still be previewed if manifest is missing.
  const img = "/human-experiment/favicon.svg";
  return [
    {
      stim_id: "preview-1",
      mode: "preview",
      reference_url: img,
      shape_match_url: img,
      texture_match_url: img
    }
  ];
}

app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(__dirname, "general_assets")));
app.use("/stimuli_pipe", express.static(STIMULI_ROOT));
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
  res.json({
    completion_code: COMPLETION_CODE,
    default_stim_set: process.env.DEFAULT_STIM_SET || "stimuli_A_auto_contrast"
  });
});

app.get("/api/stimuli", (req, res) => {
  try {
    const stimSet = String(req.query.stim_set || process.env.DEFAULT_STIM_SET || "stimuli_A_auto_contrast");
    const stimuli = loadStimuliFromManifest(stimSet);
    res.json({ stim_set: stimSet, count: stimuli.length, stimuli });
  } catch (err) {
    console.warn("[local-preview] /api/stimuli fallback:", String(err.message || err));
    const stimuli = fallbackStimuli();
    res.json({ stim_set: "preview_fallback", count: stimuli.length, stimuli, preview_fallback: true });
  }
});

app.post("/api/log", (req, res) => {
  const payload = req.body || {};
  const trialIndex = payload.trial_index;
  const stimId = payload.stim_id;
  const responseKey = payload.response_key;
  console.log("[local-preview] trial log", {
    trial_index: trialIndex,
    stim_id: stimId,
    response_key: responseKey
  });
  res.json({ ok: true, preview_mode: true });
});

app.listen(PORT, () => {
  console.log(`Local preview server running at http://localhost:${PORT}`);
  console.log("This mode skips MongoDB and is safe for UI testing.");
});
