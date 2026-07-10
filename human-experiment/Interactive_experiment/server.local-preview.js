const path = require("path");
const express = require("express");
require("dotenv").config({ path: path.resolve(__dirname, "..", "..", ".env") });
require("dotenv").config();

const {
  loadComparisonConfig,
  listAvailableModels,
  buildComparisonTrials
} = require("./comparison-survey");

const app = express();
app.use(express.json({ limit: "2mb" }));

// Default 3042 so this can run alongside the parent shape-bias experiment on 3041.
const PORT = Number(process.env.PREVIEW_PORT || process.env.PORT || 3042);
const PUBLIC_DIR = path.join(__dirname, "public");
const EXPERIMENT_ROOT = __dirname;
const PARENT_HE = path.resolve(__dirname, "..");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";

app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(PARENT_HE, "general_assets")));
app.use("/stl", express.static(path.join(EXPERIMENT_ROOT, "stl")));
app.use("/glb", express.static(path.join(EXPERIMENT_ROOT, "glb")));
app.use("/vendor/jspsych", express.static(path.join(PARENT_HE, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(PARENT_HE, "node_modules", "@jspsych")));

// Preview fallback: map expected local jsPsych URLs to CDN if parent assets are missing.
const CDN = {
  jspsych: "https://unpkg.com/jspsych@8.2.3",
  instructions: "https://unpkg.com/@jspsych/plugin-instructions@2.1.0",
  htmlButtonResponse: "https://unpkg.com/@jspsych/plugin-html-button-response@2.1.0"
};

app.get("/general_assets/jspsych/dist/jspsych.css", (_req, res) => {
  res.redirect(302, `${CDN.jspsych}/css/jspsych.css`);
});
app.get("/general_assets/jspsych/dist/jspsych.js", (_req, res) => {
  res.redirect(302, CDN.jspsych);
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
    experiment: "complexity_comparison",
    comparison_config_path: "Interactive_experiment/configs/comparison_survey.json",
    preview_mode: true
  });
});

app.get("/api/comparison-config", (_req, res) => {
  try {
    const config = loadComparisonConfig(EXPERIMENT_ROOT);
    res.json({ config });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.get("/api/models", (_req, res) => {
  try {
    const config = loadComparisonConfig(EXPERIMENT_ROOT);
    const models = listAvailableModels(EXPERIMENT_ROOT, config.model_sources);
    res.json({ count: models.length, models, model_sources: config.model_sources });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.get("/api/comparison-trials", (req, res) => {
  try {
    const participantSeed = `${String(req.query.prolific_pid || "debug_pid")}|${String(req.query.study_id || "debug_study")}|${String(req.query.session_id || "debug_session")}`;
    const result = buildComparisonTrials({
      repoRoot: EXPERIMENT_ROOT,
      participantSeed
    });
    res.json({
      config: result.config,
      model_count: result.models.length,
      count: result.trials.length,
      trials: result.trials
    });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/api/log", (req, res) => {
  const payload = req.body || {};
  console.log("[local-preview] trial log", {
    trial_index: payload.trial_index,
    experiment_type: payload.experiment_type,
    chosen_filename: payload.chosen_filename,
    response_key: payload.response_key
  });
  res.json({ ok: true, preview_mode: true });
});

app.listen(PORT, () => {
  console.log(`Complexity comparison preview at http://localhost:${PORT}`);
  console.log("This mode skips MongoDB and is safe for UI testing.");
});
