const path = require("path");
const fs = require("fs");
const express = require("express");
const mongoose = require("mongoose");
require("dotenv").config();

const ShapeBiasHumanTrial = require("./models/shapebias-human-logger");
const {
  loadComparisonConfig,
  listAvailableModels,
  buildComparisonTrials
} = require("./comparison-survey");

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = Number(process.env.PORT || 3041);
const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_ROOT = path.resolve(__dirname, "..");
const STIMULI_ROOT = path.join(REPO_ROOT, "stimuli_pipe");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";
const BENCHMARK_STIM_PACKAGE = "stimuli_per_stl_packages";
const HUMAN_UNIQUE_STIM_PACKAGES = [
  "stimuli_unique_texture_per_stl_v1",
  "stimuli_unique_texture_per_stl_v2"
];
const ALLOWED_STIM_PACKAGES = new Set([
  BENCHMARK_STIM_PACKAGE,
  ...HUMAN_UNIQUE_STIM_PACKAGES
]);

function readMongoCreds() {
  const candidatePaths = [
    path.join(__dirname, "mongo_auth.json")
  ].filter(Boolean);

  for (const p of candidatePaths) {
    if (fs.existsSync(p)) {
      const raw = fs.readFileSync(p, "utf8");
      return JSON.parse(raw);
    }
  }
  throw new Error(
    "mongo_auth.json not found in human-experiment/. Create human-experiment/mongo_auth.json."
  );
}

async function connectMongo() {
  if (process.env.MONGO_URI) {
    await mongoose.connect(process.env.MONGO_URI);
    return;
  }
  const creds = readMongoCreds();
  const username = encodeURIComponent(creds.username || creds.user || "");
  const password = encodeURIComponent(creds.password || creds.pass || "");
  if (!username || !password) {
    throw new Error("mongo_auth.json missing username/password fields");
  }
  const dbName = process.env.MONGO_DB || "samah";
  const host = process.env.MONGO_HOST || "127.0.0.1";
  const port = process.env.MONGO_PORT || "27017";
  const uri = `mongodb://${username}:${password}@${host}:${port}/${dbName}?authSource=admin`;
  await mongoose.connect(uri);
}

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

function hashString(input) {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function pickHumanStimPackage(participantSeed) {
  const idx = hashString(participantSeed) % HUMAN_UNIQUE_STIM_PACKAGES.length;
  return HUMAN_UNIQUE_STIM_PACKAGES[idx];
}

function resolveStimPackage({ requestedPkg, design, participantSeed }) {
  if (design === "human_friendly") {
    if (requestedPkg && HUMAN_UNIQUE_STIM_PACKAGES.includes(requestedPkg)) {
      return requestedPkg;
    }
    return pickHumanStimPackage(participantSeed);
  }
  if (requestedPkg === BENCHMARK_STIM_PACKAGE) {
    return requestedPkg;
  }
  return BENCHMARK_STIM_PACKAGE;
}

function loadStimuliFromManifest(stimPackage, stimSet) {
  const manifestPath = path.join(
    STIMULI_ROOT,
    stimPackage,
    stimSet,
    "manifest.csv"
  );
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`Manifest not found: ${manifestPath}`);
  }
  const rows = parseSimpleCsv(fs.readFileSync(manifestPath, "utf8"));
  const normalizeStimulusRelPath = (rawValue) => {
    const raw = String(rawValue || "").replace(/\\/g, "/").replace(/^\/+/, "");
    const parts = raw.split("/").filter(Boolean);
    let idx = 0;

    // Legacy manifests may start with stimuli_per_stl_packages/.
    if (parts[idx] === BENCHMARK_STIM_PACKAGE) idx += 1;
    // Some manifests include a package segment after the legacy prefix.
    if (parts[idx] && ALLOWED_STIM_PACKAGES.has(parts[idx])) idx += 1;
    // Use whatever remains (usually <stim_set>/<stl_id>/<file>.png).
    const tail = parts.slice(idx);
    if (tail.length === 0) return "";
    return path.posix.join(stimPackage, ...tail);
  };
  return rows.map((r) => {
    const norm = (p) => `/${p.replace(/^\/+/, "")}`;
    return {
      stim_id: r.stl_id,
      mode: r.mode,
      reference_url: norm(path.posix.join("stimuli_pipe", normalizeStimulusRelPath(r.reference))),
      shape_match_url: norm(path.posix.join("stimuli_pipe", normalizeStimulusRelPath(r.shape_match))),
      texture_match_url: norm(path.posix.join("stimuli_pipe", normalizeStimulusRelPath(r.texture_match)))
    };
  });
}

app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(__dirname, "general_assets")));
app.use("/stimuli_pipe", express.static(STIMULI_ROOT));
app.use("/stl", express.static(path.join(REPO_ROOT, "stl")));
app.use("/glb", express.static(path.join(REPO_ROOT, "glb")));
app.use("/vendor/jspsych", express.static(path.join(__dirname, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(__dirname, "node_modules", "@jspsych")));

app.get("/", (_req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

app.get("/api/config", (_req, res) => {
  res.json({
    completion_code: COMPLETION_CODE,
    default_stim_set: process.env.DEFAULT_STIM_SET || "stimuli_A_auto_contrast",
    benchmark_stim_pkg: BENCHMARK_STIM_PACKAGE,
    human_stim_pkgs: HUMAN_UNIQUE_STIM_PACKAGES,
    comparison_config_path: "configs/comparison_survey.json"
  });
});

app.get("/api/comparison-config", (_req, res) => {
  try {
    const config = loadComparisonConfig(REPO_ROOT);
    res.json({ config });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.get("/api/models", (_req, res) => {
  try {
    const config = loadComparisonConfig(REPO_ROOT);
    const models = listAvailableModels(REPO_ROOT, config.model_sources);
    res.json({ count: models.length, models, model_sources: config.model_sources });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.get("/api/comparison-trials", (req, res) => {
  try {
    const participantSeed = `${String(req.query.prolific_pid || "debug_pid")}|${String(req.query.study_id || "debug_study")}|${String(req.query.session_id || "debug_session")}`;
    const result = buildComparisonTrials({ repoRoot: REPO_ROOT, participantSeed });
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

app.get("/api/stimuli", (req, res) => {
  try {
    const design = String(req.query.design || "human_friendly");
    const stimSet = String(req.query.stim_set || process.env.DEFAULT_STIM_SET || "stimuli_A_auto_contrast");
    const requestedPkg = req.query.stim_pkg ? String(req.query.stim_pkg) : "";
    const participantSeed = `${String(req.query.prolific_pid || "debug_pid")}|${String(req.query.study_id || "debug_study")}|${String(req.query.session_id || "debug_session")}`;
    const stimPkg = resolveStimPackage({ requestedPkg, design, participantSeed });
    const stimuli = loadStimuliFromManifest(stimPkg, stimSet);
    res.json({ stim_set: stimSet, stim_pkg: stimPkg, count: stimuli.length, stimuli });
  } catch (err) {
    res.status(500).json({ error: String(err.message || err) });
  }
});

app.post("/api/log", async (req, res) => {
  try {
    const payload = req.body || {};
    await ShapeBiasHumanTrial.create(payload);
    res.json({ ok: true });
  } catch (err) {
    console.error("Failed to save trial:", err);
    res.status(500).json({ ok: false, error: String(err.message || err) });
  }
});

connectMongo()
  .then(() => {
    app.listen(PORT, () => {
      console.log(`Shape-bias human experiment server running at http://localhost:${PORT}`);
      console.log(`Alternative URL: http://127.0.0.1:${PORT}`);
    });
  })
  .catch((err) => {
    console.error("MongoDB connection failed:", err);
    process.exit(1);
  });
