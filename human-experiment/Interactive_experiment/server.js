const path = require("path");
const fs = require("fs");
const express = require("express");
const mongoose = require("mongoose");
require("dotenv").config({ path: path.resolve(__dirname, "..", "..", ".env") });
require("dotenv").config();

const ShapeBiasHumanTrial = require("./models/shapebias-human-logger");
const {
  loadComparisonConfig,
  listAvailableModels,
  buildComparisonTrials
} = require("./comparison-survey");

const app = express();
app.use(express.json({ limit: "2mb" }));

// Default 3042 so this can run alongside the parent shape-bias experiment on 3041.
const PORT = Number(process.env.PORT || 3042);
const PUBLIC_DIR = path.join(__dirname, "public");
const EXPERIMENT_ROOT = __dirname;
const PARENT_HE = path.resolve(__dirname, "..");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";

function readMongoCreds() {
  const candidatePaths = [
    path.join(__dirname, "mongo_auth.json"),
    path.join(PARENT_HE, "mongo_auth.json")
  ];

  for (const p of candidatePaths) {
    if (fs.existsSync(p)) {
      return JSON.parse(fs.readFileSync(p, "utf8"));
    }
  }
  throw new Error(
    "mongo_auth.json not found. Place it in Interactive_experiment/ or human-experiment/."
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

app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(PARENT_HE, "general_assets")));
app.use("/stl", express.static(path.join(EXPERIMENT_ROOT, "stl")));
app.use("/glb", express.static(path.join(EXPERIMENT_ROOT, "glb")));
app.use("/vendor/jspsych", express.static(path.join(PARENT_HE, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(PARENT_HE, "node_modules", "@jspsych")));

app.get("/", (_req, res) => {
  res.sendFile(path.join(PUBLIC_DIR, "index.html"));
});

app.get("/api/config", (_req, res) => {
  res.json({
    completion_code: COMPLETION_CODE,
    experiment: "complexity_comparison",
    comparison_config_path: "Interactive_experiment/configs/comparison_survey.json"
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
      console.log(
        `Complexity comparison server running at http://localhost:${PORT}`
      );
    });
  })
  .catch((err) => {
    console.error("MongoDB connection failed:", err);
    process.exit(1);
  });
