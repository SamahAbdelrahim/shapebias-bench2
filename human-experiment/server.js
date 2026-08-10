const path = require("path");
const fs = require("fs");
const express = require("express");
const mongoose = require("mongoose");
require("dotenv").config();

const ShapeBiasHumanTrial = require("./models/shapebias-human-logger");

const app = express();
app.use(express.json({ limit: "2mb" }));

const PORT = Number(process.env.PORT || 3041);
const PUBLIC_DIR = path.join(__dirname, "public");
const REPO_ROOT = path.resolve(__dirname, "..");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";
// Built offline by scripts/build_human_trial_pool.py, together with the WebP
// images it references under public/stimuli/. The server no longer walks the
// stimulus tree: the grid nests a texture level and the cue-conflict sets use
// "<shape_id>-<texture_id>" folders, neither of which the old manifest reader
// could handle, and both live on scratch rather than in the repo.
const POOL_PATH = path.join(PUBLIC_DIR, "trial_pool.json");

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

function loadPool() {
  if (!fs.existsSync(POOL_PATH)) {
    throw new Error(
      `Trial pool not found at ${POOL_PATH}. ` +
        "Run: .venv/bin/python scripts/build_human_trial_pool.py"
    );
  }
  return JSON.parse(fs.readFileSync(POOL_PATH, "utf8"));
}

// public/ also holds trial_pool.json and the stimuli/ image tree, so this one
// mount serves the experiment, the pool and every image.
app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(__dirname, "general_assets")));
app.use("/vendor/jspsych", express.static(path.join(__dirname, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(__dirname, "node_modules", "@jspsych")));

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

// The frontend prefers the static file at /human-experiment/trial_pool.json and
// falls back here, which matters for hosts that do not serve the JSON directly.
app.get("/api/pool", (_req, res) => {
  try {
    res.json(loadPool());
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
