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
const STIMULI_ROOT = path.join(REPO_ROOT, "stimuli_pipe");
const COMPLETION_CODE = process.env.PROLIFIC_COMPLETION_CODE || "TESTCODE";

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

app.use("/human-experiment", express.static(PUBLIC_DIR));
app.use("/general_assets", express.static(path.join(__dirname, "general_assets")));
app.use("/stimuli_pipe", express.static(STIMULI_ROOT));
app.use("/vendor/jspsych", express.static(path.join(__dirname, "node_modules", "jspsych")));
app.use("/vendor/@jspsych", express.static(path.join(__dirname, "node_modules", "@jspsych")));

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
