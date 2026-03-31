#!/usr/bin/env node
/*
Export logged human trial rows into an analysis-ready CSV.
Prefers MongoDB, then falls back to local CSV exports.
*/

const fs = require("fs");
const path = require("path");
require("dotenv").config();
let mongoose = null;
let ShapeBiasHumanTrial = null;

function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (typeof next === "undefined" || next.startsWith("--")) {
      out[key] = "1";
      continue;
    }
    out[key] = next;
    i += 1;
  }
  return out;
}

function readMongoCreds() {
  const candidate = path.join(__dirname, "mongo_auth.json");
  if (fs.existsSync(candidate)) {
    return JSON.parse(fs.readFileSync(candidate, "utf8"));
  }
  throw new Error(
    "mongo_auth.json not found in human-experiment/. Set MONGO_URI or create human-experiment/mongo_auth.json."
  );
}

async function connectMongo() {
  if (!mongoose) {
    // Load Mongo-related dependencies only when Mongo export is actually used.
    mongoose = require("mongoose");
    ShapeBiasHumanTrial = require("./models/shapebias-human-logger");
  }
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

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (char === "\"" && next === "\"") {
        field += "\"";
        i += 1;
      } else if (char === "\"") {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === "\"") {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char === "\r") {
      // ignore CR, LF handling above
    } else {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function csvRowsToObjects(text) {
  const rows = parseCsv(text).filter((row) => row.some((value) => String(value).length > 0));
  if (!rows.length) return [];
  const headers = rows[0];
  return rows.slice(1).map((row) => {
    const obj = {};
    for (let i = 0; i < headers.length; i += 1) {
      obj[headers[i]] = typeof row[i] === "undefined" ? "" : row[i];
    }
    return obj;
  });
}

function getFallbackCsvPaths(repoRoot, explicitPath) {
  if (explicitPath) {
    return [path.resolve(repoRoot, explicitPath)];
  }

  const defaults = [
    path.join(repoRoot, "results", "human.results"),
    path.join(repoRoot, "results")
  ];

  const discovered = [];
  for (const dir of defaults) {
    if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) continue;
    const entries = fs.readdirSync(dir)
      .filter((name) => name.toLowerCase().endsWith(".csv") && name.toLowerCase().includes("human"))
      .sort();
    for (const entry of entries) {
      discovered.push(path.join(dir, entry));
    }
  }
  return discovered;
}

function csvEscape(value) {
  if (value === null || typeof value === "undefined") return "";
  const text = String(value);
  if (text.includes(",") || text.includes("\"") || text.includes("\n")) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function normalizeTrial(doc) {
  const raw = doc.raw_trial || {};
  return {
    created_at: doc.created_at ? new Date(doc.created_at).toISOString() : "",
    prolific_pid: doc.prolific_pid || raw.prolific_pid || "",
    study_id: doc.study_id || raw.study_id || "",
    session_id: doc.session_id || raw.session_id || "",
    completion_code: doc.completion_code || raw.completion_code || "",
    design: raw.design || "",
    ordering_mode: raw.ordering_mode || "",
    condition: doc.condition || raw.condition || "",
    stim_set: doc.stim_set || raw.stim_set || "",
    stim_pkg: doc.stim_pkg || raw.stim_pkg || "",
    trial_index: doc.trial_index ?? raw.trial_index ?? "",
    stim_id: doc.stim_id || raw.stim_id || "",
    word: doc.word || raw.word || "",
    word_type: doc.word_type || raw.word_type || "",
    word_length: doc.word_length ?? raw.word_length ?? "",
    ordering: doc.ordering || raw.ordering || "",
    a_is: doc.a_is || raw.a_is || "",
    b_is: doc.b_is || raw.b_is || "",
    response_key: doc.response_key || raw.response_key || "",
    choice: doc.choice || raw.choice || "",
    rt_ms: doc.rt_ms ?? raw.rt ?? "",
    reference_url: doc.reference_url || raw.reference_url || "",
    image_a_url: doc.image_a_url || raw.image_a_url || "",
    image_b_url: doc.image_b_url || raw.image_b_url || "",
    shape_match_url: doc.shape_match_url || raw.shape_match_url || "",
    texture_match_url: doc.texture_match_url || raw.texture_match_url || "",
    browser_user_agent: doc.browser_user_agent || "",
    timezone: doc.timezone || "",
  };
}

function normalizeCsvTrial(row) {
  return {
    created_at: row.created_at || "",
    prolific_pid: row.prolific_pid || row["raw_trial.prolific_pid"] || "",
    study_id: row.study_id || row["raw_trial.study_id"] || "",
    session_id: row.session_id || row["raw_trial.session_id"] || "",
    completion_code: row.completion_code || row["raw_trial.completion_code"] || "",
    design: row.design || row["raw_trial.design"] || "",
    ordering_mode: row.ordering_mode || row["raw_trial.ordering_mode"] || "",
    condition: row.condition || row["raw_trial.condition"] || "",
    stim_set: row.stim_set || row["raw_trial.stim_set"] || "",
    stim_pkg: row.stim_pkg || row["raw_trial.stim_pkg"] || "",
    trial_index: row.trial_index || row["raw_trial.trial_index"] || "",
    stim_id: row.stim_id || row["raw_trial.stim_id"] || "",
    word: row.word || row["raw_trial.word"] || "",
    word_type: row.word_type || row["raw_trial.word_type"] || "",
    word_length: row.word_length || row["raw_trial.word_length"] || "",
    ordering: row.ordering || row["raw_trial.ordering"] || "",
    a_is: row.a_is || row["raw_trial.a_is"] || "",
    b_is: row.b_is || row["raw_trial.b_is"] || "",
    response_key: row.response_key || row["raw_trial.response"] || "",
    choice: row.choice || "",
    rt_ms: row.rt_ms || row["raw_trial.rt"] || "",
    reference_url: row.reference_url || row["raw_trial.reference_url"] || "",
    image_a_url: row.image_a_url || row["raw_trial.image_a_url"] || "",
    image_b_url: row.image_b_url || row["raw_trial.image_b_url"] || "",
    shape_match_url: row.shape_match_url || row["raw_trial.shape_match_url"] || "",
    texture_match_url: row.texture_match_url || row["raw_trial.texture_match_url"] || "",
    browser_user_agent: row.browser_user_agent || "",
    timezone: row.timezone || "",
  };
}

async function loadRowsFromMongo() {
  await connectMongo();
  const docs = await ShapeBiasHumanTrial.find({}).sort({ created_at: 1 }).lean();
  return docs.map(normalizeTrial);
}

function loadRowsFromCsv(repoRoot, explicitPath) {
  const csvPaths = getFallbackCsvPaths(repoRoot, explicitPath);
  if (!csvPaths.length) {
    throw new Error("No fallback human-results CSV files found.");
  }

  const rows = [];
  for (const csvPath of csvPaths) {
    const text = fs.readFileSync(csvPath, "utf8");
    const parsed = csvRowsToObjects(text).map(normalizeCsvTrial);
    rows.push(...parsed);
  }
  return { rows, csvPaths };
}

async function main() {
  const args = parseArgs(process.argv);
  const repoRoot = path.resolve(__dirname, "..");
  const outputPath = path.resolve(
    repoRoot,
    args.output || path.join("results", "data", "human_results.csv")
  );
  const designFilter = args.design || "";
  const completionCodeFilter = args.completion_code || "";
  const csvInputPath = args.input || "";
  let rows = [];
  let source = "mongo";
  let csvPaths = [];

  try {
    rows = await loadRowsFromMongo();
  } catch (_err) {
    source = "csv";
    const fallback = loadRowsFromCsv(repoRoot, csvInputPath);
    rows = fallback.rows;
    csvPaths = fallback.csvPaths;
  }

  if (designFilter) {
    rows = rows.filter((row) => row.design === designFilter);
  }
  if (completionCodeFilter) {
    rows = rows.filter((row) => row.completion_code === completionCodeFilter);
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, toCsv(rows), "utf8");

  const participantCount = new Set(rows.map((row) => `${row.prolific_pid}|${row.study_id}|${row.session_id}`)).size;
  console.log(JSON.stringify({
    ok: true,
    source,
    input_csv_paths: csvPaths,
    output: outputPath,
    rows: rows.length,
    participants: participantCount,
    design_filter: designFilter || null,
    completion_code_filter: completionCodeFilter || null
  }, null, 2));
}

main()
  .catch((err) => {
    console.error(err && err.stack ? err.stack : String(err));
    process.exitCode = 1;
  })
  .finally(async () => {
    if (mongoose) {
      try {
        await mongoose.disconnect();
      } catch (_err) {
        // no-op
      }
    }
  });
