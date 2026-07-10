const path = require("path");
const fs = require("fs");

const MODEL_EXTENSIONS = {
  stl: [".stl"],
  glb: [".glb", ".gltf"]
};

function hashString(input) {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace(arr, rand) {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr;
}

function loadComparisonConfig(repoRoot) {
  const configPath = path.join(repoRoot, "configs", "comparison_survey.json");
  if (!fs.existsSync(configPath)) {
    throw new Error(`Comparison config not found: ${configPath}`);
  }
  const raw = JSON.parse(fs.readFileSync(configPath, "utf8"));
  return {
    pair_mode: raw.pair_mode === "fixed" ? "fixed" : "random",
    trial_count: Math.max(1, Number(raw.trial_count) || 15),
    shuffle_trials: raw.shuffle_trials !== false,
    allow_repeat_pairs: raw.allow_repeat_pairs === true,
    fixed_pairs: Array.isArray(raw.fixed_pairs) ? raw.fixed_pairs : [],
    model_sources: Array.isArray(raw.model_sources) && raw.model_sources.length > 0
      ? raw.model_sources.filter((s) => s === "stl" || s === "glb")
      : ["stl", "glb"],
    prompt: String(raw.prompt || "Which object looks more complex?")
  };
}

function listModelsInDirectory(dirPath, source, extensions) {
  if (!fs.existsSync(dirPath)) return [];
  return fs
    .readdirSync(dirPath)
    .filter((name) => extensions.some((ext) => name.toLowerCase().endsWith(ext)))
    .map((filename) => ({
      filename,
      source,
      url: `/${source}/${encodeURIComponent(filename)}`
    }))
    .sort((a, b) => a.filename.localeCompare(b.filename, undefined, { numeric: true }));
}

function listAvailableModels(repoRoot, modelSources) {
  const models = [];
  if (modelSources.includes("stl")) {
    models.push(
      ...listModelsInDirectory(
        path.join(repoRoot, "stl"),
        "stl",
        MODEL_EXTENSIONS.stl
      )
    );
  }
  if (modelSources.includes("glb")) {
    models.push(
      ...listModelsInDirectory(
        path.join(repoRoot, "glb"),
        "glb",
        MODEL_EXTENSIONS.glb
      )
    );
  }
  return models;
}

function resolveModelReference(ref, modelsByKey) {
  if (!ref) return null;
  if (typeof ref === "string") {
    const direct = modelsByKey.get(ref.toLowerCase());
    if (direct) return direct;
    const basename = path.basename(ref);
    return modelsByKey.get(basename.toLowerCase()) || null;
  }
  if (typeof ref === "object") {
    const filename = String(ref.file || ref.filename || ref.name || "");
    const source = ref.source === "glb" ? "glb" : ref.source === "stl" ? "stl" : null;
    if (source) {
      return modelsByKey.get(`${source}/${filename}`.toLowerCase()) || null;
    }
    return resolveModelReference(filename, modelsByKey);
  }
  return null;
}

function modelLookup(models) {
  const byKey = new Map();
  for (const model of models) {
    byKey.set(model.filename.toLowerCase(), model);
    byKey.set(`${model.source}/${model.filename}`.toLowerCase(), model);
  }
  return byKey;
}

function pairKey(left, right) {
  const ids = [left.filename, right.filename].sort();
  return `${ids[0]}::${ids[1]}`;
}

function buildRandomPairs(models, trialCount, rand, allowRepeatPairs) {
  if (models.length < 2) {
    throw new Error("Need at least two model files to build random comparison trials.");
  }

  const allPairs = [];
  for (let i = 0; i < models.length; i += 1) {
    for (let j = i + 1; j < models.length; j += 1) {
      allPairs.push([models[i], models[j]]);
    }
  }
  shuffleInPlace(allPairs, rand);

  const selected = [];
  const used = new Set();
  for (const [left, right] of allPairs) {
    if (selected.length >= trialCount) break;
    const key = pairKey(left, right);
    if (!allowRepeatPairs && used.has(key)) continue;
    used.add(key);
    selected.push({ left, right });
  }

  while (selected.length < trialCount) {
    const left = models[Math.floor(rand() * models.length)];
    let right = models[Math.floor(rand() * models.length)];
    let guard = 0;
    while (right.filename === left.filename && guard < 20) {
      right = models[Math.floor(rand() * models.length)];
      guard += 1;
    }
    if (right.filename === left.filename) break;
    const key = pairKey(left, right);
    if (!allowRepeatPairs && used.has(key)) continue;
    used.add(key);
    selected.push({ left, right });
  }

  return selected;
}

function buildFixedPairs(config, modelsByKey) {
  const pairs = [];
  for (const entry of config.fixed_pairs) {
    const leftRef = entry.left || entry.a || entry.model_a;
    const rightRef = entry.right || entry.b || entry.model_b;
    const left = resolveModelReference(leftRef, modelsByKey);
    const right = resolveModelReference(rightRef, modelsByKey);
    if (!left || !right) {
      throw new Error(
        `Fixed pair could not be resolved: ${JSON.stringify({ left: leftRef, right: rightRef })}`
      );
    }
    pairs.push({ left, right });
  }
  if (pairs.length === 0) {
    throw new Error("pair_mode is fixed but fixed_pairs is empty.");
  }
  return pairs;
}

function buildComparisonTrials({ repoRoot, participantSeed, configOverride = null }) {
  const config = configOverride || loadComparisonConfig(repoRoot);
  const models = listAvailableModels(repoRoot, config.model_sources);
  const modelsByKey = modelLookup(models);
  const rand = mulberry32(hashString(participantSeed));

  let pairs =
    config.pair_mode === "fixed"
      ? buildFixedPairs(config, modelsByKey)
      : buildRandomPairs(models, config.trial_count, rand, config.allow_repeat_pairs);

  if (config.shuffle_trials) {
    pairs = shuffleInPlace([...pairs], rand);
  }

  if (config.pair_mode !== "fixed" && pairs.length > config.trial_count) {
    pairs = pairs.slice(0, config.trial_count);
  }

  const trials = pairs.map((pair, index) => {
    const swapSides = rand() < 0.5;
    const modelA = swapSides ? pair.right : pair.left;
    const modelB = swapSides ? pair.left : pair.right;
    return {
      trial_index: index,
      pair_mode: config.pair_mode,
      prompt: config.prompt,
      model_a: {
        filename: modelA.filename,
        source: modelA.source,
        url: modelA.url
      },
      model_b: {
        filename: modelB.filename,
        source: modelB.source,
        url: modelB.url
      },
      pair_left_filename: pair.left.filename,
      pair_right_filename: pair.right.filename,
      pair_left_source: pair.left.source,
      pair_right_source: pair.right.source,
      side_swapped: swapSides
    };
  });

  return {
    config,
    models,
    trials
  };
}

module.exports = {
  loadComparisonConfig,
  listAvailableModels,
  buildComparisonTrials
};
