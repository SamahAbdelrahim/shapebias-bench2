#!/usr/bin/env node
/*
Generate and analyze human-friendly experiment words by English
character transition probabilities.
*/

const fs = require("fs");
const path = require("path");

const HUMAN_UNIQUE_STIM_PACKAGES = [
  "stimuli_unique_texture_per_stl_v1",
  "stimuli_unique_texture_per_stl_v2",
];

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

function makePseudoWord(rand, length) {
  const consonants = "bcdfghjklmnpqrstvwxyz";
  const vowels = "aeiou";
  let out = "";
  for (let i = 0; i < length; i += 1) {
    const bank = i % 2 === 0 ? consonants : vowels;
    out += bank[Math.floor(rand() * bank.length)];
  }
  return out;
}

function makeRandomWord(rand, length) {
  const letters = "abcdefghijklmnopqrstuvwxyz";
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += letters[Math.floor(rand() * letters.length)];
  }
  return out;
}

function buildUniqueHumanWords(count, seedText) {
  const rand = mulberry32(hashString(seedText));
  const seen = new Set();
  const out = [];
  const lengths = [6, 7, 8, 9, 10];

  const sudoCount = Math.ceil(count / 2);
  const randomCount = Math.floor(count / 2);
  const typePool = [];
  for (let i = 0; i < sudoCount; i += 1) typePool.push("sudo");
  for (let i = 0; i < randomCount; i += 1) typePool.push("random");

  for (let i = typePool.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = typePool[i];
    typePool[i] = typePool[j];
    typePool[j] = tmp;
  }

  for (let idx = 0; idx < count; idx += 1) {
    const wordType = typePool[idx];
    const length = lengths[idx % lengths.length];
    let candidate = "";
    do {
      candidate =
        wordType === "sudo"
          ? makePseudoWord(rand, length)
          : makeRandomWord(rand, length);
    } while (seen.has(candidate));
    seen.add(candidate);
    out.push({ word: candidate, word_type: wordType, word_length: length });
  }
  return out;
}

function loadCorpusWords() {
  try {
    // eslint-disable-next-line global-require, import/no-extraneous-dependencies
    const rw = require("random-words");
    if (rw && Array.isArray(rw.wordList) && rw.wordList.length > 100) {
      return rw.wordList;
    }
  } catch (e) {
    // continue with fallback
  }

  const dictPath = "/usr/share/dict/words";
  if (fs.existsSync(dictPath)) {
    const content = fs.readFileSync(dictPath, "utf8");
    const words = content
      .split(/\r?\n/)
      .map((w) => w.trim().toLowerCase())
      .filter((w) => /^[a-z]+$/.test(w));
    if (words.length > 100) return words;
  }

  return [
    "chair",
    "table",
    "window",
    "orange",
    "planet",
    "animal",
    "garden",
    "forest",
    "simple",
    "wonder",
  ];
}

function buildBigramModel(words) {
  const letters = "abcdefghijklmnopqrstuvwxyz";
  const fromChars = `^${letters}`;
  const toChars = `${letters}$`;
  const counts = {};
  const totals = {};

  for (const a of fromChars) {
    counts[a] = {};
    for (const b of toChars) counts[a][b] = 1; // Laplace smoothing
  }

  for (const raw of words) {
    const w = String(raw).toLowerCase().replace(/[^a-z]/g, "");
    if (!w) continue;
    const seq = `^${w}$`;
    for (let i = 0; i < seq.length - 1; i += 1) {
      const a = seq[i];
      const b = seq[i + 1];
      if (counts[a] && typeof counts[a][b] === "number") {
        counts[a][b] += 1;
      }
    }
  }

  for (const a of fromChars) {
    totals[a] = 0;
    for (const b of toChars) totals[a] += counts[a][b];
  }

  function scoreWord(word) {
    const w = String(word).toLowerCase().replace(/[^a-z]/g, "");
    if (!w) return Number.NEGATIVE_INFINITY;
    const seq = `^${w}$`;
    let sum = 0;
    let n = 0;
    for (let i = 0; i < seq.length - 1; i += 1) {
      const a = seq[i];
      const b = seq[i + 1];
      const p = counts[a][b] / totals[a];
      sum += Math.log(p);
      n += 1;
    }
    return sum / n;
  }

  return { scoreWord };
}

function percentile(sortedArr, p) {
  if (!sortedArr.length) return Number.NaN;
  const idx = Math.max(0, Math.min(sortedArr.length - 1, Math.floor(p * sortedArr.length)));
  return sortedArr[idx];
}

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const esc = (v) => {
    const s = String(v);
    if (s.includes(",") || s.includes("\"") || s.includes("\n")) {
      return `"${s.replace(/"/g, "\"\"")}"`;
    }
    return s;
  };
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => esc(row[h])).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function main() {
  const args = parseArgs(process.argv);
  const prolific_pid = args.prolific_pid || "debug_pid";
  const study_id = args.study_id || "debug_study";
  const session_id = args.session_id || "debug_session";
  const stim_set = args.stim_set || "stimuli_A_auto_contrast";
  const condition = args.condition || "noun_label";
  const trial_limit = Number(args.trial_limit || 30);
  const stim_pkg =
    args.stim_pkg ||
    HUMAN_UNIQUE_STIM_PACKAGES[
      hashString(`${prolific_pid}|${study_id}|${session_id}`) %
        HUMAN_UNIQUE_STIM_PACKAGES.length
    ];

  if (condition === "no_word_category") {
    console.log("condition=no_word_category -> no words to analyze.");
    return;
  }

  const seedText = `${prolific_pid}|${study_id}|${session_id}|${stim_set}|${stim_pkg}|${condition}|words`;
  const words = buildUniqueHumanWords(trial_limit, seedText);
  const corpus = loadCorpusWords();
  const { scoreWord } = buildBigramModel(corpus);

  const baselineScores = corpus
    .map((w) => scoreWord(w))
    .filter((x) => Number.isFinite(x))
    .sort((a, b) => a - b);

  const p10 = percentile(baselineScores, 0.1);
  const p25 = percentile(baselineScores, 0.25);

  const pseudo = words.filter((w) => w.word_type === "sudo");
  const random = words.filter((w) => w.word_type === "random");

  const scored = words.map((w, idx) => {
    const s = scoreWord(w.word);
    return {
      index: idx + 1,
      word: w.word,
      word_type: w.word_type,
      word_length: w.word_length,
      transition_logprob: s.toFixed(6),
      english_like_p10: s >= p10 ? 1 : 0,
      english_like_p25: s >= p25 ? 1 : 0,
    };
  });

  const outDir = path.join(__dirname, "reports");
  fs.mkdirSync(outDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const base = `${prolific_pid}_${study_id}_${session_id}_${stamp}`;
  const csvPath = path.join(outDir, `human_words_analysis_${base}.csv`);
  const jsonPath = path.join(outDir, `human_words_analysis_${base}.json`);

  fs.writeFileSync(csvPath, toCsv(scored), "utf8");

  const pseudoLikeP10 = scored.filter((r) => r.word_type === "sudo" && r.english_like_p10).length;
  const randomLikeP10 = scored.filter((r) => r.word_type === "random" && r.english_like_p10).length;

  const summary = {
    prolific_pid,
    study_id,
    session_id,
    stim_set,
    stim_pkg,
    condition,
    trial_limit,
    seed_text: seedText,
    corpus_size: corpus.length,
    thresholds: {
      p10_logprob: p10,
      p25_logprob: p25,
    },
    counts: {
      total_words: words.length,
      pseudo_words: pseudo.length,
      random_words: random.length,
      pseudo_english_like_p10: pseudoLikeP10,
      random_english_like_p10: randomLikeP10,
      total_english_like_p10: pseudoLikeP10 + randomLikeP10,
    },
    output: {
      csv: csvPath,
      json: jsonPath,
    },
  };

  fs.writeFileSync(jsonPath, JSON.stringify(summary, null, 2), "utf8");

  console.log("Word analysis complete.");
  console.log(JSON.stringify(summary, null, 2));
}

main();

