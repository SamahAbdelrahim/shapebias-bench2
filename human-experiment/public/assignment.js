/**
 * Participant assignment for the shape-bias human experiment.
 *
 * Everything here is a pure function of the trial pool and the participant
 * seed, with no DOM and no network, so the counterbalancing can be checked in
 * Node (see human-experiment/verify_assignment.js) rather than only by clicking
 * through a browser. The frontend loads this file before experiment.js and
 * reads it off the global; Node requires it directly.
 *
 * Design in brief:
 *   - condition, ordering group and block order are hashes of the Prolific
 *     identifiers, so a reload lands in the same cell with the same items
 *   - each participant takes a contiguous, wrapping window of the pool, which
 *     is emitted round-robin across shapes and classes so any window spans both
 *   - a trial's ordering is the parity of its pool index, flipped for group B,
 *     so each triad is shape-first for one group and texture-first for the other
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.SBAssignment = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const CONDITIONS = ["noun_label", "no_word_category"];
  const ORDERING_GROUPS = ["A", "B"];
  const SET_ORDER_DEFAULT = ["grid", "cc_triads"];
  const TRIALS_PER_SET = 18;
  const CATCH_PER_SESSION = 4;

  // Frequent English bigrams with rough relative weights (higher = more
  // English-like). Used only for filtering pseudo-words.
  const ENGLISH_BIGRAM_WEIGHTS = {
    th: 1.0, he: 0.98, in: 0.96, er: 0.94, an: 0.93, re: 0.92, on: 0.91, at: 0.9,
    en: 0.9, nd: 0.89, ti: 0.88, es: 0.87, or: 0.86, te: 0.86, of: 0.85, ed: 0.85,
    is: 0.84, it: 0.84, al: 0.83, ar: 0.82, st: 0.82, to: 0.82, nt: 0.81, ng: 0.81,
    se: 0.8, ha: 0.8, as: 0.79, ou: 0.79, io: 0.78, le: 0.78, ve: 0.77, co: 0.77,
    me: 0.76, de: 0.76, hi: 0.75, ri: 0.75, ro: 0.74, ic: 0.74, ne: 0.74, ea: 0.73,
    ra: 0.73, ce: 0.72, li: 0.72, ch: 0.72, ll: 0.71, be: 0.71, ma: 0.7, si: 0.7,
    om: 0.69, ur: 0.69, ca: 0.68, el: 0.68, ta: 0.68, la: 0.67, ns: 0.67, di: 0.67,
    fo: 0.66, ho: 0.66, pe: 0.65, ec: 0.65, pr: 0.65, no: 0.64, wa: 0.64, wi: 0.64,
    us: 0.63, tr: 0.63, wh: 0.63, ge: 0.62, po: 0.62, lo: 0.62, im: 0.61, il: 0.61,
    mo: 0.61, un: 0.6, ai: 0.6, ie: 0.59, oo: 0.59, ee: 0.58, ss: 0.57, tt: 0.57
  };

  function hashString(input) {
    let h = 2166136261;
    const s = String(input);
    for (let i = 0; i < s.length; i += 1) {
      h ^= s.charCodeAt(i);
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

  /**
   * A well-mixed number in [0,1) from a seed string.
   *
   * Do not use `hashString(x) % k` to assign the between-participant factors.
   * FNV-1a ends in a multiply by an odd prime, so the low bit of the digest is
   * a near-linear function of the input bytes: hashing one seed under two
   * different salts yields coins that are perfectly correlated rather than
   * independent. Passing the digest through mulberry32 avalanches it first.
   */
  function seededUnit(seedText) {
    return mulberry32(hashString(seedText))();
  }

  function seededIndex(seedText, n) {
    return Math.min(n - 1, Math.floor(seededUnit(seedText) * n));
  }

  function seededShuffle(arr, seedText) {
    const rand = mulberry32(hashString(seedText));
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
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

  function englishTransitionScore(word) {
    const w = String(word || "").toLowerCase().replace(/[^a-z]/g, "");
    if (w.length < 2) return 0;
    let sum = 0;
    let n = 0;
    for (let i = 0; i < w.length - 1; i += 1) {
      const bg = w.slice(i, i + 2);
      // Smoothing floor keeps unseen transitions from being exactly zero.
      const p = ENGLISH_BIGRAM_WEIGHTS[bg] || 0.02;
      sum += p;
      n += 1;
    }
    return sum / n;
  }

  function buildUniqueWords(count, seedText, opts) {
    const o = opts || {};
    const minLen = o.wordMinLen || 4;
    const maxLen = o.wordMaxLen || 8;
    const threshold = typeof o.sudoThreshold === "number" ? o.sudoThreshold : 0.62;
    const rand = mulberry32(hashString(seedText));
    const seen = new Set();
    const lengths = [];
    for (let len = minLen; len <= maxLen; len += 1) lengths.push(len);

    const out = [];
    for (let idx = 0; idx < count; idx += 1) {
      const length = lengths[idx % lengths.length];
      let candidate = "";
      let bestCandidate = "";
      let bestScore = -1;
      let accepted = false;
      for (let tries = 0; tries < 400; tries += 1) {
        const maybe = makePseudoWord(rand, length);
        if (seen.has(maybe)) continue;
        const s = englishTransitionScore(maybe);
        if (s > bestScore) {
          bestScore = s;
          bestCandidate = maybe;
        }
        if (s >= threshold) {
          candidate = maybe;
          accepted = true;
          break;
        }
      }
      // Fallback: keep the best candidate seen so generation always completes.
      if (!accepted) candidate = bestCandidate || makePseudoWord(rand, length);
      seen.add(candidate);
      out.push({ name: candidate, type: "sudo", length });
    }
    return out;
  }

  function takeWindow(items, startIndex, count) {
    const out = [];
    const n = Math.min(count, items.length);
    for (let i = 0; i < n; i += 1) {
      const idx = (startIndex + i) % items.length;
      out.push({ item: items[idx], poolIndex: idx });
    }
    return out;
  }

  function pickCondition(participantSeed, override) {
    if (CONDITIONS.indexOf(override) !== -1) return override;
    return CONDITIONS[seededIndex(`condition|${participantSeed}`, CONDITIONS.length)];
  }

  function pickOrderingGroup(participantSeed, override) {
    if (ORDERING_GROUPS.indexOf(override) !== -1) return override;
    return ORDERING_GROUPS[seededIndex(`ordering_group|${participantSeed}`, ORDERING_GROUPS.length)];
  }

  function pickSetOrder(participantSeed) {
    const flipped = seededIndex(`block_order|${participantSeed}`, 2) === 1;
    return flipped ? SET_ORDER_DEFAULT.slice().reverse() : SET_ORDER_DEFAULT.slice();
  }

  function buildTestTrials(pool, participantSeed, setOrder, orderingGroup, trialsPerSet) {
    const bySet = {};
    for (const entry of pool.test) {
      (bySet[entry.stim_set_name] = bySet[entry.stim_set_name] || []).push(entry);
    }
    // Group B takes the opposite ordering of group A on every triad.
    const groupOffset = orderingGroup === "B" ? 1 : 0;

    const trials = [];
    for (let blockIndex = 0; blockIndex < setOrder.length; blockIndex += 1) {
      const setName = setOrder[blockIndex];
      const items = bySet[setName] || [];
      if (items.length === 0) continue;
      const start = seededIndex(`window|${setName}|${participantSeed}`, items.length);
      const window = takeWindow(items, start, trialsPerSet);
      for (const entry of window) {
        const item = entry.item;
        const shapeFirst = (entry.poolIndex + groupOffset) % 2 === 0;
        trials.push({
          block_index: blockIndex,
          stim_set_name: setName,
          stim_id: item.stim_id,
          stl_id: item.stl_id,
          texture_set: item.texture_set,
          pool_index: entry.poolIndex,
          is_catch: false,
          ordering: shapeFirst ? "shape_first" : "texture_first",
          a_is: shapeFirst ? "shape" : "texture",
          b_is: shapeFirst ? "texture" : "shape",
          reference_url: item.reference_url,
          image_a_url: shapeFirst ? item.shape_match_url : item.texture_match_url,
          image_b_url: shapeFirst ? item.texture_match_url : item.shape_match_url,
          shape_match_url: item.shape_match_url,
          texture_match_url: item.texture_match_url
        });
      }
    }
    return trials;
  }

  function buildCatchTrials(pool, participantSeed, setOrder, catchPerSession) {
    const catchPool = Array.isArray(pool.catch) ? pool.catch : [];
    if (catchPool.length === 0 || catchPerSession <= 0) return [];

    const shuffled = seededShuffle(catchPool, `${participantSeed}|catch`);
    // Consume from per-set queues so a session never repeats a check and the
    // checks are split across both stimulus sets.
    const queues = {};
    for (const c of shuffled) {
      (queues[c.stim_set_name] = queues[c.stim_set_name] || []).push(c);
    }
    const spare = shuffled.slice();
    const picked = [];
    for (let i = 0; i < catchPerSession; i += 1) {
      const preferred = setOrder[i % setOrder.length];
      const other = setOrder[(i + 1) % setOrder.length];
      const pick =
        (queues[preferred] && queues[preferred].shift()) ||
        (queues[other] && queues[other].shift()) ||
        spare.shift();
      if (pick && picked.indexOf(pick) === -1) picked.push(pick);
    }

    const sideRand = mulberry32(hashString(`${participantSeed}|catch_side`));
    return picked.map((c) => {
      const matchFirst = sideRand() < 0.5;
      return {
        block_index: -1,
        stim_set_name: c.stim_set_name,
        stim_id: c.stim_id,
        stl_id: c.stl_id,
        texture_set: c.texture_set,
        pool_index: -1,
        is_catch: true,
        ordering: matchFirst ? "match_first" : "foil_first",
        a_is: matchFirst ? "match" : "foil",
        b_is: matchFirst ? "foil" : "match",
        reference_url: c.reference_url,
        image_a_url: matchFirst ? c.match_url : c.foil_url,
        image_b_url: matchFirst ? c.foil_url : c.match_url,
        shape_match_url: c.match_url,
        texture_match_url: c.foil_url
      };
    });
  }

  /**
   * Resolve one participant's whole session.
   *
   * Returns the between-participant cell plus the ordered trial list, with
   * pseudo-words already attached in the noun condition.
   */
  function assignParticipant(pool, participantSeed, options) {
    const opts = options || {};
    const condition = pickCondition(participantSeed, opts.condition);
    const orderingGroup = pickOrderingGroup(participantSeed, opts.orderingGroup);
    const setOrder = pickSetOrder(participantSeed);
    const trialsPerSet = opts.trialsPerSet || TRIALS_PER_SET;
    const catchPerSession =
      typeof opts.catchPerSession === "number" ? opts.catchPerSession : CATCH_PER_SESSION;

    const testTrials = buildTestTrials(pool, participantSeed, setOrder, orderingGroup, trialsPerSet);
    const catchTrials = buildCatchTrials(pool, participantSeed, setOrder, catchPerSession);

    // Shuffle within each block so pool order is not presentation order, then
    // spread the checks evenly through the session.
    const ordered = [];
    for (let blockIndex = 0; blockIndex < setOrder.length; blockIndex += 1) {
      const block = testTrials.filter((t) => t.block_index === blockIndex);
      ordered.push(...seededShuffle(block, `${participantSeed}|block|${blockIndex}`));
    }

    const trials = ordered.slice();
    if (catchTrials.length > 0) {
      const spacing = Math.max(1, Math.floor(ordered.length / (catchTrials.length + 1)));
      for (let i = 0; i < catchTrials.length; i += 1) {
        const insertAt = Math.min(trials.length, spacing * (i + 1) + i);
        trials.splice(insertAt, 0, catchTrials[i]);
      }
    }

    const words =
      condition === "no_word_category"
        ? []
        : buildUniqueWords(trials.length, `${participantSeed}|words|${condition}`, opts);
    for (let i = 0; i < trials.length; i += 1) {
      const w = condition === "no_word_category"
        ? { name: "", type: "none", length: 0 }
        : words[i];
      trials[i].word = w.name;
      trials[i].word_type = w.type;
      trials[i].word_length = w.length;
      trials[i].condition = condition;
      trials[i].ordering_group = orderingGroup;
      trials[i].pool_version = pool.pool_version || "unknown";
    }

    return {
      condition,
      orderingGroup,
      setOrder,
      testCount: ordered.length,
      catchCount: catchTrials.length,
      trials
    };
  }

  return {
    CONDITIONS,
    ORDERING_GROUPS,
    SET_ORDER_DEFAULT,
    TRIALS_PER_SET,
    CATCH_PER_SESSION,
    hashString,
    mulberry32,
    seededUnit,
    seededIndex,
    seededShuffle,
    englishTransitionScore,
    buildUniqueWords,
    takeWindow,
    assignParticipant
  };
});
