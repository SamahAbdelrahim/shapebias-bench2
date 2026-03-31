(async function runExperiment() {
  const DBG_PREFIX = "[SB-HUMAN]";
  let jsPsychMounted = false;
  const mountWatchdogMs = 2000;
  setTimeout(() => {
    if (!jsPsychMounted) {
      console.warn(`${DBG_PREFIX} jsPsych did not mount yet`, {
        after_ms: mountWatchdogMs,
        href: window.location.href,
        readyState: document.readyState
      });
    }
  }, mountWatchdogMs);

  const ck = (label, data) => {
    if (typeof data === "undefined") {
      console.log(`${DBG_PREFIX} ${label}`);
    } else {
      console.log(`${DBG_PREFIX} ${label}`, data);
    }
  };
  const cwarn = (label, data) => {
    if (typeof data === "undefined") {
      console.warn(`${DBG_PREFIX} ${label}`);
    } else {
      console.warn(`${DBG_PREFIX} ${label}`, data);
    }
  };
  const cerr = (label, data) => {
    if (typeof data === "undefined") {
      console.error(`${DBG_PREFIX} ${label}`);
    } else {
      console.error(`${DBG_PREFIX} ${label}`, data);
    }
  };
  const setBootStatus = (text) => ck(`BOOT: ${text}`);

  ck("Experiment boot start", { href: window.location.href, ts: new Date().toISOString() });
  setBootStatus("Initializing experiment...");

  if (
    typeof initJsPsych === "undefined" ||
    typeof jsPsychPreload === "undefined" ||
    typeof jsPsychInstructions === "undefined" ||
    typeof jsPsychHtmlButtonResponse === "undefined"
  ) {
    cerr("Missing jsPsych globals", {
      initJsPsych: typeof initJsPsych,
      jsPsychPreload: typeof jsPsychPreload,
      jsPsychInstructions: typeof jsPsychInstructions,
      jsPsychHtmlButtonResponse: typeof jsPsychHtmlButtonResponse
    });
    throw new Error(
      "jsPsych core/plugins failed to load. Check CDN access and script URLs in public/index.html."
    );
  }
  ck("jsPsych globals verified");

  const BENCHMARK_WORD_PAIRS = [
    ["shiple", "afnafq", 6],
    ["clapher", "ieyiccw", 7],
    ["plailass", "orvufaig", 8],
    ["procation", "qahftrxck", 9],
    ["adinefults", "cgchqjjfgy", 10]
  ];
  const BENCHMARK_WORDS = [];
  for (const [sudo, rand, length] of BENCHMARK_WORD_PAIRS) {
    BENCHMARK_WORDS.push({ name: sudo, type: "sudo", length });
    BENCHMARK_WORDS.push({ name: rand, type: "random", length });
  }
  const BENCHMARK_STIM_PACKAGE = "stimuli_per_stl_packages";
  const HUMAN_UNIQUE_STIM_PACKAGES = [
    "stimuli_unique_texture_per_stl_v1",
    "stimuli_unique_texture_per_stl_v2"
  ];
  // Frequent English bigrams with rough relative weights (higher = more English-like).
  // Used only for filtering pseudo-words (sudo words), not random words.
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

  const url = new URL(window.location.href);
  const params = url.searchParams;
  const prolific_pid = params.get("PROLIFIC_PID") || "debug_pid";
  const study_id = params.get("STUDY_ID") || "debug_study";
  const session_id = params.get("SESSION_ID") || "debug_session";
  const design = params.get("design") || "human_friendly"; // human_friendly | benchmark
  const condition = params.get("condition") || "noun_label"; // noun_label | no_word_category
  const stimSet = params.get("stim_set") || "stimuli_A_auto_contrast";
  const stimPkgParam = params.get("stim_pkg");
  const orderMode = params.get("ordering") || (design === "human_friendly" ? "random" : "both");
  const defaultLimit = design === "human_friendly" ? 30 : 0;
  const trialLimit = Number(params.get("trial_limit") || defaultLimit);
  const shuffleTrials = params.get("shuffle") !== "0";
  const verboseTrials = params.get("verbose_trials") === "1";
  const preloadMode = params.get("preload") || "minimal"; // off | minimal | all
  const requestedWordMode = params.get("word_mode") || "sudo_only"; // sudo_only | mixed
  const wordMode = requestedWordMode === "mixed" ? "mixed" : "sudo_only";
  const wordMinLen = Math.max(1, Number(params.get("word_min_len") || 4));
  const wordMaxLen = Math.max(wordMinLen, Number(params.get("word_max_len") || 8));
  // Sudo (pseudo) words only: minimum English-transition score [0..1].
  // Higher means stricter English-like filtering.
  const sudoThreshold = Number(params.get("sudo_threshold") || 0.62);
  ck("Parsed URL parameters", {
    prolific_pid,
    study_id,
    session_id,
    design,
    condition,
    stimSet,
    stimPkgParam,
    orderMode,
    trialLimit,
    shuffleTrials,
    verboseTrials,
    preloadMode,
    wordMode,
    wordMinLen,
    wordMaxLen,
    sudoThreshold
  });

  setBootStatus("Loading config...");
  ck("Fetching /api/config");
  const configRes = await fetch("/api/config");
  ck("Config response received", { ok: configRes.ok, status: configRes.status });
  const config = await configRes.json();
  const completionCode = params.get("cc") || config.completion_code || "TESTCODE";
  const participantSeed = `${prolific_pid}|${study_id}|${session_id}`;
  const stimPackage =
    design === "human_friendly"
      ? HUMAN_UNIQUE_STIM_PACKAGES.includes(stimPkgParam)
        ? stimPkgParam
        : chooseParticipantStimPackage(participantSeed)
      : BENCHMARK_STIM_PACKAGE;
  ck("Resolved config", { completionCode, default_stim_set: config.default_stim_set });

  setBootStatus("Loading stimuli list...");
  ck("Fetching /api/stimuli", { stim_set: stimSet, stim_pkg: stimPackage, design });
  const stimRes = await fetch(
    `/api/stimuli?stim_set=${encodeURIComponent(stimSet)}&stim_pkg=${encodeURIComponent(stimPackage)}&design=${encodeURIComponent(design)}&prolific_pid=${encodeURIComponent(prolific_pid)}&study_id=${encodeURIComponent(study_id)}&session_id=${encodeURIComponent(session_id)}`
  );
  ck("Stimuli response received", { ok: stimRes.ok, status: stimRes.status });
  const stimData = await stimRes.json();
  const resolvedStimSet = stimData.stim_set || stimSet;
  const resolvedStimPackage = stimData.stim_pkg || stimPackage;
  ck("Stimuli payload parsed", {
    stim_set: resolvedStimSet,
    stim_pkg: resolvedStimPackage,
    hasStimuliArray: Array.isArray(stimData.stimuli),
    count: Array.isArray(stimData.stimuli) ? stimData.stimuli.length : null
  });
  if (!stimData.stimuli || !Array.isArray(stimData.stimuli) || stimData.stimuli.length === 0) {
    cerr("No stimuli available from /api/stimuli", stimData);
    document.body.innerHTML = "<p>Failed to load stimuli. Please contact the researcher.</p>";
    return;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildPromptParts(word) {
    if (condition === "no_word_category") {
      return {
        intro: 'This first image is a "<span class="sb-highlight">label</span>".',
        question:
          'Which of the following two is also a "<span class="sb-highlight">label</span>"?'
      };
    }
    const safeWord = escapeHtml(word);
    return {
      intro: `This first image is a "<span class="sb-highlight">${safeWord}</span>".`,
      question: `Which of the following two is also a "<span class="sb-highlight">${safeWord}</span>"?`
    };
  }

  function chooseOrderings() {
    if (orderMode === "shape_first") return ["shape_first"];
    if (orderMode === "texture_first") return ["texture_first"];
    if (orderMode === "random") return [Math.random() < 0.5 ? "shape_first" : "texture_first"];
    return ["shape_first", "texture_first"];
  }

  function shuffleInPlace(arr) {
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  function hashString(input) {
    let h = 2166136261;
    for (let i = 0; i < input.length; i += 1) {
      h ^= input.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function chooseParticipantStimPackage(seedText) {
    const idx = hashString(seedText) % HUMAN_UNIQUE_STIM_PACKAGES.length;
    return HUMAN_UNIQUE_STIM_PACKAGES[idx];
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

  function randomInt(rand, min, maxInclusive) {
    return Math.floor(rand() * (maxInclusive - min + 1)) + min;
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

  function buildUniqueHumanWords(count, seedText, mode = "sudo_only") {
    const rand = mulberry32(hashString(seedText));
    const seen = new Set();
    const out = [];
    const lengths = [];
    for (let len = wordMinLen; len <= wordMaxLen; len += 1) {
      lengths.push(len);
    }

    // Default behavior: all words are pseudo (sudo) for human-friendly mode.
    // Optional fallback: mixed 50/50 sudo+random via word_mode=mixed.
    const sudoCount = mode === "mixed" ? Math.ceil(count / 2) : count;
    const randomCount = mode === "mixed" ? Math.floor(count / 2) : 0;
    const typePool = [];
    for (let i = 0; i < sudoCount; i += 1) typePool.push("sudo");
    for (let i = 0; i < randomCount; i += 1) typePool.push("random");

    // Seeded shuffle so assignments are deterministic per participant seed.
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
      if (wordType === "sudo") {
        // Strictly filter sudo words by English-like transition score.
        let bestCandidate = "";
        let bestScore = -1;
        let accepted = false;
        const maxAttempts = 400;
        for (let tries = 0; tries < maxAttempts; tries += 1) {
          const maybe = makePseudoWord(rand, length);
          if (seen.has(maybe)) continue;
          const s = englishTransitionScore(maybe);
          if (s > bestScore) {
            bestScore = s;
            bestCandidate = maybe;
          }
          if (s >= sudoThreshold) {
            candidate = maybe;
            accepted = true;
            break;
          }
        }
        // Fallback: keep best seen candidate so generation always completes.
        if (!accepted) candidate = bestCandidate || makePseudoWord(rand, length);
      } else {
        do {
          candidate = makeRandomWord(rand, length);
        } while (seen.has(candidate));
      }
      seen.add(candidate);
      out.push({ name: candidate, type: wordType, length });
    }
    return out;
  }

  function maybeSampleStimuli(stimuli, count, seedText) {
    if (count <= 0 || count >= stimuli.length) return [...stimuli];
    const rand = mulberry32(hashString(seedText));
    const copied = [...stimuli];
    for (let i = copied.length - 1; i > 0; i -= 1) {
      const j = randomInt(rand, 0, i);
      const tmp = copied[i];
      copied[i] = copied[j];
      copied[j] = tmp;
    }
    return copied.slice(0, count);
  }

  function renderStimulusHtml(t) {
    return `
      <div class="sb-container sb-trial-view">
        <div class="sb-card sb-reference-card">
          <div class="sb-prompt sb-prompt-top">${t.prompt_intro}</div>
          <img class="sb-image sb-reference-image" src="${t.reference_url}" alt="reference image">
        </div>
        <div class="sb-prompt sb-prompt-question">${t.prompt_question}</div>
        <div class="sb-grid">
          <div class="sb-card sb-option-card">
            <div class="sb-label">Option 1</div>
            <img class="sb-image" src="${t.image_a_url}" alt="option 1 image">
          </div>
          <div class="sb-card sb-option-card">
            <div class="sb-label">Option 2</div>
            <img class="sb-image" src="${t.image_b_url}" alt="option 2 image">
          </div>
        </div>
      </div>
    `;
  }

  const trialVars = [];
  if (design === "human_friendly") {
    // Human-friendly default: one trial per object with unique labels to avoid memory carry-over.
    const seedText = `${prolific_pid}|${study_id}|${session_id}|${resolvedStimSet}|${resolvedStimPackage}|${condition}|${wordMode}`;
    const selectedStimuli = maybeSampleStimuli(stimData.stimuli, trialLimit, `${seedText}|stimuli`);
    const uniqueWords = buildUniqueHumanWords(selectedStimuli.length, `${seedText}|words`, wordMode);
    ck("Human-friendly trial builder", {
      seedPreview: `${seedText.slice(0, 20)}...`,
      selectedStimuli: selectedStimuli.length,
      uniqueWords: uniqueWords.length,
      wordMode
    });
    for (let i = 0; i < selectedStimuli.length; i += 1) {
      const stim = selectedStimuli[i];
      const w =
        condition === "no_word_category"
          ? { name: `__no_word__${i + 1}`, type: "none", length: 0 }
          : uniqueWords[i];
      const orderingChoices = chooseOrderings();
      const ordering =
        orderingChoices.length === 1
          ? orderingChoices[0]
          : orderingChoices[Math.floor(Math.random() * orderingChoices.length)];
      const shapeFirst = ordering === "shape_first";
      const promptParts = buildPromptParts(w.name);
      trialVars.push({
        prolific_pid,
        study_id,
        session_id,
        completion_code: completionCode,
        condition,
        stim_set: resolvedStimSet,
        stim_pkg: resolvedStimPackage,
        stim_id: String(stim.stim_id),
        word: w.name,
        word_type: w.type,
        word_length: w.length,
        ordering,
        a_is: shapeFirst ? "shape" : "texture",
        b_is: shapeFirst ? "texture" : "shape",
        reference_url: stim.reference_url,
        image_a_url: shapeFirst ? stim.shape_match_url : stim.texture_match_url,
        image_b_url: shapeFirst ? stim.texture_match_url : stim.shape_match_url,
        shape_match_url: stim.shape_match_url,
        texture_match_url: stim.texture_match_url,
        prompt_text:
          condition === "no_word_category"
            ? "Which of the following two images is also a label?"
            : `Which of the following two images is also a ${w.name}?`,
        prompt_intro: promptParts.intro,
        prompt_question: promptParts.question
      });
    }
  } else {
    // Benchmark mode (model-like exhaustive Cartesian product).
    ck("Benchmark trial builder", {
      stimuli: stimData.stimuli.length,
      words: BENCHMARK_WORDS.length,
      orderings: orderMode
    });
    for (const stim of stimData.stimuli) {
      for (const w of BENCHMARK_WORDS) {
        const ords = chooseOrderings();
        for (const ordering of ords) {
          const shapeFirst = ordering === "shape_first";
          const promptParts = buildPromptParts(w.name);
          trialVars.push({
            prolific_pid,
            study_id,
            session_id,
            completion_code: completionCode,
            condition,
            stim_set: resolvedStimSet,
            stim_pkg: resolvedStimPackage,
            stim_id: String(stim.stim_id),
            word: w.name,
            word_type: w.type,
            word_length: w.length,
            ordering,
            a_is: shapeFirst ? "shape" : "texture",
            b_is: shapeFirst ? "texture" : "shape",
            reference_url: stim.reference_url,
            image_a_url: shapeFirst ? stim.shape_match_url : stim.texture_match_url,
            image_b_url: shapeFirst ? stim.texture_match_url : stim.shape_match_url,
            shape_match_url: stim.shape_match_url,
            texture_match_url: stim.texture_match_url,
            prompt_text:
              condition === "no_word_category"
                ? "Which of the following two images is also a label?"
                : `Which of the following two images is also a ${w.name}?`,
            prompt_intro: promptParts.intro,
            prompt_question: promptParts.question
          });
        }
      }
    }
  }

  if (shuffleTrials) shuffleInPlace(trialVars);
  for (const t of trialVars) {
    t.stimulus_html = renderStimulusHtml(t);
  }
  const finalTrials = design === "human_friendly" ? trialVars : trialLimit > 0 ? trialVars.slice(0, trialLimit) : trialVars;
  setBootStatus(`Preparing ${finalTrials.length} trials...`);
  ck("Final trial set ready", {
    generated: trialVars.length,
    final: finalTrials.length,
    shuffled: shuffleTrials
  });
  if (finalTrials.length > 0) {
    ck("First trial preview", {
      stim_id: finalTrials[0].stim_id,
      word: finalTrials[0].word,
      ordering: finalTrials[0].ordering,
      a_is: finalTrials[0].a_is,
      b_is: finalTrials[0].b_is
    });
  } else {
    cwarn("No trials generated after filtering");
  }

  async function logTrial(data) {
    const res = await fetch("/api/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      let bodyText = "";
      try {
        bodyText = await res.text();
      } catch (_e) {
        bodyText = "<failed to read body>";
      }
      throw new Error(`POST /api/log failed (${res.status}): ${bodyText}`);
    }
    return res;
  }

  ck("Initializing jsPsych");
  const jsPsych = initJsPsych({
    on_finish: () => {
      ck("jsPsych on_finish called, redirecting to Prolific", {
        completionCode
      });
      window.location.href = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(completionCode)}`;
    }
  });

  jsPsych.data.addProperties({
    prolific_pid,
    study_id,
    session_id,
    design,
    condition,
    stim_set: resolvedStimSet,
    stim_pkg: resolvedStimPackage,
    ordering_mode: orderMode
  });

  const preload = {
    type: jsPsychPreload
  };
  if (preloadMode === "all") {
    preload.images = finalTrials.flatMap((t) => [t.reference_url, t.image_a_url, t.image_b_url]);
  } else if (preloadMode === "minimal") {
    const firstTrials = finalTrials.slice(0, Math.min(2, finalTrials.length));
    preload.images = firstTrials.flatMap((t) => [t.reference_url, t.image_a_url, t.image_b_url]);
  } else {
    preload.images = [];
  }
  preload.show_progress_bar = true;
  preload.message = "Loading media...";
  preload.continue_after_error = true;
  ck("Preload configured", {
    preloadMode,
    imageCount: preload.images.length
  });

  const intro = {
    type: jsPsychInstructions,
    pages: [
      `<div class="sb-container">
        <div class="sb-intro-logo-wrap">
          <img class="sb-intro-logo" src="/general_assets/stanford.png" alt="Stanford University logo" onerror="this.style.display='none'">
        </div>
        <h2 class="sb-intro-title">Welcome</h2>
        <div class="sb-intro-copy">
          <p>By answering the following questions, you are participating in a study being performed by cognitive scientists in the Stanford Department of Psychology.</p>
          <p>If you have questions about this research, please contact us at <a href="mailto:languagecoglab@gmail.com">languagecoglab@gmail.com</a>.</p>
          <p>You must be at least 18 years old to participate. Your participation is voluntary.</p>
          <p>You may decline to answer any question, and you may stop at any time without adverse consequences.</p>
          <p>Your responses are anonymous and will be used for research purposes only.</p>
        </div>
      </div>`,
      `<div class="sb-container sb-instruction-page">
        <h3 class="sb-intro-title">Task Instructions</h3>
        <div class="sb-intro-copy">
          <p>You will complete <b>${finalTrials.length}</b> trials. In each trial:</p>
          <ul class="sb-instruction-list">
            <li>An image of an object appears at the top. The object is labeled with a word.</li>
            <li>Two options will appear with it. Your task is to choose between these two options.</li>
          </ul>
          <div class="sb-example-wrap">
            <p class="sb-example-title">For example:</p>
            <div class="sb-card sb-reference-card sb-example-reference">
              <div class="sb-prompt sb-prompt-top">This first image is a "<span class="sb-highlight">chair</span>".</div>
              <img
                class="sb-image sb-reference-image sb-example-image"
                src="/general_assets/chair_target.jpg"
                alt="example target chair"
                onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
              >
            </div>
            <div class="sb-prompt sb-prompt-question">Which of the following is also a "<span class="sb-highlight">chair</span>"?</div>
            <div class="sb-grid sb-example-grid">
              <div class="sb-card sb-option-card">
                <div class="sb-label">Option 1</div>
                <img
                  class="sb-image sb-example-image"
                  src="/general_assets/chair_option_match.jpg"
                  alt="example matching chair option"
                  onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
                >
              </div>
              <div class="sb-card sb-option-card">
                <div class="sb-label">Option 2</div>
                <img
                  class="sb-image sb-example-image"
                  src="/general_assets/chair_option_nonmatch.jpg"
                  alt="example non-matching option"
                  onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
                >
              </div>
            </div>
          </div>
        </div>
      </div>`,
      `<div class="sb-container sb-instruction-page">
        <h3 class="sb-intro-title">Task Instructions</h3>
        <div class="sb-intro-copy">
          <div class="sb-example-wrap">
            <p class="sb-example-title">Another example:</p>
            <div class="sb-card sb-reference-card sb-example-reference">
              <div class="sb-prompt sb-prompt-top">This first image is "<span class="sb-highlight">sand</span>".</div>
              <img
                class="sb-image sb-reference-image sb-example-image"
                src="/general_assets/sand_target.jpg"
                alt="example target sand"
                onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
              >
            </div>
            <div class="sb-prompt sb-prompt-question">Which of the following is also "<span class="sb-highlight">sand</span>"?</div>
            <div class="sb-grid sb-example-grid">
              <div class="sb-card sb-option-card">
                <div class="sb-label">Option 1</div>
                <img
                  class="sb-image sb-example-image"
                  src="/general_assets/sand_option_match.jpg"
                  alt="example matching sand option"
                  onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
                >
              </div>
              <div class="sb-card sb-option-card">
                <div class="sb-label">Option 2</div>
                <img
                  class="sb-image sb-example-image"
                  src="/general_assets/sand_option_nonmatch.jpg"
                  alt="example non-matching sand option"
                  onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
                >
              </div>
            </div>
          </div>
        </div>
      </div>`,
      `<div class="sb-container sb-instruction-page">
        <h3 class="sb-intro-title">Now let's begin!</h3>
        <div class="sb-intro-copy">
          <p>You will start the task.</p>
          <p>Remember to choose between Option 1 and Option 2 on each trial.</p>
        </div>
      </div>`
    ],
    show_clickable_nav: true
  };

  const trialBlock = {
    timeline: [
      {
        type: jsPsychHtmlButtonResponse,
        stimulus: jsPsych.timelineVariable("stimulus_html"),
        choices: ["Option 1", "Option 2"],
        data: {
          prolific_pid: jsPsych.timelineVariable("prolific_pid"),
          study_id: jsPsych.timelineVariable("study_id"),
          session_id: jsPsych.timelineVariable("session_id"),
          completion_code: jsPsych.timelineVariable("completion_code"),
          condition: jsPsych.timelineVariable("condition"),
          stim_set: jsPsych.timelineVariable("stim_set"),
          stim_pkg: jsPsych.timelineVariable("stim_pkg"),
          stim_id: jsPsych.timelineVariable("stim_id"),
          word: jsPsych.timelineVariable("word"),
          word_type: jsPsych.timelineVariable("word_type"),
          word_length: jsPsych.timelineVariable("word_length"),
          ordering: jsPsych.timelineVariable("ordering"),
          a_is: jsPsych.timelineVariable("a_is"),
          b_is: jsPsych.timelineVariable("b_is"),
          reference_url: jsPsych.timelineVariable("reference_url"),
          image_a_url: jsPsych.timelineVariable("image_a_url"),
          image_b_url: jsPsych.timelineVariable("image_b_url"),
          shape_match_url: jsPsych.timelineVariable("shape_match_url"),
          texture_match_url: jsPsych.timelineVariable("texture_match_url"),
          prompt_text: jsPsych.timelineVariable("prompt_text")
        },
        on_finish: async (data) => {
          const parsed = data.response === 0 ? "1" : data.response === 1 ? "2" : null;
          const choice = parsed === "1" ? data.a_is : parsed === "2" ? data.b_is : "unclear";
          const trialNumber = Number(data.trial_index) + 1;
          if (verboseTrials || trialNumber <= 3 || trialNumber % 10 === 0) {
            ck("Trial response captured", {
              trialNumber,
              stim_id: data.stim_id,
              word: data.word,
              ordering: data.ordering,
              response_key: parsed,
              choice,
              rt_ms: data.rt
            });
          }
          const payload = {
            prolific_pid: data.prolific_pid,
            study_id: data.study_id,
            session_id: data.session_id,
            completion_code: data.completion_code,
            condition: data.condition,
            stim_set: data.stim_set,
            stim_pkg: data.stim_pkg,
            trial_index: data.trial_index,
            stim_id: data.stim_id,
            word: data.word,
            word_type: data.word_type,
            word_length: data.word_length,
            ordering: data.ordering,
            a_is: data.a_is,
            b_is: data.b_is,
            response_key: parsed,
            choice,
            rt_ms: data.rt,
            reference_url: data.reference_url,
            image_a_url: data.image_a_url,
            image_b_url: data.image_b_url,
            shape_match_url: data.shape_match_url,
            texture_match_url: data.texture_match_url,
            browser_user_agent: navigator.userAgent,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            raw_trial: data
          };
          try {
            await logTrial(payload);
            if (verboseTrials || trialNumber <= 3 || trialNumber % 10 === 0) {
              ck("Trial logged to /api/log", {
                trialNumber,
                stim_id: data.stim_id
              });
            }
          } catch (err) {
            cerr("Failed to log trial", {
              trialNumber,
              stim_id: data.stim_id,
              error: String(err && err.message ? err.message : err)
            });
          }
        }
      }
    ],
    timeline_variables: finalTrials
  };

  const end = {
    type: jsPsychInstructions,
    pages: [
      `<div class="sb-container"><h2>Thank you!</h2><p>Your responses have been recorded.</p><p>You will now be redirected to Prolific.</p></div>`
    ],
    show_clickable_nav: true
  };

  ck("Starting jsPsych timeline", {
    totalNodes: preload.images.length > 0 ? 4 : 3,
    totalTrials: finalTrials.length
  });
  const timeline = preload.images.length > 0 ? [preload, intro, trialBlock, end] : [intro, trialBlock, end];
  setBootStatus("Starting task...");
  jsPsychMounted = true;
  jsPsych.run(timeline);
  ck("jsPsych run() invoked");
})();
