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

  const url = new URL(window.location.href);
  const params = url.searchParams;
  const prolific_pid = params.get("PROLIFIC_PID") || "debug_pid";
  const study_id = params.get("STUDY_ID") || "debug_study";
  const session_id = params.get("SESSION_ID") || "debug_session";
  const design = params.get("design") || "human_friendly"; // human_friendly | benchmark
  const condition = params.get("condition") || "noun_label"; // noun_label | no_word_category
  const stimSet = params.get("stim_set") || "stimuli_A_auto_contrast";
  const orderMode = params.get("ordering") || (design === "human_friendly" ? "random" : "both");
  const defaultLimit = design === "human_friendly" ? 30 : 0;
  const trialLimit = Number(params.get("trial_limit") || defaultLimit);
  const shuffleTrials = params.get("shuffle") !== "0";
  const verboseTrials = params.get("verbose_trials") === "1";
  const preloadMode = params.get("preload") || "minimal"; // off | minimal | all
  ck("Parsed URL parameters", {
    prolific_pid,
    study_id,
    session_id,
    design,
    condition,
    stimSet,
    orderMode,
    trialLimit,
    shuffleTrials,
    verboseTrials,
    preloadMode
  });

  setBootStatus("Loading config...");
  ck("Fetching /api/config");
  const configRes = await fetch("/api/config");
  ck("Config response received", { ok: configRes.ok, status: configRes.status });
  const config = await configRes.json();
  const completionCode = params.get("cc") || config.completion_code || "TESTCODE";
  ck("Resolved config", { completionCode, default_stim_set: config.default_stim_set });

  setBootStatus("Loading stimuli list...");
  ck("Fetching /api/stimuli", { stim_set: stimSet });
  const stimRes = await fetch(`/api/stimuli?stim_set=${encodeURIComponent(stimSet)}`);
  ck("Stimuli response received", { ok: stimRes.ok, status: stimRes.status });
  const stimData = await stimRes.json();
  ck("Stimuli payload parsed", {
    hasStimuliArray: Array.isArray(stimData.stimuli),
    count: Array.isArray(stimData.stimuli) ? stimData.stimuli.length : null
  });
  if (!stimData.stimuli || !Array.isArray(stimData.stimuli) || stimData.stimuli.length === 0) {
    cerr("No stimuli available from /api/stimuli", stimData);
    document.body.innerHTML = "<p>Failed to load stimuli. Please contact the researcher.</p>";
    return;
  }

  function makePrompt(word) {
    if (condition === "no_word_category") {
      return "See this object in the first image. Can you find another one of the two images (1 or 2)?";
    }
    return `The first image is a ${word}. Which of the following two images (1 or 2) is also a ${word}?`;
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

  function buildUniqueHumanWords(count, seedText) {
    const rand = mulberry32(hashString(seedText));
    const seen = new Set();
    const out = [];
    const lengths = [6, 7, 8, 9, 10];
    while (out.length < count) {
      const idx = out.length;
      const wordType = idx % 2 === 0 ? "sudo" : "random";
      const length = lengths[idx % lengths.length];
      const candidate =
        wordType === "sudo"
          ? makePseudoWord(rand, length)
          : makeRandomWord(rand, length);
      if (seen.has(candidate)) continue;
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
      <div class="sb-container">
        <div class="sb-grid">
          <div class="sb-card">
            <div class="sb-label">Reference</div>
            <img class="sb-image" src="${t.reference_url}" alt="reference">
          </div>
          <div class="sb-card">
            <div class="sb-label">Image 1</div>
            <img class="sb-image" src="${t.image_a_url}" alt="image 1">
          </div>
          <div class="sb-card">
            <div class="sb-label">Image 2</div>
            <img class="sb-image" src="${t.image_b_url}" alt="image 2">
          </div>
        </div>
        <div class="sb-question">${t.prompt_text}</div>
      </div>
    `;
  }

  const trialVars = [];
  if (design === "human_friendly") {
    // Human-friendly default: one trial per object with unique labels to avoid memory carry-over.
    const seedText = `${prolific_pid}|${study_id}|${session_id}|${stimSet}|${condition}`;
    const selectedStimuli = maybeSampleStimuli(stimData.stimuli, trialLimit, `${seedText}|stimuli`);
    const uniqueWords = buildUniqueHumanWords(selectedStimuli.length, `${seedText}|words`);
    ck("Human-friendly trial builder", {
      seedPreview: `${seedText.slice(0, 20)}...`,
      selectedStimuli: selectedStimuli.length,
      uniqueWords: uniqueWords.length
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
      trialVars.push({
        prolific_pid,
        study_id,
        session_id,
        completion_code: completionCode,
        condition,
        stim_set: stimSet,
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
        prompt_text: makePrompt(w.name)
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
          trialVars.push({
            prolific_pid,
            study_id,
            session_id,
            completion_code: completionCode,
            condition,
            stim_set: stimSet,
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
            prompt_text: makePrompt(w.name)
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
    stim_set: stimSet,
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
        <h2>Welcome</h2>
        <p>You will see 3 images per trial: one reference image and two candidate images.</p>
        <p>Your task is to decide whether image <b>1</b> or image <b>2</b> matches the reference based on the question.</p>
        <p>Please respond carefully. There are <b>${finalTrials.length}</b> trials.</p>
      </div>`,
      `<div class="sb-container">
        <h3>Response format</h3>
        <p>Select button <b>1</b> or <b>2</b>.</p>
        <p class="sb-note">Design: <code>${design}</code> | Condition: <code>${condition}</code> | Stimulus set: <code>${stimSet}</code></p>
      </div>`
    ],
    show_clickable_nav: true
  };

  const trialBlock = {
    timeline: [
      {
        type: jsPsychHtmlButtonResponse,
        stimulus: jsPsych.timelineVariable("stimulus_html"),
        choices: ["1", "2"],
        data: {
          prolific_pid: jsPsych.timelineVariable("prolific_pid"),
          study_id: jsPsych.timelineVariable("study_id"),
          session_id: jsPsych.timelineVariable("session_id"),
          completion_code: jsPsych.timelineVariable("completion_code"),
          condition: jsPsych.timelineVariable("condition"),
          stim_set: jsPsych.timelineVariable("stim_set"),
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
