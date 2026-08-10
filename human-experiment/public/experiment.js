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
  if (typeof SBAssignment === "undefined") {
    cerr("Missing SBAssignment global");
    throw new Error("assignment.js failed to load. Check the script tag in public/index.html.");
  }
  ck("jsPsych globals verified");

  const url = new URL(window.location.href);
  const params = url.searchParams;
  const prolificPidParam = params.get("PROLIFIC_PID");
  const studyIdParam = params.get("STUDY_ID");
  const sessionIdParam = params.get("SESSION_ID");
  const isProlificSession = Boolean(prolificPidParam && studyIdParam && sessionIdParam);
  const prolific_pid = prolificPidParam || "debug_pid";
  const study_id = studyIdParam || "debug_study";
  const session_id = sessionIdParam || "debug_session";
  const participantSeed = `${prolific_pid}|${study_id}|${session_id}`;

  const verboseTrials = params.get("verbose_trials") === "1";
  const preloadMode = params.get("preload") || "minimal"; // off | minimal | all

  // Between-participant factors come from the Prolific identifiers, so a
  // participant who reloads lands in the same cell with the same items. The
  // URL params below override only for piloting.
  const assignOptions = {
    condition: params.get("condition") || undefined,
    orderingGroup: params.get("ordering_group") || undefined,
    trialsPerSet: params.get("trials_per_set")
      ? Number(params.get("trials_per_set"))
      : undefined,
    catchPerSession: params.get("catch_n") ? Number(params.get("catch_n")) : undefined,
    wordMinLen: params.get("word_min_len") ? Number(params.get("word_min_len")) : undefined,
    wordMaxLen: params.get("word_max_len") ? Number(params.get("word_max_len")) : undefined,
    sudoThreshold: params.get("sudo_threshold")
      ? Number(params.get("sudo_threshold"))
      : undefined
  };

  setBootStatus("Loading config...");
  let config = {};
  try {
    const configRes = await fetch("/api/config");
    if (configRes.ok) config = await configRes.json();
  } catch (err) {
    cwarn("Config fetch failed, continuing with defaults", {
      error: String(err && err.message ? err.message : err)
    });
  }
  const completionCode = config.completion_code || "TESTCODE";
  if (isProlificSession && completionCode === "TESTCODE") {
    // Sending the placeholder to a real participant means they cannot claim
    // payment, so this must be visible rather than silently accepted.
    cerr("PROLIFIC_COMPLETION_CODE is not set on the server. Real participants " +
      "would be redirected with a placeholder completion code and could not be paid.");
  }
  ck("Resolved config", { completionCode, isProlificSession });

  setBootStatus("Loading trial pool...");
  // The pool is a static file built offline by scripts/build_human_trial_pool.py.
  // Which triads exist is fixed there, not resolved at run time, so the app
  // never has to walk the nested grid or the triad directories.
  let pool = null;
  for (const poolUrl of ["/human-experiment/trial_pool.json", "/api/pool"]) {
    try {
      const res = await fetch(poolUrl);
      if (!res.ok) continue;
      pool = await res.json();
      ck("Trial pool loaded", { poolUrl, version: pool.pool_version, counts: pool.counts });
      break;
    } catch (err) {
      cwarn("Trial pool fetch failed", {
        poolUrl,
        error: String(err && err.message ? err.message : err)
      });
    }
  }
  if (!pool || !Array.isArray(pool.test) || pool.test.length === 0) {
    cerr("No trial pool available");
    document.body.innerHTML = "<p>Failed to load stimuli. Please contact the researcher.</p>";
    return;
  }

  const session = SBAssignment.assignParticipant(pool, participantSeed, assignOptions);
  const { condition, orderingGroup, setOrder, trials: finalTrials } = session;
  const poolVersion = pool.pool_version || "unknown";

  ck("Participant assigned", {
    condition,
    orderingGroup,
    setOrder,
    test: session.testCount,
    catch: session.catchCount
  });

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Wording tracks the model templates in evaluation_pipe/eval_core.py, so the
  // human and model conditions differ in modality rather than in what is asked.
  // The model templates close with a response-format instruction ("exactly one
  // character: 1 or 2"), dropped here because participants click a button.
  function buildPromptParts(word) {
    if (condition === "no_word_category") {
      return {
        intro: "This first image is an object.",
        question: "Which of the following two images is another one?"
      };
    }
    const safeWord = escapeHtml(word);
    return {
      intro: `This first image is a "<span class="sb-highlight">${safeWord}</span>".`,
      question: `Which of the following two images is also a "<span class="sb-highlight">${safeWord}</span>"?`
    };
  }

  function plainPromptText(word) {
    if (condition === "no_word_category") {
      return "This first image is an object. Which of the following two images is another one?";
    }
    return `This first image is a ${word}. Which of the following two images is also a ${word}?`;
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

  for (const t of finalTrials) {
    const promptParts = buildPromptParts(t.word);
    t.prompt_intro = promptParts.intro;
    t.prompt_question = promptParts.question;
    t.prompt_text = plainPromptText(t.word);
    t.prolific_pid = prolific_pid;
    t.study_id = study_id;
    t.session_id = session_id;
    t.completion_code = completionCode;
    t.stimulus_html = renderStimulusHtml(t);
  }

  setBootStatus(`Preparing ${finalTrials.length} trials...`);
  ck("Final trial set ready", {
    total: finalTrials.length,
    shapeFirstShare:
      finalTrials.filter((t) => t.ordering === "shape_first").length /
      (finalTrials.filter((t) => !t.is_catch).length || 1)
  });
  if (finalTrials.length === 0) {
    cwarn("No trials generated");
    document.body.innerHTML = "<p>Failed to build trials. Please contact the researcher.</p>";
    return;
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
      ck("jsPsych on_finish called, redirecting to Prolific", { completionCode });
      window.location.href = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(completionCode)}`;
    }
  });

  jsPsych.data.addProperties({
    prolific_pid,
    study_id,
    session_id,
    design: "matched_v2",
    condition,
    ordering_group: orderingGroup,
    pool_version: poolVersion,
    set_order: setOrder.join(",")
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
  ck("Preload configured", { preloadMode, imageCount: preload.images.length });

  const isNoWord = condition === "no_word_category";
  const exampleIntroLine = isNoWord
    ? "An image of an object appears at the top."
    : "An image of an object appears at the top. The object is labeled with a word.";
  const examplePromptTop = isNoWord
    ? "This first image is an object."
    : 'This first image is a "<span class="sb-highlight">chair</span>".';
  const examplePromptQuestion = isNoWord
    ? "Which of the following two images is another one?"
    : 'Which of the following two images is also a "<span class="sb-highlight">chair</span>"?';

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
            <li>${exampleIntroLine}</li>
            <li>Two options will appear with it. Your task is to choose between these two options.</li>
          </ul>
          <div class="sb-example-wrap">
            <p class="sb-example-title">For example:</p>
            <div class="sb-card sb-reference-card sb-example-reference">
              <div class="sb-prompt sb-prompt-top">${examplePromptTop}</div>
              <img
                class="sb-image sb-reference-image sb-example-image"
                src="/general_assets/chair_target.jpg"
                alt="example target chair"
                onerror="this.onerror=null;this.src='/human-experiment/favicon.svg';"
              >
            </div>
            <div class="sb-prompt sb-prompt-question">${examplePromptQuestion}</div>
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
          <p>Some of the objects you will see are unusual, and some are photographs. There is no trick: choose whichever option seems right to you.</p>
          <p>Please look at all three images on every trial. A few trials are easy checks that we use to confirm you are paying attention.</p>
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
          stim_set_name: jsPsych.timelineVariable("stim_set_name"),
          stim_id: jsPsych.timelineVariable("stim_id"),
          stl_id: jsPsych.timelineVariable("stl_id"),
          texture_set: jsPsych.timelineVariable("texture_set"),
          pool_index: jsPsych.timelineVariable("pool_index"),
          pool_version: jsPsych.timelineVariable("pool_version"),
          block_index: jsPsych.timelineVariable("block_index"),
          is_catch: jsPsych.timelineVariable("is_catch"),
          word: jsPsych.timelineVariable("word"),
          word_type: jsPsych.timelineVariable("word_type"),
          word_length: jsPsych.timelineVariable("word_length"),
          ordering: jsPsych.timelineVariable("ordering"),
          ordering_group: jsPsych.timelineVariable("ordering_group"),
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
          // On test trials a_is/b_is are shape/texture; on catch trials they are
          // match/foil, so one mapping codes both correctly.
          const choice = parsed === "1" ? data.a_is : parsed === "2" ? data.b_is : "unclear";
          const catchCorrect = data.is_catch ? choice === "match" : null;
          const trialNumber = Number(data.trial_index) + 1;
          if (verboseTrials || trialNumber <= 3 || trialNumber % 10 === 0) {
            ck("Trial response captured", {
              trialNumber,
              stim_set_name: data.stim_set_name,
              stim_id: data.stim_id,
              is_catch: data.is_catch,
              ordering: data.ordering,
              response_key: parsed,
              choice,
              catch_correct: catchCorrect,
              rt_ms: data.rt
            });
          }
          const payload = {
            prolific_pid: data.prolific_pid,
            study_id: data.study_id,
            session_id: data.session_id,
            completion_code: data.completion_code,
            condition: data.condition,
            design: "matched_v2",
            pool_version: data.pool_version,
            stim_set_name: data.stim_set_name,
            trial_index: data.trial_index,
            block_index: data.block_index,
            stim_id: data.stim_id,
            stl_id: data.stl_id,
            texture_set: data.texture_set,
            pool_index: data.pool_index,
            is_catch: Boolean(data.is_catch),
            catch_correct: catchCorrect,
            word: data.word,
            word_type: data.word_type,
            word_length: data.word_length,
            ordering: data.ordering,
            ordering_group: data.ordering_group,
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
              ck("Trial logged to /api/log", { trialNumber, stim_id: data.stim_id });
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
