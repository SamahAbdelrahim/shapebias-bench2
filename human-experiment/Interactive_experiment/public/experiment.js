import {
  createModelViewer,
  disposeAllModelViewers,
  resizeAllModelViewers
} from "./model-viewer.js";

(async function runExperiment() {
  const DBG_PREFIX = "[SB-COMPLEXITY]";
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
  const cerr = (label, data) => {
    if (typeof data === "undefined") {
      console.error(`${DBG_PREFIX} ${label}`);
    } else {
      console.error(`${DBG_PREFIX} ${label}`, data);
    }
  };

  ck("Experiment boot start", { href: window.location.href, ts: new Date().toISOString() });

  if (
    typeof initJsPsych === "undefined" ||
    typeof jsPsychInstructions === "undefined" ||
    typeof jsPsychHtmlButtonResponse === "undefined"
  ) {
    cerr("Missing jsPsych globals", {
      initJsPsych: typeof initJsPsych,
      jsPsychInstructions: typeof jsPsychInstructions,
      jsPsychHtmlButtonResponse: typeof jsPsychHtmlButtonResponse
    });
    throw new Error(
      "jsPsych core/plugins failed to load. Check CDN access and script URLs in public/index.html."
    );
  }

  const url = new URL(window.location.href);
  const params = url.searchParams;
  const prolificPidParam = params.get("PROLIFIC_PID");
  const studyIdParam = params.get("STUDY_ID");
  const sessionIdParam = params.get("SESSION_ID");
  const isProlificSession = Boolean(prolificPidParam && studyIdParam && sessionIdParam);
  const prolific_pid = prolificPidParam || "debug_pid";
  const study_id = studyIdParam || "debug_study";
  const session_id = sessionIdParam || "debug_session";
  const verboseTrials = params.get("verbose_trials") === "1";

  ck("Fetching comparison trials");
  const trialsRes = await fetch(
    `/api/comparison-trials?prolific_pid=${encodeURIComponent(prolific_pid)}&study_id=${encodeURIComponent(study_id)}&session_id=${encodeURIComponent(session_id)}`
  );
  if (!trialsRes.ok) {
    const errBody = await trialsRes.text();
    document.body.innerHTML = `<p>Failed to load comparison trials: ${errBody}</p>`;
    return;
  }
  const payload = await trialsRes.json();
  const finalTrials = payload.trials || [];
  const surveyConfig = payload.config || {};
  const completionCode = isProlificSession ? "CTJN09E1" : "TESTCODE";

  if (finalTrials.length === 0) {
    document.body.innerHTML = "<p>No comparison trials are available. Add STL or GLB files and update configs/comparison_survey.json.</p>";
    return;
  }

  ck("Comparison trials loaded", {
    count: finalTrials.length,
    pair_mode: surveyConfig.pair_mode,
    model_count: payload.model_count
  });

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatPromptHtml(text) {
    return escapeHtml(text).replace(/\bcomplex\b/gi, "<b>complex</b>");
  }

  function renderTrialHtml(trial) {
    const prompt = formatPromptHtml(trial.prompt);
    const modelAUrl = escapeHtml(trial.model_a.url);
    const modelBUrl = escapeHtml(trial.model_b.url);
    const modelASource = escapeHtml(trial.model_a.source);
    const modelBSource = escapeHtml(trial.model_b.source);
    const modelAName = escapeHtml(trial.model_a.filename);
    const modelBName = escapeHtml(trial.model_b.filename);
    return `
      <div class="sb-container sb-trial-view">
        <div class="sb-prompt sb-prompt-question">${prompt}</div>
        <div class="sb-grid">
          <div class="sb-card sb-option-card">
            <div class="sb-label">Object A</div>
            <div class="sb-model-stage">
              <span class="sb-drag-hint" aria-hidden="true">Drag Me</span>
              <canvas
                class="sb-model-canvas"
                data-model-slot="a"
                data-model-url="${modelAUrl}"
                data-model-source="${modelASource}"
                aria-label="${modelAName}"
              ></canvas>
            </div>
          </div>
          <div class="sb-card sb-option-card">
            <div class="sb-label">Object B</div>
            <div class="sb-model-stage">
              <span class="sb-drag-hint" aria-hidden="true">Drag Me</span>
              <canvas
                class="sb-model-canvas"
                data-model-slot="b"
                data-model-url="${modelBUrl}"
                data-model-source="${modelBSource}"
                aria-label="${modelBName}"
              ></canvas>
            </div>
          </div>
        </div>
        <p id="sb-choice-prompt" class="sb-choice-prompt">Drag both objects to explore them before making your choice.</p>
      </div>
    `;
  }

  for (const trial of finalTrials) {
    trial.stimulus_html = renderTrialHtml(trial);
    trial.prolific_pid = prolific_pid;
    trial.study_id = study_id;
    trial.session_id = session_id;
    trial.completion_code = completionCode;
  }

  function shouldShowFullscreenWarning() {
    // Mobile/tablet: survey is designed to work here; no fullscreen nag
    if (window.matchMedia("(max-width: 900px)").matches) {
      return false;
    }
    if (window.matchMedia("(pointer: coarse) and (max-width: 1024px)").matches) {
      return false;
    }

    // Desktop: window not maximized or too small for the task
    const minComfortableWidth = 900;
    const minComfortableHeight = 650;
    const windowTooSmall =
      window.innerWidth < minComfortableWidth || window.innerHeight < minComfortableHeight;
    const notMaximized =
      window.outerWidth < screen.availWidth - 80 ||
      window.outerHeight < screen.availHeight - 80;

    return windowTooSmall || notMaximized;
  }

  function setupFullscreenWarning() {
    const banner = document.createElement("div");
    banner.id = "sb-fullscreen-warning";
    banner.className = "sb-fullscreen-warning sb-fullscreen-warning--hidden";
    banner.textContent = "⚠️ You need to view in full screen!";
    document.body.prepend(banner);

    const update = () => {
      banner.classList.toggle("sb-fullscreen-warning--hidden", !shouldShowFullscreenWarning());
    };

    update();
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("resize", update);
      banner.remove();
    };
  }

  let removeFullscreenWarning = setupFullscreenWarning();

  async function prepareTrialChoice() {
    const promptEl = document.getElementById("sb-choice-prompt");
    const buttons = document.querySelectorAll("#jspsych-html-button-response-btngroup .jspsych-btn");
    const explored = new Set();
    let unlocked = false;
    let firstInteractionAt = null;

    const hideHintForCanvas = (canvas) => {
      const hint = canvas.closest(".sb-model-stage")?.querySelector(".sb-drag-hint");
      if (hint) hint.classList.add("sb-drag-hint--hidden");
    };

    const unlockChoices = () => {
      if (unlocked) return;
      unlocked = true;
      document.querySelectorAll(".sb-drag-hint").forEach((hint) => {
        hint.classList.add("sb-drag-hint--hidden");
      });
      if (promptEl) {
        promptEl.textContent = "Please make your choice.";
      }
      buttons.forEach((button) => {
        button.disabled = false;
      });
    };

    const onObjectExplored = (canvas) => {
      const slot = canvas.dataset.modelSlot;
      if (!slot || explored.has(slot)) return;
      explored.add(slot);
      hideHintForCanvas(canvas);
      if (!firstInteractionAt) {
        firstInteractionAt = Date.now();
      }
      if (explored.size >= 2) {
        unlockChoices();
      }
    };

    buttons.forEach((button) => {
      button.disabled = true;
    });
    if (promptEl) {
      promptEl.textContent = "Drag both objects to explore them before making your choice.";
    }

    await mountTrialViewers(onObjectExplored);

    const interactionWatch = window.setInterval(() => {
      if (unlocked) {
        window.clearInterval(interactionWatch);
        return;
      }
      if (explored.size >= 2) {
        unlockChoices();
        window.clearInterval(interactionWatch);
        return;
      }
      if (firstInteractionAt && Date.now() - firstInteractionAt >= 4000) {
        unlockChoices();
        window.clearInterval(interactionWatch);
      }
    }, 250);

    window.setTimeout(() => {
      unlockChoices();
      window.clearInterval(interactionWatch);
    }, 12000);
  }

  async function mountTrialViewers(onInteract = null) {
    disposeAllModelViewers();
    const canvases = document.querySelectorAll(".sb-model-canvas");
    const mounts = [...canvases].map((canvas) =>
      createModelViewer(canvas, {
        url: canvas.dataset.modelUrl,
        source: canvas.dataset.modelSource,
        label: canvas.getAttribute("aria-label") || "",
        onInteract: onInteract ? () => onInteract(canvas) : null
      }).catch((err) => {
        cerr("Failed to mount model viewer", err);
      })
    );
    await Promise.all(mounts);
    resizeAllModelViewers();
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

  const jsPsych = initJsPsych({
    on_finish: () => {
      disposeAllModelViewers();
      window.location.href = `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(completionCode)}`;
    }
  });

  jsPsych.data.addProperties({
    prolific_pid,
    study_id,
    session_id,
    experiment_type: "complexity_comparison",
    pair_mode: surveyConfig.pair_mode
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
        <h2 class="sb-intro-title">Task Instructions</h2>
        <div class="sb-intro-copy">
          <p>In this experiment, you'll see pairs of 3D abstract objects. Click and drag to rotate or move each object with your cursor.</p>
          <p>For each pair, you will be asked to decide which object appears more <b>complex</b>.</p>
          <p>Use your intuition to make these judgments — there are no right or wrong answers.</p>
          <p>Please click next to begin.</p>
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
        choices: ["Object A is more complex", "Object B is more complex"],
        button_layout: "flex",
        on_load: async () => {
          if (removeFullscreenWarning) {
            removeFullscreenWarning();
            removeFullscreenWarning = null;
          }
          await prepareTrialChoice();
        },
        on_finish: async (data) => {
          disposeAllModelViewers();
          const responseKey = data.response === 0 ? "1" : data.response === 1 ? "2" : null;
          const chosenModel = responseKey === "1" ? data.model_a : responseKey === "2" ? data.model_b : null;
          const trialNumber = Number(data.trial_index) + 1;
          if (verboseTrials || trialNumber <= 3 || trialNumber % 10 === 0) {
            ck("Trial response captured", {
              trialNumber,
              response_key: responseKey,
              chosen: chosenModel?.filename || null,
              rt_ms: data.rt
            });
          }
          const payloadData = {
            prolific_pid: data.prolific_pid,
            study_id: data.study_id,
            session_id: data.session_id,
            completion_code: data.completion_code,
            experiment_type: "complexity_comparison",
            pair_mode: data.pair_mode,
            trial_index: data.trial_index,
            prompt: data.prompt,
            model_a_filename: data.model_a?.filename,
            model_a_source: data.model_a?.source,
            model_a_url: data.model_a?.url,
            model_b_filename: data.model_b?.filename,
            model_b_source: data.model_b?.source,
            model_b_url: data.model_b?.url,
            pair_left_filename: data.pair_left_filename,
            pair_right_filename: data.pair_right_filename,
            pair_left_source: data.pair_left_source,
            pair_right_source: data.pair_right_source,
            side_swapped: data.side_swapped,
            response_key: responseKey,
            chosen_filename: chosenModel?.filename || null,
            chosen_source: chosenModel?.source || null,
            rt_ms: data.rt,
            browser_user_agent: navigator.userAgent,
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            raw_trial: data
          };
          try {
            await logTrial(payloadData);
          } catch (err) {
            cerr("Failed to log trial", {
              trialNumber,
              error: String(err && err.message ? err.message : err)
            });
          }
        },
        data: {
          prolific_pid: jsPsych.timelineVariable("prolific_pid"),
          study_id: jsPsych.timelineVariable("study_id"),
          session_id: jsPsych.timelineVariable("session_id"),
          completion_code: jsPsych.timelineVariable("completion_code"),
          trial_index: jsPsych.timelineVariable("trial_index"),
          pair_mode: jsPsych.timelineVariable("pair_mode"),
          prompt: jsPsych.timelineVariable("prompt"),
          model_a: jsPsych.timelineVariable("model_a"),
          model_b: jsPsych.timelineVariable("model_b"),
          pair_left_filename: jsPsych.timelineVariable("pair_left_filename"),
          pair_right_filename: jsPsych.timelineVariable("pair_right_filename"),
          pair_left_source: jsPsych.timelineVariable("pair_left_source"),
          pair_right_source: jsPsych.timelineVariable("pair_right_source"),
          side_swapped: jsPsych.timelineVariable("side_swapped")
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

  window.addEventListener("resize", resizeAllModelViewers);
  jsPsychMounted = true;
  jsPsych.run([intro, trialBlock, end]);
})();
