const mongoose = require("mongoose");

const ShapeBiasHumanSchema = new mongoose.Schema(
  {
    created_at: { type: Date, default: Date.now },
    prolific_pid: String,
    study_id: String,
    session_id: String,
    completion_code: String,

    condition: String,
    stim_set: String,
    trial_index: Number,

    stim_id: String,
    word: String,
    word_type: String,
    word_length: Number,
    ordering: String,
    a_is: String,
    b_is: String,

    response_key: String,
    choice: String,
    rt_ms: Number,

    reference_url: String,
    image_a_url: String,
    image_b_url: String,
    shape_match_url: String,
    texture_match_url: String,

    browser_user_agent: String,
    timezone: String,

    raw_trial: mongoose.Schema.Types.Mixed
  },
  { collection: "shape_bias_human_trials" }
);

module.exports = mongoose.model("shapeBiasHumanTrial", ShapeBiasHumanSchema);
