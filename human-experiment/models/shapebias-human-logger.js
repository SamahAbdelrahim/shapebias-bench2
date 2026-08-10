const mongoose = require("mongoose");

const ShapeBiasHumanSchema = new mongoose.Schema(
  {
    created_at: { type: Date, default: Date.now },
    prolific_pid: String,
    study_id: String,
    session_id: String,
    completion_code: String,

    condition: String,
    design: String,
    // Which stimulus set the trial came from ("grid" or "cc_triads"), and which
    // build of the offline trial pool produced it. Both are needed to keep the
    // matched_v2 sample separable from the March 2026 human_friendly pilot,
    // which used the flat 30-shape packages below.
    stim_set_name: String,
    pool_version: String,
    pool_index: Number,
    stim_set: String,
    stim_pkg: String,
    trial_index: Number,
    block_index: Number,

    stim_id: String,
    // Shape and texture identity of the triad, so item-level analysis does not
    // have to re-derive them from stim_id.
    stl_id: String,
    texture_set: String,
    word: String,
    word_type: String,
    word_length: Number,
    ordering: String,
    ordering_group: String,
    a_is: String,
    b_is: String,

    is_catch: Boolean,
    catch_correct: Boolean,

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
