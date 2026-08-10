library(readr)
library(dplyr)
library(tibble)

get_human_input_dir <- function() {
  file.path(get_results_dir(), "human.results")
}

get_human_results_path <- function(filename = "human_results.csv") {
  file.path(get_data_dir(), filename)
}

get_human_input_paths <- function() {
  c(
    get_human_results_path("human_results.csv"),
    file.path(get_human_input_dir(), "human_results.csv"),
    file.path(get_human_input_dir(), "human_trials1.csv")
  )
}

load_human_results <- function(path = NULL) {
  candidate_paths <- if (is.null(path)) get_human_input_paths() else c(path)
  existing_paths <- unique(candidate_paths[file.exists(candidate_paths)])
  if (!length(existing_paths)) {
    return(tibble())
  }

  df <- read_csv(existing_paths[[1]], show_col_types = FALSE)

  if (!"design" %in% names(df) && "raw_trial.design" %in% names(df)) {
    ordering_mode_col <- if ("raw_trial.ordering_mode" %in% names(df)) "raw_trial.ordering_mode" else NULL
    df <- df |>
      mutate(
        design = `raw_trial.design`,
        ordering_mode = if (!is.null(ordering_mode_col)) .data[[ordering_mode_col]] else NA_character_
      )
  }

  # The March 2026 human_friendly pilot predates the matched_v2 columns, so
  # every field added for the two-stimulus-set design is filled in here rather
  # than guarded at each call site.
  ensure_col <- function(d, name, value) {
    if (!name %in% names(d)) d[[name]] <- value
    d
  }
  df <- df |>
    ensure_col("stim_set_name", NA_character_) |>
    ensure_col("ordering_group", NA_character_) |>
    ensure_col("pool_version", NA_character_) |>
    ensure_col("stl_id", NA_character_) |>
    ensure_col("texture_set", NA_character_) |>
    ensure_col("is_catch", NA) |>
    ensure_col("catch_correct", NA) |>
    ensure_col("response_key", NA_character_)

  as_logical_flag <- function(x) {
    if (is.logical(x)) return(x)
    tolower(trimws(as.character(x))) %in% c("true", "1", "t", "yes")
  }

  df |>
    mutate(
      design = ifelse(is.na(design), "", design),
      condition = ifelse(is.na(condition), "noun_label", condition),
      ordering_mode = ifelse(is.na(ordering_mode), "", ordering_mode),
      is_catch = as_logical_flag(is_catch),
      catch_correct = as_logical_flag(catch_correct),
      participant_id = paste(prolific_pid, study_id, session_id, sep = "|"),
      choice = factor(choice, levels = c("shape", "texture", "match", "foil", "unclear"))
    )
}

filter_human_friendly <- function(human_df) {
  if (!nrow(human_df)) return(human_df)
  human_df |>
    filter(design == "human_friendly")
}

# ---------------------------------------------------------------------------
# matched_v2: two stimulus sets, two framings, catch trials
#
# The March 2026 pilot (design == "human_friendly") stays reachable through the
# functions above. Everything below reads the matched_v2 sample, which differs
# in ways that make the two non-poolable: different stimulus sets, a no-word
# condition that did not exist before, ordering counterbalanced between
# participants, and attention checks.
# ---------------------------------------------------------------------------

filter_matched_v2 <- function(human_df) {
  if (!nrow(human_df)) return(human_df)
  human_df |>
    filter(design == "matched_v2")
}

#' Wilson score interval, which behaves at the ceiling rates this task produces
#' where a normal approximation would run past 1.
wilson_ci <- function(successes, n, z = 1.96) {
  if (is.na(n) || n == 0) return(c(NA_real_, NA_real_))
  p <- successes / n
  denom <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / denom
  halfwidth <- (z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2))) / denom
  c(max(0, centre - halfwidth), min(1, centre + halfwidth))
}

#' Per-participant attention-check performance.
#'
#' Columns: participant_id, prolific_pid, study_id, session_id, condition,
#' ordering_group, catch_trials, catch_passed, catch_errors, excluded.
summarize_catch_by_participant <- function(human_df, max_errors = 1) {
  if (!nrow(human_df)) return(tibble())
  human_df |>
    filter(is_catch) |>
    group_by(participant_id, prolific_pid, study_id, session_id, condition, ordering_group) |>
    summarise(
      catch_trials = n(),
      catch_passed = sum(catch_correct, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      catch_errors = catch_trials - catch_passed,
      excluded = catch_errors > max_errors
    ) |>
    arrange(desc(catch_errors), participant_id)
}

#' Drop participants who failed more than `max_errors` checks, then drop the
#' check trials themselves so only the shape-versus-texture trials remain.
apply_catch_exclusions <- function(human_df, max_errors = 1) {
  if (!nrow(human_df)) return(human_df)
  catch_summary <- summarize_catch_by_participant(human_df, max_errors = max_errors)
  excluded_ids <- catch_summary$participant_id[catch_summary$excluded]
  human_df |>
    filter(!participant_id %in% excluded_ids) |>
    filter(!is_catch)
}

#' Shape-choice rates by stimulus set and condition.
#'
#' Written to results/data/human_summary_by_set.csv. Schema, which
#' analysis_pipe/full_grid_figures.py reads to place the human anchors:
#'   stim_set_name, condition, participants, trials, unique_stimuli,
#'   shape_prop, shape_lo, shape_hi, texture_prop, unclear_rate,
#'   median_rt_ms, mean_rt_ms
summarize_human_by_set_condition <- function(human_df) {
  if (!nrow(human_df)) return(tibble())
  human_df |>
    group_by(stim_set_name, condition) |>
    summarise(
      participants = n_distinct(participant_id),
      trials = n(),
      unique_stimuli = n_distinct(stim_id),
      n_shape = sum(choice == "shape", na.rm = TRUE),
      n_decided = sum(choice %in% c("shape", "texture"), na.rm = TRUE),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      texture_prop = mean(choice == "texture", na.rm = TRUE),
      unclear_rate = mean(choice == "unclear", na.rm = TRUE),
      median_rt_ms = median(rt_ms, na.rm = TRUE),
      mean_rt_ms = mean(rt_ms, na.rm = TRUE),
      .groups = "drop"
    ) |>
    rowwise() |>
    mutate(
      shape_lo = wilson_ci(n_shape, n_decided)[[1]],
      shape_hi = wilson_ci(n_shape, n_decided)[[2]]
    ) |>
    ungroup() |>
    select(
      stim_set_name, condition, participants, trials, unique_stimuli,
      shape_prop, shape_lo, shape_hi, texture_prop, unclear_rate,
      median_rt_ms, mean_rt_ms
    ) |>
    arrange(stim_set_name, condition)
}

#' Item-level means, including the human counterpart of the model tracking gate.
#'
#' Models see each triad in both orderings, and tracking asks whether the choice
#' follows content across the swap. No participant sees a triad twice here, so
#' the same question is asked between participants: an item whose shape rate is
#' the same under both orderings is being answered on content, while one whose
#' two rates are roughly complementary is being answered on position.
#'
#' Written to results/data/human_item_means.csv. Schema:
#'   stim_set_name, condition, stim_id, stl_id, texture_set, n, shape_prop,
#'   n_shape_first, shape_prop_shape_first, n_texture_first,
#'   shape_prop_texture_first, option1_rate, content_consistency
summarize_human_items <- function(human_df) {
  if (!nrow(human_df)) return(tibble())
  human_df |>
    group_by(stim_set_name, condition, stim_id, stl_id, texture_set) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      n_shape_first = sum(ordering == "shape_first", na.rm = TRUE),
      shape_prop_shape_first = mean(choice[ordering == "shape_first"] == "shape", na.rm = TRUE),
      n_texture_first = sum(ordering == "texture_first", na.rm = TRUE),
      shape_prop_texture_first = mean(choice[ordering == "texture_first"] == "shape", na.rm = TRUE),
      option1_rate = mean(response_key == "1", na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(
      content_consistency = 1 - abs(shape_prop_shape_first - shape_prop_texture_first)
    ) |>
    arrange(stim_set_name, condition, stim_id)
}

#' Roll the item-level position check up to one row per set and condition, so it
#' can be read against the models' 0.70 tracking gate.
summarize_position_check <- function(item_means, gate = 0.70) {
  if (!nrow(item_means)) return(tibble())
  item_means |>
    group_by(stim_set_name, condition) |>
    summarise(
      items = n(),
      items_with_both_orderings = sum(n_shape_first > 0 & n_texture_first > 0, na.rm = TRUE),
      mean_content_consistency = mean(content_consistency, na.rm = TRUE),
      items_above_gate = sum(content_consistency >= gate, na.rm = TRUE),
      mean_option1_rate = mean(option1_rate, na.rm = TRUE),
      .groups = "drop"
    ) |>
    mutate(prop_items_above_gate = items_above_gate / items) |>
    arrange(stim_set_name, condition)
}

#' Item-level agreement between humans and one model cell, per stimulus set.
#'
#' The old comparison correlated 30 stimuli measured under different protocols
#' and returned r = 0.08. This one joins on stim_id within a stimulus set, where
#' humans and models saw the same triads.
compute_human_model_item_correlation <- function(item_means, model_items,
                                                 model_label = "model") {
  if (!nrow(item_means) || !nrow(model_items)) return(tibble())
  item_means |>
    inner_join(model_items, by = c("stim_set_name", "stim_id")) |>
    group_by(stim_set_name, condition) |>
    summarise(
      model = model_label,
      shared_items = n(),
      human_shape_prop = mean(shape_prop, na.rm = TRUE),
      model_shape_prop = mean(model_shape_prop, na.rm = TRUE),
      mean_abs_delta = mean(abs(shape_prop - model_shape_prop), na.rm = TRUE),
      item_correlation = if (n() >= 3) {
        suppressWarnings(cor(shape_prop, model_shape_prop, use = "complete.obs"))
      } else {
        NA_real_
      },
      .groups = "drop"
    )
}

summarize_human_friendly_overall <- function(human_df) {
  if (!nrow(human_df)) {
    return(tibble())
  }

  participant_ids <- unique(paste(human_df$prolific_pid, human_df$study_id, human_df$session_id, sep = "|"))

  human_df |>
    summarise(
      participants = length(participant_ids),
      trials = n(),
      unique_stimuli = n_distinct(stim_id),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      texture_prop = mean(choice == "texture", na.rm = TRUE),
      unclear_rate = mean(choice == "unclear", na.rm = TRUE),
      median_rt_ms = median(rt_ms, na.rm = TRUE),
      mean_rt_ms = mean(rt_ms, na.rm = TRUE)
    )
}

summarize_human_friendly_by_participant <- function(human_df) {
  if (!nrow(human_df)) {
    return(tibble())
  }

  human_df |>
    mutate(participant_id = paste(prolific_pid, study_id, session_id, sep = "|")) |>
    group_by(participant_id, prolific_pid, study_id, session_id, condition, stim_pkg) |>
    summarise(
      trials = n(),
      unique_stimuli = n_distinct(stim_id),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      texture_prop = mean(choice == "texture", na.rm = TRUE),
      unclear_rate = mean(choice == "unclear", na.rm = TRUE),
      median_rt_ms = median(rt_ms, na.rm = TRUE),
      mean_rt_ms = mean(rt_ms, na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(participant_id)
}

summarize_human_friendly_by_stimulus <- function(human_df) {
  if (!nrow(human_df)) {
    return(tibble())
  }

  human_df |>
    group_by(stim_id) |>
    summarise(
      trials = n(),
      participants = n_distinct(paste(prolific_pid, study_id, session_id, sep = "|")),
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      texture_prop = mean(choice == "texture", na.rm = TRUE),
      unclear_rate = mean(choice == "unclear", na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(desc(trials), stim_id)
}

compute_human_model_approx_comparison <- function(human_df, model_df, model_subset_label = "all_models") {
  if (!nrow(human_df) || !nrow(model_df)) {
    return(tibble())
  }

  human_overall <- summarize_human_friendly_overall(human_df)

  model_overall <- model_df |>
    summarise(
      model_trials = n(),
      model_trial_weighted_shape_prop = mean(choice == "shape", na.rm = TRUE),
      model_trial_weighted_texture_prop = mean(choice == "texture", na.rm = TRUE),
      model_unclear_rate = mean(choice == "unclear", na.rm = TRUE)
    )

  model_by_model <- model_df |>
    group_by(model) |>
    summarise(
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      .groups = "drop"
    )

  human_by_stim <- summarize_human_friendly_by_stimulus(human_df)
  model_by_stim <- model_df |>
    group_by(stim_id) |>
    summarise(
      model_shape_prop = mean(choice == "shape", na.rm = TRUE),
      model_trials = n(),
      .groups = "drop"
    )

  overlap <- human_by_stim |>
    inner_join(model_by_stim, by = "stim_id") |>
    mutate(abs_delta = abs(shape_prop - model_shape_prop))

  stim_correlation <- if (nrow(overlap) >= 2) {
    suppressWarnings(cor(overlap$shape_prop, overlap$model_shape_prop, use = "complete.obs"))
  } else {
    NA_real_
  }

  tibble(
    metric = c(
      "model_subset",
      "human_participants",
      "human_trials",
      "human_shape_prop",
      "human_texture_prop",
      "human_unclear_rate",
      "model_trials",
      "model_trial_weighted_shape_prop",
      "model_mean_of_model_shape_prop",
      "shared_stimulus_count",
      "shared_stimulus_mean_abs_delta",
      "shared_stimulus_shape_prop_correlation"
    ),
    value = c(
      model_subset_label,
      human_overall$participants[[1]],
      human_overall$trials[[1]],
      human_overall$shape_prop[[1]],
      human_overall$texture_prop[[1]],
      human_overall$unclear_rate[[1]],
      model_overall$model_trials[[1]],
      model_overall$model_trial_weighted_shape_prop[[1]],
      mean(model_by_model$shape_prop, na.rm = TRUE),
      nrow(overlap),
      if (nrow(overlap)) mean(overlap$abs_delta, na.rm = TRUE) else NA_real_,
      stim_correlation
    ),
    interpretation = c(
      "Which model subset was used for this comparison.",
      "Pilot sample size for the human-friendly protocol.",
      "Logged human-friendly trials included in the pilot summary.",
      "Overall human-friendly proportion of shape choices.",
      "Overall human-friendly proportion of texture choices.",
      "Overall human-friendly unclear-response rate.",
      "Total model trials in the canonical benchmark dataset.",
      "Benchmark model shape-choice proportion across all canonical trials.",
      "Average benchmark shape-choice proportion across models.",
      "Number of stimulus IDs observed in both human-friendly and model datasets.",
      "Average absolute difference between human and model stimulus-level shape-choice rates on overlapping stimuli.",
      "Correlation of stimulus-level shape-choice rates across overlapping human and model stimuli."
    )
  )
}
