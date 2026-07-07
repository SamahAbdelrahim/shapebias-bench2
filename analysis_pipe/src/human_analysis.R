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

  df |>
    mutate(
      design = ifelse(is.na(design), "", design),
      condition = ifelse(is.na(condition), "noun_label", condition),
      ordering_mode = ifelse(is.na(ordering_mode), "", ordering_mode),
      choice = factor(choice, levels = c("shape", "texture", "unclear"))
    )
}

filter_human_friendly <- function(human_df) {
  if (!nrow(human_df)) return(human_df)
  human_df |>
    filter(design == "human_friendly")
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
