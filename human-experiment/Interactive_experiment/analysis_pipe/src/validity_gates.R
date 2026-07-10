library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(scales)

# Gate thresholds for cognitive-model interpretability
VALIDITY_THRESHOLDS <- list(
  image_tracking_valid = 0.70,
  image_tracking_borderline = 0.50,
  word_sensitivity_valid = 0.20,
  parse_quality_valid = 0.97
)

#' Compute image-tracking rate from paired deterministic orderings
compute_image_tracking_rate <- function(df) {
  paired <- df |>
    filter(ordering %in% c("shape_first", "texture_first"),
           order_method == "deterministic") |>
    select(model, stim_id, word, ordering, parsed_answer) |>
    pivot_wider(names_from = ordering, values_from = parsed_answer, values_fn = list) |>
    filter(lengths(shape_first) == 1, lengths(texture_first) == 1) |>
    mutate(
      shape_first = unlist(shape_first),
      texture_first = unlist(texture_first),
      tracks_image = case_when(
        shape_first == "1" & texture_first == "2" ~ 1,
        shape_first == "2" & texture_first == "1" ~ 1,
        TRUE ~ 0
      )
    )

  paired |>
    group_by(model) |>
    summarise(
      n_pairs = n(),
      image_tracking_rate = mean(tracks_image),
      .groups = "drop"
    )
}

#' Compute share of (model,stimulus,ordering) groups that vary by word
compute_word_sensitivity_rate <- function(df) {
  per_group <- df |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(model, stim_id, ordering) |>
    summarise(
      unique_choices = n_distinct(choice),
      .groups = "drop"
    ) |>
    mutate(word_sensitive = unique_choices > 1)

  per_group |>
    group_by(model) |>
    summarise(
      n_groups = n(),
      word_sensitivity_rate = mean(word_sensitive),
      .groups = "drop"
    )
}

#' Compute parse quality diagnostics from retry and unclear signals
compute_parse_quality <- function(df) {
  df |>
    group_by(model) |>
    summarise(
      n_trials = n(),
      unclear_rate = mean(choice == "unclear"),
      retry_rate = mean(attempts > 1),
      parse_quality = 1 - unclear_rate,
      .groups = "drop"
    )
}

#' Merge validity metrics and assign gate labels
compute_model_validity <- function(df) {
  tracking <- compute_image_tracking_rate(df)
  word <- compute_word_sensitivity_rate(df)
  parse <- compute_parse_quality(df)

  metrics <- tracking |>
    full_join(word, by = "model") |>
    full_join(parse, by = "model")

  metrics |>
    mutate(
      validity_label = case_when(
        image_tracking_rate >= VALIDITY_THRESHOLDS$image_tracking_valid &
          word_sensitivity_rate >= VALIDITY_THRESHOLDS$word_sensitivity_valid &
          parse_quality >= VALIDITY_THRESHOLDS$parse_quality_valid ~ "valid",
        image_tracking_rate >= VALIDITY_THRESHOLDS$image_tracking_borderline ~ "borderline",
        TRUE ~ "invalid"
      )
    )
}

#' Keep only valid models for cognitive-interpretation analyses
filter_valid_models <- function(df, validity_df = NULL) {
  if (is.null(validity_df)) validity_df <- compute_model_validity(df)
  valid_models <- validity_df |>
    filter(validity_label == "valid") |>
    pull(model)
  df |>
    filter(model %in% valid_models)
}

#' Keep valid and borderline models for sensitivity analyses
filter_valid_and_borderline_models <- function(df, validity_df = NULL) {
  if (is.null(validity_df)) validity_df <- compute_model_validity(df)
  selected_models <- validity_df |>
    filter(validity_label %in% c("valid", "borderline")) |>
    pull(model)
  df |>
    filter(model %in% selected_models)
}

#' Plot model validity labels by image tracking performance
plot_model_validity <- function(df, validity_df = NULL) {
  if (is.null(validity_df)) validity_df <- compute_model_validity(df)
  meta <- get_model_metadata()
  plot_df <- validity_df |>
    left_join(meta, by = "model") |>
    mutate(
      model_label = paste0(model, "\n(", param_b, "B)"),
      model_label = reorder(model_label, image_tracking_rate),
      validity_label = factor(validity_label, levels = c("invalid", "borderline", "valid"))
    )

  ggplot(plot_df, aes(x = model_label, y = image_tracking_rate, color = validity_label)) +
    geom_hline(yintercept = VALIDITY_THRESHOLDS$image_tracking_valid, linetype = "dashed", color = "grey40") +
    geom_point(size = 3.5) +
    scale_color_manual(values = c(invalid = "#D32F2F", borderline = "#F9A825", valid = "#2E7D32")) +
    scale_y_continuous(labels = label_percent(), limits = c(0, 1)) +
    labs(
      x = "Model",
      y = "Image tracking rate",
      color = "Validity label",
      title = "Model validity gate",
      subtitle = "Cognitive interpretation requires strong image tracking, non-trivial word sensitivity, and high parse quality"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
