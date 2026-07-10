library(ggplot2)
library(dplyr)
library(scales)

SHAPE_COL   <- "#6B8E23"
TEXTURE_COL <- "#FFB6C1"
UNCLEAR_COL <- "#AAAAAA"

#' Main plot: proportion shape choice by stimulus
plot_stimulus_bias_main <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  df <- filter_trials(df, ordering = ordering, order_method = order_method, word_type = word_type)
  subtitle <- make_filter_subtitle(ordering = ordering, order_method = order_method, word_type = word_type)

  summary_df <- df |>
    group_by(stim_id) |>
    summarise(
      shape_prop = mean(choice == "shape", na.rm = TRUE),
      .groups = "drop"
    ) |>
    arrange(stim_id) |>
    mutate(stim_id = factor(stim_id))

  p <- ggplot(summary_df, aes(x = stim_id, y = shape_prop)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(colour = SHAPE_COL, size = 3) +
    geom_text(aes(label = sprintf("%.2f", shape_prop)),
              vjust = -1.2, size = 3) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    labs(
      x = "Stimulus ID",
      y = "Proportion shape choice",
      title = "Shape bias by stimulus",
      subtitle = subtitle
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9))

  p
}

#' Supplementary plot: stacked breakdown by stimulus
plot_stimulus_bias_supplement <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  df <- filter_trials(df, ordering = ordering, order_method = order_method, word_type = word_type)
  subtitle <- make_filter_subtitle(ordering = ordering, order_method = order_method, word_type = word_type)

  summary_df <- df |>
    group_by(stim_id, choice) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup() |>
    mutate(stim_id = factor(stim_id))

  p <- ggplot(summary_df, aes(x = stim_id, y = prop, fill = choice)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    scale_fill_manual(
      values = c(shape = SHAPE_COL, texture = TEXTURE_COL, unclear = UNCLEAR_COL),
      name = "Choice"
    ) +
    scale_y_continuous(labels = label_percent()) +
    labs(
      x = "Stimulus ID",
      y = "Proportion",
      title = "Response breakdown by stimulus",
      subtitle = subtitle
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9))

  p
}
