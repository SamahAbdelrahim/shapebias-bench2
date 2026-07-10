library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(ggrepel)

TRACKS_COL  <- "#4CAF50"
BIAS_COL    <- "#F44336"
MISSING_COL <- "#AAAAAA"

#' Classify each stimulus x word x model pair as "tracks image" or "position bias"
#'
#' Requires both orderings (shape_first + texture_first) to be present.
#' Logic: if the model picks image-1 when shape is first and image-2 when shape
#' is second (or vice-versa for texture), it is tracking the image content.
#' If it gives the same answer regardless of ordering, it has position bias.
classify_position_bias <- function(df) {
  paired <- df |>
    filter(ordering %in% c("shape_first", "texture_first"),
           order_method == "deterministic") |>
    select(model, model_label, model_family, stim_id, word, ordering, parsed_answer) |>
    pivot_wider(names_from = ordering, values_from = parsed_answer,
                values_fn = list)

  # Keep only rows where we have exactly one answer per ordering
  paired <- paired |>
    filter(
      lengths(shape_first) == 1,
      lengths(texture_first) == 1
    ) |>
    mutate(
      shape_first = unlist(shape_first),
      texture_first = unlist(texture_first)
    )

  paired |>
    mutate(
      validity = case_when(
        # shape_first: 1=shape,2=texture; texture_first: 1=texture,2=shape
        # Tracks shape: picks shape position in both orderings
        shape_first == "1" & texture_first == "2" ~ "tracks_image",
        # Tracks texture: picks texture position in both orderings
        shape_first == "2" & texture_first == "1" ~ "tracks_image",
        # Same answer regardless → position bias
        shape_first == texture_first ~ "position_bias",
        TRUE ~ "other"
      )
    )
}

#' Main plot: proportion of trials that track image vs show position bias, by model
plot_position_bias_main <- function(df) {
  classified <- classify_position_bias(df)

  summary_df <- classified |>
    group_by(model_label, model_family, validity) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n), total = sum(n)) |>
    ungroup()

  # Add label for tracks_image proportion
  label_df <- summary_df |>
    filter(validity == "tracks_image") |>
    select(model_label, prop)

  # Build axis color vector matching factor level order
  axis_colors <- FAMILY_COLORS[MODEL_FAMILIES[sub("\\n.*", "", levels(droplevels(summary_df$model_label)))]]

  ggplot(summary_df, aes(x = model_label, y = prop, fill = validity)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_text(data = label_df,
              aes(x = model_label, y = prop, label = sprintf("%.0f%%", prop * 100),
                  fill = NULL),
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(
      values = c(tracks_image = TRACKS_COL, position_bias = BIAS_COL, other = MISSING_COL),
      labels = c(tracks_image = "Tracks image", position_bias = "Position bias", other = "Other"),
      name = "Validity"
    ) +
    scale_y_continuous(limits = c(0, 1.08), labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion of paired trials",
      title = "Position bias validation",
      subtitle = "Do models track the correct image when positions are swapped?"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9, angle = 45, hjust = 1,
                                     colour = axis_colors))
}

#' Supplementary: position bias broken down by stimulus
plot_position_bias_by_stimulus <- function(df) {
  classified <- classify_position_bias(df)

  summary_df <- classified |>
    group_by(model_label, stim_id, validity) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup() |>
    filter(validity == "tracks_image") |>
    mutate(stim_id = factor(stim_id))

  ggplot(summary_df, aes(x = stim_id, y = prop, colour = model_label)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    scale_colour_viridis_d(name = "Model") +
    labs(
      x = "Stimulus ID",
      y = "Proportion tracking image",
      title = "Image tracking by stimulus and model"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9))
}

#' Rank-ordered position bias: each model's stimuli sorted independently by tracking rate
plot_position_bias_ranked <- function(df) {
  classified <- classify_position_bias(df)

  # Get all model x stim_id combinations so 0% tracking stimuli aren't dropped
  all_combos <- classified |>
    distinct(model_label, stim_id)

  summary_df <- classified |>
    group_by(model_label, stim_id, validity) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup() |>
    filter(validity == "tracks_image") |>
    right_join(all_combos, by = c("model_label", "stim_id")) |>
    mutate(prop = replace_na(prop, 0)) |>
    group_by(model_label) |>
    arrange(prop, .by_group = TRUE) |>
    mutate(rank = row_number()) |>
    ungroup()

  ggplot(summary_df, aes(x = rank, y = prop, colour = model_label, group = model_label)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_line(linewidth = 0.6, alpha = 0.6) +
    geom_point(size = 2.5) +
    geom_text_repel(aes(label = stim_id), size = 2.3, show.legend = FALSE,
                    max.overlaps = 20, segment.size = 0.2, segment.alpha = 0.4,
                    box.padding = 0.2, point.padding = 0.15, seed = 42) +
    scale_y_continuous(limits = c(-0.05, 1.1), labels = label_percent()) +
    scale_colour_viridis_d(name = "Model") +
    labs(
      x = "Stimulus rank (sorted by tracking rate per model)",
      y = "Proportion tracking image",
      title = "Rank-ordered image tracking by model",
      subtitle = "Each model's stimuli sorted independently; labels = stimulus ID"
    ) +
    theme_minimal(base_size = 13)
}
