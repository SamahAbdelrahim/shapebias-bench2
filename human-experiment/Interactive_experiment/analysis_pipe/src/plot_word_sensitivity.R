library(ggplot2)
library(dplyr)
library(scales)

SENSITIVE_COL   <- "#FF9800"
INSENSITIVE_COL <- "#607D8B"

#' Compute word sensitivity per (model, stim_id, ordering) group.
#'
#' A group is "word-sensitive" if the model gives different choices for
#' different words on the same stimulus+ordering. If it always picks the
#' same choice regardless of word, it is "word-insensitive".
compute_word_sensitivity <- function(df) {
  df |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(model, model_label, model_family, stim_id, ordering) |>
    summarise(
      n_words       = n(),
      unique_choices = n_distinct(choice),
      .groups = "drop"
    ) |>
    mutate(
      word_sensitive = ifelse(unique_choices > 1, "word-sensitive", "word-insensitive")
    )
}

#' Main plot: proportion of stimulus-ordering groups that are word-sensitive, by model
plot_word_sensitivity_main <- function(df) {
  ws <- compute_word_sensitivity(df)

  summary_df <- ws |>
    group_by(model_label, model_family, word_sensitive) |>
    summarise(n = n(), .groups = "drop_last") |>
    mutate(prop = n / sum(n)) |>
    ungroup()

  # Label for sensitive proportion
  label_df <- summary_df |>
    filter(word_sensitive == "word-sensitive") |>
    select(model_label, prop)

  # Build axis color vector matching factor level order
  axis_colors <- FAMILY_COLORS[MODEL_FAMILIES[sub("\\n.*", "", levels(droplevels(summary_df$model_label)))]]

  ggplot(summary_df, aes(x = model_label, y = prop, fill = word_sensitive)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_text(data = label_df,
              aes(x = model_label, y = prop, label = sprintf("%.0f%%", prop * 100),
                  fill = NULL),
              vjust = -0.5, size = 3.5) +
    scale_fill_manual(
      values = c("word-sensitive" = SENSITIVE_COL, "word-insensitive" = INSENSITIVE_COL),
      name = "Sensitivity"
    ) +
    scale_y_continuous(limits = c(0, 1.08), labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion of stimulus-ordering groups",
      title = "Word sensitivity validation",
      subtitle = "Does the model's choice change when different words are used?"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9, angle = 45, hjust = 1,
                                     colour = axis_colors))
}

#' By-stimulus: word sensitivity rate per stimulus, coloured by model
plot_word_sensitivity_by_stimulus <- function(df) {
  ws <- compute_word_sensitivity(df)

  summary_df <- ws |>
    group_by(model_label, stim_id) |>
    summarise(
      sensitive_prop = mean(word_sensitive == "word-sensitive"),
      .groups = "drop"
    ) |>
    mutate(stim_id = factor(stim_id))

  ggplot(summary_df, aes(x = stim_id, y = sensitive_prop, colour = model_label)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    scale_colour_viridis_d(name = "Model") +
    labs(
      x = "Stimulus ID",
      y = "Proportion word-sensitive",
      title = "Word sensitivity by stimulus and model"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9))
}

#' Word type effect: mean shape-bias by word_type (sudo vs random) per model
plot_word_type_effect <- function(df) {
  summary_df <- df |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(model_label, model_family, word_type) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      se = sqrt(shape_prop * (1 - shape_prop) / n),
      .groups = "drop"
    )

  # Build axis color vector matching factor level order
  axis_colors <- FAMILY_COLORS[MODEL_FAMILIES[sub("\\n.*", "", levels(droplevels(summary_df$model_label)))]]

  ggplot(summary_df, aes(x = model_label, y = shape_prop, fill = word_type)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6) +
    geom_errorbar(
      aes(ymin = pmax(shape_prop - se, 0), ymax = pmin(shape_prop + se, 1)),
      position = position_dodge(width = 0.7), width = 0.2
    ) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    scale_fill_manual(
      values = c(sudo = "#5C6BC0", random = "#AB47BC"),
      labels = c(sudo = "Pseudo-words", random = "Random strings"),
      name = "Word type"
    ) +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    labs(
      x = "Model (parameters)",
      y = "Proportion shape choice",
      title = "Shape bias by word type",
      subtitle = "Do pseudo-words vs random strings affect model responses?"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 9, angle = 45, hjust = 1,
                                     colour = axis_colors))
}

#' Shape bias by individual word, faceted by word type
plot_word_bias <- function(df) {
  summary_df <- df |>
    filter(choice %in% c("shape", "texture")) |>
    mutate(word_length = nchar(word)) |>
    group_by(word, word_type, word_length) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      se = sqrt(shape_prop * (1 - shape_prop) / n),
      .groups = "drop"
    ) |>
    mutate(
      word = reorder(word, word_length),
      word_type_label = ifelse(word_type == "sudo", "Pseudo-words", "Random strings")
    )

  ggplot(summary_df, aes(x = word, y = shape_prop)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(colour = "#6B8E23", size = 3) +
    geom_errorbar(
      aes(ymin = pmax(shape_prop - se, 0), ymax = pmin(shape_prop + se, 1)),
      width = 0.2, colour = "#6B8E23"
    ) +
    geom_text(aes(label = sprintf("%.0f%%", shape_prop * 100)),
              vjust = -1.2, size = 3.2) +
    facet_wrap(~ word_type_label, scales = "free_x") +
    scale_y_continuous(limits = c(0, 1.08), labels = label_percent()) +
    labs(
      x = "Word",
      y = "Proportion shape choice",
      title = "Shape bias by word",
      subtitle = "Averaged across all models and stimuli"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 10, angle = 30, hjust = 1))
}

#' Shape bias by individual word, broken down by model, faceted by word type
plot_word_bias_by_model <- function(df) {
  summary_df <- df |>
    filter(choice %in% c("shape", "texture")) |>
    mutate(word_length = nchar(word)) |>
    group_by(model_label, word, word_type, word_length) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      .groups = "drop"
    ) |>
    mutate(
      word = reorder(word, word_length),
      word_type_label = ifelse(word_type == "sudo", "Pseudo-words", "Random strings")
    )

  ggplot(summary_df, aes(x = word, y = shape_prop, colour = model_label)) +
    geom_hline(yintercept = 0.5, linetype = "dashed", colour = "grey40") +
    geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
    facet_wrap(~ word_type_label, scales = "free_x") +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    scale_colour_viridis_d(name = "Model") +
    labs(
      x = "Word",
      y = "Proportion shape choice",
      title = "Shape bias by word and model",
      subtitle = "Per-model breakdown; words ordered by length"
    ) +
    theme_minimal(base_size = 13) +
    theme(axis.text.x = element_text(size = 10, angle = 30, hjust = 1))
}
