library(dplyr)
library(tidyr)
library(tibble)
library(ggplot2)
library(scales)

#' Compute noun-label vs no-word contrasts per model.
compute_condition_contrasts <- function(df) {
  base <- df |>
    filter(choice %in% c("shape", "texture")) |>
    mutate(prompt_condition = ifelse(is.na(prompt_condition), "noun_label", prompt_condition))

  shape_df <- base |>
    group_by(model, prompt_condition) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      .groups = "drop"
    ) |>
    pivot_wider(names_from = prompt_condition, values_from = c(n, shape_prop))

  order_df <- base |>
    filter(ordering %in% c("shape_first", "texture_first")) |>
    group_by(model, prompt_condition, ordering) |>
    summarise(shape_prop = mean(choice == "shape"), .groups = "drop") |>
    pivot_wider(names_from = ordering, values_from = shape_prop) |>
    mutate(ordering_delta = shape_first - texture_first) |>
    select(model, prompt_condition, ordering_delta) |>
    pivot_wider(names_from = prompt_condition, values_from = ordering_delta, names_prefix = "ord_delta_")

  contrast <- shape_df |>
    left_join(order_df, by = "model") |>
    mutate(
      delta_shape_prop = shape_prop_no_word_category - shape_prop_noun_label,
      delta_ordering_effect = ord_delta_no_word_category - ord_delta_noun_label,
      se_noun = sqrt(shape_prop_noun_label * (1 - shape_prop_noun_label) / n_noun_label),
      se_no_word = sqrt(shape_prop_no_word_category * (1 - shape_prop_no_word_category) / n_no_word_category),
      se_delta = sqrt(se_noun^2 + se_no_word^2),
      ci_low = delta_shape_prop - 1.96 * se_delta,
      ci_high = delta_shape_prop + 1.96 * se_delta
    ) |>
    arrange(desc(abs(delta_shape_prop)))

  contrast
}

#' Plot shape-bias deltas between noun-label and no-word conditions.
plot_condition_delta_shape <- function(df) {
  contrast <- compute_condition_contrasts(df) |>
    left_join(get_model_metadata(), by = "model") |>
    mutate(model_label = paste0(model, "\n(", param_b, "B)")) |>
    mutate(model_label = reorder(model_label, delta_shape_prop))

  ggplot(contrast, aes(x = model_label, y = delta_shape_prop, color = model_family)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.2) +
    geom_point(size = 3) +
    scale_color_manual(values = FAMILY_COLORS, drop = FALSE) +
    scale_y_continuous(labels = label_percent()) +
    labs(
      x = "Model",
      y = "Delta shape rate (no-word - noun-label)",
      color = "Family",
      title = "No-word vs noun-label contrast by model"
    ) +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}
