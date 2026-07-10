library(dplyr)
library(tidyr)
library(tibble)

#' Robustness diagnostics for ordering and response format stability
compute_robustness_diagnostics <- function(df) {
  base <- df |>
    filter(choice %in% c("shape", "texture"))

  ordering_effect <- base |>
    filter(ordering %in% c("shape_first", "texture_first")) |>
    group_by(model, ordering) |>
    summarise(shape_prop = mean(choice == "shape"), .groups = "drop") |>
    pivot_wider(names_from = ordering, values_from = shape_prop) |>
    mutate(
      ordering_delta = shape_first - texture_first,
      ordering_stable = abs(ordering_delta) <= 0.10
    )

  parse_stability <- df |>
    group_by(model) |>
    summarise(
      unclear_rate = mean(choice == "unclear"),
      retry_rate = mean(attempts > 1),
      parse_stable = unclear_rate <= 0.01 & retry_rate <= 0.10,
      .groups = "drop"
    )

  ordering_effect |>
    left_join(parse_stability, by = "model") |>
    mutate(
      robust_overall = ordering_stable & parse_stable
    )
}

#' Build a balanced frozen subset for human benchmarking
#'
#' Returns one row per selected stimulus id.
propose_human_benchmark_subset <- function(df_with_cov, n_stimuli = 12, seed = 42) {
  set.seed(seed)
  stim_df <- df_with_cov |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(stim_id, mode, texture_family) |>
    summarise(
      shape_prop = mean(choice == "shape"),
      n_trials = n(),
      .groups = "drop"
    ) |>
    mutate(
      shape_band = case_when(
        shape_prop < 0.40 ~ "texture_leaning",
        shape_prop > 0.60 ~ "shape_leaning",
        TRUE ~ "ambiguous"
      )
    )

  # Balance across mode x shape_band first, then fill up to n_stimuli.
  target_cells <- stim_df |>
    group_by(mode, shape_band) |>
    slice_sample(n = min(2, n())) |>
    ungroup()

  selected <- target_cells
  if (nrow(selected) < n_stimuli) {
    remaining <- stim_df |>
      anti_join(selected, by = "stim_id")
    selected <- bind_rows(
      selected,
      remaining |> slice_sample(n = min(n_stimuli - nrow(selected), nrow(remaining)))
    )
  }

  selected |>
    slice_head(n = n_stimuli) |>
    arrange(mode, shape_band, stim_id)
}

#' Human protocol specification table aligned with benchmark design
build_human_protocol_spec <- function(
  word_pairs = list(
    c("shiple", "afnafq"),
    c("clapher", "ieyiccw"),
    c("plailass", "orvufaig"),
    c("procation", "qahftrxck"),
    c("adinefults", "cgchqjjfgy")
  ),
  trials_per_stimulus = 10
) {
  words <- unlist(lapply(word_pairs, function(x) c(x[1], x[2])))
  tibble(
    component = c(
      "Task format",
      "Instruction",
      "Stimulus set",
      "Words",
      "Counterbalancing",
      "Trials per stimulus",
      "Response coding"
    ),
    specification = c(
      "3-image 2AFC",
      "First image is a {word}; choose 1 or 2 for same {word}",
      "Frozen subset from packaged benchmark stimuli",
      paste(length(words), "novel words (5 pseudo + 5 random, length matched)"),
      "shape_first and texture_first both included",
      as.character(trials_per_stimulus),
      "Map human 1/2 to shape/texture with same a_is/b_is mapping used for models"
    )
  )
}
