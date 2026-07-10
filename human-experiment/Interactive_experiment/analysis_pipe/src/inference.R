library(dplyr)
library(tibble)

#' Prepare binary shape-choice rows for inference
prepare_inference_data <- function(df) {
  df |>
    filter(choice %in% c("shape", "texture")) |>
    mutate(
      shape_binary = ifelse(choice == "shape", 1, 0),
      word_type = factor(word_type),
      ordering = factor(ordering),
      model = factor(model),
      stim_id = factor(stim_id)
    )
}

#' Fit shape-choice model; uses mixed effects when lme4 is available
fit_shape_choice_model <- function(df) {
  d <- prepare_inference_data(df)

  if (requireNamespace("lme4", quietly = TRUE)) {
    formula <- shape_binary ~ ordering + word_type + scale(param_b) + model_family +
      (1 | stim_id) + (1 | word)
    fit <- lme4::glmer(formula, data = d, family = binomial())
    attr(fit, "fit_type") <- "glmer"
    return(fit)
  }

  # Fallback with fixed effects when mixed-effects package is unavailable.
  formula <- shape_binary ~ ordering + word_type + scale(param_b) + model_family + stim_id
  fit <- glm(formula, data = d, family = binomial())
  attr(fit, "fit_type") <- "glm_fixed_effects"
  fit
}

#' Return coefficient table with odds ratios and confidence intervals
summarize_shape_choice_model <- function(fit) {
  coef_df <- as.data.frame(summary(fit)$coefficients)
  coef_df <- tibble::rownames_to_column(coef_df, "term")
  names(coef_df) <- c("term", "estimate", "std_error", "statistic", "p_value")

  ci <- suppressMessages(suppressWarnings(confint.default(fit)))
  ci_df <- tibble(
    term = rownames(ci),
    conf_low = ci[, 1],
    conf_high = ci[, 2]
  )

  coef_df |>
    left_join(ci_df, by = "term") |>
    mutate(
      odds_ratio = exp(estimate),
      or_low = exp(conf_low),
      or_high = exp(conf_high),
      fit_type = attr(fit, "fit_type")
    ) |>
    arrange(desc(abs(estimate)))
}

#' Model-wise shape bias summary with Wald CIs
compute_model_shape_ci <- function(df) {
  d <- df |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(model, model_family, param_b, deployment, architecture) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      se = sqrt(shape_prop * (1 - shape_prop) / n),
      ci_low = pmax(0, shape_prop - 1.96 * se),
      ci_high = pmin(1, shape_prop + 1.96 * se),
      .groups = "drop"
    ) |>
    arrange(desc(shape_prop))

  d
}

#' Ordering-effect summary by model with confidence intervals
compute_ordering_effect_ci <- function(df) {
  d <- df |>
    filter(choice %in% c("shape", "texture"),
           ordering %in% c("shape_first", "texture_first")) |>
    group_by(model, ordering) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(names_from = ordering, values_from = c(n, shape_prop)) |>
    mutate(
      delta_shape = shape_prop_shape_first - shape_prop_texture_first,
      se_delta = sqrt(
        (shape_prop_shape_first * (1 - shape_prop_shape_first) / n_shape_first) +
          (shape_prop_texture_first * (1 - shape_prop_texture_first) / n_texture_first)
      ),
      ci_low = delta_shape - 1.96 * se_delta,
      ci_high = delta_shape + 1.96 * se_delta
    ) |>
    arrange(desc(abs(delta_shape)))

  d
}
