library(dplyr)
library(readr)
library(stringr)
library(tibble)

#' Load engineered stimulus metadata from available manifests/audits.
#'
#' Optional manual metadata can be provided at results/data/stimulus_metadata.csv
#' with columns: stim_id, complexity, novelty, realism, abstractness, artifact_like.
load_stimulus_covariates <- function(
  repo_root = REPO_ROOT,
  optional_metadata_path = get_data_path("stimulus_metadata.csv")
) {
  manifest_path <- file.path(
    repo_root, "stimuli_pipe", "stimuli_per_stl_packages", "combined_benchmark_manifest.csv"
  )
  texture_audit_path <- file.path(
    repo_root, "stimuli_pipe", "stimuli_per_stl_packages", "texture_sync_audit.csv"
  )

  manifest <- read_csv(manifest_path, show_col_types = FALSE) |>
    transmute(
      mode,
      stim_id = as.character(stl_id)
    )

  texture <- read_csv(texture_audit_path, show_col_types = FALSE) |>
    transmute(
      mode,
      stim_id = as.character(stl_id),
      forced_texture_set,
      texture_consistent = as.logical(consistent_in_logs),
      texture_family = case_when(
        str_detect(str_to_lower(forced_texture_set), "metal|steel") ~ "metal",
        str_detect(str_to_lower(forced_texture_set), "fabric|denim") ~ "fabric",
        TRUE ~ "other"
      )
    )

  base_cov <- manifest |>
    left_join(texture, by = c("mode", "stim_id")) |>
    distinct(mode, stim_id, .keep_all = TRUE)

  if (file.exists(optional_metadata_path)) {
    custom <- read_csv(optional_metadata_path, show_col_types = FALSE) |>
      mutate(stim_id = as.character(stim_id))
    base_cov <- base_cov |>
      left_join(custom, by = "stim_id")
  } else {
    template <- base_cov |>
      distinct(stim_id) |>
      mutate(
        complexity = NA_real_,
        novelty = NA_real_,
        realism = NA_real_,
        abstractness = NA_real_,
        artifact_like = NA_real_
      )
    template_dir <- dirname(optional_metadata_path)
    if (!dir.exists(template_dir)) dir.create(template_dir, recursive = TRUE)
    write_csv(template, optional_metadata_path)
    base_cov <- base_cov |>
      left_join(template, by = "stim_id")
  }

  base_cov
}

#' Join stimulus covariates to trial-level benchmark data
join_stimulus_covariates <- function(df, covariates_df = NULL) {
  if (is.null(covariates_df)) covariates_df <- load_stimulus_covariates()
  df |>
    mutate(stim_id = as.character(stim_id)) |>
    left_join(covariates_df, by = "stim_id")
}

#' Fit covariate model to test predictors of shape-vs-texture choices
fit_stimulus_covariate_model <- function(df_with_cov) {
  d <- df_with_cov |>
    filter(choice %in% c("shape", "texture")) |>
    mutate(
      shape_binary = ifelse(choice == "shape", 1, 0),
      texture_family = factor(texture_family),
      mode = factor(mode),
      model = factor(model)
    )

  # Keep complete rows for optional scalar annotations.
  d <- d |>
    mutate(
      complexity = ifelse(is.na(complexity), 0, complexity),
      novelty = ifelse(is.na(novelty), 0, novelty),
      realism = ifelse(is.na(realism), 0, realism),
      abstractness = ifelse(is.na(abstractness), 0, abstractness),
      artifact_like = ifelse(is.na(artifact_like), 0, artifact_like)
    )

  glm(
    shape_binary ~ mode + texture_family + texture_consistent +
      complexity + novelty + realism + abstractness + artifact_like + model,
    data = d,
    family = binomial()
  )
}

#' Summarize covariate-level shape preference with uncertainty
summarize_covariate_effects <- function(df_with_cov) {
  df_with_cov |>
    filter(choice %in% c("shape", "texture")) |>
    group_by(mode, texture_family) |>
    summarise(
      n = n(),
      shape_prop = mean(choice == "shape"),
      se = sqrt(shape_prop * (1 - shape_prop) / n),
      ci_low = pmax(0, shape_prop - 1.96 * se),
      ci_high = pmin(1, shape_prop + 1.96 * se),
      .groups = "drop"
    ) |>
    arrange(desc(shape_prop))
}
