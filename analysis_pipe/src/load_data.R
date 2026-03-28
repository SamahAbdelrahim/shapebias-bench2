library(readr)
library(dplyr)
library(tibble)
library(stringr)

# ---------------------------------------------------------------------------
# Paths from .env
# ---------------------------------------------------------------------------
# here::here() points to analysis_pipe/ (renv project root).
# Repo root is one level up.
REPO_ROOT <- normalizePath(file.path(here::here(), ".."))

read_dotenv <- function(path = file.path(REPO_ROOT, ".env")) {
  if (!file.exists(path)) stop(".env file not found at: ", path)
  lines <- readLines(path, warn = FALSE)
  lines <- lines[grepl("=", lines) & !grepl("^\\s*#", lines)]
  for (line in lines) {
    parts <- strsplit(line, "=", fixed = TRUE)[[1]]
    key   <- trimws(parts[1])
    value <- trimws(paste(parts[-1], collapse = "="))
    do.call(Sys.setenv, setNames(list(value), key))
  }
}

#' Get the results directory (absolute) from .env RESULTS_DIR
get_results_dir <- function() {
  if (file.exists(file.path(REPO_ROOT, ".env"))) {
    read_dotenv()
  }
  rd <- Sys.getenv("RESULTS_DIR", unset = "results")
  # Resolve relative paths against repo root
  if (!startsWith(rd, "/")) rd <- file.path(REPO_ROOT, rd)
  rd
}

#' Get the data subdirectory inside results, creating it if needed
get_data_dir <- function() {
  data_dir <- file.path(get_results_dir(), "data")
  if (!dir.exists(data_dir)) dir.create(data_dir, recursive = TRUE)
  data_dir
}

#' Get the figures output directory, creating it if needed
get_figures_dir <- function() {
  fig_dir <- file.path(get_results_dir(), "figures")
  if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)
  fig_dir
}

#' Get the default CSV data path
get_data_path <- function(filename = "local_eval.csv") {
  file.path(get_data_dir(), filename)
}

# Model sizes in billions of parameters
MODEL_SIZES <- c(
  "qwen3.5-0.8b"     = 0.8,
  "internvl"          = 2,
  "qwen3-vl-2b"      = 2,
  "smolvlm"           = 2.2,
  "tinyllava"         = 3.1,
  "qwen3-vl-4b"      = 4,
  "qwen3.5-4b"        = 4,
  "qwen3.5-9b"        = 9,
  "qwen3.5-27b"       = 27,
  "qwen3.5-35b-a3b"   = 35,
  "llama4-scout"       = 109,
  "qwen3.5-122b-a10b" = 122
)

# Model family mapping (for color-coding plots)
MODEL_FAMILIES <- c(
  "qwen3.5-0.8b"     = "Qwen",
  "internvl"          = "InternVL",
  "qwen3-vl-2b"      = "Qwen",
  "smolvlm"           = "SmolVLM",
  "tinyllava"         = "TinyLLaVA",
  "qwen3-vl-4b"      = "Qwen",
  "qwen3.5-4b"        = "Qwen",
  "qwen3.5-9b"        = "Qwen",
  "qwen3.5-27b"       = "Qwen",
  "qwen3.5-35b-a3b"   = "Qwen",
  "llama4-scout"       = "LLaMA",
  "qwen3.5-122b-a10b" = "Qwen"
)

MODEL_DEPLOYMENT <- c(
  "qwen3.5-0.8b"     = "local",
  "internvl"         = "local",
  "qwen3-vl-2b"      = "local",
  "smolvlm"          = "local",
  "tinyllava"        = "local",
  "qwen3-vl-4b"      = "local",
  "qwen3.5-4b"       = "local",
  "qwen3.5-9b"       = "remote",
  "qwen3.5-27b"      = "remote",
  "qwen3.5-35b-a3b"  = "remote",
  "llama4-scout"     = "remote",
  "qwen3.5-122b-a10b"= "remote"
)

MODEL_ARCH <- c(
  "qwen3.5-0.8b"      = "dense",
  "internvl"          = "dense",
  "qwen3-vl-2b"       = "dense",
  "smolvlm"           = "dense",
  "tinyllava"         = "dense",
  "qwen3-vl-4b"       = "dense",
  "qwen3.5-4b"        = "dense",
  "qwen3.5-9b"        = "dense",
  "qwen3.5-27b"       = "dense",
  "qwen3.5-35b-a3b"   = "moe",
  "llama4-scout"      = "moe",
  "qwen3.5-122b-a10b" = "moe"
)

FAMILY_COLORS <- c(
  "Qwen"     = "#1f77b4",
  "InternVL" = "#ff7f0e",
  "SmolVLM"  = "#2ca02c",
  "TinyLLaVA"= "#9467bd",
  "LLaMA"    = "#d62728"
)

#' Load and clean results CSV(s)
load_results <- function(csv_paths) {
  existing_paths <- csv_paths[file.exists(csv_paths)]
  if (length(existing_paths) == 0) {
    stop("No input CSV files found. Checked:\n", paste(csv_paths, collapse = "\n"))
  }
  if (length(existing_paths) == 1) {
    df <- read_csv(existing_paths, show_col_types = FALSE)
  } else {
    df <- bind_rows(lapply(existing_paths, read_csv, show_col_types = FALSE))
  }
  df <- df |>
    mutate(
      choice = factor(choice, levels = c("shape", "texture", "unclear")),
      order_method = ifelse(order_method == "fixed", "deterministic", order_method)
    )
  df
}

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------
#' Filter trials by ordering, order_method, and/or word_type
filter_trials <- function(df, ordering = NULL, order_method = NULL, word_type = NULL) {
  if (!is.null(ordering))     df <- df |> filter(ordering %in% !!ordering)
  if (!is.null(order_method)) df <- df |> filter(order_method %in% !!order_method)
  if (!is.null(word_type))    df <- df |> filter(word_type %in% !!word_type)
  df
}

#' Build a subtitle string describing active filters (NULL if no filters)
make_filter_subtitle <- function(ordering = NULL, order_method = NULL, word_type = NULL) {
  parts <- c()
  if (!is.null(ordering))     parts <- c(parts, paste("Ordering:", paste(ordering, collapse = ", ")))
  if (!is.null(order_method)) parts <- c(parts, paste("Method:", paste(order_method, collapse = ", ")))
  if (!is.null(word_type))    parts <- c(parts, paste("Word type:", paste(word_type, collapse = ", ")))
  if (length(parts) == 0) return(NULL)
  paste(parts, collapse = " | ")
}

#' Add model size info and create ordered factor for plotting
add_model_size <- function(df) {
  size_df <- tibble(
    model = names(MODEL_SIZES),
    param_b = unname(MODEL_SIZES)
  )
  if (!"param_b" %in% names(df)) {
    df <- df |>
      left_join(size_df, by = "model")
  }
  if (!"model_family" %in% names(df)) df$model_family <- NA_character_
  if (!"deployment" %in% names(df)) df$deployment <- NA_character_
  if (!"architecture" %in% names(df)) df$architecture <- NA_character_
  df <- df |>
    mutate(
      model_label = paste0(model, "\n(", param_b, "B)"),
      model_label = factor(
        model_label,
        levels = size_df |>
          arrange(param_b) |>
          mutate(label = paste0(model, "\n(", param_b, "B)")) |>
          pull(label)
      ),
      model_family = ifelse(is.na(model_family), MODEL_FAMILIES[model], model_family),
      deployment = ifelse(is.na(deployment), MODEL_DEPLOYMENT[model], deployment),
      architecture = ifelse(is.na(architecture), MODEL_ARCH[model], architecture)
    )
  df
}

#' Build model metadata table for joining/export
get_model_metadata <- function() {
  tibble(
    model = names(MODEL_SIZES),
    param_b = as.numeric(unname(MODEL_SIZES)),
    model_family = unname(MODEL_FAMILIES[names(MODEL_SIZES)]),
    deployment = unname(MODEL_DEPLOYMENT[names(MODEL_SIZES)]),
    architecture = unname(MODEL_ARCH[names(MODEL_SIZES)])
  )
}

#' Canonical result paths in priority order
get_candidate_result_paths <- function() {
  c(
    get_data_path("local_eval.csv"),
    get_data_path("remote_all_fixed.csv"),
    get_data_path("remote_all.csv"),
    file.path(get_results_dir(), "local_eval.csv"),
    file.path(get_results_dir(), "remote_all_fixed.csv"),
    file.path(get_results_dir(), "remote_all.csv")
  )
}

#' Build one canonical combined dataset for all available runs
build_canonical_dataset <- function(output_path = get_data_path("canonical_combined_eval.csv")) {
  candidate_paths <- get_candidate_result_paths()
  existing_paths <- unique(candidate_paths[file.exists(candidate_paths)])
  if (length(existing_paths) == 0) {
    stop("No result CSVs found in expected paths.")
  }

  source_dfs <- lapply(existing_paths, function(path) {
    df <- read_csv(path, show_col_types = FALSE)
    filename <- basename(path)
    run_source <- case_when(
      str_detect(filename, "local") ~ "local",
      str_detect(filename, "remote") ~ "remote",
      TRUE ~ "other"
    )
    df |>
      mutate(
        source_file = filename,
        source_path = path,
        run_source = run_source
      )
  })

  combined <- bind_rows(source_dfs) |>
    mutate(order_method = ifelse(order_method == "fixed", "deterministic", order_method)) |>
    distinct(model, stim_id, word, ordering, source_file, .keep_all = TRUE) |>
    left_join(get_model_metadata(), by = "model")

  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  write_csv(combined, output_path)
  combined
}
