# Shared helpers for per-source preprocessing scripts.
#
# Per-source scripts in rcts/ and surveys/ compose these helpers with their own
# dataset-specific cleaning logic. Helpers harmonize what is harmonizable; per-
# source idiosyncrasies (variable selection, study-specific recodes, panel
# pivots, country loops) stay in the per-source script.
#
# Output conventions enforced here:
#   - First column is `subject_id`; remaining columns follow original-study
#     order, with `treatment` renamed in place. Outcome variables keep their
#     original study-specific names — no rename to `outcome` / `outcome_1` etc.
#   - Row 1 is the question header. For trivial questions (gender, etc.) the
#     header is the bare label. For non-trivial questions with discrete options
#     the header is `<label> Answer one of the following options: <options>`.
#     For continuous-scale variables with a prose preface in `label_for_options`
#     but no enumerated `options`, the prose is preserved.
#   - Raw variable-code labels are kept only for `subject_id` and `treatment`.
#   - Output is written with default CSV quoting and `na = NA` (not `na = ""`).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(stringr)
  library(tibble)
})

# ---------------------------------------------------------------------------
# Mapping-CSV helpers (canonical schema: name, label, options, selected,
# label_for_options).
# ---------------------------------------------------------------------------

read_mapping_csv <- function(path) {
  required_cols <- c("name", "label", "options", "selected", "label_for_options")
  mapping <- readr::read_csv(
    path,
    show_col_types = FALSE,
    col_types = readr::cols(.default = readr::col_character())
  )
  missing <- setdiff(required_cols, names(mapping))
  if (length(missing) > 0) {
    stop("Mapping CSV is missing required columns: ",
         paste(missing, collapse = ", "), call. = FALSE)
  }
  mapping %>%
    mutate(selected = suppressWarnings(as.integer(selected)))
}

# Parse "1=Yes, 2=No" -> "Yes, No". Returns NA for empty/NA inputs.
parse_options <- function(options_string) {
  if (length(options_string) != 1) {
    return(vapply(options_string, parse_options, character(1)))
  }
  if (is.na(options_string) || options_string == "") return(NA_character_)
  options_string %>%
    str_split(",") %>%
    unlist() %>%
    str_remove(".*=") %>%
    str_trim() %>%
    paste(collapse = ", ")
}

# Build a per-variable named-vector lookup from the mapping. Each entry maps
# the raw code (e.g. "1") to the human label (e.g. "Yes").
build_options_lookup <- function(mapping) {
  mapping %>%
    filter(selected == 1) %>%
    distinct(name, .keep_all = TRUE) %>%
    select(name, options) %>%
    mutate(
      options_vec = map(options, function(opt) {
        if (is.na(opt) || opt == "") return(NULL)
        parts <- str_split(opt, ",\\s*")[[1]]
        keys <- str_extract(parts, "^[^=]+")
        labels <- str_remove(parts, "^[^=]+=")
        setNames(labels, keys)
      })
    ) %>%
    select(name, options_vec)
}

# Replace coded values in a single column with their human-readable labels.
# Preserves NA. Handles comma-separated multi-codes by mapping each.
replace_with_human_readable <- function(x, col_name, lookup) {
  idx <- which(lookup$name == col_name)
  if (length(idx) == 0) return(x)
  label_vec <- lookup$options_vec[[idx]]
  if (is.null(label_vec)) return(x)
  map_chr(x, function(.x) {
    if (is.na(.x)) return(NA_character_)
    keys <- str_trim(str_split(.x, ",")[[1]])
    labels <- label_vec[keys]
    if (any(is.na(labels))) .x else paste(labels, collapse = ",")
  })
}

# Apply replace_with_human_readable across all columns of a data frame using
# the supplied lookup. Columns absent from the lookup are left unchanged.
apply_human_readable <- function(df, lookup) {
  df %>%
    mutate(across(everything(),
                  ~ replace_with_human_readable(.x, cur_column(), lookup)))
}

# Cleanup applied to `label_for_options` when it is being included in the
# question header. Default policy: minimize adaptations — start with the
# verbatim survey label, only attach `label_for_options` content if it is
# substantive (e.g., scale-anchor description). Synthetic format instructions
# like "Please, answer with a number between 1 and 100." are dropped.
clean_label_for_options <- function(s) {
  if (is.na(s) || s == "") return(NA_character_)
  # Synthetic format instruction (no scale-anchor content): drop entirely.
  if (grepl("^[Pp]lease,?\\s*[Aa]nswer", s)) return(NA_character_)
  # Strip leading "Please " / "Please, " (politeness)
  s <- sub("^[Pp]lease,?\\s*", "", s)
  # Capitalize first letter after the strip
  s <- sub("^(.)", "\\U\\1", s, perl = TRUE)
  # Normalize curly double quotes to straight single quotes
  s <- gsub("“", "'", s, fixed = TRUE)
  s <- gsub("”", "'", s, fixed = TRUE)
  # American-style comma inside closing quote -> outside (do so,' -> do so',)
  s <- gsub(",'", "',", s, fixed = TRUE)
  # Drop redundant trailing scale enumeration:
  #   "You can [also] use any number between 0 and 10 ...".
  s <- sub("\\s*You can( also)?\\s+use\\s+any\\s+number\\s+between.*$", "", s)
  trimws(s)
}

# Build a 1-row tibble of question-header text, indexed by variable name.
# Rule:
#   - For names in `raw_label_vars`: header is the variable name itself.
#   - If `options` is populated: "<label> <response_phrase>: <options>"
#     (universal preface; replaces whatever per-row preface previously lived
#     in `label_for_options`).
#   - If `options` is empty but `label_for_options` is substantive (i.e.,
#     `clean_label_for_options` returns non-NA): "<label> <cleaned_lfo>"
#     (e.g., scale-anchor descriptions for continuous scales).
#   - Otherwise: the bare label, verbatim from the original survey.
build_question_header <- function(mapping,
                                  response_phrase = "Answer one of the following options",
                                  raw_label_vars = c("subject_id", "treatment")) {
  mapping %>%
    filter(selected == 1) %>%
    distinct(name, .keep_all = TRUE) %>%
    mutate(
      parsed_opts = parse_options(options),
      cleaned_lfo = vapply(label_for_options, clean_label_for_options, character(1)),
      header_text = case_when(
        name %in% raw_label_vars     ~ name,
        is.na(label)                 ~ NA_character_,
        !is.na(parsed_opts)          ~ paste0(label, " ", response_phrase, ": ", parsed_opts),
        !is.na(cleaned_lfo)          ~ paste0(label, " ", cleaned_lfo),
        TRUE                         ~ label
      )
    ) %>%
    select(name, header_text) %>%
    pivot_wider(names_from = name, values_from = header_text)
}

# ---------------------------------------------------------------------------
# Inline-questions helper (for sources that hardcode their own question
# vector + response-level lookup, e.g. HPS, Brailovskaia, CANDOUR/Duch 2025,
# Duch 2023, Afrobarometer, Meriggi).
# ---------------------------------------------------------------------------

# Build a 1-row tibble of question-header text from an inline `spec` tibble
# with columns `name`, `question`, `response_levels`.
#
# Rule:
#   - For names in `raw_label_vars`: header is the variable name itself.
#   - If `response_levels` is NA or empty: bare question (verbatim).
#   - If `response_levels` contains ";" (discrete enumerated options): append
#         "<question> Answer one of the following options: <opts>"
#     where opts have the semicolons replaced with commas.
#   - Otherwise (continuous-scale description): append
#         "<question> Answer with <response_levels>".
build_inline_question_header <- function(spec,
                                         response_phrase = "Answer one of the following options",
                                         continuous_phrase = "Answer with",
                                         raw_label_vars = c("subject_id", "treatment")) {
  spec %>%
    mutate(
      header_text = case_when(
        name %in% raw_label_vars
                                                 ~ name,
        is.na(question)
                                                 ~ NA_character_,
        is.na(response_levels) | response_levels == ""
                                                 ~ question,
        grepl(";", response_levels, fixed = TRUE)
                                                 ~ paste0(question, " ", response_phrase, ": ",
                                                          gsub(";\\s*", ", ", response_levels)),
        TRUE
                                                 ~ paste0(question, " ", continuous_phrase, " ", response_levels)
      )
    ) %>%
    select(name, header_text) %>%
    pivot_wider(names_from = name, values_from = header_text)
}

# ---------------------------------------------------------------------------
# Output helpers (universal across mapping-path and inline-path scripts)
# ---------------------------------------------------------------------------

# Coerce both header (1-row tibble) and data to character, then bind. The
# result has the question-header row first, data rows below. Embedded
# carriage-return / line-feed runs in header cells (common when mapping CSVs
# have multi-line label cells with Windows line endings) are collapsed to a
# single space.
inject_question_header <- function(df, header_row) {
  data_chr <- df %>% mutate(across(everything(), as.character))
  header_chr <- header_row %>%
    mutate(across(everything(),
                  ~ gsub("[\r\n]+", " ", as.character(.x))))
  bind_rows(header_chr[, names(data_chr), drop = FALSE], data_chr)
}

# Move `subject_id` to position 1; preserve all other columns in current
# order. If subject_id is absent, append it at position 1 with sequential row
# numbers AFTER the header row (the header gets the literal "subject_id").
ensure_subject_id_first <- function(df, header_already_injected = TRUE) {
  if (!"subject_id" %in% names(df)) {
    n <- nrow(df)
    if (header_already_injected) {
      df$subject_id <- c("subject_id", as.character(seq_len(n - 1)))
    } else {
      df$subject_id <- as.character(seq_len(n))
    }
  }
  df[, c("subject_id", setdiff(names(df), "subject_id")), drop = FALSE]
}

# Write a cleaned CSV. Uses default quoting and `na = NA` (so missing values
# emit as a literal "NA" rather than the Stata-style empty string convention
# that earlier scripts used). UTF-8 encoding.
write_clean_csv <- function(df, path) {
  dir <- dirname(path)
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(df, path, na = "NA")
}
