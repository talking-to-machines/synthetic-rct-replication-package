# Data cleaning script translated from data_cleaning_Fink.do
# Supplementary-materials version: retains the original Stata recodes,
# variable names, labels, and spelling/typographical choices as closely as possible.

# Required packages -----------------------------------------------------------
# install.packages(c("dplyr", "haven", "readr"))
library(dplyr)
library(haven)
library(readr)

# Paths -----------------------------------------------------------------------
if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human", "rcts", "fink_et_al_2018")
saving_path <- file.path("data", "processed", "rcts", "fink_et_al_2018")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

baseline_file <- file.path(data_path, "baseline_cleaned_all.dta")
outcome_file  <- file.path(data_path, "04_2_ShortAnalysis_C4D&M4D.dta")
output_file   <- file.path(saving_path, "fink_et_al_2018_data.csv")

# Helper functions ------------------------------------------------------------
tagged_na_safe <- function(x) {
  out <- rep(FALSE, length(x))
  try(out <- haven::is_tagged_na(x), silent = TRUE)
  out
}

tag_safe <- function(x) {
  out <- rep(NA_character_, length(x))
  try(out <- haven::na_tag(x), silent = TRUE)
  out
}

to_num <- function(x) {
  suppressWarnings(as.numeric(haven::zap_labels(x)))
}

num_eq <- function(x, value) {
  y <- to_num(x)
  !is.na(y) & y == value
}

num_in <- function(x, values) {
  y <- to_num(x)
  !is.na(y) & y %in% values
}

num_lt <- function(x, value) {
  y <- to_num(x)
  !is.na(y) & y < value
}

is_missing_any <- function(x) {
  is.na(x)
}

to_stata_string <- function(x) {
  y <- haven::zap_labels(x)
  out <- suppressWarnings(as.character(y))
  tagged <- tagged_na_safe(x)
  tags <- tag_safe(x)
  out[tagged & !is.na(tags)] <- paste0(".", tags[tagged & !is.na(tags)])
  out[is.na(x) & !tagged] <- "."
  out
}

set_na_by_condition <- function(x, condition) {
  out <- x
  condition[is.na(condition)] <- FALSE
  out[condition] <- NA
  out
}

replace_string_values <- function(x, from, to) {
  out <- as.character(x)
  out[!is.na(out) & out %in% from] <- to
  out
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

order_any <- function(df, vars) {
  ordered <- intersect(vars, names(df))
  df[, c(ordered, setdiff(names(df), ordered)), drop = FALSE]
}

move_after <- function(df, vars, after) {
  vars <- intersect(vars, names(df))
  if (!length(vars) || !(after %in% names(df))) return(df)
  others <- setdiff(names(df), vars)
  pos <- match(after, others)
  df[, append(others, vars, after = pos), drop = FALSE]
}

recode_to_character <- function(df, var, map, missing_any = FALSE,
                                missing_codes = NULL, less_than = NULL,
                                missing_label = "N/A") {
  if (!(var %in% names(df))) return(df)
  source <- df[[var]]
  out <- to_stata_string(source)
  for (value in names(map)) {
    out[num_eq(source, as.numeric(value))] <- unname(map[[value]])
  }
  missing <- rep(FALSE, length(out))
  if (missing_any) missing <- missing | is_missing_any(source)
  if (!is.null(missing_codes)) missing <- missing | num_in(source, missing_codes)
  if (!is.null(less_than)) missing <- missing | num_lt(source, less_than)
  out[missing] <- missing_label
  df[[var]] <- out
  df
}

stringify_variable <- function(df, var, missing_any = FALSE,
                               missing_codes = NULL, less_than = NULL,
                               replace_codes = NULL,
                               missing_label = "N/A") {
  if (!(var %in% names(df))) return(df)
  source <- df[[var]]
  numeric_source <- source
  if (!is.null(replace_codes)) {
    for (from in names(replace_codes)) {
      numeric_source[num_eq(numeric_source, as.numeric(from))] <- as.numeric(replace_codes[[from]])
    }
  }
  out <- to_stata_string(numeric_source)
  missing <- rep(FALSE, length(out))
  if (missing_any) missing <- missing | is_missing_any(source)
  if (!is.null(missing_codes)) missing <- missing | num_in(source, missing_codes)
  if (!is.null(less_than)) missing <- missing | num_lt(source, less_than)
  out[missing] <- missing_label
  df[[var]] <- out
  df
}

recode_many_to_character <- function(df, vars, map, missing_any = FALSE,
                                     missing_codes = NULL, less_than = NULL) {
  for (var in vars) {
    df <- recode_to_character(
      df, var, map,
      missing_any = missing_any,
      missing_codes = missing_codes,
      less_than = less_than
    )
  }
  df
}

collapse_any_yes <- function(df, out_var, vars, yes_value = 1) {
  vars <- intersect(vars, names(df))
  if (!length(vars)) return(df)
  any_yes <- Reduce(`|`, lapply(vars, function(v) num_eq(df[[v]], yes_value)))
  all_missing <- Reduce(`&`, lapply(vars, function(v) is_missing_any(df[[v]])))
  out <- ifelse(any_yes, "Yes", "No")
  out[all_missing] <- "N/A"
  df[[out_var]] <- out
  df
}

add_question_row <- function(df, questions) {
  df <- df %>% mutate(across(everything(), as.character))
  header <- as.list(setNames(rep("", ncol(df)), names(df)))
  for (i in seq_along(questions)) {
    var <- names(questions)[i]
    if (var %in% names(header)) header[[var]] <- unname(questions[[i]])
  }
  bind_rows(as_tibble(header), df)
}

# Baseline covariates ---------------------------------------------------------
baseline_keep <- c(
  "household_id",
  "community_id",
  "district",
  "region",
  "urbanrural",
  "b_q5",
  "b_q6_whose_number",
  "b_q6_whose_numbe_ospec",
  "b_q1_1",
  "b_q1_2",
  "b_q1_3",
  "b_q1_4",
  "b_q1_5",
  "b_q1_5_other_relative",
  "b_q1_5_other_nrelative",
  "b_q1_8",
  "b_q1_8_ospecify",
  "b_q1_9",
  "b_q1_11",
  "b_q1_11_ospecify",
  "b_q1_12",
  "b_q1_13",
  "b_q1_14",
  "b_q1_15",
  "b_q1_16",
  "b_q2_1",
  "b_q2_2",
  "b_q2_3",
  "b_q2_4_1",
  "b_q2_4_2",
  "b_q2_4_3",
  "b_q2_4_4",
  "b_q2_4_5",
  "b_q2_4_6",
  "b_q2_4_7",
  "b_q2_4_8",
  "b_q2_4_9",
  "b_q2_4_9_ospecify",
  "b_q2_4_888",
  "b_q2_4_999",
  "b_q2_5",
  "b_q2_5_ospecify",
  "b_q2_6",
  "b_q2_7",
  "b_q2_8",
  "b_q2_8_ospecify",
  "b_q2_9_1",
  "b_q2_9_2",
  "b_q2_9_3",
  "b_q2_9_4",
  "b_q2_9_5",
  "b_q2_9_6",
  "b_q2_9_7",
  "b_q2_9_8",
  "b_q2_9_9",
  "b_q2_9_9_problem",
  "b_q2_9_10",
  "b_q2_9_10_explain",
  "b_q2_9_11",
  "b_q2_9_11_ospecify",
  "b_q2_9_888",
  "b_q2_10",
  "b_q2_11",
  "b_q2_12_1",
  "b_q2_12_2",
  "b_q2_12_3",
  "b_q2_12_4",
  "b_q2_12_5",
  "b_q2_12_6",
  "b_q2_12_7",
  "b_q2_12_8",
  "b_q2_12_9",
  "b_q2_12_10",
  "b_q2_12_11",
  "b_q2_12_12",
  "b_q2_12_13",
  "b_q2_12_14",
  "b_q2_12_15",
  "b_q2_12_16",
  "b_q2_12_16_ospecify",
  "b_q2_12_666",
  "b_q2_12_888",
  "b_q2_13_1",
  "b_q2_13_2",
  "b_q2_13_3",
  "b_q2_13_4",
  "b_q2_13_5",
  "b_q2_13_6",
  "b_q2_13_7",
  "b_q2_13_8",
  "b_q2_13_9",
  "b_q2_13_10",
  "b_q2_13_11",
  "b_q2_13_12",
  "b_q2_13_13",
  "b_q2_14",
  "b_q2_15_hr",
  "b_q2_15_min",
  "b_q2_16",
  "b_q2_16_ospecify",
  "b_q2_17",
  "b_q2_18",
  "b_q3_1",
  "b_q3_2",
  "b_q3_3",
  "b_q3_4",
  "b_q3_5",
  "b_q3_6",
  "b_q4_1_1",
  "b_q4_1_2",
  "b_q4_1_3",
  "b_q4_1_4",
  "b_q4_1_5",
  "b_q4_1_6",
  "b_q4_1_7",
  "b_q4_1_8",
  "b_q4_1_9",
  "b_q4_1_10",
  "b_q4_1_11",
  "b_q4_1_12",
  "b_q4_1_13",
  "b_q4_1_14",
  "b_q4_1_15",
  "b_q4_1_15_ospecify",
  "b_q4_2",
  "b_q4_3_1",
  "b_q4_3_2",
  "b_q4_3_3",
  "b_q4_3_3_ospecify",
  "b_q4_3_4",
  "b_q4_3_888",
  "b_q4_4",
  "b_q4_5",
  "b_q4_6",
  "b_q4_7",
  "b_q4_8",
  "b_q4_9_1",
  "b_q4_9_2",
  "b_q4_9_3",
  "b_q4_9_4",
  "b_q4_9_5",
  "b_q4_9_6",
  "b_q4_9_7",
  "b_q4_9_8",
  "b_q4_9_9",
  "b_q4_9_10",
  "b_q4_9_11",
  "b_q4_9_11_ospecify",
  "b_q4_9_12",
  "b_q4_9_13",
  "b_q4_9_14",
  "b_q4_9_666",
  "b_q4_9_888",
  "b_q4_10_1",
  "b_q4_10_2",
  "b_q4_10_3",
  "b_q4_10_4",
  "b_q4_10_5",
  "b_q4_10_6",
  "b_q4_10_7",
  "b_q4_10_8",
  "b_q4_10_9",
  "b_q4_10_10",
  "b_q4_10_11",
  "b_q4_10_11_ospecify",
  "b_q4_10_12",
  "b_q4_10_13",
  "b_q4_10_666",
  "b_q4_10_888",
  "b_q4_11",
  "b_q4_12",
  "b_q4_13_1",
  "b_q4_13_2",
  "b_q4_13_3",
  "b_q4_13_4",
  "b_q4_13_5",
  "b_q4_13_6",
  "b_q4_13_7",
  "b_q4_13_8",
  "b_q4_13_8_ospecify",
  "b_q4_13_888",
  "b_q4_13_999",
  "b_q4_14_1",
  "b_q4_14_2",
  "b_q4_14_3",
  "b_q4_14_4",
  "b_q4_14_5",
  "b_q4_14_6",
  "b_q4_14_6_ospecify",
  "b_q4_14_7",
  "b_q4_14_888",
  "b_q4_15",
  "b_q4_16",
  "b_q4_17",
  "b_q4_18",
  "b_q4_19_1",
  "b_q4_19_2",
  "b_q4_19_3",
  "b_q4_19_4",
  "b_q4_19_5",
  "b_q4_19_6",
  "b_q4_19_7",
  "b_q4_19_8",
  "b_q4_19_9",
  "b_q4_19_9_ospecify",
  "b_q4_19_10",
  "b_q4_19_888",
  "b_q4_20_1",
  "b_q4_20_2",
  "b_q4_20_3",
  "b_q4_20_4",
  "b_q4_20_5",
  "b_q4_20_6",
  "b_q4_20_7",
  "b_q4_20_8",
  "b_q4_20_8_ospecify",
  "b_q4_20_888",
  "b_q4_21",
  "b_q4_22",
  "b_q4_23",
  "b_q4_24",
  "b_q4_25_1",
  "b_q4_25_2",
  "b_q4_25_3",
  "b_q4_25_4",
  "b_q4_25_5",
  "b_q4_25_5_ospecify",
  "b_q4_25_6",
  "b_q4_25_7",
  "b_q4_25_888",
  "b_q4_26_1",
  "b_q4_26_2",
  "b_q4_26_3",
  "b_q4_26_4",
  "b_q4_26_4_ospecify",
  "b_q4_26_5",
  "b_q4_26_888",
  "b_q4_27_1",
  "b_q4_27_2",
  "b_q4_27_3",
  "b_q4_27_4",
  "b_q4_27_5",
  "b_q4_27_6",
  "b_q4_27_7",
  "b_q4_27_8",
  "b_q4_27_9",
  "b_q4_27_10",
  "b_q4_27_11",
  "b_q4_27_12",
  "b_q4_27_13",
  "b_q4_27_14",
  "b_q4_27_15",
  "b_q4_27_16",
  "b_q4_27_16_ospecify",
  "b_q4_27_666",
  "b_q4_27_888",
  "b_q5_1",
  "b_q5_2",
  "b_q5_2_ospecify",
  "b_q5_3",
  "b_q5_4",
  "b_q5_5",
  "b_q5_5_ospecify",
  "b_q5_6",
  "b_q5_6_ospecify",
  "b_q5_7",
  "b_q5_8",
  "b_q5_8_ospecify",
  "b_q5_9",
  "b_q5_10_1",
  "b_q5_10_2",
  "b_q5_10_3",
  "b_q5_10_4",
  "b_q5_10_5",
  "b_q5_10_5_ospecify",
  "b_q5_10_6",
  "b_q5_10_888",
  "b_q5_11_1",
  "b_q5_11_2",
  "b_q5_11_3",
  "b_q5_11_4",
  "b_q5_11_5",
  "b_q5_11_6",
  "b_q5_11_7",
  "b_q5_11_8",
  "b_q5_11_9",
  "b_q5_11_10",
  "b_q5_11_11",
  "b_q5_11_11_ospecify",
  "b_q5_11_12",
  "b_q5_11_888",
  "b_q5_12",
  "b_q5_13_1",
  "b_q5_13_2",
  "b_q5_13_3",
  "b_q5_13_4",
  "b_q5_13_5",
  "b_q5_13_6",
  "b_q5_13_7",
  "b_q5_13_8",
  "b_q5_13_9",
  "b_q5_13_10",
  "b_q5_13_10_ospecify",
  "b_q5_14_1",
  "b_q5_14_2",
  "b_q5_14_3",
  "b_q5_14_4",
  "b_q5_14_5",
  "b_q5_14_6",
  "b_q5_14_7",
  "b_q5_14_8",
  "b_q5_14_9",
  "b_q5_14_9_ospecify",
  "b_q5_15",
  "b_q5_16",
  "b_q5_17",
  "b_q5_18",
  "b_q5_19_1",
  "b_q5_19_2",
  "b_q5_19_3",
  "b_q5_19_4",
  "b_q5_19_5",
  "b_q5_19_6",
  "b_q5_19_7",
  "b_q5_19_7_ospecify",
  "b_q5_20",
  "b_q5_21",
  "b_q5_22_1",
  "b_q5_22_2",
  "b_q5_22_3",
  "b_q5_22_4",
  "b_q5_22_5",
  "b_q5_22_6",
  "b_q5_22_7",
  "b_q5_22_8",
  "b_q5_22_9",
  "b_q5_22_10",
  "b_q5_22_11",
  "b_q5_22_12",
  "b_q5_22_13",
  "b_q5_22_14",
  "b_q5_22_15",
  "b_q5_22_16",
  "b_q5_22_16_ospecify",
  "b_q5_22_666",
  "b_q5_22_888",
  "b_q6_1_1",
  "b_q6_1_2",
  "b_q6_1_3",
  "b_q6_1_4",
  "b_q6_1_5",
  "b_q6_1_6",
  "b_q6_1_7",
  "b_q6_1_8",
  "b_q6_1_9",
  "b_q6_1_10",
  "b_q6_1_11",
  "b_q6_1_12",
  "b_q6_1_13",
  "b_q6_1_14",
  "b_q6_1_15",
  "b_q6_1_16",
  "b_q6_1_16_ospecify",
  "b_q6_1_18",
  "b_q6_1_19",
  "b_q6_1_20",
  "b_q6_1_666",
  "b_q6_2",
  "b_q6_3",
  "b_q6_3_ospecify",
  "b_q6_4",
  "b_q6_5",
  "b_q6_6",
  "b_q6_7_1",
  "b_q6_7_2",
  "b_q6_7_2_lstools",
  "b_q6_7_3",
  "b_q6_7_4",
  "b_q6_7_5",
  "b_q6_7_6",
  "b_q6_7_7",
  "b_q6_7_8",
  "b_q6_7_9",
  "b_q6_7_10",
  "b_q6_7_10_ospecify",
  "b_q6_8",
  "b_q6_9",
  "b_q6_10_1",
  "b_q6_10_2",
  "b_q6_10_3",
  "b_q6_10_4",
  "b_q6_10_5",
  "b_q6_10_6",
  "b_q6_10_7",
  "b_q6_10_8",
  "b_q6_10_9",
  "b_q6_10_10",
  "b_q6_10_10_ospecify",
  "b_q6_10_11",
  "b_q6_10_888",
  "b_q6_11_1",
  "b_q6_11_2",
  "b_q6_11_3",
  "b_q6_11_4",
  "b_q6_11_5",
  "b_q6_11_6",
  "b_q6_11_7",
  "b_q6_11_8",
  "b_q6_11_9",
  "b_q6_11_10",
  "b_q6_11_11",
  "b_q6_11_12",
  "b_q6_11_13",
  "b_q6_11_14",
  "b_q6_11_15",
  "b_q6_11_16",
  "b_q6_11_17",
  "b_q6_11_18",
  "b_q6_11_18_ospecify",
  "b_q6_11_19",
  "b_q6_11_888",
  "b_q6_12",
  "b_q6_13",
  "b_q6_13_num_stools",
  "b_q6_13_ospecify",
  "b_q6_14",
  "b_q6_14_ospecify",
  "b_q6_15",
  "b_q6_16",
  "b_q6_16_ospecify",
  "b_q6_17",
  "b_q6_18_1",
  "b_q6_18_2",
  "b_q6_18_3",
  "b_q6_18_4",
  "b_q6_18_5",
  "b_q6_18_6",
  "b_q6_18_6_ospecify",
  "b_q6_19",
  "b_q6_20",
  "b_q6_21",
  "b_q6_22",
  "b_q6_23",
  "b_q6_24_1",
  "b_q6_24_2",
  "b_q6_24_3",
  "b_q6_24_4",
  "b_q6_24_5",
  "b_q6_24_6",
  "b_q6_24_7",
  "b_q6_24_8",
  "b_q6_24_9",
  "b_q6_24_10",
  "b_q6_24_11",
  "b_q6_24_12",
  "b_q6_24_13",
  "b_q6_24_14",
  "b_q6_24_15",
  "b_q6_24_16",
  "b_q6_24_16_ospecify",
  "b_q6_24_666",
  "b_q6_24_888",
  "b_q7_1",
  "b_q7_2",
  "b_q7_3",
  "b_q7_4",
  "b_q7_5",
  "b_q7_6",
  "b_q7_7",
  "b_q7_8",
  "b_q7_9",
  "b_q7_10",
  "b_q7_10_ospecify",
  "b_q7_11",
  "b_q7_12",
  "b_q7_13",
  "b_q7_14",
  "b_q7_15",
  "b_q7_16",
  "b_q7_17",
  "b_q7_18",
  "b_q7_18_ospecify",
  "b_q7_19",
  "b_q7_20_1",
  "b_q7_20_2",
  "b_q7_20_3",
  "b_q7_20_4",
  "b_q7_20_5",
  "b_q7_20_6",
  "b_q7_20_7",
  "b_q7_20_8",
  "b_q7_20_9",
  "b_q7_20_9_ospecify",
  "b_q7_20_888",
  "b_q7_21",
  "b_q7_22_i",
  "b_q7_23",
  "b_q7_24",
  "b_q7_25_1",
  "b_q7_25_2",
  "b_q7_25_3",
  "b_q7_25_4",
  "b_q7_25_5",
  "b_q7_25_6",
  "b_q7_25_7",
  "b_q7_25_8",
  "b_q7_25_9",
  "b_q7_25_10",
  "b_q7_25_10_ospecify",
  "b_q7_25_888",
  "b_q7_26_1",
  "b_q7_26_2",
  "b_q7_26_3",
  "b_q7_26_4",
  "b_q7_26_5",
  "b_q7_26_6",
  "b_q7_26_7",
  "b_q7_26_8",
  "b_q7_26_9",
  "b_q7_26_10",
  "b_q7_26_11",
  "b_q7_26_12",
  "b_q7_26_13",
  "b_q7_26_14",
  "b_q7_26_15",
  "b_q7_26_16",
  "b_q7_26_16_ospecify",
  "b_q7_26_666",
  "b_q7_26_888",
  "b_q8_1",
  "b_q8_1_ospecify",
  "b_q8_2",
  "b_q8_2_ospecify",
  "b_q8_3_1",
  "b_q8_3_2",
  "b_q8_3_3",
  "b_q8_3_4",
  "b_q8_3_5",
  "b_q8_3_6",
  "b_q8_3_7",
  "b_q8_3_8",
  "b_q8_3_8_ospecify",
  "b_q8_3_888",
  "b_q8_4_1",
  "b_q8_4_2",
  "b_q8_4_3",
  "b_q8_4_4",
  "b_q8_4_5",
  "b_q8_4_6",
  "b_q8_4_6_ospecify",
  "b_q8_4_888",
  "b_q8_4_7",
  "b_q8_4_8",
  "b_q8_5",
  "b_q8_5_ospecify",
  "b_q8_6",
  "b_q8_6_ospecify",
  "b_q8_7",
  "b_q8_7_ospecify",
  "b_q8_8_1",
  "b_q8_8_2",
  "b_q8_8_3",
  "b_q8_8_4",
  "b_q8_8_5",
  "b_q8_8_6",
  "b_q8_8_7",
  "b_q8_8_8",
  "b_q8_8_8_ospecify",
  "b_q8_9",
  "b_q8_10",
  "b_q8_11",
  "b_q8_12",
  "b_q8_13",
  "b_q8_14_1",
  "b_q8_14_2",
  "b_q8_14_3",
  "b_q8_14_4",
  "b_q8_14_5",
  "b_q8_14_6_ospecify",
  "b_q8_14_888",
  "b_q8_14_6",
  "b_q8_15",
  "b_q8_16",
  "b_q8_17_1",
  "b_q8_17_2",
  "b_q8_17_3",
  "b_q8_17_4",
  "b_q8_17_5",
  "b_q8_17_6",
  "b_q8_17_7",
  "b_q8_17_8",
  "b_q8_17_9",
  "b_q8_17_10",
  "b_q8_17_10_ospecify",
  "b_q8_17_11",
  "b_q8_17_12",
  "b_q8_18_1",
  "b_q8_18_2",
  "b_q8_18_3",
  "b_q8_18_4",
  "b_q8_18_5",
  "b_q8_18_6",
  "b_q8_18_7",
  "b_q8_18_8",
  "b_q8_18_9",
  "b_q8_18_10",
  "b_q8_18_10_ospecify",
  "b_q8_18_11",
  "b_q8_18_12",
  "b_q8_19",
  "b_q8_20",
  "b_q8_21",
  "b_q8_22_1",
  "b_q8_22_2",
  "b_q8_22_3",
  "b_q8_22_4",
  "b_q8_22_5",
  "b_q8_22_6",
  "b_q8_22_7",
  "b_q8_22_8",
  "b_q8_22_9",
  "b_q8_22_10",
  "b_q8_22_11",
  "b_q8_22_12",
  "b_q8_22_13",
  "b_q8_22_14",
  "b_q8_22_15",
  "b_q8_22_16",
  "b_q8_22_16_ospecify",
  "b_q8_22_666",
  "b_q8_22_888",
  NULL
)

individual_clean <- read_dta(baseline_file) %>% haven::zap_labels()
individual_clean <- individual_clean %>% select(all_of(baseline_keep))
individual_clean <- drop_any(individual_clean, "community_id")

# Demographics and baseline covariates ---------------------------------------
individual_clean <- stringify_variable(individual_clean, "b_q1_1", missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q1_2", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q1_3", c(
  "1" = "Neighboring community",
  "2" = "Other Northern Region",
  "3" = "Other Upper East",
  "4" = "Other Upper West",
  "5" = "Accra",
  "6" = "Tamale",
  "7" = "Wa",
  "8" = "Bolgatanga",
  "9" = "Other Ghana",
  "10" = "Cote d'Ivoire",
  "11" = "Burkina Faso",
  "13" = "Other African country"
), missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q1_4", missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q1_5", c(
  "1" = "Spouse", "2" = "Respondent", "3" = "Father or Mother",
  "4" = "Aunt or Uncle", "5" = "Grandparent", "6" = "Friend",
  "7" = "Employer", "8" = "Other family relative",
  "9" = "Other non-family relative"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q1_5_other_relative", "b_q1_5_other_nrelative"))

religion_map <- c(
  "1" = "Catholic", "2" = "Anglican", "3" = "Presbyterian",
  "4" = "Methodist", "5" = "Pentecostals", "6" = "Other Christian",
  "7" = "Spiritualist", "8" = "Muslim", "9" = "Traditional",
  "10" = "No Religion", "11" = "Other non-Christian"
)
individual_clean <- recode_to_character(individual_clean, "b_q1_8", religion_map, missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q1_8_ospecify")
individual_clean <- recode_to_character(individual_clean, "b_q1_9", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q1_11", religion_map, missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q1_11_ospecify")

education_map <- c(
  "1" = "None", "2" = "Some primary", "3" = "Completed primary",
  "4" = "Some Junior Secondary", "5" = "Completed Junior High School",
  "6" = "Some Senior High School", "7" = "Completed Senior High School",
  "8" = "Some Tertiary", "9" = "Completed Tertiary"
)
individual_clean <- recode_to_character(individual_clean, "b_q1_12", education_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q1_13", education_map, missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q1_14", missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q1_15", missing_any = TRUE, less_than = 0)
individual_clean <- recode_to_character(individual_clean, "b_q1_16", c(
  "1" = "Cannot read",
  "2" = "Can read letters",
  "3" = "Can read part(s) of the sentence",
  "4" = "Able to read whole sentence",
  "6" = "Blind/mute or visually impaired"
), missing_any = TRUE)

# Health information and CBA variables ---------------------------------------
individual_clean <- drop_any(individual_clean, c("b_q2_1", "b_q2_2"))
individual_clean <- recode_to_character(individual_clean, "b_q2_3", c("1" = "Yes", "2" = "No"), missing_any = TRUE)

info_vars <- paste0("b_q2_4_", 1:9)
individual_clean <- collapse_any_yes(individual_clean, "info", info_vars)
individual_clean <- drop_any(individual_clean, c(info_vars, "b_q2_4_9_ospecify", "b_q2_4_888", "b_q2_4_999"))

individual_clean <- recode_to_character(individual_clean, "b_q2_5", c(
  "1" = "Heard something about it somewhere",
  "2" = "Thought it was a good idea",
  "3" = "Husband encouraged",
  "4" = "Other family/friends encouraged",
  "5" = "Pregnant",
  "6" = "Respondent or child was sick",
  "7" = "Other"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q2_5_ospecify")
individual_clean <- recode_to_character(individual_clean, "b_q2_6", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q2_7", c(
  "1" = "Don't know at all", "2" = "Know somewhat",
  "3" = "Know a little", "4" = "Know quite a bit",
  "5" = "Know extremely well"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q2_8", c(
  "1" = "Today", "2" = "Within a week", "3" = "Within a month",
  "4" = "Within a year", "5" = "Never", "6" = "Other"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q2_8_ospecify")

cba_vars <- paste0("b_q2_9_", 1:11)
individual_clean <- collapse_any_yes(individual_clean, "cba_talk", cba_vars)
individual_clean <- drop_any(individual_clean, c(
  cba_vars, "b_q2_9_9_problem", "b_q2_9_10_explain",
  "b_q2_9_11_ospecify", "b_q2_9_888"
))

freq_map <- c("1" = "Never", "2" = "Occasionally", "3" = "Sometimes", "4" = "Usually", "5" = "Always")
individual_clean <- recode_to_character(individual_clean, "b_q2_10", freq_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q2_11", freq_map, missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q2_12_", 1:16), "b_q2_12_666", "b_q2_12_16_ospecify", "b_q2_12_888"))
individual_clean <- recode_many_to_character(individual_clean, paste0("b_q2_13_", 1:13), c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q2_14", c(
  "1" = "On Foot", "2" = "Bicycle", "3" = "Cart", "4" = "Motorcycle",
  "5" = "Truck/tractor", "6" = "Car/taxi", "7" = "Tro Tro", "8" = "Bus"
), missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q2_15_hr", missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q2_15_min")
individual_clean <- recode_to_character(individual_clean, "b_q2_16", c(
  "1" = "Within the last week",
  "2" = "Within the last month",
  "3" = "Within the last 6 months",
  "4" = "Within the last year",
  "5" = "Other",
  "6" = "Never"
), missing_any = TRUE, missing_codes = 888)
individual_clean <- drop_any(individual_clean, "b_q2_16_ospecify")
individual_clean <- recode_to_character(individual_clean, "b_q2_17", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q2_18", missing_any = TRUE)
individual_clean <- drop_any(individual_clean, paste0("b_q3_", 1:6))

# Malaria and insecticide-treated net variables ------------------------------
malaria_vars <- paste0("b_q4_1_", 1:15)
individual_clean <- collapse_any_yes(individual_clean, "malaria_prev", malaria_vars)
individual_clean <- drop_any(individual_clean, c(malaria_vars, "b_q4_1_15_ospecify"))
individual_clean <- recode_to_character(individual_clean, "b_q4_2", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q4_3_1", "b_q4_3_2", "b_q4_3_3", "b_q4_3_3_ospecify", "b_q4_3_4", "b_q4_3_888"))
individual_clean <- recode_to_character(individual_clean, "b_q4_4", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_5", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_6", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- stringify_variable(individual_clean, "b_q4_7", missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_8", freq_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_9_666", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q4_9_", 1:14), "b_q4_9_11_ospecify", "b_q4_9_888"))
individual_clean <- recode_to_character(individual_clean, "b_q4_10_666", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q4_10_", 1:13), "b_q4_10_11_ospecify", "b_q4_10_888"))
individual_clean <- recode_to_character(individual_clean, "b_q4_11", c(
  "1" = "Extremely unimportant",
  "2" = "Unimportant",
  "3" = "Neither serious nor unimportant",
  "4" = "Serious",
  "5" = "Extremely serious"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q4_12", paste0("b_q4_13_", 1:8), "b_q4_13_8_ospecify", "b_q4_13_888", "b_q4_13_999"))
individual_clean <- recode_many_to_character(individual_clean, paste0("b_q4_14_", 1:7), c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q4_14_6_ospecify", "b_q4_14_888"))
individual_clean <- recode_to_character(individual_clean, "b_q4_15", c(
  "1" = "Not likely at all", "2" = "Somewhat likely", "3" = "A little likely",
  "4" = "Likely", "5" = "Extremely likely"
), missing_any = TRUE)
individual_clean <- recode_many_to_character(individual_clean, c("b_q4_16", "b_q4_17", "b_q4_18"), c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_many_to_character(
  individual_clean,
  c(paste0("b_q4_19_", 1:10), paste0("b_q4_20_", 1:8)),
  c("1" = "Yes", "0" = "No"), missing_any = TRUE
)
individual_clean <- drop_any(individual_clean, c("b_q4_19_9_ospecify", "b_q4_19_888", "b_q4_20_8_ospecify", "b_q4_20_888"))
friends_map <- c("1" = "None", "2" = "A few", "3" = "Some", "4" = "A lot", "5" = "All")
individual_clean <- recode_to_character(individual_clean, "b_q4_21", friends_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_22", friends_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_23", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q4_24", c("1" = "Yes", "2" = "No"), missing_any = TRUE, missing_codes = -777)
individual_clean <- recode_to_character(individual_clean, "b_q4_25_888", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q4_25_1", "b_q4_25_2", "b_q4_25_3", "b_q4_25_4", "b_q4_25_5", "b_q4_25_5_ospecify", "b_q4_25_6", "b_q4_25_7"))
individual_clean <- recode_to_character(individual_clean, "b_q4_26_888", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c("b_q4_26_1", "b_q4_26_2", "b_q4_26_3", "b_q4_26_4", "b_q4_26_4_ospecify", "b_q4_26_5"))
individual_clean <- recode_to_character(individual_clean, "b_q4_27_666", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q4_27_", 1:16), "b_q4_27_16_ospecify", "b_q4_27_888"))

# Handwashing and soap variables ---------------------------------------------
individual_clean <- recode_to_character(individual_clean, "b_q5_1", freq_map, missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q5_2", c(
  "0" = "No", "1" = "Water only", "2" = "Soap and water", "3" = "Ash"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q5_2_ospecify")
individual_clean <- stringify_variable(individual_clean, "b_q5_4", missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q5_5", c(
  "1" = "Yes, with water", "2" = "Yes, with soap and water", "3" = "No", "4" = "Other"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q5_5_ospecify")
individual_clean <- recode_to_character(individual_clean, "b_q5_6", c(
  "1" = "Yes, with water", "2" = "Yes, with soap and water", "3" = "No", "4" = "Other"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q5_6_ospecify")
individual_clean <- drop_any(individual_clean, c("b_q5_7", "b_q5_8", "b_q5_8_ospecify"))
individual_clean <- recode_to_character(individual_clean, "b_q5_9", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_many_to_character(
  individual_clean,
  c(paste0("b_q5_10_", 1:6), "b_q5_10_888", paste0("b_q5_11_", 1:12), "b_q5_11_888"),
  c("1" = "Yes", "0" = "No"), missing_any = TRUE
)
individual_clean <- drop_any(individual_clean, c("b_q5_10_5_ospecify", "b_q5_11_11_ospecify"))
individual_clean <- recode_to_character(individual_clean, "b_q5_12", c(
  "1" = "Never", "2" = "A little of the time", "3" = "Some of the time",
  "4" = "Most of the time", "5" = "All the time"
), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q5_13_", 1:10), "b_q5_13_10_ospecify"))
clean_imp_vars <- paste0("b_q5_14_", 1:9)
individual_clean <- collapse_any_yes(individual_clean, "clean_imp", clean_imp_vars)
individual_clean <- drop_any(individual_clean, c(clean_imp_vars, "b_q5_14_9_ospecify", "b_q5_16", "b_q5_17", "b_q5_18"))
individual_clean <- recode_to_character(individual_clean, "b_q5_15", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
encourage_vars <- paste0("b_q5_19_", 1:7)
individual_clean <- collapse_any_yes(individual_clean, "encourage", encourage_vars)
individual_clean <- drop_any(individual_clean, encourage_vars)
individual_clean <- recode_to_character(individual_clean, "b_q5_20", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q5_21", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q5_22_666", c("1" = "Yes", "0" = "No"), missing_any = FALSE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q5_22_", 1:16), "b_q5_22_16_ospecify", "b_q5_22_888"))

# Diarrhea and ORS variables --------------------------------------------------
individual_clean <- recode_to_character(individual_clean, "b_q6_1_666", c("1" = "Yes", "0" = "No"), missing_any = FALSE)
individual_clean <- drop_any(individual_clean, c(paste0("b_q6_1_", 1:16), "b_q6_1_16_ospecify", "b_q6_1_18", "b_q6_1_19", "b_q6_1_20"))
individual_clean <- recode_to_character(individual_clean, "b_q6_2", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_3", c(
  "1" = "Cost", "2" = "Didn't know where to buy", "3" = "Child wasn't sick enough",
  "4" = "Child never had diarrhea", "5" = "Didn't have the ingredients",
  "6" = "Prefer alternative treatment", "7" = "Other"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_4", c(
  "1" = "Yes, near here", "2" = "Yes, but it is far away", "3" = "No, I don't know a place"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_5", c("1" = "Made at home", "2" = "Purchased packet"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_6", c(
  "1" = "Less than 1 day", "2" = "2 days", "3" = "3 days", "4" = "4 days", "5" = "5 or more days"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_8", c(
  "1" = "Extremely unimportant", "2" = "A little unimportant",
  "3" = "Neither serious nor unimportant", "4" = "Serious", "5" = "Extremely serious"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_9", c("1" = "Yes", "2" = "No"), missing_any = TRUE, missing_codes = -777)
individual_clean <- recode_to_character(individual_clean, "b_q6_10_888", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_11_888", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_12", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, "b_q6_13")
individual_clean <- recode_to_character(individual_clean, "b_q6_14", c(
  "1" = "Older than 5 years", "2" = "Older than 1 year", "3" = "Older than 6 months",
  "4" = "Any time", "5" = "Other"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_15", c("1" = "Yes", "2" = "No"), missing_any = FALSE)
individual_clean <- drop_any(individual_clean, "b_q6_16")
individual_clean <- recode_to_character(individual_clean, "b_q6_18_5", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_22", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_21", c(
  "1" = "Extremely unlikely", "2" = "Somewhat unlikely", "3" = "Neither likely nor unlikely",
  "4" = "Somewhat likely", "5" = "Extremely likely"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_23", c("1" = "Yes", "2" = "No"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_24_666", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- drop_any(individual_clean, c(
  "b_q6_3_ospecify", "b_q6_7_1", "b_q6_7_2", "b_q6_7_2_lstools", "b_q6_7_3", "b_q6_7_4", "b_q6_7_5", "b_q6_7_6", "b_q6_7_7", "b_q6_7_8", "b_q6_7_9", "b_q6_7_10", "b_q6_7_10_ospecify",
  paste0("b_q6_10_", 1:11), "b_q6_10_10_ospecify",
  paste0("b_q6_11_", 1:19), "b_q6_11_18_ospecify",
  "b_q6_13_num_stools", "b_q6_13_ospecify", "b_q6_14_ospecify", "b_q6_16_ospecify", "b_q6_17",
  "b_q6_18_1", "b_q6_18_2", "b_q6_18_3", "b_q6_18_4", "b_q6_18_6", "b_q6_18_6_ospecify", "b_q6_19", "b_q6_20",
  paste0("b_q6_24_", 1:16), "b_q6_24_16_ospecify", "b_q6_24_888"
))

# Breastfeeding and skilled-birth-attendant variables ------------------------
individual_clean <- recode_many_to_character(individual_clean, c("b_q7_1", "b_q7_2", "b_q7_4", "b_q7_6"), c("1" = "Yes", "2" = "No"), missing_any = TRUE, less_than = 0)
individual_clean <- stringify_variable(individual_clean, "b_q7_7", missing_any = TRUE, less_than = 0)
individual_clean <- stringify_variable(individual_clean, "b_q7_8", missing_any = TRUE, replace_codes = c("-777" = "0"))
individual_clean <- stringify_variable(individual_clean, "b_q7_9", missing_any = TRUE, replace_codes = c("-777" = "0"))
individual_clean <- stringify_variable(individual_clean, "b_q7_18", missing_any = TRUE, less_than = 0)
individual_clean <- stringify_variable(individual_clean, "b_q7_17", missing_any = TRUE, less_than = 0)
individual_clean <- recode_to_character(individual_clean, "b_q7_19", c(
  "1" = "Extremely bad", "2" = "Somewhat bad", "3" = "Neither good or bad",
  "4" = "Somewhat good", "5" = "Extremely good"
), missing_any = TRUE, less_than = 0)
individual_clean <- recode_to_character(individual_clean, "b_q7_20_888", c("1" = "Yes", "0" = "No"), missing_any = TRUE)
individual_clean <- recode_many_to_character(individual_clean, c("b_q7_21", "b_q7_23"), c("1" = "Yes", "2" = "No"), missing_any = TRUE, less_than = 0)
individual_clean <- drop_any(individual_clean, "b_q7_22_i")
individual_clean <- recode_to_character(individual_clean, "b_q7_24", friends_map, missing_any = TRUE)
individual_clean <- recode_many_to_character(individual_clean, c("b_q7_25_9", "b_q7_26_666"), c("1" = "Yes", "0" = "No"), missing_any = TRUE, less_than = 0)

individual_clean <- recode_to_character(individual_clean, "b_q8_1", c(
  "1" = "Doctor", "2" = "Nurse", "3" = "Midwife/Nurse-midwife",
  "4" = "Traditional birth attendant", "5" = "Community health worker/CBA",
  "6" = "Relative/Friends", "7" = "Other", "8" = "No one"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q8_2", c("1" = "At home", "2" = "In a hospital/clinic/facility", "3" = "Other"), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q8_5", c(
  "1" = "At home", "2" = "In a health facility", "3" = "Other", "4" = "Anywhere", "5" = "Up to God"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q8_6", c(
  "1" = "Female relatives", "2" = "Husband", "3" = "Traditional birth attendant", "4" = "Midwife",
  "5" = "Doctor", "6" = "Nurse", "7" = "No one", "8" = "Other", "9" = "CBA"
), missing_any = TRUE, missing_codes = -777)
individual_clean <- recode_to_character(individual_clean, "b_q8_7", c("1" = "At home", "2" = "In a health facility", "3" = "Other"), missing_any = TRUE)
individual_clean <- recode_many_to_character(individual_clean, c("b_q8_10", "b_q8_11", "b_q8_13", "b_q8_19"), c("1" = "Yes", "2" = "No"), missing_any = TRUE, less_than = 0)
individual_clean <- stringify_variable(individual_clean, "b_q8_12", missing_any = TRUE)
individual_clean <- recode_many_to_character(individual_clean, c("b_q8_18_1", "b_q8_22_666"), c("1" = "Yes", "0" = "No"), missing_any = TRUE, less_than = 0)

individual_clean <- recode_to_character(individual_clean, "b_q5", c(
  "0" = "No access to a mobile phone", "1" = "Gave a number", "2" = "Refused to share number"
), missing_any = TRUE)
individual_clean <- recode_to_character(individual_clean, "b_q6_whose_number", c(
  "1" = "Respondent's personal phoen",
  "2" = "Immediate family member, within community",
  "3" = "Immediate family member, not living in the community",
  "4" = "Someone else's living within community",
  "5" = "Someone else's living near the community",
  "6" = "Respondent's active SIM for use in some cases",
  "7" = "Other"
), missing_any = TRUE, missing_codes = -777)
individual_clean <- drop_any(individual_clean, "b_q6_whose_numbe_ospec")
individual_clean <- drop_any(individual_clean, c(
  "b_q7_3", "b_q7_5", "b_q7_10", "b_q7_10_ospecify", "b_q7_11", "b_q7_12", "b_q7_13", "b_q7_14", "b_q7_15", "b_q7_16", "b_q7_18_ospecify",
  paste0("b_q7_20_", 1:9), "b_q7_20_9_ospecify", "b_q7_24", paste0("b_q7_25_", 1:8), "b_q7_25_10", "b_q7_25_10_ospecify", "b_q7_25_888",
  paste0("b_q7_26_", 1:16), "b_q7_26_16_ospecify", "b_q7_26_888",
  "b_q8_1_ospecify", "b_q8_2_ospecify", paste0("b_q8_3_", 1:8), "b_q8_3_8_ospecify", "b_q8_3_888",
  paste0("b_q8_4_", 1:8), "b_q8_4_6_ospecify", "b_q8_4_888", "b_q8_6_ospecify", "b_q8_5_ospecify", "b_q8_7_ospecify",
  paste0("b_q8_8_", 1:8), "b_q8_8_8_ospecify", "b_q8_9", paste0("b_q8_14_", 1:6), "b_q8_14_6_ospecify", "b_q8_14_888", "b_q8_15", "b_q8_16",
  paste0("b_q8_17_", 1:12), "b_q8_17_10_ospecify", paste0("b_q8_18_", 2:12), "b_q8_18_10_ospecify", "b_q8_20", "b_q8_21",
  paste0("b_q8_22_", 1:16), "b_q8_22_16_ospecify", "b_q8_22_888", "b_q5_3"
))

# Outcome dataset -------------------------------------------------------------
outcome <- read_dta(outcome_file) %>% haven::zap_labels()
if (!("m4d_treatment" %in% names(outcome)) && "m4d_treat" %in% names(outcome)) {
  outcome <- outcome %>% rename(m4d_treatment = m4d_treat)
}
outcome_keep <- c(
  "hhid", "hh_comm_id", "hh_dis_id", "hh_reg_id", "treatment", "live", "video",
  "comradio", "m4d_treatment", "b_child_used_itn", "m_child_used_itn",
  "child_used_itn", "b_mom_used_itn", "m_mom_used_itn", "mom_used_itn",
  "b_sba", "m_sba", "sba", "b_exclusive", "m_exclusive", "exclusive",
  "m_exclusive2", "b_exclusive2", "exclusive2", "b_ors_clinic",
  "m_ors_clinic", "ors_clinic", "b_soap_observed", "m_soap_observed",
  "soap_observed"
)
outcome <- outcome %>% select(all_of(outcome_keep))

treat_source <- outcome$treatment
outcome$treatment <- to_stata_string(treat_source)
outcome$treatment[num_eq(treat_source, 1) & num_eq(outcome$live, 1)] <- "Live drama shows by Center for National Culture"
outcome$treatment[num_eq(treat_source, 1) & num_eq(outcome$video, 1)] <- "Video by Center for National Culture"
outcome$treatment[num_eq(treat_source, 0)] <- "No"
outcome <- drop_any(outcome, c("live", "video"))
outcome <- recode_to_character(outcome, "comradio", c("1" = "Yes", "0" = "No"), missing_any = FALSE)

for (var in c("b_mom_used_itn", "m_mom_used_itn", "mom_used_itn")) {
  outcome <- recode_to_character(outcome, var, c("1" = "Yes, I slept under the net", "0" = "No, I do not own a net"), missing_any = TRUE)
}
for (var in c("b_child_used_itn", "m_child_used_itn", "child_used_itn")) {
  outcome <- recode_to_character(outcome, var, c("1" = "Yes, my youngest child slept under the net", "0" = "No, we do not have a net in the household"), missing_any = TRUE)
}
for (var in c("b_sba", "m_sba", "sba", "b_exclusive", "m_exclusive", "exclusive", "b_exclusive2", "m_exclusive2", "exclusive2", "b_ors_clinic", "m_ors_clinic", "ors_clinic", "b_soap_observed", "m_soap_observed", "soap_observed")) {
  outcome <- recode_to_character(outcome, var, c("1" = "Yes", "0" = "No"), missing_any = TRUE)
}
outcome <- recode_to_character(outcome, "m4d_treatment", c(
  "0" = "Did not receive phone calls",
  "1" = "Always called by the same person",
  "2" = "Called by different people"
), missing_any = TRUE)
outcome <- move_after(outcome, "m4d_treatment", "comradio")

# Merge individual-level and outcome datasets --------------------------------
outcome <- outcome %>% rename(household_id = hhid)
dat <- full_join(outcome, individual_clean, by = "household_id")

# Question row/codebook -------------------------------------------------------
dat <- drop_any(dat, c("hh_dis_id", "hh_reg_id"))
questions <- c(
  household_id = "Household ID",
  hh_comm_id = "Community ID",
  treatment = "In your community, did the Centre for National Culture showed a video or a live drama about health behaviors (e.g., washing hand, exclusive breastfeeding, ORS)?",
  comradio = "Did your community radio broadcast programs about health behaviors?",
  m4d_treatment = "Did you received phone calls from healthcare workers informing you about health behaviours? If so, was the person who called always the same?",
  b_mom_used_itn = "In 2012: if you own an insecticide-treated net, did you sleep under it last night?",
  m_mom_used_itn = "In 2014: if you own an insecticide-treated net, did you sleep under it last night?",
  mom_used_itn = "In 2016: if you own an insecticide-treated net, did you sleep under it last night?",
  b_child_used_itn = "In 2012: if your household owns an insecticide-treated net, did your youngest child sleep under it last night?",
  m_child_used_itn = "In 2014: if your household owns an insecticide-treated net, did your youngest child sleep under it last night?",
  child_used_itn = "In 2016: if your household owns an insecticide-treated net, did your youngest child sleep under it last night?",
  b_sba = "In 2012: were you assisted by someone (e.g., a doctor, nurse, midwife, SBA, or community health worker) during your last delivery?",
  m_sba = "In 2014: were you assisted by someone (e.g., a doctor, nurse, midwife, SBA, or community health worker) during your last delivery?",
  sba = "In 2016: were you assisted by someone (e.g., a doctor, nurse, midwife, SBA, or community health worker) during your last delivery?",
  b_exclusive = "In 2012: did you exclusively breastfeed your youngest child when they were younger than six months?",
  m_exclusive = "In 2014: did you exclusively breastfeed your youngest child when they were younger than six months?",
  exclusive = "In 2016: did you exclusively breastfeed your youngest child when they were younger than six months?",
  b_ors_clinic = "In 2012: did you give your child ORS or go to the clinic the last time they had diarrhea?",
  m_ors_clinic = "In 2014: did you give your child ORS or go to the clinic the last time they had diarrhea?",
  ors_clinic = "In 2016: did you give your child ORS or go to the clinic the last time they had diarrhea?",
  b_soap_observed = "In 2012: if you have a handwashing station in your household, is there soap available to wash your hands?",
  m_soap_observed = "In 2014: if you have a handwashing station in your household, is there soap available to wash your hands?",
  soap_observed = "In 2016: if you have a handwashing station in your household, is there soap available to wash your hands?",
  m_exclusive2 = "In 2014: did you exclusively breastfeed your youngest child when they were younger than two years?",
  b_exclusive2 = "In 2012: did you exclusively breastfeed your youngest child when they were younger than two years?",
  exclusive2 = "In 2016: did you exclusively breastfeed your youngest child when they were younger than two years?",
  district = "In which district do you live?",
  region = "In which region do you live?",
  urbanrural = "Do you live in a urban or rural area?",
  b_q5 = "Do you have a mobile phone?",
  b_q6_whose_number = "Whose number is this?",
  b_q1_1 = "What is your age?",
  b_q1_2 = "Were you born in this community?",
  b_q1_3 = "Where were you born?",
  b_q1_4 = "How many years have you lived in this community?",
  b_q1_5 = "Who is the head of the household?",
  b_q1_8 = "What is your main religious denomination?",
  b_q1_9 = "Are you currently married?",
  b_q1_11 = "What is your spouse's religious denomination?",
  b_q1_12 = "What was the highest level of education attained by you?",
  b_q1_13 = "What was the highest level of education attained by your spouse?",
  b_q1_14 = "How many times have you ever given birth in your lifetime?",
  b_q1_15 = "How many children would you like to have in your lifetime?",
  b_q1_16 = "Are you able to read a sentence?",
  b_q2_3 = "In the last year have you sought health-related information?",
  b_q2_5 = "What made you decide to look for that information?",
  b_q2_6 = "Do you know any CBA in your community?",
  b_q2_7 = "How well do you know the CBA that you know the best?",
  b_q2_8 = "When was the last time you talked to this CBA?",
  cba_talk = "Have you talked about health-related topics (e.g., diarrhea, exclusive/complementary breastfeeding, insecticide treated nets, malaria) with this CBA?",
  b_q2_10 = "How frequently do you talk to this CBA about being healthy?",
  b_q2_11 = "Do you trust the information this CBA gives?",
  b_q2_13_1 = "Family is top 3 source to hear about staying healty",
  b_q2_13_2 = "Community is top 3 source to hear about staying healty",
  b_q2_13_3 = "Radio is top 3 source to hear about staying healty",
  b_q2_13_4 = "TV is top 3 source to hear about staying healty",
  b_q2_13_5 = "Clinic/Hospital is top 3 source to hear about staying healty",
  b_q2_13_6 = "Traditional healer is top 3 source to hear about staying healty",
  b_q2_13_7 = "Newspaper is top 3 source to hear about staying healty",
  b_q2_13_8 = "Books/magazine is top 3 source to hear about staying healty",
  b_q2_13_9 = "Internet is top 3 source to hear about staying healty",
  b_q2_13_10 = "Mobile phone is top 3 source to hear about staying healty",
  b_q2_13_11 = "Poster/Billboard phone is top 3 source to hear about staying healty",
  b_q2_13_12 = "NGO/Outreach phone is top 3 source to hear about staying healty",
  b_q2_13_13 = "Friends phone is top 3 source to hear about staying healty",
  b_q2_14 = "What means of transportation do you most often use to go to clinc/hospital?",
  b_q2_15_hr = "How many hours does it take to get to the nearest clinic/hospital?",
  b_q2_16 = "Not including CBAs, when was the last time you saw a health worker?",
  b_q2_17 = "Have you ever sent or received an SMS text message?",
  b_q2_18 = "How many SMS messages do you send or receive in a typical week?",
  malaria_prev = "Have you done something to prevent malaria (e.g., slept under mosquito treated nets, sprayed the house, used anti-insect body lotion)?",
  b_q4_2 = "Have you ever heard about an insecticide treated net?",
  b_q4_4 = "Do you understand an insecticide treated net is a special type of net?",
  b_q4_5 = "Last night, did you sleep under an insecticide treated net?",
  b_q4_6 = "Last night, did your youngest child sleep under an insecticide treated net?",
  b_q4_7 = "Last night, how many of your own children under 5 slept under an insecticide treated net?",
  b_q4_8 = "During your last pregnancy, how frequently did you sleep under an insecticide treated net?",
  b_q4_9_666 = "Do you always sleep under an insecticide treated net?",
  b_q4_10_666 = "Do your children always sleep under an insecticide treated net?",
  b_q4_11 = "How serious or unimportant of a problem is malaria?",
  b_q4_14_1 = "Is mortality in children a major problem malaria causes?",
  b_q4_14_2 = "Is mortality in pregnant women a major problem malaria causes?",
  b_q4_14_3 = "Is mortality a major problem malaria causes?",
  b_q4_14_4 = "Is poor health a major problem malaria causes?",
  b_q4_14_5 = "Is anemia a major problem malaria causes?",
  b_q4_14_6 = "Others are the major problems that malaria causes",
  b_q4_14_7 = "Are symptoms like vomit, diarrhea, and fever the major problems that malaria causes?",
  b_q4_15 = "In the next six months, how likely it is that one of your children will get malaria?",
  b_q4_16 = "Have you ever been scared for the life of one of your children due to malaria?",
  b_q4_17 = "Have you ever known a child that died from malaria?",
  b_q4_18 = "Was the child who died from malaria your own child?",
  b_q4_19_1 = "Do mosquito bites cause malaria?",
  b_q4_19_2 = "Do unsanitary conditions cause malaria?",
  b_q4_19_3 = "Do weather conditions cause malaria?",
  b_q4_19_4 = "Is malaria caused by eating dirty food?",
  b_q4_19_5 = "Is malaria caused by drinking dirty water?",
  b_q4_19_6 = "Do houseflies cause malaria?",
  b_q4_19_7 = "Is malaria caused by the air?",
  b_q4_19_8 = "Is malaria caused by witchcraft?",
  b_q4_19_9 = "Is malaria caused by other factors?",
  b_q4_19_10 = "Is malaria caused by lack of personal or environmental hygiene?",
  b_q4_20_1 = "Should everyone sleep under a insecticide treated net?",
  b_q4_20_2 = "Should only infants sleep under a insecticide treated net?",
  b_q4_20_3 = "Should only children sleep under a insecticide treated net?",
  b_q4_20_4 = "Should only adults sleep under a insecticide treated net?",
  b_q4_20_5 = "Should only elderly sleep under a insecticide treated net?",
  b_q4_20_6 = "Should only pregnant women sleep under a insecticide treated net?",
  b_q4_20_7 = "Should only sick people sleep under a insecticide treated net?",
  b_q4_20_8 = "Should only other categories of people not previously mentioned sleep under a insecticide treated net?",
  b_q4_21 = "How many of your friends do you think sleep under an insecticide treated net?",
  b_q4_22 = "How many children in your community do you think sleep under an insecticide treated net?",
  b_q4_23 = "Have you ever advise anyone in this community on insecticide treated nets?",
  b_q4_24 = "Has anyone in your family ever encouraged the use of insecticide treated nets?",
  b_q4_25_888 = "Do you know any way to take care of an insecticide treated net?",
  b_q4_26_888 = "You do not know how an insecticide treated net is treated",
  b_q4_27_666 = "You do not know anything about insecticide treated nets",
  b_q5_1 = "How often do you wash your hand with only water?",
  b_q5_2 = "The last time you cleaned your hands, what did you use?",
  b_q5_4 = "How many times did you wash your hands yesterday?",
  b_q5_5 = "The last time you ate, did you wash your hands before eating?",
  b_q5_6 = "After the last time you defecated, did you wash your hands?",
  b_q5_9 = "Are there times when you think you should wash your hands but you do not?",
  b_q5_10_1 = "I did not wash my hands with soap because there was not enough water",
  b_q5_10_2 = "I did not wash my hands with soap because there was not enough money to buy some",
  b_q5_10_3 = "I did not wash my hands with soap because there was not enough time",
  b_q5_10_4 = "I did not wash my hands with soap because there was no place where to wash my hands",
  b_q5_10_5 = "I did not wash my hands with soap because of other reasons",
  b_q5_10_6 = "I did not wash my hands with soap because I forgot or I was lazy",
  b_q5_10_888 = "I don't know why I did not wash my hand with soap",
  b_q5_11_1 = "Do you wash your hand because you have heard that cleanliness is important?",
  b_q5_11_2 = "Do you wash your hand because you have heard that it kills germs?",
  b_q5_11_3 = "Do you wash your hand because you have heard that it prevents diarrhea?",
  b_q5_11_4 = "Do you wash your hand because you have heard that it prevents food contamination?",
  b_q5_11_5 = "Do you wash your hand because you have heard that it prevents stomach being upset?",
  b_q5_11_6 = "Do you wash your hand because you have heard that it prevents water contamination?",
  b_q5_11_7 = "Do you wash your hand because you have heard that it prevents the spread of germs?",
  b_q5_11_8 = "Do you wash your hand because you have heard that it makes the hands smell nice?",
  b_q5_11_9 = "Do you wash your hand because you have heard that it prevents other illnesses?",
  b_q5_11_10 = "Do you wash your hand because you have heard that it helps staying healthy?",
  b_q5_11_11 = "Do you wash your hand because you have heard of other reasons?",
  b_q5_11_12 = "Do you wash your hand because you have heard that reduces risk of malaria?",
  b_q5_11_888 = "Do you wash your hand because you do not know?",
  b_q5_12 = "How often is okay to wash your hands with only water?",
  clean_imp = "Do you think it is important to wash your hands before some activities such as eating, feeding a child, etc.?",
  b_q5_15 = "Is there a place in your community where you can buy soap?",
  encourage = "Has anyone ever encouraged you to wash your hands with soap?",
  b_q5_20 = "Have you ever encouraged someone in your community to wash their hands with soap?",
  b_q5_21 = "Do you teach your children to wash their hands with soap?",
  b_q5_22_666 = "Have you really never heard anyone mention that washing hands is a good health practice?",
  b_q6_1_666 = "Has your child really never had diarrhea?",
  b_q6_2 = "Have you ever used ORS to treat a child of yours?",
  b_q6_3 = "Why have you never used ORS to treat a child of yours?",
  b_q6_4 = "Do you know a place where you could buy a packet of ORS?",
  b_q6_5 = "The last time you used ORS, did you make it at home or buy a packet?",
  b_q6_6 = "The last time you used ORS, how long the child was sick before you used it?",
  b_q6_8 = "How serious or unimportant of a problem do you think diarrhea is?",
  b_q6_9 = "Have you ever been scared of the life of one of your children due to diarrhea?",
  b_q6_10_888 = "Have you never really known which are the causes of diarrhea?",
  b_q6_11_888 = "Have you never really known what to do when a child gets diarrhea?",
  b_q6_12 = "Do you understand what an ORS is?",
  b_q6_14 = "At what age do you think it is alright to give a child ORS?",
  b_q6_15 = "Can you think of a time you ever wanted to use ORS but did not?",
  b_q6_18_5 = "Has no one really encouraged you to use ORS?",
  b_q6_21 = "In the next 6 months, how likely it is that one of your children will get diarrhea?",
  b_q6_22 = "Have you ever known a child that died from diarrhea?",
  b_q6_23 = "Was the child who died from diarrhea your own child?",
  b_q6_24_666 = "Have you really never heard anything about treating diarrhea in the last year?",
  b_q7_1 = "Have you ever breastfed one of your children?",
  b_q7_2 = "Are you currently breastfeeding your youngest child?",
  b_q7_4 = "Yesterday, did your youngest child drink any fluids (besides water, ORS, or breast milk)?",
  b_q7_6 = "Yesterday, did your youngest child eat any food?",
  b_q7_7 = "In the last week, how many times have you lef your youngest child with someone else for more than 1 hour?",
  b_q7_8 = "At how many months did your youngest child first drink water/other liquids?",
  b_q7_9 = "At how many months did your youngest child first taste any food?",
  b_q7_17 = "If you give birth again, at how many months do you intend to give child food?",
  b_q7_18 = "If you give birth again, at how many months do you think a baby should start eating?",
  b_q7_19 = "If a baby under 6 months is too hot, how good/bad it is to give water?",
  b_q7_20_888 = "Have you never really heard that breastfeeding is beneficial?",
  b_q7_21 = "Have you ever heard of exclusive breastfeeding?",
  b_q7_23 = "Do you understand what exclusive breastfeeding is?",
  b_q7_25_9 = "Has no one really encouraged you to exclusively breastfeed your child in their first 6 months?",
  b_q7_26_666 = "Have you never really heard anything about exclusive breastfeeding?",
  b_q8_1 = "For your last pregnancy, who helped you to deliver?",
  b_q8_2 = "For your last pregnancy, where did you deliver?",
  b_q8_5 = "If you deliver again, where do you intend to deliver?",
  b_q8_6 = "If you deliver again, who do you intend to have to help you?",
  b_q8_7 = "Where do you think is the best place to deliver?",
  b_q8_10 = "There is something called skilled birth attendant. Have you ever heard of it?",
  b_q8_11 = "Do you understand what a skilled birth attendant is?",
  b_q8_12 = "How many times have you had a skilled birth attendant at your delivery?",
  b_q8_13 = "In the past, have you ever wanted to deliver in a health facility but not been able to?",
  b_q8_18_1 = "Has no one really encouraged you to deliver in a health facility?",
  b_q8_19 = "Have you ever had a very bad problem from delivery?",
  b_q8_22_666 = "Have you never really heard anything about using a skilled birth attendant?",
  NULL
)

dat <- add_question_row(dat, questions)
dat <- drop_any(dat, c("info", "b_q5_19_7_ospecify"))

final_order <- c(
  "household_id",
  "hh_comm_id",
  "district",
  "region",
  "urbanrural",
  "b_q5",
  "b_q6_whose_number",
  "b_q1_1",
  "b_q1_2",
  "b_q1_3",
  "b_q1_4",
  "b_q1_5",
  "b_q1_8",
  "b_q1_9",
  "b_q1_11",
  "b_q1_12",
  "b_q1_13",
  "b_q1_14",
  "b_q1_15",
  "b_q1_16",
  "b_q2_3",
  "b_q2_5",
  "b_q2_6",
  "b_q2_7",
  "b_q2_8",
  "cba_talk",
  "b_q2_10",
  "b_q2_11",
  "b_q2_13_1",
  "b_q2_13_2",
  "b_q2_13_3",
  "b_q2_13_4",
  "b_q2_13_5",
  "b_q2_13_6",
  "b_q2_13_7",
  "b_q2_13_8",
  "b_q2_13_9",
  "b_q2_13_10",
  "b_q2_13_11",
  "b_q2_13_12",
  "b_q2_13_13",
  "b_q2_14",
  "b_q2_15_hr",
  "b_q2_16",
  "b_q2_17",
  "b_q2_18",
  "malaria_prev",
  "b_q4_2",
  "b_q4_4",
  "b_q4_5",
  "b_q4_6",
  "b_q4_7",
  "b_q4_8",
  "b_q4_9_666",
  "b_q4_10_666",
  "b_q4_11",
  "b_q4_14_1",
  "b_q4_14_2",
  "b_q4_14_3",
  "b_q4_14_4",
  "b_q4_14_5",
  "b_q4_14_6",
  "b_q4_14_7",
  "b_q4_15",
  "b_q4_16",
  "b_q4_17",
  "b_q4_18",
  "b_q4_19_1",
  "b_q4_19_2",
  "b_q4_19_3",
  "b_q4_19_4",
  "b_q4_19_5",
  "b_q4_19_6",
  "b_q4_19_7",
  "b_q4_19_8",
  "b_q4_19_9",
  "b_q4_19_10",
  "b_q4_20_1",
  "b_q4_20_2",
  "b_q4_20_3",
  "b_q4_20_4",
  "b_q4_20_5",
  "b_q4_20_6",
  "b_q4_20_7",
  "b_q4_20_8",
  "b_q4_21",
  "b_q4_22",
  "b_q4_23",
  "b_q4_24",
  "b_q4_25_888",
  "b_q4_26_888",
  "b_q4_27_666",
  "b_q5_1",
  "b_q5_2",
  "b_q5_4",
  "b_q5_5",
  "b_q5_6",
  "b_q5_9",
  "b_q5_10_1",
  "b_q5_10_2",
  "b_q5_10_3",
  "b_q5_10_4",
  "b_q5_10_5",
  "b_q5_10_6",
  "b_q5_10_888",
  "b_q5_11_1",
  "b_q5_11_2",
  "b_q5_11_3",
  "b_q5_11_4",
  "b_q5_11_5",
  "b_q5_11_6",
  "b_q5_11_7",
  "b_q5_11_8",
  "b_q5_11_9",
  "b_q5_11_10",
  "b_q5_11_11",
  "b_q5_11_12",
  "b_q5_11_888",
  "b_q5_12",
  "clean_imp",
  "b_q5_15",
  "encourage",
  "b_q5_20",
  "b_q5_21",
  "b_q5_22_666",
  "b_q6_1_666",
  "b_q6_2",
  "b_q6_3",
  "b_q6_4",
  "b_q6_5",
  "b_q6_6",
  "b_q6_8",
  "b_q6_9",
  "b_q6_10_888",
  "b_q6_11_888",
  "b_q6_12",
  "b_q6_14",
  "b_q6_15",
  "b_q6_18_5",
  "b_q6_21",
  "b_q6_22",
  "b_q6_23",
  "b_q6_24_666",
  "b_q7_1",
  "b_q7_2",
  "b_q7_4",
  "b_q7_6",
  "b_q7_7",
  "b_q7_8",
  "b_q7_9",
  "b_q7_17",
  "b_q7_18",
  "b_q7_19",
  "b_q7_20_888",
  "b_q7_21",
  "b_q7_23",
  "b_q7_25_9",
  "b_q7_26_666",
  "b_q8_1",
  "b_q8_2",
  "b_q8_5",
  "b_q8_6",
  "b_q8_7",
  "b_q8_10",
  "b_q8_11",
  "b_q8_12",
  "b_q8_13",
  "b_q8_18_1",
  "b_q8_19",
  "b_q8_22_666",
  "treatment",
  "comradio",
  "m4d_treatment",
  "b_mom_used_itn",
  "m_mom_used_itn",
  "mom_used_itn",
  "b_child_used_itn",
  "m_child_used_itn",
  "child_used_itn",
  "b_sba",
  "m_sba",
  "sba",
  "b_exclusive",
  "m_exclusive",
  "exclusive",
  "b_ors_clinic",
  "m_ors_clinic",
  "ors_clinic",
  "b_soap_observed",
  "m_soap_observed",
  "soap_observed",
  "m_exclusive2",
  "b_exclusive2",
  "exclusive2",
  NULL
)

dat <- order_any(dat, final_order)

# Export ----------------------------------------------------------------------
if ("household_id" %in% names(dat)) {
  names(dat)[names(dat) == "household_id"] <- "subject_id"
  dat$subject_id[1] <- "subject_id"
  dat <- dat[, c("subject_id", setdiff(names(dat), "subject_id")), drop = FALSE]
}
write_clean_csv(dat, output_file)
