# Milkman et al. (2024) — Megastudy on COVID-19 booster nudges (CVS Pharmacy).
#
# Reads:
#   data/human/rcts/milkman_et_al_2024/megastudy_clean.dta
# Writes:
#   data/processed/rcts/milkman_et_al_2024/milkman_et_al_2024_data.csv
#
# Combines: (a) per-arm subsampling of 1800 observations from megastudy_clean
# for the three arms used in the synthetic replication (Control, Waiting,
# Lyft), seed = 9614 — previously a separate
# `milkman_et_al_analysis_preprocessing_TOCHECK.R` step that has been folded
# in here so no intermediate CSV is written. (b) Stata-translated cleaning
# logic from `data_cleaning_Milkman.do`.

library(dplyr)
library(readr)
library(haven)

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

data_path   <- file.path("data", "human", "rcts", "milkman_et_al_2024")
saving_path <- file.path("data", "processed", "rcts", "milkman_et_al_2024")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

raw_dta_file <- file.path(data_path, "megastudy_clean.dta")
output_file  <- file.path(saving_path, "milkman_et_al_2024_data.csv")

# Helper functions ------------------------------------------------------------
to_num <- function(x) {
  suppressWarnings(as.numeric(x))
}

num_eq <- function(x, value) {
  y <- to_num(x)
  !is.na(y) & y == value
}

to_stata_string <- function(x, numeric_missing = ".", string_missing = "") {
  # Mirrors Stata's tostring workflow for the variables in this file:
  # numeric missing values become "."; string missing values remain blank.
  if (is.factor(x)) x <- as.character(x)

  if (is.numeric(x) || is.integer(x)) {
    out <- rep(numeric_missing, length(x))
    nonmissing <- !is.na(x)
    out[nonmissing] <- format(x[nonmissing], scientific = FALSE, trim = TRUE, justify = "none")
  } else {
    out <- as.character(x)
    out[is.na(out)] <- string_missing
  }

  out
}

move_to_end <- function(df, var) {
  if (!(var %in% names(df))) return(df)
  df[, c(setdiff(names(df), var), var), drop = FALSE]
}

stringify_move_to_end <- function(df, var, force = FALSE) {
  # Stata pattern used throughout the .do file:
  #   tostring var, gen(var_)
  #   drop var
  #   rename var_ var
  # This converts to string and moves the variable to the end of the dataset.
  if (!(var %in% names(df))) return(df)

  df[[var]] <- to_stata_string(df[[var]])
  move_to_end(df, var)
}

recode_binary_yes_no_move_to_end <- function(df, var) {
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)
  out[num_eq(source, 1)] <- "Yes"
  out[num_eq(source, 0)] <- "No"

  df[[var]] <- out
  move_to_end(df, var)
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

drop_prefixes <- function(df, prefixes) {
  keep <- !Reduce(`|`, lapply(prefixes, function(prefix) startsWith(names(df), prefix)))
  df[, keep, drop = FALSE]
}

blank_character <- function(n) {
  rep("", n)
}

assign_if <- function(out, condition, value) {
  condition[is.na(condition)] <- FALSE
  out[condition] <- value
  out
}

add_question_row_stata_style <- function(df, questions) {
  # Mirrors the Stata sequence:
  #   insobs 1, before(1)
  #   replace var = "question" if var == ""
  # This intentionally replaces all blank string values in each listed variable,
  # not only the inserted first row, because that is what the Stata code does.
  df <- df %>% mutate(across(everything(), as.character))
  header <- as_tibble(as.list(setNames(rep("", ncol(df)), names(df))))
  out <- bind_rows(header, df)

  for (var in names(questions)) {
    if (var %in% names(out)) {
      is_blank <- !is.na(out[[var]]) & out[[var]] == ""
      out[[var]][is_blank] <- unname(questions[[var]])
    }
  }

  out
}

# Cleaning the dataset --------------------------------------------------------
# Step (a): per-arm subsampling — 1800 observations per arm with seed 9614.
data_for_sampling <- haven::read_dta(raw_dta_file)
arms_selected <- c("Control", "Waiting", "Lyft")
sampling_arm <- function(arm, data, n = 1800) {
  data %>%
    filter(.data[[arm]] == 1) %>%
    slice_sample(n = n, replace = FALSE)
}
set.seed(9614)
df <- bind_rows(lapply(arms_selected, sampling_arm,
                       data = data_for_sampling, n = 1800))

# The arm indicator columns in the .dta are title-cased; normalise to lowercase
# so all downstream processing (binary recoding, treatment assignment, drop) works.
df <- df %>% rename_with(tolower, any_of(c("Control", "Waiting", "Lyft", "Planning",
                                            "Transmission", "Pharmacist", "Recommendation",
                                            "Holidays", "Misinformation")))

# Binary variables converted to Yes/No ---------------------------------------
binary_vars <- c(
  "booster_outcome_30", "booster_outcome_90", "flu_outcome_30",
  "control", "waiting", "lyft", "planning", "transmission",
  "pharmacist", "recommendation", "holidays", "misinformation", "male",
  "age_geq_50", "med_age", "medicare", "medicaid", "unknown", "commercial",
  "priorflu", "day_dum1", "day_dum2", "day_dum3", "dec_priorflu",
  "feb_priorflu", "med_white", "med_black", "med_asian", "med_hispanic",
  "med_res_sq_mile", "med_gopvoteshare", "med_median_income",
  "med_vaxxed_perc", "med_boosted_perc", "med_education", "med_strpsqmi",
  "mis_white", "mis_black", "mis_asian", "mis_hispanic", "mis_res_sq_mile",
  "mis_gopvoteshare", "mis_median_income", "mis_vaxxed_perc",
  "mis_boosted_perc", "mis_education", "mis_strpsqmi", "previous_booster"
)

for (var in binary_vars) {
  df <- recode_binary_yes_no_move_to_end(df, var)
}

# Numeric/count identifiers converted to strings -----------------------------
string_vars <- c(
  "de_id", "age", "booster_count", "dec_covid_booster_count",
  "feb_covid_booster_count", "dec_covid_shot_count", "feb_covid_shot_count"
)

for (var in string_vars) {
  df <- stringify_move_to_end(df, var)
}

# Demographic/contextual variables converted to strings, equivalent to
# tostring ..., force in the Stata file.
force_string_vars <- c(
  "hispanic", "black", "asian", "white", "median_income", "education",
  "vaxxed_perc", "boosted_perc", "gopvoteshare", "res_sq_mile", "strpsqmi"
)

for (var in force_string_vars) {
  df <- stringify_move_to_end(df, var, force = TRUE)
}

# Mark values as N/A where the corresponding missingness flag is Yes ----------
context_vars <- c(
  "white", "black", "asian", "hispanic", "res_sq_mile", "gopvoteshare",
  "median_income", "vaxxed_perc", "boosted_perc", "education", "strpsqmi"
)

for (var in context_vars) {
  missing_flag <- paste0("mis_", var)
  if (var %in% names(df) && missing_flag %in% names(df)) {
    flag_yes <- !is.na(df[[missing_flag]]) & df[[missing_flag]] == "Yes"
    df[[var]][flag_yes] <- "N/A"
  }
}

# Insurance ------------------------------------------------------------------
df <- drop_any(df, "previous_booster")

insurance <- blank_character(nrow(df))
if ("medicare" %in% names(df)) insurance <- assign_if(insurance, df$medicare == "Yes", "Medicare")
if ("medicaid" %in% names(df)) insurance <- assign_if(insurance, df$medicaid == "Yes", "Medicaid")
if ("commercial" %in% names(df)) insurance <- assign_if(insurance, df$commercial == "Yes", "Commercial insurance")
if ("unknown" %in% names(df)) insurance <- assign_if(insurance, df$unknown == "Yes", "Don't know")
df$insurance <- insurance

df <- drop_any(df, c("medicare", "medicaid", "unknown", "commercial"))

# Treatment -------------------------------------------------------------------
treatment <- blank_character(nrow(df))
if ("control" %in% names(df)) treatment <- assign_if(treatment, df$control == "Yes", "Control")
if ("waiting" %in% names(df)) treatment <- assign_if(treatment, df$waiting == "Yes", "Baseline")
if ("lyft" %in% names(df)) treatment <- assign_if(treatment, df$lyft == "Yes", "Free ride")
if ("planning" %in% names(df)) treatment <- assign_if(treatment, df$planning == "Yes", "Default plan")
if ("transmission" %in% names(df)) treatment <- assign_if(treatment, df$transmission == "Yes", "Infection rates")
if ("pharmacist" %in% names(df)) treatment <- assign_if(treatment, df$pharmacist == "Yes", "Pharmacy team message")
if ("recommendation" %in% names(df)) treatment <- assign_if(treatment, df$recommendation == "Yes", "CDC recommended")
if ("holidays" %in% names(df)) treatment <- assign_if(treatment, df$holidays == "Yes", "Holiday protection")
if ("misinformation" %in% names(df)) treatment <- assign_if(treatment, df$misinformation == "Yes", "Misinformation resources")
df$treatment <- treatment

df <- drop_any(
  df,
  c("control", "waiting", "lyft", "planning", "transmission", "pharmacist",
    "recommendation", "holidays", "misinformation")
)

# Gender ----------------------------------------------------------------------
if ("male" %in% names(df)) {
  df$male[df$male == "Yes"] <- "Male"
  df$male[df$male == "No"] <- "Female"
}

# Insert first-row questions and drop variables exactly as in the Stata file --
questions <- c(
  de_id = "Patient ID",
  booster_outcome_30 = "Did you receive a COVID-19 booster shot within 30 days after getting the first message from your CVS Pharmacy?",
  booster_outcome_90 = "Did you receive a COVID-19 booster shot within 90 days after getting the first message from your CVS Pharmacy",
  flu_outcome_30 = "Did you receive a flu shot within 30 days after getting the first message from your CVS Pharmacy",
  treatment = "Treatment",
  male = "What is your gender?",
  age = "How old were you in October 2022?",
  booster_count = "How many COVID-19 booster shots had you received by October 2022?",
  insurance = "Which medical insurance do you have?",
  priorflu = "Did you receive a flu shot for the 2021-2022 flu season by October 2022?",
  dec_covid_booster_count = "How many COVID-19 booster shots had you received by December 2022?",
  feb_covid_booster_count = "How many COVID-19 booster shots had you received by February 2023?",
  dec_covid_shot_count = "How many COVID-19 vaccine doses had you received by December 2022?",
  feb_covid_shot_count = "How many COVID-19 vaccine doses had you received by February 2023?",
  dec_priorflu = "Did you receive a flu shot for the 2021-2022 flu season by December 2022?",
  feb_priorflu = "Did you receive a flu shot for the 2021-2022 flu season by February 2023?",
  hispanic = "What percentage of residents in the ZIP code of your nearest CVS Pharmacy are Hispanic?",
  black = "What percentage of residents in the ZIP code of your nearest CVS Pharmacy are Black?",
  asian = "What percentage of residents in the ZIP code of your nearest CVS Pharmacy are Asian?",
  white = "What percentage of residents in the ZIP code of your nearest CVS Pharmacy are White?",
  median_income = "What is the median income in the ZIP code of your nearest CVS Pharmacy?",
  education = "What percentage of residents in the ZIP code of your nearest CVS Pharmacy has a bachelor degree?",
  vaxxed_perc = "What percentage of people in the county of your nearest CVS Pharmacy have completed the primary COVID vaccine series?",
  boosted_perc = "What percentage of people in the county of your nearest CVS Pharmacy have completed the primary COVID vaccine series and received at least booster dose?",
  gopvoteshare = "What percentage of residents in the county of your nearest CVS Pharmacy voted for the Republican candidate?",
  res_sq_mile = "What is the population density (residents per square mile) in the ZIP code of your nearest CVS Pharmacy?",
  strpsqmi = "What is the number of CVS Pharmacies per square mile in the ZIP code of your nearest CVS Pharmacy?",
  day_dum1 = "Did you receive the message from your CVS Pharmacy on 3 Novemeber?",
  day_dum2 = "Did you receive the message from your CVS Pharmacy on 5 Novemeber?",
  day_dum3 = "Did you receive the message from your CVS Pharmacy on 8 Novemeber?"
)

df <- add_question_row_stata_style(df, questions)

df <- drop_any(df, c("age_geq_50", "med_age"))
df <- drop_prefixes(df, c("mis_", "med_"))

# Export ----------------------------------------------------------------------
if ("de_id" %in% names(df)) {
  names(df)[names(df) == "de_id"] <- "ID"
  df$ID[1] <- "ID"
  df <- df[, c("ID", setdiff(names(df), "ID")), drop = FALSE]
}
write_clean_csv(df, output_file)
