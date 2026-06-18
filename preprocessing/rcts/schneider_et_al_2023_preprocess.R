# Data cleaning script translated from data_cleaning_Schneider.do
# Supplementary-materials version: retains the original Stata recodes,
# variable names, inserted question row, output filename, and spelling/
# typographical choices as closely as possible.

# Required packages -----------------------------------------------------------
# install.packages(c("dplyr", "haven", "readr", "tibble"))
library(dplyr)
library(haven)
library(readr)
library(tibble)

# Paths -----------------------------------------------------------------------
if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human", "rcts", "schneider_et_al_2023")
saving_path <- file.path("data", "processed", "rcts", "schneider_et_al_2023")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

input_file  <- file.path(data_path, "US_study.dta")
output_file <- file.path(saving_path, "schneider_et_al_2023_data.csv")

# Helper functions ------------------------------------------------------------
read_dta_flexible <- function(path) {
  if (file.exists(path)) return(haven::read_dta(path))

  without_ext <- sub("\\.dta$", "", path)
  if (file.exists(without_ext)) return(haven::read_dta(without_ext))

  stop("Could not find Stata file: ", path, " or ", without_ext, call. = FALSE)
}

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

is_system_missing <- function(x) {
  is.na(x) & !tagged_na_safe(x)
}

to_stata_string <- function(x) {
  # Mirrors Stata's tostring for the variables in this script: numeric system
  # missing values become "."; tagged missing values are retained as ".a", etc.
  y <- haven::zap_labels(x)

  if (is.factor(y)) y <- as.character(y)

  if (is.numeric(y) || is.integer(y)) {
    out <- rep(".", length(y))
    nonmissing <- !is.na(y)
    out[nonmissing] <- format(y[nonmissing], scientific = FALSE, trim = TRUE, justify = "none")
  } else {
    out <- as.character(y)
    out[is.na(out)] <- "."
  }

  tagged <- tagged_na_safe(x)
  tags <- tag_safe(x)
  out[tagged & !is.na(tags)] <- paste0(".", tags[tagged & !is.na(tags)])
  out[is_system_missing(x)] <- "."
  out
}

move_to_end <- function(df, var) {
  if (!(var %in% names(df))) return(df)
  df[, c(setdiff(names(df), var), var), drop = FALSE]
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

recode_to_character_move_to_end <- function(df, var, map, missing_label = NULL) {
  # Stata pattern used throughout the .do file:
  #   tostring var, gen(var_)
  #   replace var_ = "..." if var == ...
  #   drop var
  #   rename var_ var
  # This converts to string and moves the variable to the end of the dataset.
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)

  for (value in names(map)) {
    out[num_eq(source, as.numeric(value))] <- unname(map[[value]])
  }

  if (!is.null(missing_label)) {
    out[is_system_missing(source)] <- missing_label
  }

  df[[var]] <- out
  move_to_end(df, var)
}

binary_yes_no_na_move_to_end <- function(df, var) {
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)
  out[num_eq(source, 1)] <- "Yes"
  out[num_eq(source, 0)] <- "No"
  out[is_system_missing(source)] <- "N/A"

  df[[var]] <- out
  move_to_end(df, var)
}

likert_agreement_move_to_end <- function(df, var) {
  recode_to_character_move_to_end(
    df,
    var,
    c(
      "1" = "Completely disagree",
      "2" = "Disagree",
      "3" = "Neither agree nor disagree",
      "4" = "Agree",
      "5" = "Completely agree"
    ),
    missing_label = "N/A"
  )
}

set_variable_label <- function(df, var, label) {
  # Stata variable labels do not appear in the exported CSV, but keeping them
  # as R attributes documents the same metadata in the intermediate object.
  if (var %in% names(df)) attr(df[[var]], "label") <- label
  df
}

add_question_row_stata_style <- function(df, questions) {
  # Mirrors the Stata pattern:
  #   insobs 1, before(1)
  #   replace var = "question" if var == var[1]
  # This intentionally replaces every blank value in each listed variable with
  # the question text, not only the inserted first row, because that is what the
  # Stata code does after inserting a blank first observation.
  df <- df %>% mutate(across(everything(), as.character))
  header <- tibble::as_tibble(as.list(setNames(rep("", ncol(df)), names(df))))
  out <- bind_rows(header, df)

  for (var in names(questions)) {
    if (var %in% names(out)) {
      first_value <- out[[var]][1]
      matches_first <- !is.na(out[[var]]) & out[[var]] == first_value
      out[[var]][matches_first] <- unname(questions[[var]])
    }
  }

  out
}

# Cleaning the dataset --------------------------------------------------------
df <- read_dta_flexible(input_file)

df$country <- "US"

df <- recode_to_character_move_to_end(df, "age", c())

df <- recode_to_character_move_to_end(
  df,
  "gender",
  c(
    "1" = "Female",
    "2" = "Male",
    "3" = "Neither female or male"
  )
)
df <- drop_any(df, "female")

df <- recode_to_character_move_to_end(
  df,
  "ethnicity",
  c(
    "1" = "European American, White",
    "2" = "African American, Black",
    "3" = "Hispanic, Latino",
    "4" = "Asian, Asian American",
    "5" = "Other"
  ),
  missing_label = "N/A"
)
df <- drop_any(df, "white")

df <- recode_to_character_move_to_end(
  df,
  "education",
  c(
    "1" = "Eighth Grade or less",
    "2" = "Some High School",
    "3" = "High School degree or GED",
    "4" = "Some College",
    "5" = "2-year College Degree",
    "6" = "4-year College Degree",
    "7" = "Master's Degree",
    "8" = "Doctoral Degree",
    "9" = "Professional Degree (JD, MD, MBA)"
  )
)
df <- drop_any(df, "college")

df <- recode_to_character_move_to_end(
  df,
  "employ_status",
  c(
    "1" = "Full-time employee",
    "2" = "Part-time employee",
    "3" = "Self-employed or small business owner",
    "4" = "Unemployed looking for work",
    "5" = "Student",
    "6" = "Not in labor force (for example: retired, or full-time parent)"
  )
)

df <- recode_to_character_move_to_end(
  df,
  "income",
  c(
    "1" = "$0-$9,999",
    "2" = "$10,000-$14,999",
    "3" = "$15,000-$19,999",
    "4" = "$20,000-$29,999",
    "5" = "$30,000-$39,999",
    "6" = "$40,000-$49,999",
    "7" = "$50,000-$74,999",
    "8" = "$75,000-$99,999",
    "9" = "$100,000-$124,999",
    "10" = "$125,000-$149,999",
    "11" = "$150,000-$199,999",
    "12" = "$200,000+"
  ),
  missing_label = "N/A"
)

df <- recode_to_character_move_to_end(
  df,
  "pol_position",
  c(
    "1" = "Republican",
    "2" = "Democrat",
    "3" = "Independent"
  )
)
df <- drop_any(df, c("pol_republican", "pol_democrat", "pol_independent"))

df <- recode_to_character_move_to_end(
  df,
  "vote_trump",
  c(
    "1" = "Joe Biden",
    "2" = "Donald Trump",
    "3" = "Other"
  ),
  missing_label = "N/A"
)
df <- drop_any(df, "vote_for_trump")

df <- recode_to_character_move_to_end(
  df,
  "pol_position_2",
  c(
    "1" = "Democratic Party",
    "2" = "Republican Party"
  ),
  missing_label = "N/A"
)

df <- recode_to_character_move_to_end(
  df,
  "vaccination_status",
  c(
    "1" = "None",
    "2" = "1 shot",
    "3" = "2 shots",
    "4" = "3 shots",
    "5" = "4 shots"
  )
)
df <- drop_any(df, "unvaccinated")

df <- recode_to_character_move_to_end(
  df,
  "vaccine_safety",
  c(
    "1" = "Completely safe",
    "2" = "Quite safe",
    "3" = "Not very safe",
    "4" = "Not safe at all"
  )
)

df <- recode_to_character_move_to_end(
  df,
  "vaccine_efficacy",
  c(
    "1" = "Very effective",
    "2" = "Quite effective",
    "3" = "Not very effective",
    "4" = "Not effective at all"
  )
)

df <- recode_to_character_move_to_end(
  df,
  "willigness_add_shot",
  c(
    "1" = "Very willing",
    "2" = "Willing",
    "3" = "Neither willing nor unwilling",
    "4" = "Unwilling",
    "5" = "Very unwilling"
  )
)

for (var in c("flu_shot", "donate_blood", "nextdose", "dose_context", "treatment", "shot_20")) {
  df <- binary_yes_no_na_move_to_end(df, var)
}

df <- drop_any(df, c("nextdose_std", "dose_context_std", "flu_shot_std", "donate_blood_std"))

if ("payments_state" %in% names(df)) {
  source <- df$payments_state
  df$payments_state[num_eq(source, 6)] <- 5
}
df <- recode_to_character_move_to_end(
  df,
  "payments_state",
  c(
    "1" = "Definitely yes",
    "2" = "Probably yes",
    "3" = "I am not sure",
    "4" = "Probably no",
    "5" = "Definitely no"
  )
)
df <- set_variable_label(df, "payments_state", "Aware of state incentive programs")

df <- recode_to_character_move_to_end(
  df,
  "heared_before_fu",
  c(
    "1" = "Definitely yes",
    "2" = "Probably yes",
    "3" = "I am not sure",
    "4" = "Probably no",
    "5" = "Definitely no"
  )
)
df <- set_variable_label(df, "heared_before_fu", "Aware of specific state incentive program")

df <- recode_to_character_move_to_end(
  df,
  "state",
  c(
    "1" = "California",
    "2" = "Florida",
    "3" = "Illinois",
    "4" = "Kentucky",
    "5" = "Louisiana",
    "6" = "Michigan",
    "7" = "Missouri",
    "8" = "New York",
    "9" = "North Carolina",
    "10" = "Ohio",
    "11" = "Pennsylvania",
    "12" = "Texas"
  )
)

df <- recode_to_character_move_to_end(
  df,
  "trust",
  c(
    "1" = "A great deal",
    "2" = "A fair amount",
    "3" = "Not very much",
    "4" = "None at all"
  ),
  missing_label = "N/A"
)

for (var in c(
  "trust_1", "trust_2", "trust_3", "trust_4",
  "safety_index_1", "safety_index_2", "safety_index_3",
  "morals_index_1", "morals_index_2", "morals_index_3"
)) {
  df <- likert_agreement_move_to_end(df, var)
}

df <- set_variable_label(df, "trust_1", "Diseases like autism, multiple sclerosis, and diabetes might be triggered through vaccination")
df <- set_variable_label(df, "trust_2", "When it comes to the COVID-19 vaccines, I trust pharmaceutical companies that develop the vaccines")
df <- set_variable_label(df, "trust_3", "When it comes to the COVID-19 vaccines, I trust the researchers who are studying the effects of the vaccines")
df <- set_variable_label(df, "trust_4", "When it comes to the COVID-19 vaccine, I trust the US Center for Disease Control")

df <- set_variable_label(df, "safety_index_1", "In general, COVID-19 vaccines are safe")
df <- set_variable_label(df, "safety_index_2", "I am worried about the side effects from COVID-19 vaccines")
df <- set_variable_label(df, "safety_index_3", "In general, COVID-19 vaccines are highly effective at protecting my health")

df <- set_variable_label(df, "morals_index_1", "I would be willing to take the personal costs of getting an additional COVID-19 vaccine shot (e.g., time, discomfort, mild side effects) for the greater good of society")
df <- set_variable_label(df, "morals_index_2", "I think people would have a civic duty or a moral obligation to get an additional COVID19 vaccine shot")
df <- set_variable_label(df, "morals_index_3", "Not taking the COVID-19 vaccine shot would be generally viewed as socially inappropriate in this situation")

df <- drop_any(df, c("state_fu", "prior", "state_paid", "heared_before", "d_aware"))

df <- drop_any(
  df,
  c(
    "safety_index_1_std", "safety_index_2_std", "safety_index_3_std",
    "morals_index_1_std", "morals_index_2_std", "morals_index_3_std",
    "shot_20_std", "trust_std", "safety_std", "morals_std", "safety", "morals"
  )
)

# Adding questions ------------------------------------------------------------
questions <- c(
  country = "What was your country of residence in 2021?",
  state = "What was your state of residence in 2021?",
  age = "What is your age?",
  gender = "Do you identify yourself as a female or a male?",
  ethnicity = "How would you describe your ethnicity/race?",
  education = "Which category best describes your highest level of education?",
  employ_status = "What is your current employment status?",
  income = "What was your TOTAL household income, before taxes, in 2021? Select an amount between $0 and $200,000+",
  pol_position = "In politics, as of today, do you consider yourself a Republican, a Democrat or an independent?",
  vote_trump = "Who did you support in the presidential election in 2020? If you did not vote, just choose the person you wanted to win the election at that time.",
  pol_position_2 = "This question applies only if you voted for Independent in the last election. As of today, do you lean more to the Democratic Party or the Republican Party?",
  vaccination_status = "How many COVID-19 vaccine shots have you taken?",
  vaccine_safety = "How safe do you think the first two doses of the currently approved COVID-19 vaccines are? Choose one of the following responses: completely safe, quite safe, not very safe, not safe at all.",
  vaccine_efficacy = "In terms of reducing the risk of becoming seriously ill from COVID-19, how effective do you think the first two doses of the currently approved COVID-19 vaccines are? Choose one of the following responses: very effective, quite effective, not very effective, not effective at all.",
  willigness_add_shot = "How willing would you be to take another COVID-19 vaccine shot within the next year if it was recommended that you do so? Choose one of the following responses: very willing, willing, neither willing nor unwilling, unwilling, very unwilling.",
  trust_1 = "To what extent do you agree with the following statement? Diseases like autism, multiple sclerosis, and diabetes might be triggered through vaccination. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  trust_2 = "To what extent do you agree with the following statement? When it comes to the COVID-19 vaccines, I trust pharmaceutical companies that develop the vaccines. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  trust_3 = "To what extent do you agree with the following statement? When it comes to the COVID-19 vaccines, I trust the researchers who are studying the effects of the vaccines. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  trust_4 = "To what extent do you agree with the following statement? When it comes to the COVID-19 vaccine, I trust the US Center for Disease Control. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  treatment = "Did you receive detailed information about your state's COVID-19 incentive program?",
  nextdose = "Do you plan to take a COVID-19 vaccine shot (regardless of whether it is your first, second, third, or fourth shot) within the next 6 months?",
  dose_context = "Suppose that there would be a new outbreak of the COVID-19 pandemic in 6 months and the Center for Disease Control would recommend people to take an additional COVID-19 vaccine shot (regardless of the number of shots they got in the past). Thinking of such a situation, would you take an additional shot?",
  safety_index_1 = "To what extent do you agree with the following statement? In general, COVID-19 vaccines are safe. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  safety_index_2 = "To what extent do you agree with the following statement? I am worried about the side effects from COVID-19 vaccines. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  safety_index_3 = "To what extent do you agree with the following statements? In general, COVID-19 vaccines are highly effective at protecting my health.",
  morals_index_1 = "Suppose that there would be a new outbreak of the COVID-19 pandemic in 6 months and the Center for Disease Control would recommend people to take an additional COVID-19 vaccine shot (regardless of the number of shots they got in the past). In this situation, to what extent do you agree with the following statement? I would be willing to take the personal costs of getting an additional COVID-19 vaccine shot (e.g., time, discomfort, mild side effects) for the greater good of society. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  morals_index_2 = "Suppose that there would be a new outbreak of the COVID-19 pandemic in 6 months and the Center for Disease Control would recommend people to take an additional COVID-19 vaccine shot (regardless of the number of shots they got in the past). In this situation, to what extent do you agree with the following statement? I think people would have a civic duty or a moral obligation to get an additional COVID19 vaccine shot. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  morals_index_3 = "Suppose that there would be a new outbreak of the COVID-19 pandemic in 6 months and the Center for Disease Control would recommend people to take an additional COVID-19 vaccine shot (regardless of the number of shots they got in the past). Not taking the COVID-19 vaccine shot would be generally viewed as socially inappropriate in this situation. Choose one of the following responses: completely agree, agree, neither agree nor disagree, disagree, completely disagree.",
  flu_shot = "Do you plan to take a flu vaccine in the 2022-2023 winter?",
  donate_blood = "Do you plan to donate blood in the next 6 months?",
  trust = "How much trust and confidence do you have in the government of the state where you live when it comes to handling state problems \u2013 a great deal, a fair amount, not very much or none at all?  Choose one of the following responses: a great deal, a fair amount, not very much, not at all.",
  shot_20 = "Suppose that there would be a new outbreak of the COVID-19 pandemic in 6 months, the Center for Disease Control would recommend people to take an additional COVID-19 vaccine shot (regardless of the number of shots they got in the past) and that every person getting an additional shot would receive $20. Thinking of such a situation, would you take an additional shot?",
  payments_state = "In 2021, did any governmental organization in your state offer any monetary compensation (for example, participation in a vax lottery or payments) to people who got a COVID-19 shot? Choose one of the following responses: definitely yes, probably yes, I am not sure, probably no, definitely no.",
  heared_before_fu = "This only applies if you received detailed information about your the state incentive program of your residence state. Had you heard about the state incentive program specific to your residence state before today? Choose one of the following responses: definitely yes, probably yes, I am not sure, probably no, definitely no."
)

df <- add_question_row_stata_style(df, questions)

# Export ----------------------------------------------------------------------
df$ID <- c("ID", as.character(seq_len(nrow(df) - 1)))
df <- df[, c("ID", setdiff(names(df), "ID")), drop = FALSE]
write_clean_csv(df, output_file)
