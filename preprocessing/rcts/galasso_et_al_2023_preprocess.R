# Data cleaning script translated from data_cleaning_Galasso.do
# Supplementary-materials version: retains the original Stata recodes,
# variable names, output ordering, and spelling/typographical choices as closely as possible.

# Required packages -----------------------------------------------------------
# install.packages(c("dplyr", "haven", "readr"))
library(dplyr)
library(haven)
library(readr)

# Paths -----------------------------------------------------------------------
if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human", "rcts", "galasso_et_al_2023")
saving_path <- file.path("data", "processed", "rcts", "galasso_et_al_2023")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

us_file     <- file.path(data_path, "vaccine_US2022.dta")
sweden_file <- file.path(data_path, "vaccine_SW2022.dta")
output_file <- file.path(saving_path, "galasso_et_al_2023_data.csv")

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

num_in <- function(x, values) {
  y <- to_num(x)
  !is.na(y) & y %in% values
}

is_missing_any <- function(x) {
  is.na(x)
}

to_stata_string <- function(x) {
  y <- haven::zap_labels(x)

  if (is.numeric(y)) {
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
  out[is.na(x) & !tagged] <- "."
  out
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

order_any <- function(df, vars) {
  ordered <- intersect(vars, names(df))
  df[, c(ordered, setdiff(names(df), ordered)), drop = FALSE]
}

recode_to_character <- function(df, var, map, missing_any = FALSE,
                                missing_codes = NULL, missing_label = "N/A") {
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)

  for (value in names(map)) {
    out[num_eq(source, as.numeric(value))] <- unname(map[[value]])
  }

  missing <- rep(FALSE, length(out))
  if (missing_any) missing <- missing | is_missing_any(source)
  if (!is.null(missing_codes)) missing <- missing | num_in(source, missing_codes)
  out[missing] <- missing_label

  df[[var]] <- out
  df
}

recode_many_to_character <- function(df, vars, map, missing_any = FALSE,
                                     missing_codes = NULL) {
  for (var in vars) {
    df <- recode_to_character(
      df, var, map,
      missing_any = missing_any,
      missing_codes = missing_codes
    )
  }
  df
}

stringify_variable <- function(df, var, missing_any = FALSE,
                               missing_codes = NULL, missing_label = "N/A") {
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)

  missing <- rep(FALSE, length(out))
  if (missing_any) missing <- missing | is_missing_any(source)
  if (!is.null(missing_codes)) missing <- missing | num_in(source, missing_codes)
  out[missing] <- missing_label

  df[[var]] <- out
  df
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
  # Mirrors the Stata pattern:
  #   insobs 1, before(1)
  #   replace var = "question" if var == var[1]
  # This intentionally replaces all blank string values with the question text,
  # not only the inserted first row, because that is what the Stata code does.
  df <- df %>% mutate(across(everything(), as.character))
  header <- as_tibble(as.list(setNames(rep("", ncol(df)), names(df))))
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
us_data <- read_dta_flexible(us_file) %>%
  mutate(across(everything(), haven::zap_labels))
us_data$country <- "US"

sweden_data <- read_dta_flexible(sweden_file) %>%
  mutate(across(everything(), haven::zap_labels))

df <- bind_rows(us_data, sweden_data)

df$country[is.na(df$country) | df$country == ""] <- "Sweden"

# Age groups ------------------------------------------------------------------
df <- recode_to_character(
  df,
  "agegroup_1",
  map = c(
    "1" = "18-34",
    "2" = "35-49",
    "3" = "50-59",
    "4" = "60+"
  )
)
df <- drop_any(df, "agegroup_2")

# Education -------------------------------------------------------------------
education <- blank_character(nrow(df))
if ("nohs_1" %in% names(df)) education <- assign_if(education, num_eq(df$nohs_1, 1), "Less than high school")
if ("hs_1" %in% names(df)) education <- assign_if(education, num_eq(df$hs_1, 1), "High school")
if ("college_1" %in% names(df)) education <- assign_if(education, num_eq(df$college_1, 1), "Bachelor degree or higher")
df$education <- education

df <- drop_any(df, c("nohs_1", "hs_1", "college_1", "nohs_2", "hs_2", "college_2"))

# Occupation ------------------------------------------------------------------
df <- recode_to_character(
  df,
  "occupation_1",
  map = c(
    "1" = "White collar occupation",
    "2" = "Service worker",
    "3" = "Blue collar occupation",
    "4" = "Inactive"
  )
)
df <- drop_any(df, "occupation_2")

# Sex -------------------------------------------------------------------------
df <- recode_to_character(
  df,
  "female_1",
  map = c(
    "1" = "Female",
    "0" = "Male"
  )
)
df <- drop_any(df, "female_2")

# Information sources ---------------------------------------------------------
info_vars <- c(
  "info_tv_1",
  "info_radio_1",
  "info_newpaper_1",
  "info_social_1",
  "info_internet_1"
)

df <- recode_many_to_character(
  df,
  info_vars,
  map = c(
    "1" = "Never",
    "2" = "Rarely",
    "3" = "Sometimes",
    "4" = "Often",
    "5" = "Always"
  ),
  missing_any = TRUE
)

# Yes/no variables ------------------------------------------------------------
yes_no_vars <- c(
  "vac_infect_1",
  "vac_contag_1",
  "vac_country_1",
  "vac_econ_1",
  "vaccinedone",
  "trustscientists_1",
  "gentrust_1",
  "trustscientists_2",
  "gentrust_2",
  "obbvaccino_1",
  "obbvaccino_2",
  "serioushealthcons_1",
  "serioushealthcons_2",
  "agreeclose_1",
  "covidyou_1",
  "covidhome_1",
  "covidfamily_1",
  "covidfriends_1",
  "covidfriends_2",
  "covidno_1",
  "covidyou_2",
  "covidhome_2",
  "covidfamily_2"
)

df <- recode_many_to_character(
  df,
  yes_no_vars,
  map = c(
    "0" = "No",
    "1" = "Yes"
  ),
  missing_any = TRUE
)

# Vaccination probability -----------------------------------------------------
df <- stringify_variable(df, "probvaccino_1", missing_any = TRUE)
df <- stringify_variable(df, "probvaccino_2", missing_any = TRUE)

# Living arrangement ----------------------------------------------------------
live <- blank_character(nrow(df))
if ("livealone_1" %in% names(df)) live <- assign_if(live, num_eq(df$livealone_1, 1), "I live alone")
if ("livefamily_1" %in% names(df)) live <- assign_if(live, num_eq(df$livefamily_1, 1), "I live with my family")
if ("liveothers_1" %in% names(df)) live <- assign_if(live, num_eq(df$liveothers_1, 1), "I live with other people who are not my family")
live[live == ""] <- "N/A"
df$live <- live

df <- drop_any(df, c("livefamily_1", "livealone_1", "liveothers_1"))

live_6 <- blank_character(nrow(df))
if ("livealone_2" %in% names(df)) live_6 <- assign_if(live_6, num_eq(df$livealone_2, 1), "I live alone")
if ("livefamily_2" %in% names(df)) live_6 <- assign_if(live_6, num_eq(df$livefamily_2, 1), "I live with my family")
if ("liveothers_2" %in% names(df)) live_6 <- assign_if(live_6, num_eq(df$liveothers_2, 1), "I live with other people who are not my family")
# Preserves original Stata code: replace live_6 = "N/A" if live == ""
live_6[df$live == ""] <- "N/A"
df$live_6 <- live_6

df <- drop_any(df, c("livefamily_2", "livealone_2", "liveothers_2"))

# Perceived infection and illness probabilities -------------------------------
df <- stringify_variable(df, "probinfected_1", missing_any = TRUE)
df <- stringify_variable(df, "probinfected_2", missing_any = TRUE)
df <- stringify_variable(df, "probsill_1", missing_any = TRUE)
df <- stringify_variable(df, "probsill_2", missing_any = TRUE)

# Political affiliation -------------------------------------------------------
political_aff <- blank_character(nrow(df))
if ("liberal_1" %in% names(df)) political_aff <- assign_if(political_aff, num_eq(df$liberal_1, 1), "Liberal")
if ("centrist_1" %in% names(df)) political_aff <- assign_if(political_aff, num_eq(df$centrist_1, 1), "Centrist")
if ("conservative_1" %in% names(df)) political_aff <- assign_if(political_aff, num_eq(df$conservative_1, 1), "Conservative")
if ("ideologydontknow_1" %in% names(df)) political_aff <- assign_if(political_aff, num_eq(df$ideologydontknow_1, 1), "I don't know")
df$political_aff <- political_aff

df <- drop_any(df, c("liberal_1", "centrist_1", "conservative_1", "ideologydontknow_1"))

political_aff_6 <- blank_character(nrow(df))
if ("liberal_2" %in% names(df)) political_aff_6 <- assign_if(political_aff_6, num_eq(df$liberal_2, 1), "Liberal")
if ("centrist_2" %in% names(df)) political_aff_6 <- assign_if(political_aff_6, num_eq(df$centrist_2, 1), "Centrist")
if ("conservative_2" %in% names(df)) political_aff_6 <- assign_if(political_aff_6, num_eq(df$conservative_2, 1), "Conservative")
if ("ideologydontknow_2" %in% names(df)) political_aff_6 <- assign_if(political_aff_6, num_eq(df$ideologydontknow_2, 1), "I don't know")
df$political_aff_6 <- political_aff_6

df <- drop_any(df, c("liberal_2", "centrist_2", "conservative_2", "ideologydontknow_2"))

# Attribution, compliance, vaccination solution, and related scales -----------
df <- stringify_variable(df, "multin_fault_1", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "multin_fault_2", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "virus_china_1", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "virus_china_2", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "noblame_1", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "blame_people_1", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "compliant_1", missing_any = TRUE)
df <- stringify_variable(df, "compliant_2", missing_any = TRUE)

df <- drop_any(df, c("overall_compliance_1", "overall_compliance_2"))

df <- stringify_variable(df, "solvaccino_1", missing_any = TRUE, missing_codes = 99)
df <- stringify_variable(df, "solvaccino_2", missing_any = TRUE, missing_codes = 99)

# Mental-health frequency variables ------------------------------------------
feeling_vars <- c(
  "feeling_nointerest_2",
  "feeling_nointerest_1",
  "feeling_low_1",
  "feeling_low_2"
)

df <- recode_many_to_character(
  df,
  feeling_vars,
  map = c(
    "1" = "Never",
    "2" = "Sometimes",
    "3" = "Often",
    "4" = "Always"
  ),
  missing_any = TRUE
)

df <- stringify_variable(df, "noenoughknowledge_2", missing_any = TRUE)
df <- stringify_variable(df, "riskadv_1", missing_any = TRUE)
# Preserves original Stata temporary name reuse:
#   tostring riskadv_2, gen(riskadv_1_)
#   rename riskadv_1_ riskadv_2
# The final exported variable is riskadv_2.
df <- stringify_variable(df, "riskadv_2", missing_any = TRUE)
df <- stringify_variable(df, "socdist_2", missing_any = TRUE)
df <- stringify_variable(df, "blame_EU_1", missing_any = TRUE)

# Drop unused variables -------------------------------------------------------
df <- drop_any(
  df,
  c(
    "mask_1",
    "goout_1",
    "blame_measure_1",
    "hide_info_gvt_1",
    "hide_info_scientists_1",
    "virus_bat_1",
    "region",
    "weights_1",
    "covidno_1",
    "mask_2",
    "goout_2",
    "hide_info_gvt_2",
    "hide_info_scientists_2",
    "virus_bat_2",
    "weights_2",
    "country_code",
    "ds_1",
    "vac_plain_1",
    "vac_altruism_1",
    "vac_own_1",
    "ds_2",
    "schoolkids_1",
    "noschoolclose_1",
    "covidno_2"
  )
)

# Treatment assignment --------------------------------------------------------
treatment <- blank_character(nrow(df))
if ("vac_infect_1" %in% names(df)) treatment <- assign_if(treatment, df$vac_infect_1 == "Yes", "Self-protection")
if ("vac_contag_1" %in% names(df)) treatment <- assign_if(treatment, df$vac_contag_1 == "Yes", "Protecting others")
if ("vac_country_1" %in% names(df)) treatment <- assign_if(treatment, df$vac_country_1 == "Yes", "Health risk")
if ("vac_econ_1" %in% names(df)) treatment <- assign_if(treatment, df$vac_econ_1 == "Yes", "Economic protection")
treatment[treatment == ""] <- "Control"
df$treatment <- treatment

# Adding questions ------------------------------------------------------------
questions <- c(
  country = "What was your country of residence at the beginning of December 2020?",
  trustscientists_1 = "Do you trust scientists?",
  gentrust_1 = "Do you generally trust others?",
  info_tv_1 = "How often do you watch news on TV?",
  info_radio_1 = "How often do you listen to news on the radio?",
  info_newpaper_1 = "How often do you read newspapers?",
  info_social_1 = "How often do you read news on social media?",
  info_internet_1 = "How often do you read news on the internet?",
  noblame_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that nobody is responsible for the outbreak of the virus?",
  blame_people_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that other people responsible for the outbreak of the virus?",
  virus_china_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that China is responsible for the outbreak of the virus?",
  feeling_nointerest_1 = "How often do you experience low interest or pleasure in general?",
  feeling_low_1 = "How often do you feel down or low in mood?",
  probinfected_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that you will get infected if you resume the usual daily activities?",
  probsill_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that you will get seriously ill if you get infected?",
  serioushealthcons_1 = "Do you think Covid-19 pandemic is having serious consequences on health in your country?",
  probvaccino_1 = "Assuming the vaccine becomes available, on a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely are you to get vaccinated?",
  multin_fault_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that the virus was created by 'big pharma companies' primarily to generate profit?",
  solvaccino_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that the vaccine is the solution?",
  obbvaccino_1 = "Have you received all mandatory vaccinations?",
  agegroup_1 = "How old are you?",
  occupation_1 = "What is your occupation?",
  covidyou_1 = "Have you ever had covid?",
  covidhome_1 = "Has anyone in your household had covid?",
  covidfamily_1 = "Has anyone in your family had covid?",
  covidfriends_1 = "Has any of your friends had covid?",
  trustscientists_2 = "After 6 months: Do you trust scientists?",
  gentrust_2 = "After 6 months: Do you generally trust others?",
  virus_china_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that China is responsible for the outbreak of the virus?",
  feeling_nointerest_2 = "After 6 months: How often do you experience low interest or pleasure in general?",
  feeling_low_2 = "After 6 months: How often do you feel down or low in mood?",
  probinfected_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that you will get infected if you resume the usual daily activities?",
  probsill_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that you will get seriously ill if you get infected?",
  serioushealthcons_2 = "After 6 months: Do you think Covid-19 pandemic is having serious consequences on health in your country?",
  probvaccino_2 = "After 6 months: Assuming the vaccine becomes available, on a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely are you to get vaccinated?",
  multin_fault_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that the virus was created by 'big pharma companies' primarily to generate profit?",
  solvaccino_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that the vaccine is the solution?",
  noenoughknowledge_2 = "After 6 months: Due to the expedition of clinical trials for the Covid-19 vaccines, the possible side effects of the vaccine are unknown. On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that the vaccine has side effects?",
  obbvaccino_2 = "After 6 months: Have you received all mandatory vaccinations?",
  socdist_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is that strictly complying with social distancing and other health measures reduces the risk of being infected?",
  covidyou_2 = "After 6 months: Have you ever had covid?",
  covidhome_2 = "After 6 months: Has anyone in your household had covid?",
  covidfamily_2 = "After 6 months: Has anyone in your family had covid?",
  covidfriends_2 = "After 6 months: Has any of your friends had covid?",
  compliant_1 = "On a scale from 0 to 100 - where 0 means 'not at all compliant' and 100 'fully compliant' - how compliant do you think others are with restrictions?",
  female_1 = "Are you a male or a female?",
  vac_infect_1 = "Given that the only way to become immune to COVID-19 in the long run is by vaccination, do you agree with the following statement? If you were vaccinated, you might be able to avoid passing the virus on to others.",
  vac_contag_1 = "Given that the only way to become immune to COVID-19 in the long run is by vaccination, do you agree with the following statement? If you were vaccinated, you might be able to avoid passing the virus on to others.",
  vac_country_1 = "Given that the only way to become immune to COVID-19 in the long run is by vaccination, do you agree with the following statement? If a person was vaccinated, they could avoid getting infected with the virus. This would protect the health of people in the US.",
  vac_econ_1 = "Given that the only way to become immune to COVID-19 in the long run is by vaccination, do you agree with the following statement? If a person was vaccinated, they could avoid getting infected with the virus. It would allow a return to normal economic activity and reduce unemployment.",
  riskadv_1 = "On a scale from 0 to 10 - where 0 means 'not at all risk-averse' and 10 means 'very risk-averse' - How risk-averse are you?",
  agreeclose_1 = "Do you agree with the decision to close schools?",
  compliant_2 = "After 6 months: On a scale from 0 to 100 - where 0 means 'not at all compliant' and 100 'fully compliant' - how compliant do you think others are with restrictions",
  vaccinedone = "Did you get vaccinated against COVID-19 by July 2021?",
  riskadv_2 = "After 6 months: On a scale from 0 to 10 - where 0 means 'not at all risk-averse' and 10 means 'very risk-averse' - How risk-averse are you?",
  education = "What is your level of education?",
  live_6 = "After 6 months: Do you live alone, with your family, or with other people who are not your family?",
  live = "Do you live alone, with your family, or with other people who are not your family?",
  political_aff = "What is your political affilitation? Choose one of the following options: liberal, centrist, conservative, or I don't know.",
  political_aff_6 = "After 6 months: What is your political affilitation? Choose one of the following options: liberal, centrist, conservative, or I don't know.",
  blame_EU_1 = "On a scale from 0 to 10 - where 0 means 'not at all likely' and 10 means 'completely certain' - how likely do you think it is the European Union is responsible for the outbreak of the virus?",
  treatment = "Treatment"
)

df <- add_question_row_stata_style(df, questions)

# Final order and export ------------------------------------------------------
final_order <- c(
  "country",
  "female_1",
  "education",
  "live",
  "political_aff",
  "trustscientists_1",
  "gentrust_1",
  "info_tv_1",
  "info_radio_1",
  "info_newpaper_1",
  "info_social_1",
  "info_internet_1",
  "noblame_1",
  "blame_people_1",
  "virus_china_1",
  "feeling_nointerest_1",
  "feeling_low_1",
  "probinfected_1",
  "probsill_1",
  "serioushealthcons_1",
  "multin_fault_1",
  "solvaccino_1",
  "obbvaccino_1",
  "agegroup_1",
  "occupation_1",
  "covidyou_1",
  "covidhome_1",
  "covidfamily_1",
  "covidfriends_1",
  "compliant_1",
  "riskadv_1",
  "agreeclose_1",
  "blame_EU_1",
  "treatment",
  "probvaccino_1",
  "trustscientists_2",
  "gentrust_2",
  "virus_china_2",
  "feeling_nointerest_2",
  "feeling_low_2",
  "probinfected_2",
  "probsill_2",
  "serioushealthcons_2",
  "multin_fault_2",
  "solvaccino_2",
  "noenoughknowledge_2",
  "socdist_2",
  "covidyou_2",
  "covidhome_2",
  "covidfamily_2",
  "covidfriends_2",
  "compliant_2",
  "riskadv_2",
  "live_6",
  "political_aff_6",
  "vaccinedone",
  "obbvaccino_2",
  "probvaccino_2"
)

df <- order_any(df, final_order)
df <- drop_any(df, c("vac_infect_1", "vac_contag_1", "vac_country_1", "vac_econ_1"))

df$subject_id <- c("subject_id", as.character(seq_len(nrow(df) - 1)))
df <- df[, c("subject_id", setdiff(names(df), "subject_id")), drop = FALSE]
write_clean_csv(df, output_file)
