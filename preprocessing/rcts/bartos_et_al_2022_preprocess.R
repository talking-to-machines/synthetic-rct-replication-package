# Bartos et al. (2022) — Communicating doctors' consensus persistently
# increases COVID-19 vaccinations.
#
# Reads:
#   data/human/rcts/bartos_et_al_2022/communicating_consensus_clean.dta
# Writes:
#   data/processed/rcts/bartos_et_al_2022/bartos_et_al_2022_data.csv
#
# Keeps subjects present in Wave 0 (vlna 25) and Wave 11 (vlna 36) who were
# unvaccinated at baseline. Profile comes entirely from Wave 0; outcome is
# nQ275_r1_w11 at Wave 11 (4-category vaccination status: No / Yes one dose /
# Yes both doses / Yes three doses). nQ275_r1_w0 (always "No") is retained as
# a profile variable. 48 subjects with missing Wave 11 outcome are dropped.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(stringr)
  library(tibble)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id     <- "bartos_et_al_2022"
human_dir     <- file.path("data", "human", "rcts", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

raw_data_path <- file.path(human_dir, "communicating_consensus_clean.dta")
output_path   <- file.path(processed_dir, paste0(source_id, "_data.csv"))

data <- read_dta(raw_data_path)

# Coalesce dummy sets (education, region, townsize, estat, HHincome) into
# single categorical columns using the Stata-generated column label attributes.
extract_value_label <- function(value) {
  lbl <- attr(value, "label")
  lbl <- sub(".*\\|\\s*([^\\(]+).*", "\\1", lbl)
  str_trim(str_to_title(lbl))
}

for (prefix in c("d_educ", "d_region", "d_townsize", "d_estat", "d_HHincome")) {
  new_var <- sub("^d_", "", prefix)
  data <- data %>%
    mutate(across(starts_with(prefix),
                  ~ ifelse(. == 1, extract_value_label(.), NA_character_),
                  .names = "{.col}_lbl")) %>%
    mutate(!!new_var := coalesce(!!!select(., ends_with("_lbl")))) %>%
    select(-ends_with("_lbl"), -starts_with(prefix))
}

# str_to_title() downcases the currency abbreviation CZK → Czk; restore it.
data <- data %>% mutate(HHincome = gsub("Czk", "CZK", HHincome))

# Subjects present in both Wave 0 (vlna 25) and Wave 11 (vlna 36).
eligible_ids <- data %>%
  group_by(respondentId) %>%
  filter(all(c(25L, 36L) %in% vlna)) %>%
  pull(respondentId) %>%
  unique()

# Wave 0 profile: unvaccinated respondents only.
w0 <- data %>%
  filter(respondentId %in% eligible_ids, vlna == 25L) %>%
  arrange(respondentId) %>%
  mutate(
    sex       = as.character(haven::as_factor(sex,      levels = "labels")),
    nQ275_r1  = as.character(haven::as_factor(nQ275_r1, levels = "labels")),
    nQ276_r1  = as.character(haven::as_factor(nQ276_r1, levels = "labels")),
    nQ277_r1  = as.character(haven::as_factor(nQ277_r1, levels = "labels")),
    treatment = as.character(haven::as_factor(nQ302,    levels = "labels")),
    nQ276_r1  = ifelse(nQ276_r1 == "Inaplicable", NA_character_, nQ276_r1),
    nQ277_r1  = ifelse(nQ277_r1 == "Inaplicable", NA_character_, nQ277_r1)
  ) %>%
  filter(nQ275_r1 == "No") %>%
  rename(subject_id = respondentId, nQ275_r1_w0 = nQ275_r1) %>%
  select(subject_id, treatment,
         sex, age, hsize, children,
         educ, region, townsize, estat, HHincome,
         nQ275_r1_w0, nQ276_r1, nQ277_r1, nQ300_1_1, nQ301_1_1)

# Wave 11 outcome.
w11 <- data %>%
  filter(respondentId %in% w0$subject_id, vlna == 36L) %>%
  arrange(respondentId) %>%
  mutate(nQ275_r1_w11 = as.character(haven::as_factor(nQ275_r1, levels = "labels"))) %>%
  rename(subject_id = respondentId) %>%
  select(subject_id, nQ275_r1_w11)

data_clean <- w0 %>%
  left_join(w11, by = "subject_id") %>%
  relocate(nQ275_r1_w11, .after = treatment) %>%
  filter(!is.na(nQ275_r1_w11)) %>%
  mutate(country = "Czech Republic") %>%
  relocate(country, .after = nQ275_r1_w11) %>%
  mutate(across(everything(), as.character))

# Inline question headers.
spec <- tribble(
  ~name,                 ~question,                                                                                                                                                              ~response_levels,
  "subject_id",          NA_character_,                                                                                                                                                          NA_character_,
  "treatment",           NA_character_,                                                                                                                                                          NA_character_,
  "nQ275_r1_w11",        "Have you already been vaccinated with the coronavirus vaccine?",                                                                                                    "No; Yes, one dose; Yes, both doses; Yes, I've had three doses",
  "country",             "Which country do you live in?",                                                                                                                                        NA_character_,
  "sex",                 "What is your gender?",                                                                                                                                                 "Male; Female",
  "age",                 "What is your age in years?",                                                                                                                                           NA_character_,
  "hsize",               "How many members are there in your household?",                                                                                                                        NA_character_,
  "children",            "How many children under 18 or students are there in your household?",                                                                                                  NA_character_,
  "educ",                "What is your education level?",                                                                                                                                        "Elementary; High School No Degree; High School Degree; University",
  "region",              "Which region do you live in?",                                                                                                                                         "Central Bohemia; Hradec Králové; Karlovy Vary; Liberec; Moravia-Silesia; Olomouc; Pardubice; Plzeň; Prague; South Bohemia; South Moravia; Ústí; Vysočina; Zlín",
  "townsize",            "How many people live in your town?",                                                                                                                                   "Below 999; 1,000-1,999; 2,000-4,999; 5,000-19,999; 20,000-49,999; 50,000-99,999; Above 100,000",
  "estat",               "What is your economic status?",                                                                                                                                        "Employee; Entrepreneur; Other; Parental Leave; Retired; Student; Unemployed",
  "HHincome",            "What is your monthly net household income as provided by the Czech National Panel (pre-crisis levels)?",                                                               "Up To 10,000 CZK; 10,001 - 15,000 CZK; 15,001 - 20,000 CZK; 20,001 - 25,000 CZK; 25,001 - 30,000 CZK; 30,001 - 35,000 CZK; 35,001 - 40,000 CZK; 40,001 - 50,000 CZK; 50,001 - 60,000 CZK; Over 60,000 CZK; I Don't Know",
  "nQ275_r1_w0",         "Have you already been vaccinated with the coronavirus vaccine?",                                                                                                    "No",
  "nQ276_r1",            "Have you already registered with the Central Reservation System for COVID-19 vaccination?",                                                                            "Registration YES + term YES; Registration YES + term NO; Registration NO + term YES; Registration NO + term NO",
  "nQ277_r1",            "If the COVID-19 vaccine was available (for free), would you get vaccinated?",                                                                                          "Definitely YES; Rather YES; Rather NO; Definitely NO; Don't know",
  "nQ300_1_1",           "What do you think is the percentage of doctors who would be interested in getting vaccinated, voluntarily, and free of charge, with an approved vaccine against Covid-19?", "a number from 0 to 100",
  "nQ301_1_1",           "What do you think is the percentage of doctors who trust Covid-19 vaccines that have been approved by the European Medicines Agency (EMA) approval process?",         "a number from 0 to 100"
)

header_row <- build_inline_question_header(spec)
df <- inject_question_header(data_clean, header_row)
df <- ensure_subject_id_first(df)

write_clean_csv(df, output_path)
