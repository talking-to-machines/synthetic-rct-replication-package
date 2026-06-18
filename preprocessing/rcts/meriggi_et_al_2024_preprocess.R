# Meriggi et al. (2024) — "Last Mile" RCT (Sierra Leone, COVID-19 vaccination).
#
# Reads:
#   data/human/rcts/meriggi_et_al_2024/individual_level.dta
#   data/human/rcts/meriggi_et_al_2024/codebook_individual_level.xlsx
# Writes:
#   data/processed/rcts/meriggi_et_al_2024/meriggi_et_al_2024_data.csv
#
# Treatment:  treatment (0=Control, 1=Individual Mobilization, 2=Group Mobilization)
# Outcome:    vaccinated_endline (verifiably vaccinated by endline)
# Profile:    baseline demographics and COVID-19 beliefs only; endline variables
#             excluded to avoid outcome leakage.

suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(haven)
  library(labelled)
})

if (!exists("build_inline_question_header")) source("preprocessing/utils.R")

source_id     <- "meriggi_et_al_2024"
human_dir     <- file.path("data", "human", "rcts", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

individual_dta <- file.path(human_dir, "individual_level.dta")
if (!file.exists(individual_dta)) {
  stop("Required raw file not found: ", individual_dta,
       " — drop individual_level.dta and the xlsx codebooks into ", human_dir)
}

individual_data <- haven::read_dta(individual_dta) %>%
  mutate(across(everything(), labelled::to_factor))

individual_codebook <- read_excel(
  file.path(human_dir, "codebook_individual_level.xlsx")
)

# Extract variable -> question pairs from codebook (rows with real question text).
codebook_spec <- individual_codebook %>%
  filter(!is.na(`Variable Name`),
         !(`Original Question` %in% c("N/A", NA))) %>%
  select(`Variable Name`, `Original Question`) %>%
  distinct(`Variable Name`, .keep_all = TRUE) %>%
  rename(name = `Variable Name`, question = `Original Question`) %>%
  mutate(response_levels = NA_character_)

# Manual spec: outcome (N/A in codebook), plus overrides for vars whose codebook
# question texts do not match their derived/binary nature in the data.
# When a name appears here AND in codebook_spec, the manual entry wins because
# bind_rows puts manual_spec first and distinct() keeps the first occurrence.
manual_spec <- tribble(
  ~name,                  ~question,                                                                              ~response_levels,
  "ID",                   "ID",                                                                                   NA_character_,
  "treatment",            "treatment",                                                                            NA_character_,
  "vaccinated_endline",   "Has this person been verifiably vaccinated by endline?",                               "No; Yes",
  "anyschooling",         "Has the head of household ever attended school?",                                      NA_character_,
  "farmer",               "Is farming the main source of income for the household?",                              NA_character_,
  "BSL_reduced_portions", "Has your household reduced food portions on more than one day over the past week?",    NA_character_,
  "religion",             "What is the religion of the head of household?",                                       NA_character_,
  "BSL_trust",            "Who do you most trust getting information about COVID-19?",                            NA_character_,
  "BSL_safe_stragree",    "Do you strongly agree with this statement: COVID-19 vaccines are safe?",              NA_character_,
  "BSL_effect_stragree",  "Do you strongly agree with this statement: COVID-19 vaccines are effective?",         NA_character_
)

full_spec  <- bind_rows(manual_spec, codebook_spec) %>%
  distinct(name, .keep_all = TRUE)
header_row <- build_inline_question_header(full_spec)

# Baseline profile variables (exclude endline / survey-logistics variables).
# christian + muslim are raw dummies in the DTA; they are selected here so they
# can be combined into a single `religion` column, then dropped.
# BSL_trust_* are five select-one dummies; similarly collapsed into `BSL_trust`.
profile_vars <- c(
  "age", "female", "hh_gender", "preg", "breast",
  "anyschooling", "farmer", "christian", "muslim",
  "BSL_owns_land", "BSL_reduced_portions", "BSL_assets",
  "vaccinated_baseline",
  "BSL_covid_believe", "BSL_covid_know", "BSL_covid_wouldtake",
  "BSL_trust_chc", "BSL_trust_famfriend", "BSL_trust_socmedia",
  "BSL_trust_media", "BSL_trust_mohs",
  "BSL_safe_stragree", "BSL_effect_stragree"
)

data_clean <- individual_data %>%
  rename(ID = master_person_id) %>%
  select(ID, treatment, vaccinated_endline, any_of(profile_vars)) %>%
  mutate(
    # Outcome: no Stata value labels (int8 0/1); recode explicitly.
    vaccinated_endline = case_when(
      as.character(vaccinated_endline) == "0" ~ "No",
      as.character(vaccinated_endline) == "1" ~ "Yes",
      TRUE                                    ~ NA_character_
    ),
    # Profile binary vars with no Stata value labels.
    anyschooling = case_when(
      as.character(anyschooling) == "0" ~ "No",
      as.character(anyschooling) == "1" ~ "Yes",
      TRUE                              ~ NA_character_
    ),
    farmer = case_when(
      as.character(farmer) == "0" ~ "No",
      as.character(farmer) == "1" ~ "Yes",
      TRUE                        ~ NA_character_
    ),
    BSL_reduced_portions = case_when(
      as.character(BSL_reduced_portions) == "0" ~ "No",
      as.character(BSL_reduced_portions) == "1" ~ "Yes",
      TRUE                                      ~ NA_character_
    ),
    # Combine christian/muslim dummies into a single religion column.
    religion = case_when(
      as.character(christian) == "1"                                          ~ "Christian",
      as.character(muslim)    == "1"                                          ~ "Muslim",
      as.character(christian) == "0" & as.character(muslim) == "0"            ~ "Other",
      TRUE                                                                     ~ NA_character_
    ),
    # Likert-derived dummies: 1 = "strongly agree", 0 = did not strongly agree.
    BSL_safe_stragree = case_when(
      as.character(BSL_safe_stragree) == "0" ~ "No",
      as.character(BSL_safe_stragree) == "1" ~ "Yes",
      TRUE                                   ~ NA_character_
    ),
    BSL_effect_stragree = case_when(
      as.character(BSL_effect_stragree) == "0" ~ "No",
      as.character(BSL_effect_stragree) == "1" ~ "Yes",
      TRUE                                     ~ NA_character_
    ),
    # Reconstruct BSL_trust from the five select-one dummies.
    # Respondents with all dummies == 0 (chose an option outside these five) → NA.
    BSL_trust = case_when(
      as.character(BSL_trust_chc)       == "1" ~ "Community Health Centre",
      as.character(BSL_trust_famfriend) == "1" ~ "Family and friends",
      as.character(BSL_trust_socmedia)  == "1" ~ "Social media",
      as.character(BSL_trust_media)     == "1" ~ "Media (i.e. news/radio/tv)",
      as.character(BSL_trust_mohs)      == "1" ~ "Ministry of Health and Sanitation",
      TRUE                                      ~ NA_character_
    )
  ) %>%
  # Drop the dummy columns now that derived columns exist.
  select(-c(christian, muslim,
            BSL_trust_chc, BSL_trust_famfriend, BSL_trust_socmedia,
            BSL_trust_media, BSL_trust_mohs)) %>%
  # Place derived columns where the dummies were.
  relocate(religion,   .after = farmer) %>%
  relocate(BSL_trust,  .after = BSL_covid_wouldtake) %>%
  mutate(across(everything(), as.character))

df <- inject_question_header(data_clean, header_row)
df <- ensure_ID_first(df)

write_clean_csv(df, file.path(processed_dir, paste0(source_id, "_data.csv")))
