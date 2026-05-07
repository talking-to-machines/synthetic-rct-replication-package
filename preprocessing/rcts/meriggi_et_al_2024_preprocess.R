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

# Manually add vaccinated_endline (N/A in codebook but is the outcome variable).
manual_spec <- tribble(
  ~name,                ~question,                                                 ~response_levels,
  "subject_id",         "subject_id",                                              NA_character_,
  "treatment",          "treatment",                                               NA_character_,
  "vaccinated_endline", "Has this person been verifiably vaccinated by endline?",  "No; Yes"
)

full_spec  <- bind_rows(manual_spec, codebook_spec)
header_row <- build_inline_question_header(full_spec)

# Baseline profile variables (exclude endline / survey-logistics variables).
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
  rename(subject_id = master_person_id) %>%
  select(subject_id, treatment, vaccinated_endline, any_of(profile_vars)) %>%
  mutate(across(everything(), as.character))

df <- inject_question_header(data_clean, header_row)
df <- ensure_subject_id_first(df)

write_clean_csv(df, file.path(processed_dir, paste0(source_id, "_data.csv")))
