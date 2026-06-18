# Campos-Mercade et al. (2021) — Monetary incentives increase COVID-19 vaccinations
#
# Reads:
#   data/human/rcts/campos_mercade_et_al_2021/data_analysis.dta
#   data/human/rcts/campos_mercade_et_al_2021/campos_mercade_et_al_2021_mapping.csv
# Writes:
#   data/processed/rcts/campos_mercade_et_al_2021/campos_mercade_et_al_2021_data.csv
#
# Outcome: vaccinated (received first COVID-19 dose within 30 days of availability).
# Intention variables (intention1, intention2_1, intention3) are excluded: they were
# measured immediately after treatment assignment in the same survey wave (post-treatment
# intermediate outcomes, not pre-treatment covariates).
# belief_prot_death / belief_cause_death are excluded: ~87% missing, measured only
# within one specific treatment arm (arm-specific post-treatment outcomes).

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(tibble)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id     <- "campos_mercade_et_al_2021"
human_dir     <- file.path("data", "human", "rcts", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

raw_data_path <- file.path(human_dir, "data_analysis.dta")
mapping_path  <- file.path(human_dir, paste0(source_id, "_mapping.csv"))
output_path   <- file.path(processed_dir, paste0(source_id, "_data.csv"))

raw     <- haven::read_dta(raw_data_path)
mapping <- read_mapping_csv(mapping_path)

# Exclude post-treatment intention variables; add week_num (enrollment timing).
post_treatment_vars <- c("intention1", "intention2_1", "intention3")
selected_vars <- mapping %>%
  filter(selected == 1, !name %in% post_treatment_vars) %>%
  pull(name)

data_subset <- raw %>%
  select(all_of(selected_vars), week_num) %>%
  # Falk scales are stored 1–11 in Stata (1-indexed); subtract 1 to restore 0–10.
  mutate(across(starts_with("falk"), ~ as.integer(.) - 1L)) %>%
  # Convert Stata weekly codes to ISO week labels (e.g. "2021-W22").
  # Stata weekly epoch: weeks since 1960-01-01; week n => year 1960 + n%/%52,
  # ISO week (n%%52)+1.
  mutate(week_num = {
    w <- as.integer(week_num)
    sprintf("%d-W%02d", 1960L + w %/% 52L, (w %% 52L) + 1L)
  }) %>%
  mutate(across(everything(), as.character))

lookup       <- build_options_lookup(mapping)
data_decoded <- apply_human_readable(data_subset, lookup) %>%
  mutate(across(everything(), ~ na_if(.x, "")))

# Inline spec: controls question text and whether options appear in the header
# independently of the value-decoding step above.
# Plain discrete categoricals have response_levels = NA (no options list).
# Likert agreement scales (covid2_*) keep their option lists.
# Continuous scales (falk*) use a prose anchor without enumerated options.
spec <- tribble(
  ~name,          ~question,                                                                                                                                                                                                             ~response_levels,
  "ID",   NA_character_,                                                                                                                                                                                                          NA_character_,
  "treatment",    NA_character_,                                                                                                                                                                                                          NA_character_,
  "vaccinated",   "Have you got a first shot of a COVID-19 vaccine within the first 30 days after the vaccine became available to you? Available means that vaccinations started for people in your age group in your region.",            NA_character_,
  "largeregions", "In which region do you live?",                                                                                                                                                                                         NA_character_,
  "age",          "What year were you born?",                                                                                                                                                                                             NA_character_,
  "week_num",     "In which calendar week did you participate in the survey?",                                                                                                                                                              NA_character_,
  "female",       "Do you identify yourself as a woman or a man?",                                                                                                                                                                        NA_character_,
  "civilstatus",  "What describes your civil status best?",                                                                                                                                                                               NA_character_,
  "education",    "What education do you have (fill in the highest you have)?",                                                                                                                                                           NA_character_,
  "occupation",   "What is your employment status?",                                                                                                                                                                                                      "Work; Unemployed; Student; Pensioner; Other",
  "income",       "How much in Swedish kronor is your household's total income per month after taxes including public benefits? Calculate also your loan if you are a student. Please answer even if you are not sure.",                  NA_character_,
  "haschildren",  "Does any child live in your household?",                                                                                                                                                                               NA_character_,
  "mother",       "Where was your mother born?",                                                                                                                                                                                          NA_character_,
  "father",       "Where was your father born?",                                                                                                                                                                                          NA_character_,
  "falk1_1",      "How willing are you to give to good causes without expecting anything in return?",                                                                                                                                     "a number from 0 to 10, where 0 means completely unwilling and 10 means very willing",
  "falk1_2",      "In general, how willing are you to take risks?",                                                                                                                                                                       "a number from 0 to 10, where 0 means completely unwilling and 10 means very willing",
  "falk1_3",      "How willing are you to give up something that is beneficial for you today in order to benefit more from that in the future?",                                                                                          "a number from 0 to 10, where 0 means completely unwilling and 10 means very willing",
  "falk2_1",      "How well do the following statements describe you as a person? When someone does me a favor, I am willing to return it.",                                                                                              "a number from 0 to 10, where 0 means does not describe me at all and 10 means describes me perfectly",
  "falk2_2",      "How well do the following statements describe you as a person? I assume that people have only the best intentions.",                                                                                                    "a number from 0 to 10, where 0 means does not describe me at all and 10 means describes me perfectly",
  "falk2_3",      "How well do the following statements describe you as a person? I postpone starting on things I dislike to do.",                                                                                                        "a number from 0 to 10, where 0 means does not describe me at all and 10 means describes me perfectly",
  "falk2_4",      "How well do the following statements describe you as a person? It is important for me to always behave properly and to avoid doing anything people would say is wrong.",                                               "a number from 0 to 10, where 0 means does not describe me at all and 10 means describes me perfectly",
  "covid1_1",     "Have you ever tested positive for COVID-19 or COVID-19 antibodies?",                                                                                                                                                   NA_character_,
  "covid1_2",     "Are you in an at-risk group for COVID-19?",                                                                                                                                                                            NA_character_,
  "covid2_1",     "To what extent do you agree with the following statement: In general, COVID-19 vaccines are safe.",                                                                                                                    "Completely disagree; Disagree; Neither agree nor disagree; Agree; Completely agree",
  "covid2_2",     "To what extent do you agree with the following statement: Diseases like autism, multiple sclerosis, and diabetes might be triggered through vaccination.",                                                             "Completely disagree; Disagree; Neither agree nor disagree; Agree; Completely agree",
  "covid2_3",     "To what extent do you agree with the following statement: I am worried about the side effects from COVID-19 vaccines.",                                                                                                "Completely disagree; Disagree; Neither agree nor disagree; Agree; Completely agree",
  "covid2_4",     "To what extent do you agree with the following statement: I am afraid of the needles used for vaccination.",                                                                                                           "Completely disagree; Disagree; Neither agree nor disagree; Agree; Completely agree"
)

header_row <- build_inline_question_header(spec)
df <- inject_question_header(data_decoded, header_row)
df <- ensure_ID_first(df)

write_clean_csv(df, output_path)
