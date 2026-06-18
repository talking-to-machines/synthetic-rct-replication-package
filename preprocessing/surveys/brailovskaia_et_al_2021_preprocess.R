# Brailovskaia et al. (2021) — Multi-country COVID-19 vaccination survey.
#
# Reads:
#   data/human/surveys/brailovskaia_et_al_2021/pone.0260230.s001.sav
# Writes (one per country):
#   data/processed/surveys/brailovskaia_et_al_2021/brailovskaia_et_al_2021_<country>_data.csv

suppressPackageStartupMessages({
  library(foreign)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(tibble)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "brailovskaia_et_al_2021"
human_dir     <- file.path("data", "human", "surveys", source_id)
processed_dir <- file.path("data", "processed", "surveys", source_id)
raw_data_path <- file.path(human_dir, "pone.0260230.s001.sav")

# Load once; per-country filtering done in the loop.
raw_all <- read.spss(raw_data_path, to.data.frame = TRUE)

# Per-variable question text and response-level descriptions.
spec <- tribble(
  ~name,                                  ~question,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ~response_levels_template,
  "country_all",                          "Which country do you live in?",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            NA_character_,
  "gender_all",                           "What is your gender?",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     "DYNAMIC",
  "age_grp_all",                          "What is your age group?",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  "18 - 24; 25 - 34; 35 - 44; 45 - 54; 55+",
  "marital_status",                       "What is your martial status?",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             "DYNAMIC",
  "social_class",                         "Where would you place your social class on a scale from 1 (lower class) to 6 (upper class)?",                                                                                                                                                                                                                                                                                                                                                                                                                                NA_character_,
  "urbanicity",                           "Do you live in a small city/rural community or a large city?",                                                                                                                                                                                                                                                                                                                                                                                                                                                              "DYNAMIC",
  "vaccination",                          "Have you already been vaccinated against COVID-19 at least once?",                                                                                                                                                                                                                                                                                                                                                                                                                                                          "Yes; No, but I would like to be vaccinated; No, and I do not want to be vaccinated",
  "risk_group",                           "Do you belong to the COVID-19 risk group?",                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 "Yes; No",
  "affectedness_physicalhealth",          "To what extent were you affected by the COVID-19 situation in terms of physical health?",                                                                                                                                                                                                                                                                                                                                                                                                                                  "an integer between 0 (not at all) and 4 (very strong)",
  "affectedness_economically",            "To what extent were you affected by the COVID-19 situation economically?",                                                                                                                                                                                                                                                                                                                                                                                                                                                  "an integer between 0 (not at all) and 4 (very strong)",
  "affectedness_mentally",                "To what extent were you affected by the COVID-19 situation mentally?",                                                                                                                                                                                                                                                                                                                                                                                                                                                      "an integer between 0 (not at all) and 4 (very strong)",
  "information_TV",                       "How frequently do you use news reports on television as a COVID-19 information source?",                                                                                                                                                                                                                                                                                                                                                                                                                                    "an integer between 1 (not at all) and 7 (very intensively)",
  "information_print",                    "How frequently do you use newspaper articles (print media) as a COVID-19 information source?",                                                                                                                                                                                                                                                                                                                                                                                                                              "an integer between 1 (not at all) and 7 (very intensively)",
  "information_officialsites",            "How frequently do you use official sites of the national government and authorities as a COVID-19 information source?",                                                                                                                                                                                                                                                                                                                                                                                                    "an integer between 1 (not at all) and 7 (very intensively)",
  "information_socialmedia",              "How frequently do you use social media (e.g., Twitter, Facebook) as a COVID-19 information source?",                                                                                                                                                                                                                                                                                                                                                                                                                        "an integer between 1 (not at all) and 7 (very intensively)",
  "commu_govern_clear_understandable",    "To what extent did you assess the communication of the national government and authorities regarding the COVID-19 situation as clear and understandable?",                                                                                                                                                                                                                                                                                                                                                                  "an integer between 1 (not at all true) and 5 (very true)",
  "commu_govern_credible_conest",         "To what extent did you assess the communication of the national government and authorities regarding the COVID-19 situation as credible and honest?",                                                                                                                                                                                                                                                                                                                                                                       "an integer between 1 (not at all true) and 5 (very true)",
  "commu_govern_interest_people",         "To what extent did you assess the communication of the national government and authorities regarding the COVID-19 situation as guided by the interests of the people?",                                                                                                                                                                                                                                                                                                                                                    "an integer between 1 (not at all true) and 5 (very true)",
  "action_gov_supported",                 "To what extent did you feel well supported by the national government and authorities?",                                                                                                                                                                                                                                                                                                                                                                                                                                    "an integer between 1 (not at all true) and 5 (very true)",
  "action_gov_informed",                  "To what extent did you feel well informed by the national government and authorities?",                                                                                                                                                                                                                                                                                                                                                                                                                                     "an integer between 1 (not at all true) and 5 (very true)",
  "action_gov_seriously",                 "To what extent did you feel taken seriously by the national government and authorities?",                                                                                                                                                                                                                                                                                                                                                                                                                                   "an integer between 1 (not at all true) and 5 (very true)",
  "action_gov_alone",                     "To what extent did you feel left alone by the national government and authorities?",                                                                                                                                                                                                                                                                                                                                                                                                                                        "an integer between 1 (not at all true) and 5 (very true)",
  "measures_usefulness",                  "To what extent did you consider the introduced measures to combat the COVID-19 crisis as useful?",                                                                                                                                                                                                                                                                                                                                                                                                                          "an integer between 0 (not at all) and 4 (very strong)",
  "measures_adherence",                   "How much did you adhere to the measures introduced to combat the COVID-19 crisis?",                                                                                                                                                                                                                                                                                                                                                                                                                                         "an integer between 0 (not at all) and 4 (very strong)",
  "dass21_depression",                    "What is your depression sum score in the Depression Anxiety Stress Scales 21 scale? The Depression subscale has seven items (e.g., 'I felt that life was meaningless'), each rated on a 4-point Likert-type scale (0 = did not apply to me at all, 3 = applies to me very much or most of the time). The higher the sum score, the higher the negative symptom.",                                                                                                                                                          "an integer between 0 and 21",
  "dass21_anxiety",                       "What is your anxiety sum score in the Depression Anxiety Stress Scales 21 scale? The Anxiety subscale has seven items (e.g., 'I felt scared without any good reason'), each rated on a 4-point Likert-type scale (0 = did not apply to me at all, 3 = applies to me very much or most of the time). The higher the sum score, the higher the negative symptom.",                                                                                                                                                          "an integer between 0 and 21",
  "dass21_stress",                        "What is your stress sum score in the Depression Anxiety Stress Scales 21 scale? The Stress subscale has seven items (e.g., 'I found it hard to wind down'), each rated on a 4-point Likert-type scale (0 = did not apply to me at all, 3 = applies to me very much or most of the time). The higher the sum score, the higher the negative symptom.",                                                                                                                                                                     "an integer between 0 and 21",
  "Covid_Burden",                         "What is your average score in the Covid-19 Burden Scale, which assesses the psychological burden caused by the Covid-19 situation? The scale has six items (e.g., 'I am burdened by the current social situation'), each rated on a 7-point Likert-type scale (1 = I do not agree, 7 = I totally agree). Higher average scores indicate higher levels of burden.",                                                                                                                                                       "a real number between 1 and 7",
  "pmh9_all",                             "What is your sum score in the Positive Mental Health Scale, which asseses psychological, emotional, and social well-being? The scale has nine items, each rated on a 4-point Likert-type scale (e.g., 'I enjoy my life'; 0 = do not agree, 3 = agree). Higher sum scores indicate higher levels of positive mental health.",                                                                                                                                                                                              "an integer between 0 and 27"
)

# Recode raw data: collapse anchor labels to numeric codes (per original script).
recode_country <- function(country) {
  df <- raw_all %>%
    subset(country_all == country) %>%
    mutate(
      social_class                 = ifelse(social_class == "lower class", 1,
                                            ifelse(social_class == "upper class", 6, social_class)),
      affectedness_physicalhealth  = ifelse(grepl("not at all", affectedness_physicalhealth), "0",
                                            ifelse(grepl("very strong", affectedness_physicalhealth), "4", affectedness_physicalhealth)),
      affectedness_economically    = ifelse(grepl("not at all", affectedness_economically), "0",
                                            ifelse(grepl("very strong", affectedness_economically), "4", affectedness_economically)),
      affectedness_mentally        = ifelse(grepl("not at all", affectedness_mentally), "0",
                                            ifelse(grepl("very strong", affectedness_mentally), "4", affectedness_mentally)),
      information_TV               = ifelse(grepl("not at all", information_TV), "1",
                                            ifelse(grepl("very intensively", information_TV), "7", information_TV)),
      information_print            = ifelse(grepl("not at all", information_print), "1",
                                            ifelse(grepl("very intensively", information_print), "7", information_print)),
      information_officialsites    = ifelse(grepl("not at all", information_officialsites), "1",
                                            ifelse(grepl("very intensively", information_officialsites), "7", information_officialsites)),
      information_socialmedia      = ifelse(grepl("not at all", information_socialmedia), "1",
                                            ifelse(grepl("very intensively", information_socialmedia), "7", information_socialmedia)),
      commu_govern_clear_understandable = ifelse(grepl("not at all", commu_govern_clear_understandable), "1",
                                                 ifelse(grepl("very true", commu_govern_clear_understandable), "5", commu_govern_clear_understandable)),
      commu_govern_credible_conest = ifelse(grepl("not at all", commu_govern_credible_conest), "1",
                                            ifelse(grepl("very true", commu_govern_credible_conest), "5", commu_govern_credible_conest)),
      commu_govern_interest_people = ifelse(grepl("not at all", commu_govern_interest_people), "1",
                                            ifelse(grepl("very true", commu_govern_interest_people), "5", commu_govern_interest_people)),
      action_gov_supported         = ifelse(grepl("not at all", action_gov_supported), "1",
                                            ifelse(grepl("very true", action_gov_supported), "5", action_gov_supported)),
      action_gov_informed          = ifelse(grepl("not at all", action_gov_informed), "1",
                                            ifelse(grepl("very true", action_gov_informed), "5", action_gov_informed)),
      action_gov_seriously         = ifelse(grepl("not at all", action_gov_seriously), "1",
                                            ifelse(grepl("very true", action_gov_seriously), "5", action_gov_seriously)),
      action_gov_alone             = ifelse(grepl("not at all", action_gov_alone), "1",
                                            ifelse(grepl("very true", action_gov_alone), "5", action_gov_alone)),
      measures_usefulness          = ifelse(grepl("not at all", measures_usefulness), "0", measures_usefulness),
      measures_adherence           = ifelse(grepl("not at all", measures_adherence), "0", measures_adherence)
    ) %>%
    select(-vaccination_willingness)
  df %>% select(all_of(spec$name))
}

build_country_spec <- function(df, base_spec) {
  base_spec %>%
    rowwise() %>%
    mutate(response_levels = if (is.na(response_levels_template)) {
      NA_character_
    } else if (response_levels_template == "DYNAMIC") {
      paste(levels(as.factor(df[[name]])), collapse = "; ")
    } else {
      response_levels_template
    }) %>%
    ungroup() %>%
    select(name, question, response_levels)
}

for (country in c("Sweden", "US")) {
  df_country <- recode_country(country) %>%
    mutate(across(everything(), as.character))

  country_spec <- build_country_spec(df_country, spec)
  header_row   <- build_inline_question_header(country_spec)

  df <- inject_question_header(df_country, header_row)
  df <- ensure_ID_first(df)

  output_path <- file.path(processed_dir,
                           paste0(source_id, "_", tolower(country), "_data.csv"))
  write_clean_csv(df, output_path)
}
