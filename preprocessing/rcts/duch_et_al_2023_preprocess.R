# Duch et al. (2023) — Financial incentives for COVID-19 vaccination, Ghana.
#
# Reads:
#   data/human/rcts/duch_et_al_2023/FinalFinal20062023Full1-6.RData
#   data/human/rcts/duch_et_al_2023/Ghana_Vaccine_Incentives_Baseline.csv
#   data/human/rcts/duch_et_al_2023/Ghana_Vaccine_Incentives_Endline_Phone.csv
#   data/human/rcts/duch_et_al_2023/Ghana_Vaccine_Incentives_Endline_In_Person.csv
# Writes:
#   data/processed/rcts/duch_et_al_2023/duch_et_al_2023_data.csv

suppressPackageStartupMessages({
  library(tidyverse)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "duch_et_al_2023"
human_dir     <- file.path("data", "human", "rcts", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

dat_baseline         <- read.csv(file.path(human_dir, "Ghana_Vaccine_Incentives_Baseline.csv"))
dat_endline_phone    <- read.csv(file.path(human_dir, "Ghana_Vaccine_Incentives_Endline_Phone.csv"))
dat_endline_in_person<- read.csv(file.path(human_dir, "Ghana_Vaccine_Incentives_Endline_In_Person.csv"))

load(file.path(human_dir, "FinalFinal20062023Full1-6.RData"))
dat_clean <- final_finalV2

dat_baseline          <- dat_baseline          %>% rename_with(~ paste0("p_i_", .x))
dat_endline_phone     <- dat_endline_phone     %>% rename_with(~ paste0("p_ii_", .x))
dat_endline_in_person <- dat_endline_in_person %>% rename_with(~ paste0("p_iii_", .x))
dat_clean             <- dat_clean             %>% rename_with(.cols = c(3:97), ~ paste0("p_i_", .x))

survey_items <- c(
  unlist(dat_baseline[1, ]),
  unlist(dat_endline_phone[1, ]),
  unlist(dat_endline_in_person[1, ])
)
survey_items <- tibble(
  !!!unlist(map(names(dat_clean), ~ survey_items[.x] %||% NA)),
  .name_repair = "unique"
)
colnames(survey_items) <- colnames(dat_clean)

dat_clean <- dat_clean %>% mutate(across(everything(), as.character))
dat_clean <- bind_rows(survey_items, dat_clean)

# Adapt question labels for constructed variables.
dat_clean[1, 'SubjectID']         <- "ID"
dat_clean[1, 'individual_treatment'] <- "treatment"
dat_clean[1, 'Village_Population']<- "How many people live in your village?"
dat_clean[1, 'clinic_distance']   <- "What is the distance in km of the nearest health clinic from where you live?"
dat_clean[1, 'Names of Community']<- "What is the name of the community you live in?"

# Merge dummy multi-choice columns.
baseline_multiple_choices_to_merge <- list(
  paste0("p_i_Q154_", 1:9),
  paste0("p_i_Q152_", 1:9),
  paste0("p_i_Q155_", 1:9),
  paste0("p_i_Q156_", 1:9)
)
for (i in seq_along(baseline_multiple_choices_to_merge)) {
  merged_col_name <- gsub("_[^_]+$", "", baseline_multiple_choices_to_merge[[i]][1])
  dat_clean <- dat_clean %>%
    rowwise() %>%
    mutate(!!merged_col_name := {
      vals <- c_across(all_of(baseline_multiple_choices_to_merge[[i]]))
      vals <- vals[!(is.na(vals)) & vals != ""]
      paste(vals, collapse = ", ")
    }) %>%
    ungroup()
  dat_clean[1, merged_col_name] <- "Why will you NOT get vaccinated against COVID-19?"
}

dat_clean[dat_clean == "" | dat_clean == "NA"] <- NA

# Coalesce vaccine intention / reasons / chance variables across treatments.
baseline_vaccine_hesitancy_to_merge <- list(
  c('p_i_Q101', 'p_i_Q109', 'p_i_Q160', 'p_i_Q164'),
  c('p_i_Q154', 'p_i_Q152', 'p_i_Q155', 'p_i_Q156'),
  c('p_i_Q102_4', 'p_i_Q110_4', 'p_i_Q161_4', 'p_i_Q161_4_1')
)
baseline_vaccine_hesitancy_colnames <- c('vaccine_intention', 'vaccine_reasons_no', 'vaccine_chance')
for (i in seq_along(baseline_vaccine_hesitancy_to_merge)) {
  merged_col_name <- baseline_vaccine_hesitancy_colnames[[i]]
  dat_clean <- dat_clean %>%
    mutate(!!merged_col_name := coalesce(!!!select(., baseline_vaccine_hesitancy_to_merge[[i]])))
}

dat_clean <- dat_clean %>%
  mutate(p_i_Q146 = case_when(p_i_Q146 == '1' ~ 'Very good',
                              p_i_Q146 == '2' ~ 'Good',
                              p_i_Q146 == '3' ~ 'Neither good nor bad',
                              p_i_Q146 == '4' ~ 'Bad',
                              p_i_Q146 == '5' ~ 'Very bad',
                              TRUE ~ NA_character_))

# Question label adaptations for variables without survey items.
dat_clean[1, 'p_i_Q2.3']   <- "What is your gender?"
dat_clean[1, 'p_i_Q2.4_1'] <- "Which region do you live in?"
dat_clean[1, 'p_i_Q2.4_2'] <- "Which district do you live in?"
dat_clean[1, 'p_i_Q144']   <- "How much (in Ghanaian Cedis) on average does your household spend in a typical week on food?"
dat_clean[1, 'p_i_Q145']   <- "How much (in Ghanaian Cedis) on average does your household spend in a typical week on non-food items (electricity, water, rent, school fees)?"
dat_clean[1, 'p_i_Q146']   <- "How would you rate the overall economic or financial condition of your household today?"
dat_clean[1, 'p_i_Q91']    <- "Do you have a registered mobile number?"

# Endline outcomes recoded.
dat_clean <- dat_clean %>%
  mutate(
    vaccine_reported_combo = case_when(vaccine_reported_combo == '1' ~ 'Yes',
                                       vaccine_reported_combo == '0' ~ 'No',
                                       TRUE ~ NA_character_),
    ActVacApril = case_when(ActVacApril == '1' ~ 'Yes',
                            ActVacApril == '0' ~ 'No',
                            TRUE ~ NA_character_)
  )
dat_clean[1, 'vaccine_reported_combo'] <- dat_clean[1, 'p_ii_Q8.1']
dat_clean[1, 'ActVacApril'] <- 'Have you received a COVID-19 vaccine, as verified in the records of the Ghanaian District Health Offices? Answer one of the following options: Yes, No'
dat_clean[1, 'FamilyVillages'] <- dat_clean[1, 'p_ii_Q27']

vars_selected <- c(
  'SubjectID', 'individual_treatment',
  'p_i_Q2.2', 'p_i_Q2.3', 'p_i_Q2.5', 'p_i_Q2.4_1', 'p_i_Q2.4_2',
  'Names of Community', 'Village_Population', 'clinic_distance',
  'p_i_Q141', 'p_i_Q142', 'p_i_Q143',
  'p_i_Q144', 'p_i_Q145', 'p_i_Q146',
  'p_i_Q91',
  'vaccine_intention', 'vaccine_reasons_no', 'vaccine_chance',
  'p_ii_Q2.2', 'p_ii_Q2.3',
  'p_ii_Q28', 'p_ii_Q30', 'p_ii_Q4.1',
  'p_ii_Q2.6',
  'vaccine_reported_combo',
  'ActVacApril'
)
dat_clean <- dat_clean %>% select(!!!vars_selected)

# Standardize required columns.
names(dat_clean)[names(dat_clean) == "SubjectID"] <- "subject_id"
names(dat_clean)[names(dat_clean) == "individual_treatment"] <- "treatment"
dat_clean$subject_id[1] <- "subject_id"
dat_clean$treatment[1]  <- "treatment"

dat_clean <- dat_clean[, c("subject_id", setdiff(names(dat_clean), "subject_id")), drop = FALSE]

write_clean_csv(dat_clean, file.path(processed_dir, paste0(source_id, "_data.csv")))
