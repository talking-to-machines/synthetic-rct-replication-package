# Levine et al. (2021) — Mobile phone reminders and incentives for childhood
# vaccination (Ghana).
#
# Reads:
#   data/human/rcts/levine_et_al_2021/levine_et_al_2021.xlsx
# Writes:
#   data/processed/rcts/levine_et_al_2021/levine_et_al_2021_data.csv
#
# Treatment:  treatment (Control / Reminder / Incentive)
# Outcome:    child_vacc_opv1_yn (received first dose of OPV, a scheduled
#             second-contact vaccine requiring a return clinic visit)
# Profile:    child characteristics and maternal demographics/phone access

suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(tibble)
})

if (!exists("build_inline_question_header")) source("preprocessing/utils.R")

source_id  <- "levine_et_al_2021"
human_dir  <- file.path("data", "human", "rcts", source_id)
output_path <- file.path("data", "processed", "rcts", source_id,
                          paste0(source_id, "_data.csv"))

raw <- read_excel(file.path(human_dir, paste0(source_id, ".xlsx")))

# Trim any trailing whitespace from column names introduced by Excel.
names(raw) <- trimws(names(raw))

data_clean <- raw %>%
  rename(
    subject_id       = baby_id,
    treatment        = `Treatment assigment`,
    antenatal_care   = `c_ant (any antenatal care)`,
    birth_time       = `child_birth_time (for transport)`,
    has_vacc_record  = `child_vacc_doc_yn (vaccine records booklet or paper record)`,
    mother_age       = `(067) How old were you at your last birthday?`,
    mother_schooling = `(069) Have you ever attended school?`,
    mother_edu       = `Mother's highest level of edu attended`,
    has_electricity  = `(077) Does your household have electricity?`,
    has_tv           = `(078) Does your household have a color TV?`,
    owns_phone       = `(079) Do you have your own cell/mobile phone?`,
    network_strength = `(082) How strong is the network coverage at your household for the phone that you primarily use?`,
    has_mobile_money = `(083) Does the phone you primarily use have a 'mobile money' account registered to it?`,
    has_internet     = `(084) Is the phone you primarily use able to access the internet or use Wifi?`
  ) %>%
  select(
    subject_id, treatment,
    child_vacc_opv1_yn,
    child_gender, child_birth_type, antenatal_care, child_birth_place, birth_time,
    has_vacc_record, mother_age, mother_schooling, mother_edu,
    has_electricity, has_tv, owns_phone, network_strength,
    has_mobile_money, has_internet
  ) %>%
  mutate(across(everything(), as.character))

spec <- tribble(
  ~name,               ~question,                                                                         ~response_levels,
  "subject_id",        "subject_id",                                                                      NA_character_,
  "treatment",         "treatment",                                                                        NA_character_,
  "child_vacc_opv1_yn","Did the child receive the first dose of oral polio vaccine (OPV1)?",               "Yes; No",
  "child_gender",      "What is the child's gender?",                                                      "Boy; Girl",
  "child_birth_type",  "What was the type of birth?",                                                      NA_character_,
  "antenatal_care",    "Did the mother receive any antenatal care during pregnancy?",                       "Yes; No",
  "child_birth_place", "Where was the child born?",                                                        NA_character_,
  "birth_time",        "How long did it take to travel to the birth location?",                            NA_character_,
  "has_vacc_record",   "Does the child have a vaccine records booklet or paper record?",                   "Yes; No",
  "mother_age",        "How old were you at your last birthday?",                                          NA_character_,
  "mother_schooling",  "Have you ever attended school?",                                                   "Yes; No",
  "mother_edu",        "What is the mother's highest level of education attended?",                        NA_character_,
  "has_electricity",   "Does your household have electricity?",                                            "Yes; No",
  "has_tv",            "Does your household have a color TV?",                                             "Yes; No",
  "owns_phone",        "Do you have your own cell/mobile phone?",                                          "Yes; No",
  "network_strength",  "How strong is the network coverage at your household for the phone that you primarily use?",  NA_character_,
  "has_mobile_money",  "Does the phone you primarily use have a mobile money account registered to it?",   "Yes; No",
  "has_internet",      "Is the phone you primarily use able to access the internet or use Wifi?",          "Yes; No"
)

header_row <- build_inline_question_header(spec)
df <- inject_question_header(data_clean, header_row)
df <- ensure_subject_id_first(df)

write_clean_csv(df, output_path)
