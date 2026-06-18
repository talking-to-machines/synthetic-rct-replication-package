# Levine et al. (2021) — Mobile phone reminders and incentives for childhood
# vaccination (Ghana).
# PLOS ONE 16(5), e0247485. https://doi.org/10.1371/journal.pone.0247485
#
# Reads:
#   data/human/rcts/levine_et_al_2021/levine_et_al_2021.xlsx
# Writes:
#   data/processed/rcts/levine_et_al_2021/levine_et_al_2021_data.csv
#
# Treatment:  treatment (Control / Reminder / Incentive)
# Outcome:    vacc_outcome — combined on-time vaccination status, constructed
#             from the paper's primary outcome definition:
#               "Yes"      — OPV0 within 14 days AND BCG within 28 days
#               "OPV0 only"— OPV0 on time, BCG not on time
#               "BCG only" — BCG on time, OPV0 not on time
#               "No"       — neither vaccine received on time
#             Source variables: "OPV0 Received within first 14 days of life"
#             and "BCG Received within first 28 days of life". No missing
#             values in either source variable (N = 690).
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
    ID               = baby_id,
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
    has_internet     = `(084) Is the phone you primarily use able to access the internet or use Wifi?`,
    opv0_ontime      = `OPV0 Received within first 14 days of life`,
    bcg_ontime       = `BCG Received within first 28 days of life`
  ) %>%
  mutate(
    vacc_outcome = case_when(
      opv0_ontime == "Yes" & bcg_ontime == "Yes" ~ "Yes",
      opv0_ontime == "Yes" & bcg_ontime == "No"  ~ "OPV0 only",
      opv0_ontime == "No"  & bcg_ontime == "Yes" ~ "BCG only",
      opv0_ontime == "No"  & bcg_ontime == "No"  ~ "No"
    )
  ) %>%
  select(
    ID, treatment,
    vacc_outcome,
    child_gender, child_birth_type, antenatal_care, child_birth_place, birth_time,
    has_vacc_record, mother_age, mother_schooling, mother_edu,
    has_electricity, has_tv, owns_phone, network_strength,
    has_mobile_money, has_internet
  ) %>%
  mutate(across(everything(), as.character)) %>%
  mutate(
    network_strength = if_else(network_strength == "999", NA_character_, network_strength),
    birth_time       = if_else(birth_time == ".d",        NA_character_, birth_time)
  )

spec <- tribble(
  ~name,               ~question,                                                                         ~response_levels,
  "ID",            "ID",                                                                                           NA_character_,
  "treatment",     "treatment",                                                                                NA_character_,
  "vacc_outcome",  "Did the child receive OPV0 within 14 days of life and BCG within 28 days of life?",                                                                  NA_character_,
  "child_gender",      "What is the child's gender?",                                                      NA_character_,
  "child_birth_type",  "What was the type of birth?",                                                      NA_character_,
  "antenatal_care",    "Did the mother receive any antenatal care during pregnancy?",                       NA_character_,
  "child_birth_place", "Where was the child born?",                                                        NA_character_,
  "birth_time",        "How long did it take to travel to the birth location?",                            NA_character_,
  "has_vacc_record",   "Does the child have a vaccine records booklet or paper record?",                   NA_character_,
  "mother_age",        "How old were you at your last birthday, in years?",                                NA_character_,
  "mother_schooling",  "Have you ever attended school?",                                                   NA_character_,
  "mother_edu",        "What is your highest level of education attended?",                                NA_character_,
  "has_electricity",   "Does your household have electricity?",                                            NA_character_,
  "has_tv",            "Does your household have a color TV?",                                             NA_character_,
  "owns_phone",        "Do you have your own cell/mobile phone?",                                          NA_character_,
  "network_strength",  "How strong is the network coverage at your household for the phone that you primarily use?",  NA_character_,
  "has_mobile_money",  "Does the phone you primarily use have a mobile money account registered to it?",   NA_character_,
  "has_internet",      "Is the phone you primarily use able to access the internet or use Wifi?",          NA_character_
)

header_row <- build_inline_question_header(spec)
df <- inject_question_header(data_clean, header_row)
df <- ensure_ID_first(df)

write_clean_csv(df, output_path)
