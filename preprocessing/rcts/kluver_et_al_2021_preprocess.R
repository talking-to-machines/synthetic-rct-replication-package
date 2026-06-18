# Klüver et al. 2021 — Incentives can spur COVID-19 vaccination uptake (Germany).
# PNAS 118(36), e2109543118. https://doi.org/10.1073/pnas.2109543118
#
# Reads:
#   data/human/surveys/kluver_et_al_2021/wave_1.csv
#   data/human/surveys/kluver_et_al_2021/kluver_et_al_2021_mapping.csv
# Writes:
#   data/processed/rcts/kluver_et_al_2021/kluver_et_al_2021_data.csv
#
# Mapping CSV uses extra columns (var_name/survey_label/etc.); transformed
# in-script to the canonical schema (name, label, options, selected,
# label_for_options).
#
# Experimental design: within-subjects factorial survey experiment. Each
# respondent was randomly assigned to two independent policy vignettes in the
# same survey session (Round 1: c_0031/v_74; Round 2: c_0032/v_77). The
# vignette varied along three dimensions: freedoms for vaccinated people
# (yes/no), financial incentives (none/EUR 25/EUR 50), and vaccination at
# local doctors (yes/no) — 2x3x2 = 12 conditions. The primary outcome
# (v_74/v_77) is self-reported vaccination probability (0-10 scale).
#
# Design choice: we retain only Round 1 (c_0031/v_74) per participant,
# giving a clean between-subjects structure. The paper reports that treatment
# effects estimated from Round 1 and Round 2 separately are "largely
# identical", so no information is lost. Round 2 variables (c_0032, v_77)
# are dropped. c_0031 is renamed to `treatment` and v_74 to `outcome`.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringi)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "kluver_et_al_2021"
human_dir     <- file.path("data", "human", "surveys", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

raw_data_path <- file.path(human_dir, "wave_1.csv")
mapping_path  <- file.path(human_dir, paste0(source_id, "_mapping.csv"))
output_path   <- file.path(processed_dir, paste0(source_id, "_data.csv"))

raw_mapping <- read_csv(
  mapping_path,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
)

# Transform Klüver's bespoke mapping schema -> canonical schema.
# Drop the existing `label` column first to avoid a duplicate when renaming
# `survey_label` -> `label`.
mapping <- raw_mapping %>%
  select(-label) %>%
  rename(name = var_name, label = survey_label) %>%
  mutate(
    label_for_options = NA_character_,
    selected = suppressWarnings(as.integer(selected))
  ) %>%
  select(name, label, options, selected, label_for_options)

selected_vars <- mapping %>% filter(selected == 1) %>% pull(name)

raw <- read_csv(raw_data_path, show_col_types = FALSE,
                col_types = cols(.default = col_character()))

data_subset <- raw %>%
  select(any_of(selected_vars)) %>%
  mutate(across(everything(), as.character))

# UTF-8 validation on the free-text city column (kept from the original
# Klüver cleaning script).
if ("city" %in% names(data_subset)) {
  data_subset$city <- stringi::stri_enc_toutf8(data_subset$city, validate = TRUE)
}

lookup <- build_options_lookup(mapping)
data_decoded <- apply_human_readable(data_subset, lookup) %>%
  mutate(across(everything(), ~ na_if(.x, "")))

header_row <- build_question_header(mapping)
df <- inject_question_header(data_decoded, header_row)
df <- ensure_ID_first(df)

# Retain only Round 1: drop Round 2 treatment (c_0032) and outcome (v_77).
# Rename Round 1 treatment (c_0031) to `treatment` and outcome (v_74) to
# `outcome` for consistency with other RCT datasets.
df <- df[, !names(df) %in% c("c_0032", "v_77"), drop = FALSE]
names(df)[names(df) == "c_0031"] <- "treatment"
names(df)[names(df) == "v_74"]   <- "outcome"

write_clean_csv(df, output_path)
