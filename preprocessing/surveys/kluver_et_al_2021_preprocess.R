# Klüver et al. 2021 — Vaccine hesitancy (Germany).
#
# Reads:
#   data/human/surveys/kluver_et_al_2021/wave_1.csv
#   data/human/surveys/kluver_et_al_2021/kluver_et_al_2021_mapping.csv
# Writes:
#   data/processed/surveys/kluver_et_al_2021/kluver_et_al_2021_data.csv
#
# Mapping CSV uses extra columns (var_name/survey_label/etc.); transformed
# in-script to the canonical schema (name, label, options, selected,
# label_for_options).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringi)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "kluver_et_al_2021"
human_dir     <- file.path("data", "human", "surveys", source_id)
processed_dir <- file.path("data", "processed", "surveys", source_id)

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
df <- ensure_subject_id_first(df)

write_clean_csv(df, output_path)
