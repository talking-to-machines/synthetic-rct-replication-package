# Duflo et al. (2019) — HIV prevention among youth: A randomized controlled
# trial of voluntary counseling and testing for HIV and male condom
# distribution in rural Kenya.
#
# Reads:
#   data/human/rcts/duflo_et_al_2019/bio_for_analysis.dta
#   data/human/rcts/duflo_et_al_2019/duflo_et_al_2019_mapping.csv
# Writes:
#   data/processed/rcts/duflo_et_al_2019/duflo_et_al_2019_data.csv
#
# Multiple primary + secondary outcomes; outcome variables retain their
# original study-specific names (no rename to outcome / outcome_1 / etc.).

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "duflo_et_al_2019"
human_dir     <- file.path("data", "human", "rcts", source_id)
processed_dir <- file.path("data", "processed", "rcts", source_id)

raw_data_path <- file.path(human_dir, "bio_for_analysis.dta")
mapping_path  <- file.path(human_dir, paste0(source_id, "_mapping.csv"))
output_path   <- file.path(processed_dir, paste0(source_id, "_data.csv"))

raw     <- haven::read_dta(raw_data_path)
mapping <- read_mapping_csv(mapping_path)

# Filter to participants with non-missing endline survey, per the original
# cleaning script's logic.
selected_vars <- mapping %>% filter(selected == 1) %>% pull(name)
data_subset <- raw %>%
  filter(an2_surveyed == 1) %>%
  select(all_of(selected_vars)) %>%
  mutate(across(everything(), as.character))

lookup <- build_options_lookup(mapping)
data_decoded <- apply_human_readable(data_subset, lookup) %>%
  mutate(across(everything(), ~ na_if(.x, "")))

# The original raw file uses ISO-8859-1 encoded strings; convert to UTF-8 so
# the output is consistent with the rest of the corpus.
data_decoded <- data_decoded %>%
  mutate(across(everything(), ~ iconv(.x, from = "ISO-8859-1", to = "UTF-8")))

data_decoded <- data_decoded %>% rename(treatment = group)
mapping <- mapping %>% mutate(name = if_else(name == "group", "treatment", name))

header_row <- build_question_header(mapping)
df <- inject_question_header(data_decoded, header_row)
df <- ensure_subject_id_first(df)

write_clean_csv(df, output_path)
