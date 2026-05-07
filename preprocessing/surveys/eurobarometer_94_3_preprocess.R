# Eurobarometer 94.3 — Sweden subset.
#
# Reads:
#   data/human/surveys/eurobarometer_94_3/ZA7780_v2-0-0.dta
#   data/human/surveys/eurobarometer_94_3/eurobarometer_94_3_mapping.csv
# Writes:
#   data/processed/surveys/eurobarometer_94_3/eurobarometer_94_3_data.csv
#
# The mapping CSV uses a bespoke schema (variable, question only). It is
# kept as-is; the question column is treated as the verbatim survey header
# and consumed without options-enumeration logic.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(stringr)
  library(purrr)
  library(tidyr)
  library(readr)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "eurobarometer_94_3"
human_dir     <- file.path("data", "human", "surveys", source_id)
processed_dir <- file.path("data", "processed", "surveys", source_id)

raw_data_path <- file.path(human_dir, "ZA7780_v2-0-0.dta")
mapping_path  <- file.path(human_dir, paste0(source_id, "_mapping.csv"))
output_path   <- file.path(processed_dir, paste0(source_id, "_data.csv"))

# --- Variable selection (study-specific; preserved from original script) ---
vars_selected <- c(
  "uniqid",
  "country",
  "qa1a_", "qa2a_", "qa3a_", "qa4a_", "qa5_", "qa6a_", "qa6b_",
  "qa10_", "qa11_", "qa12", "qa13_", "qa14", "qa15", "qa16", "qa17",
  "qa18_", "qa20_",
  "sd18a", "qb2_", "qb4_3",
  "qc1a_1", "qc1a_2", "qc5_1", "qc6_",
  "qd3_", "qd7_", "qd8_",
  "d7", "d9", "d10", "d15a", "d15b", "d25", "d40a", "d40b", "d40c",
  "d60", "d63", "d70", "d1", "d71_", "d73_",
  "qa19"
)
prefixes <- vars_selected[str_ends(vars_selected, "_")]
dummy_vars_to_merge <- c("qa3a", "qa4a", "qa5", "qa11", "qa20")

extract_value_label <- function(value) {
  attr(value, "label") %>%
    sub(".*\\|\\s*([^\\(]+).*", "\\1", .) %>%
    str_trim() %>% str_to_title()
}

merge_dummies_eb <- function(data, prefix) {
  data %>%
    mutate(across(starts_with(prefix),
                  ~ if_else(as.character(.) == "1", extract_value_label(.), NA_character_),
                  .names = "{.col}_value_label")) %>%
    mutate(!!prefix := coalesce(!!!select(., ends_with("_value_label")))) %>%
    select(all_of(prefix))
}

# --- Mapping (bespoke schema preserved) ---
raw_mapping <- read_csv(mapping_path, show_col_types = FALSE,
                        col_types = cols(.default = col_character()))
questionnaire <- pivot_wider(raw_mapping, names_from = "variable",
                             values_from = "question")

# --- Raw data ---
raw_data <- read_dta(raw_data_path)

df <- raw_data %>%
  subset(country == 17) %>%   # Sweden
  select(starts_with(prefixes),
         all_of(vars_selected[!vars_selected %in% prefixes]))

df <- df %>%
  bind_cols(map_dfc(dummy_vars_to_merge, ~ merge_dummies_eb(df, .x))) %>%
  select(-starts_with(paste0(dummy_vars_to_merge, "_"))) %>%
  mutate(d40d = d40b + d40c) %>%
  select(-d40b, -d40c) %>%
  mutate(across(where(is.labelled), ~ as_factor(.x, levels = "labels"))) %>%
  mutate(
    across(where(~ is.character(.x) | is.factor(.x)),
           ~ str_remove(.x, "\\s*\\(SPONTANEOUS\\)")),
    country = str_remove(country, "SE - "),
    d40a = df$d40a,
    d9 = str_remove(d9, "SWE:"),
    d15b = if_else(d15b %in% c("NA", "Inap. (not 1 to 4 in d15a)") | is.na(d15b),
                   NA, d15b),
    d1 = df$d1,
    d1 = case_when(d1 == 97 ~ "Refuse to respond",
                   d1 == 98 ~ "Don't know",
                   TRUE ~ as.character(d1)),
    qa3a = str_remove(qa3a, "Important Issues Cntry: "),
    qa4a = str_remove(qa4a, "Important Issues Pers: "),
    qa5  = str_remove(qa5,  "Important Issues Eu: "),
    qa11 = str_remove(qa11, "Eu Corona Response Priority: "),
    qa20 = str_remove(qa20, "Trust Vaccine Info Source: ")
  ) %>%
  mutate(across(everything(), as.character))

# Inject the questionnaire row (verbatim from mapping).
df <- bind_rows(questionnaire, df) %>%
  mutate(across(everything(),
                ~ str_replace_all(.x, c(
                  "\\(NATIONALITY\\)" = "Swedish",
                  "\\(OUR COUNTRY\\)" = "Sweden"
                ))))

# Subject ID = uniqid (native).
names(df)[names(df) == "uniqid"] <- "subject_id"
df <- df[, c("subject_id", setdiff(names(df), "subject_id")), drop = FALSE]

# Header row's subject_id cell should be the literal "subject_id".
df$subject_id[1] <- "subject_id"

write_clean_csv(df, output_path)
