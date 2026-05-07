################################################################################
## data_cleaning.R
##
##
##
## Required packages: readr, readxl
##   install.packages(c("readr", "readxl"))
################################################################################

library(readr)
library(readxl)

## ---------------------------------------------------------------------------
## setting computer directories
## ---------------------------------------------------------------------------

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human",     "surveys", "arce_et_al_2021")
saving_path <- file.path("data", "processed", "surveys", "arce_et_al_2021")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

## ---------------------------------------------------------------------------
## cleaning the dataset
## ---------------------------------------------------------------------------

{

  ## Read the raw combined survey file.
  ## The original Stata script uses `import delimited`, which treats columns
  ## with mixed/non-numeric content as strings. To reproduce that behaviour
  ## faithfully we read every column as character and convert explicitly
  ## where needed. This matches the Stata logic, which compares dummy
  ## variables to the *string* "1" / "0".
  combined <- read_csv(
    file.path(data_path, "combined.csv"),
    col_types = cols(.default = col_character()),
    na = character()                # do not auto-convert "" or "NA" to <NA>
  )
  combined <- as.data.frame(combined, stringsAsFactors = FALSE)

  ## drop irrelevant variables
  combined[, c("study", "cluster", "weight", "take_vaccine_num")] <- NULL

  ## yes and no vaccine cannot be combined as multiple replies can be chosen
  ## replacing 0-1 dummies into strings
  yes_no_vars <- c(
    "yes_vaccine_1",  "yes_vaccine_2",  "yes_vaccine_3",  "yes_vaccine_4",
    "yes_vaccine_5",  "yes_vaccine_666",
    "no_vaccine_1",   "no_vaccine_2",   "no_vaccine_3",   "no_vaccine_4",
    "no_vaccine_5",   "no_vaccine_6",   "no_vaccine_7",   "no_vaccine_8",
    "no_vaccine_9",   "no_vaccine_666"
  )

  for (v in yes_no_vars) {
    x <- combined[[v]]
    x[x == "1"] <- "Yes"
    x[x == "0"] <- "No"
    combined[[v]] <- x
  }

  ## combining trust vaccine variables into a single variable
  ## (mutually exclusive reply)
  ##
  ## NOTE on Stata semantics: the original code uses chained
  ##     replace trust_vaccine = k if trust_vaccine_k == "1" & trust_vaccine == .
  ## so that earlier categories take precedence over later ones whenever a
  ## respondent has multiple "1"s across the trust_vaccine_* dummies. We
  ## reproduce that precedence rule below.
  combined$trust_vaccine <- NA_real_

  trust_assignments <- list(
    list(col = "trust_vaccine_1",      val =  1),
    list(col = "trust_vaccine_2",      val =  2),
    list(col = "trust_vaccine_3",      val =  3),
    list(col = "trust_vaccine_4",      val =  4),
    list(col = "trust_vaccine_5",      val =  5),
    list(col = "trust_vaccine_6",      val =  6),
    list(col = "trust_vaccine_7",      val =  7),
    list(col = "trust_vaccine_8",      val =  8),
    list(col = "trust_vaccine_9",      val =  9),
    list(col = "trust_vaccine_dk",     val = 90),
    list(col = "trust_vaccine_refuse", val = 91),
    list(col = "trust_vaccine_nr",     val = 92),
    list(col = "trust_vaccine_666",    val = 93),
    list(col = "trust_vaccine_other",  val = 94)
  )

  for (a in trust_assignments) {
    idx <- which(combined[[a$col]] == "1" & is.na(combined$trust_vaccine))
    combined$trust_vaccine[idx] <- a$val
  }

  ## drop the component dummies
  trust_component_vars <- vapply(trust_assignments, `[[`, character(1), "col")
  combined[, trust_component_vars] <- NULL

  ## Build the string version of trust_vaccine and replace the numeric one,
  ## matching the Stata `tostring ... gen(trust_vaccine_str)` step followed by
  ## the recoded labels.
  trust_vaccine_str <- rep(NA_character_, nrow(combined))
  trust_vaccine_str[combined$trust_vaccine == 1]  <- "Family"
  trust_vaccine_str[combined$trust_vaccine == 2]  <- "Friends"
  trust_vaccine_str[combined$trust_vaccine == 3]  <- "Religious leader"
  trust_vaccine_str[combined$trust_vaccine == 4]  <- "Famous person"
  trust_vaccine_str[combined$trust_vaccine == 5]  <- "Health workers"
  trust_vaccine_str[combined$trust_vaccine == 6]  <- "Government or Ministry of Health"
  trust_vaccine_str[combined$trust_vaccine == 7]  <- "Traditional healers"
  trust_vaccine_str[combined$trust_vaccine == 8]  <- "Traditional media (e.g., newspapers, radio)"
  trust_vaccine_str[combined$trust_vaccine == 9]  <- "Online medical discussion groups"
  trust_vaccine_str[combined$trust_vaccine == 90] <- "Don't know"
  trust_vaccine_str[combined$trust_vaccine == 91] <- "Refuse"
  trust_vaccine_str[combined$trust_vaccine == 92] <- "No response"
  trust_vaccine_str[combined$trust_vaccine == 93 |
                    combined$trust_vaccine == 94] <- "Other"
  trust_vaccine_str[is.na(combined$trust_vaccine)] <- "NA"

  combined$trust_vaccine <- trust_vaccine_str

  ## replacing values
  combined$take_vaccine[combined$take_vaccine == "DK"] <- "Don't know"

  ## saving the file (Stata save -> R .rds, preserving types)
  saveRDS(combined, file = file.path(saving_path, "combined.rds"))

}

## ---------------------------------------------------------------------------
## cleaning the codebook
## ---------------------------------------------------------------------------

{

  ## use the dictionary presenting the harmonised variable labels
  ## (Stata: import excel ... firstrow allstring clear)
  dictionary <- read_excel(
    file.path(data_path, "dictionary.xlsx"),
    sheet = "Sheet1",
    col_names = TRUE,
    col_types = "text"
  )
  dictionary <- as.data.frame(dictionary, stringsAsFactors = FALSE)

  ## transpose the dataset (Stata: sxpose, clear)
  ##
  ## We replicate Stata's `sxpose` behaviour exactly. `sxpose` takes a
  ## string dataset and returns the transpose; the original column names
  ## are NOT preserved in the result -- the new columns are simply named
  ## _var1, _var2, ... with row 1 of the new data holding the values that
  ## were originally in row 1 of the source.
  ##
  ## The two subsequent Stata commands -
  ##     foreach var of varlist _all { rename `var' `=`var'[1]' }
  ##     drop in 1
  ## - then promote the values in row 1 of the transposed data to be the
  ## new column headers, and drop that now-redundant first row. We do
  ## exactly the same below.
  transposed <- as.data.frame(
    t(as.matrix(dictionary)),
    stringsAsFactors = FALSE
  )
  rownames(transposed) <- NULL
  colnames(transposed) <- paste0("_var", seq_len(ncol(transposed)))

  ## rename the variable with the values in the first row
  colnames(transposed) <- as.character(unlist(transposed[1, ]))

  ## drop the first row
  transposed <- transposed[-1, , drop = FALSE]
  rownames(transposed) <- NULL

  ## drop excluded variables
  excluded <- c(
    "study", "cluster", "weight", "take_vaccine_num",
    "age_groups", "age_groups_binary", "educ_binary",
    "trust_vaccine_2", "trust_vaccine_3", "trust_vaccine_4",
    "trust_vaccine_5", "trust_vaccine_6", "trust_vaccine_7",
    "trust_vaccine_8", "trust_vaccine_9",
    "trust_vaccine_dk", "trust_vaccine_refuse", "trust_vaccine_nr",
    "trust_vaccine_666", "trust_vaccine_other",
    "trust_recode_1", "trust_recode_2", "trust_recode_3",
    "trust_recode_4", "trust_recode_5"
  )
  transposed[, excluded] <- NULL

  ## rename trust_vaccine_1 -> trust_vaccine
  colnames(transposed)[colnames(transposed) == "trust_vaccine_1"] <- "trust_vaccine"

  ## replace first row with the full question reported in tables S10-13 of
  ## the online supplement of the paper
  ## [https://static-content.springer.com/esm/art%3A10.1038%2Fs41591-021-01454-y/MediaObjects/41591_2021_1454_MOESM1_ESM.pdf]
  ##
  ## Stata `replace var = "..."` without an `if` clause replaces every row.
  ## After the cleaning above, the codebook contains a single label row, so
  ## these statements set the value of that single row. We assign to all
  ## rows of `transposed` to reproduce that behaviour exactly.
  transposed$country <- "Country where the study took place"
  transposed$age     <- "What is your age?"
  transposed$educ    <- "What is your highest level of education?"
  transposed$gender  <- "What is your gender?"

  ## adaptation from the original survey question "Why would you take it?"
  transposed$yes_vaccine_1   <- "Would you take the vaccine for self-protection?"
  transposed$yes_vaccine_2   <- "Would you take the vaccine to protect your family?"
  transposed$yes_vaccine_3   <- "Would you take the vaccine to protect your community?"
  transposed$yes_vaccine_4   <- "Would you take the vaccine if a health worker recommends it?"
  transposed$yes_vaccine_5   <- "Would you take the vaccine if the government recommends it?"
  transposed$yes_vaccine_666 <- "Would you take the vaccine for other reasons?"

  ## adaptation from the original survey question "Why would you not take it?"
  transposed$no_vaccine_1   <- "Would you not take the vaccine because you are concerned about side effects?"
  transposed$no_vaccine_2   <- "Would you not take the vaccine because you are concerned about getting coronavirus from the vaccine"
  transposed$no_vaccine_3   <- "Would you not take the vaccine because you are not concerned about getting seriously ill?"
  transposed$no_vaccine_4   <- "Would you not take the vaccine because you do not think vaccines are effective?"
  transposed$no_vaccine_5   <- "Would you not take the vaccine because you do not think coronavirus outbreak is as serious as people say?"
  transposed$no_vaccine_6   <- "Would you not take the vaccine because you do not like needles?"
  transposed$no_vaccine_7   <- "Would you not take the vaccine because you are allergic to vaccines?"
  transposed$no_vaccine_8   <- "Would you not take the vaccine because you will not have time to get vaccinated?"
  transposed$no_vaccine_9   <- "Would you not take the vaccine because you mention a conspiracy theory?"
  transposed$no_vaccine_666 <- "Would you not take the vaccine for other reasons?"

  ## adaptation from the original survey question "Which of the following
  ## people would you trust MOST to help you decide whether you would get a
  ## covid-19 vaccine, if one becomes available?"
  transposed$trust_vaccine <- "Who would you trust MOST to help you decide whether you would get a covid-19 vaccine, if one becomes available?"

  codebook <- transposed

}

## ---------------------------------------------------------------------------
## appending the two cleaned files and exporting in CSV
## ---------------------------------------------------------------------------

## The Stata pipeline ends with `append using combined`, which stacks the
## previously saved `combined` data file *underneath* the codebook that is
## currently in memory. Stata's `append` aligns columns by name and fills
## missing columns with missing values. We replicate that semantic here.

combined <- readRDS(file.path(saving_path, "combined.rds"))

## Align columns: union of names, preserving codebook order first, then any
## columns unique to combined.
all_cols <- union(colnames(codebook), colnames(combined))

add_missing <- function(df, cols) {
  missing_cols <- setdiff(cols, colnames(df))
  for (m in missing_cols) df[[m]] <- NA_character_
  df[, cols, drop = FALSE]
}

## Coerce both data frames to character so the stacked output matches
## Stata's `export delimited` behaviour, which writes everything as text.
codebook[]  <- lapply(codebook,  as.character)
combined[]  <- lapply(combined,  as.character)

codebook_aligned <- add_missing(codebook, all_cols)
combined_aligned <- add_missing(combined, all_cols)

final <- rbind(codebook_aligned, combined_aligned)

# Synthesize subject_id (Arce has no native participant identifier).
final$subject_id <- c("subject_id", as.character(seq_len(nrow(final) - 1)))
final <- final[, c("subject_id", setdiff(names(final), "subject_id")), drop = FALSE]

write_clean_csv(final, file.path(saving_path, "arce_et_al_2021_data.csv"))

################################################################################
## End of file
################################################################################
