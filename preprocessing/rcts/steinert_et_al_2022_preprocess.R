# Data cleaning script translated from data_cleaning_Steneirt.do
# Supplementary-materials version: retains the original Stata recodes,
# variable names, inserted question row, merge sequence, output filename,
# and spelling/typographical choices as closely as possible.

# Required packages -----------------------------------------------------------
# install.packages(c("dplyr", "haven", "readxl", "readr", "tibble"))
library(dplyr)
library(haven)
library(readxl)
library(readr)
library(tibble)

# Paths -----------------------------------------------------------------------
if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human", "rcts", "steinert_et_al_2022")
saving_path <- file.path("data", "processed", "rcts", "steinert_et_al_2022")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

qualitative_file   <- file.path(data_path, "Sweden_Qualitative Coding.xlsx")
main_file          <- file.path(data_path, "Experiment_maineffects_all_countries_merged.dta")
heterogeneity_file <- file.path(data_path, "Experiment_heterogeneitymrp_dataset")
output_file        <- file.path(saving_path, "steinert_et_al_2022_data.csv")

# Helper functions ------------------------------------------------------------
read_dta_flexible <- function(path) {
  if (file.exists(path)) return(haven::read_dta(path))
  if (file.exists(paste0(path, ".dta"))) return(haven::read_dta(paste0(path, ".dta")))

  without_ext <- sub("\\.dta$", "", path)
  if (file.exists(without_ext)) return(haven::read_dta(without_ext))

  stop("Could not find Stata file: ", path, " or ", paste0(path, ".dta"), call. = FALSE)
}

tagged_na_safe <- function(x) {
  out <- rep(FALSE, length(x))
  try(out <- haven::is_tagged_na(x), silent = TRUE)
  out
}

tag_safe <- function(x) {
  out <- rep(NA_character_, length(x))
  try(out <- haven::na_tag(x), silent = TRUE)
  out
}

to_num <- function(x) {
  suppressWarnings(as.numeric(haven::zap_labels(x)))
}

num_eq <- function(x, value) {
  y <- to_num(x)
  !is.na(y) & y == value
}

is_system_missing <- function(x) {
  is.na(x) & !tagged_na_safe(x)
}

to_stata_string <- function(x) {
  # Mirrors Stata's tostring for the variables in this script: numeric system
  # missing values become "."; tagged missing values become ".a", ".b", etc.
  y <- haven::zap_labels(x)
  if (is.factor(y)) y <- as.character(y)

  if (is.numeric(y) || is.integer(y)) {
    out <- rep(".", length(y))
    nonmissing <- !is.na(y)
    out[nonmissing] <- format(y[nonmissing], scientific = FALSE, trim = TRUE, justify = "none")
  } else {
    out <- as.character(y)
    out[is.na(out)] <- "."
  }

  tagged <- tagged_na_safe(x)
  tags <- tag_safe(x)
  out[tagged & !is.na(tags)] <- paste0(".", tags[tagged & !is.na(tags)])
  out[is_system_missing(x)] <- "."
  out
}

as_excel_string <- function(x) {
  out <- as.character(x)
  out[is.na(out)] <- ""
  out
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

order_any <- function(df, vars) {
  ordered <- intersect(vars, names(df))
  df[, c(ordered, setdiff(names(df), ordered)), drop = FALSE]
}

move_to_end <- function(df, var) {
  if (!(var %in% names(df))) return(df)
  df[, c(setdiff(names(df), var), var), drop = FALSE]
}

move_before <- function(df, var, before) {
  if (!(var %in% names(df)) || !(before %in% names(df))) return(df)
  vars <- names(df)
  vars <- vars[vars != var]
  idx <- match(before, vars)
  vars <- append(vars, values = var, after = idx - 1)
  df[, vars, drop = FALSE]
}

rename_by_map <- function(df, rename_map) {
  # rename_map is a named character vector: names are old names, values are new names.
  for (old in names(rename_map)) {
    new <- unname(rename_map[[old]])
    if (old %in% names(df)) names(df)[names(df) == old] <- new
  }
  df
}

replace_values <- function(df, vars, from, to) {
  vars <- intersect(vars, names(df))
  for (var in vars) {
    out <- as.character(df[[var]])
    out[is.na(out)] <- ""
    for (i in seq_along(from)) {
      out[out == from[[i]]] <- to[[i]]
    }
    df[[var]] <- out
  }
  df
}

apply_replacement_table <- function(df, repl) {
  if (nrow(repl) == 0) return(df)
  for (i in seq_len(nrow(repl))) {
    var <- repl$var[[i]]
    if (!(var %in% names(df))) next
    out <- as.character(df[[var]])
    out[is.na(out)] <- ""
    out[out == repl$from[[i]]] <- repl$to[[i]]
    df[[var]] <- out
  }
  df
}

recode_numeric_to_character_move_to_end <- function(df, var, values, labels, missing_label = NULL) {
  if (!(var %in% names(df))) return(df)

  source <- df[[var]]
  out <- to_stata_string(source)
  for (i in seq_along(values)) {
    out[num_eq(source, values[[i]])] <- labels[[i]]
  }
  if (!is.null(missing_label)) out[is_system_missing(source)] <- missing_label

  df[[var]] <- out
  move_to_end(df, var)
}

yes_no_na_move_to_end <- function(df, var) {
  recode_numeric_to_character_move_to_end(
    df, var,
    values = c(1, 0),
    labels = c("Yes", "No"),
    missing_label = "N/A"
  )
}

add_blank_first_row <- function(df) {
  df <- df %>% mutate(across(everything(), as.character))
  header <- tibble::as_tibble(as.list(setNames(rep("", ncol(df)), names(df))))
  bind_rows(header, df)
}

stata_merge_master_using <- function(master, using, by) {
  # Stata merge behavior for overlapping non-key variables: master values take
  # precedence for matched observations. For using-only observations, using values
  # are retained. This function also avoids dplyr's .x/.y suffixes in the result.
  common <- intersect(setdiff(names(master), by), setdiff(names(using), by))
  merged <- full_join(master, using, by = by, suffix = c(".master", ".using"))

  for (var in common) {
    master_name <- paste0(var, ".master")
    using_name <- paste0(var, ".using")
    merged[[var]] <- ifelse(!is.na(merged[[master_name]]), merged[[master_name]], merged[[using_name]])
    merged[[master_name]] <- NULL
    merged[[using_name]] <- NULL
  }

  merged
}

# Data-driven replacement tables translated from Stata ------------------------
qual_simple_replacements <- tibble::tribble(
  ~var, ~from, ~to,
  "M8DAreyoucurrentlyorhavey", "1", "Yes, and I have/had a serious or severe case of COVID-19 and/or serious delayed or long-term health consequences",
  "M8DAreyoucurrentlyorhavey", "2", "Yes, but I have/had only a mild infection (no more than a cold) and also no serious delayed or long-term health consequences",
  "M8DAreyoucurrentlyorhavey", "3", "No, not as far as I know",
  "MDIHowwellinformeddoyoufe", "1", "Not at all well-informed",
  "MDIHowwellinformeddoyoufe", "2", "Somewhat well-informed",
  "MDIHowwellinformeddoyoufe", "3", "Well-informed",
  "MDIHowwellinformeddoyoufe", "4", "Very well-informed",
  "MDIHowwellinformeddoyoufe", "", "N/A",
  "SimondecidedtoinvestEURO", "1", "broken even in the stock market",
  "SimondecidedtoinvestEURO", "2", "is ahead of where he began",
  "SimondecidedtoinvestEURO", "3", "has lost money",
  "SimondecidedtoinvestEURO", "", "N/A",
  "children", "", "Please indicate how many children aged below 18 live in your household (i.e., sleep in your household an average of at least 5 nights per week)",
  "S6Doyouworkinessentialser", "v_26", "Do you work in essential services (e.g. as a cashier, pharmacist, nurse, doctor, etc.)?",
  "S7Doyoupracticeaprofessio", "v_27", "Do you practice a profession in one of the following areas: Culture, gastronomy, entertainment, retail?",
  "M1CHowmanypeoplediedfromC", "v_46", "How many people died from COVID-19 in the AstraZeneca group?",
  "M1DHowmanypeoplediedfromC", "v_47", "How many people died from COVID-19 in the BioNTec group?",
  "M1EWhichaspectsofpublichea", "v_48", "Which aspects of public health can be restored once everyone is vaccinated against COVID-19? Going to the cinema",
  "AO", "v_49", "Which aspects of public health can be restored once everyone is vaccinated against COVID-19? Grocery shopping",
  "AP", "v_50", "Which aspects of public health can be restored once everyone is vaccinated against COVID-19? Seeing a doctor",
  "AQ", "v_51", "Which aspects of public health can be restored once everyone is vaccinated against COVID-19? Going to a restaurant",
  "AR", "v_52", "Which aspects of public health can be restored once everyone is vaccinated against COVID-19? Travelling abroad",
  "M1FWhichactivitieswillonly", "v_53", "Which activity will only be allowed for people who can show a valid vaccination passport? Travelling without quarantine",
  "AT", "v_54", "Which activity will only be allowed for people who can show a valid vaccination passport? Travelling everywhere",
  "AU", "v_55", "Which activity will only be allowed for people who can show a valid vaccination passport? Travelling within your country",
  "AV", "v_56", "Which activity will only be allowed for people who can show a valid vaccination passport? Travelling in the EU",
  "M2BPleaseindicateyouragreem", "v_64", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am confident that the vaccination against COVID-19 is safe and has been sufficiently tested.",
  "BE", "v_65", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am concerned regarding the safety of COVID-19 vaccines because of the speed with which they have been developed and produced.",
  "BF", "v_66", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: The side effects that are associated with vaccination against COVID-19 are downplayed by health authorities on purpose.",
  "M2CPleaseindicateyouragreem", "v_67_1", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am confident that the Moderna vaccine is safe and has been sufficiently tested.",
  "BH", "v_67_2", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am confident that the Astra Zeneca vaccine is safe and has been sufficiently tested.",
  "BI", "v_67_3", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am confident that the Johnson & Johnson vaccine is safe and has been sufficiently tested.",
  "BJ", "v_67_4", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am confident that the BioNTech/Pfizer vaccine is safe and has been sufficiently tested.",
  "BK", "v_68_1", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am concerned regarding the safety of the Moderna because of the speed with which they have been developed and produced.",
  "BL", "v_68_2", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am concerned regarding the safety of the Astra Zeneca because of the speed with which they have been developed and produced.",
  "BM", "v_68_3", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am concerned regarding the safety of the Johnson & Johnson because of the speed with which they have been developed and produced.",
  "BN", "v_68_4", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: I am concerned regarding the safety of the BioNTech/Pfizer because of the speed with which they have been developed and produced.",
  "BO", "v_69_1", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: The side effects that are associated with the Moderna is downplayed by health authorities on purpose.",
  "BP", "v_69_2", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: The side effects that are associated with the Astra Zeneca is downplayed by health authorities on purpose.",
  "BQ", "v_69_3", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: The side effects that are associated with the Johnson & Johnson is downplayed by health authorities on purpose.",
  "BR", "v_69_4", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: The side effects that are associated with the BioNTech/Pfizer is downplayed by health authorities on purpose.",
  "M3APleaseindicateyouragreeme", "v_70", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: When everyone is getting vaccinated against COVID-19, it is not essential for myself to get vaccinated.",
  "BT", "v_71", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: Vaccination against COVID-19 helps primarily to overcome the disadvantages that I am experiencing due to the pandemic, e.g., social distancing and lockdown.",
  "BU", "v_72", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: If I get vaccinated against COVID-19, I can also protect people around me from an infection with the virus.",
  "BV", "v_73", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please indicate your agreement/disagreement to the following statement: If all risk groups are vaccinated against COVID-19, it no longer matters whether I also get vaccinated or not.",
  "M5AAOurbodiesareinawar", "v_75", "It is something of a mystery how vaccines work to protect us from viral diseases like COVID-19 Please read the following idea as to how vaccines work. We would like you to rank it from 1 to 4, where 1 is closest to your understanding about how vaccines work and 4 is the most distant from your understanding about how vaccines work: 'Our bodies are in a war against hostile viruses forces like COVID-19. Vaccines strengthen our armed forces to fight and defeat viruses.'",
  "M5ABVaccinesarelikesecret", "v_76", "It is something of a mystery how vaccines work to protect us from viral diseases like COVID-19 Please read the following idea as to how vaccines work. We would like you to rank it from 1 to 4, where 1 is closest to your understanding about how vaccines work and 4 is the most distant from your understanding about how vaccines work: 'Vaccines are like secret service agents, they tell the body where the enemy is going to attack so the body is ready, prepared, and able to stop them before they cause harm.'",
  "M5ACVaccinesarelikeabull", "v_77", "It is something of a mystery how vaccines work to protect us from viral diseases like COVID-19 Please read the following idea as to how vaccines work. We would like you to rank it from 1 to 4, where 1 is closest to your understanding about how vaccines work and 4 is the most distant from your understanding about how vaccines work: 'Vaccines are like a bullet proof vest protecting the body from the viruses and most of the damage that they could do to the body.'",
  "M5ADVaccinesarelikeherbic", "v_78", "It is something of a mystery how vaccines work to protect us from viral diseases like COVID-19 Please read the following idea as to how vaccines work. We would like you to rank it from 1 to 4, where 1 is closest to your understanding about how vaccines work and 4 is the most distant from your understanding about how vaccines work: 'Vaccines are like herbicides for killing weeds, they prevent the virus from taking root in the body.'",
  "M6ATodevelopanewCOVID19v", "v_79", "There have been several variants (types) of the COVID-19 virus: the `English', the `South African', the `Brazilian', the `Nigerian' and the `Indian'. How effective do you think the following strategy would be against these new variants? To develop a new COVID-19 vaccine every year and offer everyone an annual injection.",
  "M6AToensurethatCOVID19vac", "v_80", "There have been several variants (types) of the COVID-19 virus: the `English', the `South African', the `Brazilian', the `Nigerian' and the `Indian'. How effective do you think the following strategy would be against these new variants? To ensure that COVID-19 vaccines are available in every corner of the world.",
  "M6ATostrengthenhealthsystem", "v_81", "There have been several variants (types) of the COVID-19 virus: the `English', the `South African', the `Brazilian', the `Nigerian' and the `Indian'. How effective do you think the following strategy would be against these new variants? To strengthen health systems in the developing countries.",
  "M6ATohavetightregulationsf", "v_82", "There have been several variants (types) of the COVID-19 virus: the `English', the `South African', the `Brazilian', the `Nigerian' and the `Indian'. How effective do you think the following strategy would be against these new variants? To have tight regulations for travellers from affected countries coming to Europe.",
  "M6BWhyisitsuggestedthatpe", "v_83", "Why is it suggested that people get vaccinated every year against the COVID-19 variants? Because the COVID-19 vaccine only lasts for about 12 months.",
  "CN", "v_84", "Why is it suggested that people get vaccinated every year against the COVID-19 variants? Because more effective vaccines will be available in the near future.",
  "CO", "v_85", "Why is it suggested that people get vaccinated every year against the COVID-19 variants? Because the COVID-19 virus might mutate every year and new protective vaccines are needed.",
  "CP", "v_86", "Why is it suggested that people get vaccinated every year against the COVID-19 variants? It makes a lot of money for the pharmaceutical companies.",
  "M7AInyouropinionhowbelieva", "v_87", "In your opinion how believable is the following account about the origin of COVID-19? 'The COVID-19 virus is just one of those accidents that happen in nature.'",
  "CR", "v_88", "In your opinion how believable is the following account about the origin of COVID-19? 'The COVID-19 virus escaped from a laboratory studying genetics.'",
  "CS", "v_89", "In your opinion how believable is the following account about the origin of COVID-19? 'The COVID-19 virus was caused by people eating wild animals' meat.'",
  "CT", "v_90", "In your opinion how believable is the following account about the origin of COVID-19? 'The COVID-19 virus is due to humans' destruction of the natural world.'",
  "CU", "v_91", "In your opinion how believable is the following account about the origin of COVID-19? 'The COVID-19 virus is the result of people not following the path of God.'",
  "M7Slider", "v_92", "In response to the COVID-19 pandemic governments faced the trade-off between saving human life and saving the national economy. On a scale from 1 to 7, where 1 means 'saving human lifes at the cost of the economy' and 7 means 'saving the economy at the cost of loss of human life', which one would you say?",
  "M8AGeneralvaccineattitudes_I", "v_94", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please evaluate how much you disagree or agree with the following statement: 'I am completely confident that vaccines are safe.'",
  "M8AGeneralvaccineattitudes_V", "v_95", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please evaluate how much you disagree or agree with the following statement: 'Vaccination is unnecessary because vaccine-preventable diseases are not common anymore.'",
  "M8AGeneralvaccineattitudes_E", "v_96", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please evaluate how much you disagree or agree with the following statement: 'Everyday stress prevents me from getting vaccinated.'",
  "M8AGeneralvaccineattitudes_W", "v_97", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please evaluate how much you disagree or agree with the following statement: 'When I think about getting vaccinated, I weigh benefits and risks to make the best decision possible.'",
  "DI", "v_98", "On a scale from 1 ('Strongly disagree') to 7 ('Strongly agree'), please evaluate how much you disagree or agree with the following statement: 'When everyone is vaccinated, I don't have to get vaccinated, too.'",
  "M8BWhichofthefollowingdise", "v_99", "Are you vaccinated against measles?",
  "DK", "v_100", "Are you vaccinated against flu (vaccination within the last three years)?",
  "DL", "v_101", "Are you vaccinated against pneumococci?",
  "DM", "v_102", "Are you vaccinated against hepatitis A?",
  "DN", "v_103", "Are you vaccinated against hepatitis B?",
  "DO", "v_104", "Are you vaccinated against tetanus?",
  "DP", "v_105", "Are you vaccinated against diphteria?",
  "M8CCOVID19threatperceptions", "v_106", "On a scale from 1 to 5, where 1 means 'something I never think about' and 5 means 'something I think about all the time', how did you perceive the Coronavirus situation?",
  "DR", "v_107", "On a scale from 1 to 5, where 1 means 'not frightening' and 5 means 'frightening', how did you perceive the Coronavirus situation?",
  "DS", "v_108", "On a scale from 1 to 5, where 1 means 'not worrisome' and 5 means 'worrisome', how did you perceive the Coronavirus situation?",
  "M8DAreyoucurrentlyorhavey", "v_109", "Are you currently or have you in the past been infected with the coronavirus?",
  "M8EWhichstatementsapplytoy", "v_110", "Please think about how the Corona pandemic has affected people in your social network - your friends, colleagues, relatives and other people you know. Which statements apply to your social network? 'There are or have been untested suspected cases of the coronavirus.'",
  "DV", "v_111", "Please think about how the Corona pandemic has affected people in your social network - your friends, colleagues, relatives and other people you know. Which statements apply to your social network? 'There are or have been confirmed cases of people infected with the coronavirus.'",
  "DW", "v_112", "Please think about how the Corona pandemic has affected people in your social network - your friends, colleagues, relatives and other people you know. Which statements apply to your social network? 'There are or have been individuals who have recovered from a coronavirus infection.'",
  "DX", "v_113", "Please think about how the Corona pandemic has affected people in your social network - your friends, colleagues, relatives and other people you know. Which statements apply to your social network? 'There are or have been individuals who have died as a result of a coronavirus infection.'",
  "M8EWhichstatementsapplyto", "v_114", "Please think about how the Corona pandemic has affected people in your social network - your friends, colleagues, relatives and other people you know. Which statements apply to your social network? 'None of the above.'",
  "M8fPerceivedsocialpressure_A", "v_115", "What do you think: how many out of 100 people in your environment will get vaccinated against COVID-19?",
  "M8gDoyouthinkmostofyourf", "v_116", "Do you think most of your family members want you to get vaccinated against COVID-19?",
  "MDHDoyouthinkmostofyourc", "v_117", "Do you think most of your close friends want you to get vaccinated against COVID-19?",
  "MDIDoyouthinkyouremployer", "v_118", "Do you think your employer wants you to get vaccinated against COVID-19?",
  "MDIHowwellinformeddoyoufe", "v_120", "How well informed do you feel about the COVID-19 vaccination?",
  "Module3_Idontlikesituations", "v_121", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I don't like situations that are uncertain.'",
  "Module3_Idislikequestionswhi", "v_122", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I dislike questions which could be answered in many different ways.'",
  "Module3_Ifindthatawellorde", "v_123", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I find that a well-ordered life with regular hours suits my temperament.'",
  "Module3_Ifeeluncomfortablewh", "v_124", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I feel uncomfortable when I don't understand the reason why an event occurred in my life.'",
  "Module3_Ifeelirritatedwheno", "v_125", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I feel irritated when one person disagrees with what everyone else in a group believes.'",
  "Module3_Idontliketogointo", "v_126", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I don't like to go into a situation without knowing what I can expect from it.'",
  "Module3_WhenIhavemadeadeci", "v_127", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'When I have made a decision, I feel relieved.'",
  "Module3_WhenIamconfrontedwi", "v_128", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'When I am confronted with a problem, I'm dying to reach a solution very quickly.'",
  "Module3_Iwouldquicklybecome", "v_129", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I would quickly become impatient and irritated if I would not find a solution to a problem immediately.'",
  "Module3_Idontliketobewith", "v_130", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I don't like to be with people who are capable of unexpected actions.'",
  "Module3_Idislikeitwhenaper", "v_131", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I dislike it when a person's statement could mean many different things.'",
  "Module3_Ifindthatestablishin", "v_132", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I find that establishing a consistent routine enables me to enjoy life more.'",
  "Module3_Ienjoyhavingaclear", "v_133", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I enjoy having a clear and structured mode of life.'",
  "Module3_Idonotusuallyconsul", "v_134", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I do not usually consult many different opinions before forming my own view.'",
  "Module3_Idislikeunpredictable", "v_135", "On a scale from 1 to 6, where 1 means 'completely disagree' and 6 means 'completely agree', how much do you agree with the following statement? 'I dislike unpredictable situations.'",
  "AbatandaballcostEuro1", "v_136", "A bat and a ball cost #c_0068# 1.10 in total. The bat costs a bit more than the ball. How much does the ball cost?",
  "Ifittakes5machines5minu", "v_137", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
  "Inalakethereisapatcho", "v_138", "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
  "IfJohncandrinkonebarrel", "v_139", "If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 days, how long would it take them to drink one barrel of water together?",
  "Jerryreceivedboththe15th", "v_140", "Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are in the class?",
  "AmanbuysapigforEURO60", "v_141", "A man buys a pig for EURO 60, sells it for EURO 70, buys it back for EURO 80, and sells it finally for EURO 90. How much has he made?",
  "SimondecidedtoinvestEURO", "v_142", "Simon decided to invest EURO 8,000 in the stock market one day early in 2008. Six months after he invested, on July 17, the stocks he had purchased were down 50%. Fortunately for Simon, from July 17 to October 17, the stocks he had purchased went up 75%. At this point, Simon has:",
  "userid", "userid", ""
)
region_replacements <- tibble::tribble(
  ~var, ~from, ~to,
  "regionfr", "1", "Ile-de-France",
  "regionfr", "2", "Haute-Normandie NORD",
  "regionfr", "3", "Picardie",
  "regionfr", "4", "Nord-Pas-de-Calais",
  "regionfr", "5", "Champagne-Ardenne",
  "regionfr", "6", "Lorraine",
  "regionfr", "7", "Alsace",
  "regionfr", "8", "Haute-Normandie",
  "regionfr", "9", "Basse-Normandie",
  "regionfr", "10", "Bretagne",
  "regionfr", "11", "Pays-de-la-Loire",
  "regionfr", "12", "Poitou-Charentes",
  "regionfr", "13", "Centre OUEST",
  "regionfr", "14", "Centre EST",
  "regionfr", "15", "Bourgogne",
  "regionfr", "16", "Franche-Comte",
  "regionfr", "17", "Rhone-Alpes NORD",
  "regionfr", "18", "Auvergne",
  "regionfr", "19", "Limousin",
  "regionfr", "20", "Aquitaine",
  "regionfr", "21", "Midi-Pyrenees",
  "regionfr", "22", "Languedoc-Roussillion",
  "regionfr", "23", "PACA",
  "regionfr", "24", "Rhone-Alpes SUD",
  "regionfr", "25", "Corse",
  "regionfr", "26", "I don't live in France",
  "regionita", "1", "Piemonte",
  "regionita", "2", "Val D'Aosta",
  "regionita", "3", "Liguria",
  "regionita", "4", "Lombardia",
  "regionita", "5", "Trentino-Alto Adige",
  "regionita", "6", "Veneto",
  "regionita", "7", "Friuli-Venezia Giulia",
  "regionita", "8", "Emilia Romagna",
  "regionita", "9", "Toscana",
  "regionita", "10", "Umbria",
  "regionita", "11", "Marche",
  "regionita", "12", "Lazio",
  "regionita", "13", "Sardegna",
  "regionita", "14", "Abruzzo",
  "regionita", "15", "Molise",
  "regionita", "16", "Puglia",
  "regionita", "17", "Campania",
  "regionita", "18", "Basilicata",
  "regionita", "19", "Calabria",
  "regionita", "20", "Sicilia",
  "regionita", "21", "I don't live in Italy",
  "regionspa", "1", "Galicia",
  "regionspa", "2", "Principado de Asturias",
  "regionspa", "3", "Cantabria",
  "regionspa", "4", "Pasi Vasco",
  "regionspa", "5", "Comunidad Foral de Navarra",
  "regionspa", "6", "La Rioja",
  "regionspa", "7", "Aragon",
  "regionspa", "8", "Comunidad de Madrid",
  "regionspa", "9", "Castilla y Leon",
  "regionspa", "10", "Castilla-la Mancia",
  "regionspa", "11", "Extremadura",
  "regionspa", "12", "Cataluna",
  "regionspa", "13", "Comunidad Valenciana",
  "regionspa", "14", "Illes Baleares",
  "regionspa", "15", "Andalucia",
  "regionspa", "16", "Region de Murcia",
  "regionspa", "17", "Canaries",
  "regionspa", "18", "I don't live in Spain",
  "regionswe", "1", "Stockholm",
  "regionswe", "2", "Uppsala",
  "regionswe", "3", "Sodermanlands",
  "regionswe", "4", "Ostergotlands",
  "regionswe", "5", "Orebro",
  "regionswe", "6", "Vastmanslands",
  "regionswe", "7", "Jonkopings",
  "regionswe", "8", "Kronobergs",
  "regionswe", "9", "Kalmar",
  "regionswe", "10", "Gotlands",
  "regionswe", "11", "Blekinge",
  "regionswe", "12", "Skane",
  "regionswe", "13", "Hallands",
  "regionswe", "14", "Vastra Gotalands",
  "regionswe", "15", "Varmlands",
  "regionswe", "16", "Dalarnas",
  "regionswe", "17", "Gavleborgs",
  "regionswe", "18", "Vasternorrlands",
  "regionswe", "19", "Jamtlands",
  "regionswe", "20", "Vasterbottens",
  "regionswe", "21", "Norrbottens",
  "regionswe", "22", "I don't live in Sweden",
  "regionbul", "1", "Vidin",
  "regionbul", "2", "Montana",
  "regionbul", "3", "Vratsa",
  "regionbul", "4", "Pleven",
  "regionbul", "5", "Lovech",
  "regionbul", "6", "Veliko Tarnovo",
  "regionbul", "7", "Grabrovo",
  "regionbul", "8", "Ruse",
  "regionbul", "9", "Razgrad",
  "regionbul", "10", "Silistra",
  "regionbul", "11", "Varna",
  "regionbul", "12", "Dobrich",
  "regionbul", "13", "Shumen",
  "regionbul", "14", "Targovishte",
  "regionbul", "15", "Burgas",
  "regionbul", "16", "Sliven",
  "regionbul", "17", "Yambol",
  "regionbul", "18", "Stara Zagora",
  "regionbul", "19", "Sofia (capital)",
  "regionbul", "20", "Sofia",
  "regionbul", "21", "Blagoevgrad",
  "regionbul", "22", "Gingerbread",
  "regionbul", "23", "Kyustendil",
  "regionbul", "24", "Plovdiv",
  "regionbul", "25", "Haskovo",
  "regionbul", "26", "Pazardzhik",
  "regionbul", "27", "Smolyan",
  "regionbul", "28", "Kardhali",
  "regionbul", "29", "I don't live in Bulgaria",
  "regionpol", "1", "Malopolskie",
  "regionpol", "2", "Slaskie",
  "regionpol", "3", "Wielkopolskie",
  "regionpol", "4", "Zachodniopomorskie",
  "regionpol", "5", "Lubuskie",
  "regionpol", "6", "Dolnoslaskie",
  "regionpol", "7", "Opolskie",
  "regionpol", "8", "Kujawsko-Pomorskie",
  "regionpol", "9", "Warminsko-Mazurskie",
  "regionpol", "10", "Pomorskie",
  "regionpol", "11", "Lodzkie",
  "regionpol", "12", "Swietokrzyskie",
  "regionpol", "13", "Lubelskie",
  "regionpol", "14", "Podkarpackie",
  "regionpol", "15", "Podlaskie",
  "regionpol", "16", "Warszawski stoleczny",
  "regionpol", "17", "Mazowiecki regionalny",
  "regionpol", "18", "I don't live in Poland",
  "regionuk", "1", "England - East Midlands",
  "regionuk", "2", "England - East of England",
  "regionuk", "3", "England - Greater London",
  "regionuk", "4", "England - North East England",
  "regionuk", "5", "England - North West England",
  "regionuk", "6", "England - South East England",
  "regionuk", "7", "England - South West England",
  "regionuk", "8", "England - West Midlands",
  "regionuk", "9", "England - Yorkshire and the Humber",
  "regionuk", "10", "Northern Ireland",
  "regionuk", "11", "Scotland",
  "regionuk", "12", "Wales",
  "regionuk", "13", "I don't live in teh UK"
)
income_replacements <- tibble::tribble(
  ~var, ~from, ~to,
  "incomefr", "1", "0-17999 EUR",
  "incomefr", "2", "18000-21999 EUR",
  "incomefr", "3", "22000-25999 EUR",
  "incomefr", "4", "26000-28999 EUR",
  "incomefr", "5", "29000-32999 EUR",
  "incomefr", "6", "33000-36999 EUR",
  "incomefr", "7", "37000-40999 EUR",
  "incomefr", "8", "41000-48999 EUR",
  "incomefr", "9", "49000-60999 EUR",
  "incomefr", "10", "61000 EUR or higher",
  "incomeita", "1", "0-11999 EUR",
  "incomeita", "2", "12000-15999 EUR",
  "incomeita", "3", "16000-19999 EUR",
  "incomeita", "4", "20000-22999 EUR",
  "incomeita", "5", "23000-26999 EUR",
  "incomeita", "6", "27000-30999 EUR",
  "incomeita", "7", "31000-35999 EUR",
  "incomeita", "8", "36000-41999 EUR",
  "incomeita", "9", "42000-52999 EUR",
  "incomeita", "10", "53000 EUR or higher",
  "incomepol", "1", "0-22999 PLN",
  "incomepol", "2", "23000-29999 PLN",
  "incomepol", "3", "30000-36999 PLN",
  "incomepol", "4", "37000-41999 PLN",
  "incomepol", "5", "42000-47999 PLN",
  "incomepol", "6", "48000-54999 PLN",
  "incomepol", "7", "55000-62999 PLN",
  "incomepol", "8", "63000-72999 PLN",
  "incomepol", "9", "73000-90999 PLN",
  "incomepol", "10", "91000 PLN or higher",
  "incomebul", "1", "0-6999 BGN",
  "incomebul", "2", "7000-9999 BGN",
  "incomebul", "3", "10000-10999 BGN",
  "incomebul", "4", "11000-11999 BGN",
  "incomebul", "5", "12000-13999 BGN",
  "incomebul", "6", "14000-14999 BGN",
  "incomebul", "7", "15000-16999 BGN",
  "incomebul", "8", "17000-18999 BGN",
  "incomebul", "9", "19000-22999 BGN",
  "incomebul", "10", "23000 BGN or higher",
  "incomesp", "1", "0-9999 EUR",
  "incomesp", "2", "10000-13999 EUR",
  "incomesp", "3", "14000-16999 EUR",
  "incomesp", "4", "17000-19999 EUR",
  "incomesp", "5", "20000-23999 EUR",
  "incomesp", "6", "24000-27999 EUR",
  "incomesp", "7", "28000-32999 EUR",
  "incomesp", "8", "33000-38999 EUR",
  "incomesp", "9", "39000-49999 EUR",
  "incomesp", "10", "50000 EUR or higher",
  "incomesw", "1", "0-207999 SEK",
  "incomesw", "2", "208000-260999 SEK",
  "incomesw", "3", "261000-313999 SEK",
  "incomesw", "4", "314000-361999 SEK",
  "incomesw", "5", "362000-406999 SEK",
  "incomesw", "6", "407000-454999 SEK",
  "incomesw", "7", "455000-509999 SEK",
  "incomesw", "8", "510000-579999 SEK",
  "incomesw", "9", "580000-694999 SEK",
  "incomesw", "10", "695000 SEK or higher"
)
final_question_replacements <- tibble::tribble(
  ~var, ~from, ~to,
  "userid", "", "USED_ID",
  "female", "", "What is your gender?",
  "age", "", "What is your age?",
  "education", "", "What is your highest educational attainmet?",
  "region", "", "In which region do you live?",
  "employed", "", "Are you currently employed?",
  "relationship", "", "Are you living in a long-term relationship (including marriage)?",
  "hhsize", "", "How many people live permanently in your household, including yourself?",
  "religion", "", "Do you identify with a religious group?",
  "vaccinated", "", "Have you already been vaccinated against COVID-19?",
  "vaccineintent", "", "How would you decide if you had the opportunity to get vaccinated against COVID-19 next week?",
  "Moderna", "", "If you have certainly decided to get vaccinated next week, will the vaccine be Moderna?",
  "AZ", "", "If you have certainly decided to get vaccinated next week, will the vaccine be Astra Zeneca?",
  "JJ", "", "If you have certainly decided to get vaccinated next week, will the vaccine be Johnson&Johnson?",
  "Pfizer", "", "If you have certainly decided to get vaccinated next week, will the vaccine be BioNTech/Pfizer?",
  "income", "", "If you take all the incomes together: what is the current montly household income of all household members?",
  "vaccine_will", "", "Will you definitely get vaccinated against COVID-19?",
  "country", "", "Country",
  "curfew", "", "Is a curfew in place in your country?",
  "trustgov", "", "What is the level of trust in the government in your country?",
  "healthlit", "", "What is the level of health literacy in your country?",
  "misinform", "", "What is the level of misinformation exposure in your country?",
  "hesitancy", "", "How  certain are you that you will get vaccinated against COVID-19? Please select one of these options: i) I am unsure/I will not get vaccinated, ii) I will get vaccinated",
  "treatment", "", "Treatment",
  "reason1", "", "Please name reasons for which you are undecided.",
  "reason2", "", "Please name reasons why you wouldn't get vaccinated under any circumstances."
)
children_conditions <- tibble::tribble(
  ~S10, ~X, ~Y, ~Z, ~AA, ~children,
  "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
  "0", "0", "0", "0", "0", "0",
  "1", "0", "0", "0", "0", "1",
  "0", "1", "0", "0", "0", "1",
  "0", "0", "1", "0", "0", "1",
  "0", "0", "0", "1", "0", "1",
  "0", "0", "0", "0", "1", "1",
  "2", "0", "0", "0", "0", "2",
  "0", "2", "0", "0", "0", "2",
  "0", "0", "2", "0", "0", "2",
  "0", "0", "0", "2", "0", "2",
  "0", "0", "0", "0", "2", "2",
  "1", "1", "0", "0", "0", "2",
  "1", "0", "1", "0", "0", "2",
  "1", "0", "0", "1", "0", "2",
  "1", "0", "0", "0", "1", "2",
  "0", "1", "1", "0", "0", "2",
  "0", "1", "0", "1", "0", "2",
  "0", "1", "0", "0", "1", "2",
  "0", "0", "1", "1", "0", "2",
  "0", "0", "0", "1", "1", "2",
  "0", "0", "0", "0", "3", "3",
  "0", "0", "0", "1", "2", "3",
  "1", "1", "0", "1", "0", "3",
  "0", "0", "0", "2", "1", "3",
  "0", "0", "0", "3", "0", "3",
  "1", "1", "1", "0", "0", "3",
  "0", "0", "1", "1", "1", "3",
  "0", "1", "2", "0", "0", "3",
  "0", "2", "1", "0", "0", "3",
  "2", "1", "0", "0", "0", "3",
  "0", "2", "0", "1", "0", "3",
  "1", "0", "0", "1", "1", "3",
  "0", "1", "1", "1", "0", "3",
  "1", "0", "1", "1", "0", "3",
  "0", "0", "2", "1", "0", "3",
  "1", "0", "2", "0", "0", "3",
  "0", "0", "1", "0", "2", "3",
  "0", "1", "0", "2", "0", "3",
  "1", "2", "0", "0", "0", "3",
  "0", "0", "3", "0", "0", "3",
  "0", "1", "0", "1", "1", "3",
  "0", "0", "1", "2", "0", "3",
  "2", "2", "0", "0", "0", "4",
  "1", "1", "1", "0", "1", "4",
  "0", "0", "2", "1", "1", "4",
  "1", "2", "1", "0", "0", "4",
  "0", "1", "0", "1", "2", "4",
  "0", "0", "0", "1", "3", "4",
  "1", "1", "1", "1", "0", "4",
  "1", "0", "1", "2", "0", "4",
  "1", "1", "1", "1", "1", "5",
  "0", "4", "0", "0", "1", "5",
  "0", "1", "2", "1", "1", "5",
  "1", "0", "0", "2", "2", "5",
  "0", "1", "2", "0", "2", "5",
  "1", "1", "0", "1", "2", "5",
  "2", "0", "0", "1", "3", "6",
  "0", "0", "2", "2", "2", "5",
  "0", "0", "0", "0", "9", "9",
  "0", "0", "9", "0", "2", "11",
  "1", "11", "0", "0", "0", "12"
)
initial_rename_map <- c(
  "Z" = "X",
  "AA" = "Y",
  "AB" = "Z",
  "AC" = "AA",
  "AQ" = "AO",
  "AR" = "AP",
  "AS" = "AQ",
  "AT" = "AR",
  "AV" = "AT",
  "AW" = "AU",
  "AX" = "AV",
  "BG" = "BE",
  "BH" = "BF",
  "BJ" = "BH",
  "BK" = "BI",
  "BL" = "BJ",
  "BM" = "BK",
  "BN" = "BL",
  "BO" = "BM",
  "BP" = "BN",
  "BQ" = "BO",
  "BR" = "BP",
  "BS" = "BQ",
  "BT" = "BR",
  "BV" = "BT",
  "BW" = "BU",
  "BX" = "BV",
  "CP" = "CN",
  "CQ" = "CO",
  "CR" = "CP",
  "CT" = "CR",
  "CU" = "CS",
  "CV" = "CT",
  "CW" = "CU",
  "DK" = "DI",
  "DM" = "DK",
  "DN" = "DL",
  "DO" = "DM",
  "DP" = "DN",
  "DQ" = "DO",
  "DR" = "DP",
  "DT" = "DR",
  "DU" = "DS",
  "DX" = "DV",
  "DY" = "DW",
  "DZ" = "DX"
)
qualitative_final_rename_map <- c(
  "S6Doyouworkinessentialser" = "S6",
  "S7Doyoupracticeaprofessio" = "S7",
  "children" = "S10",
  "M1CHowmanypeoplediedfromC" = "M1C",
  "M1DHowmanypeoplediedfromC" = "M1D",
  "M1EWhichaspectsofpublichea" = "M1E_48",
  "AO" = "M1E_49",
  "AP" = "M1E_50",
  "AQ" = "M1E_51",
  "AR" = "M1E_52",
  "M1FWhichactivitieswillonly" = "M1F_53",
  "AT" = "M1F_54",
  "AU" = "M1F_55",
  "AV" = "M1F_56",
  "M2BPleaseindicateyouragreem" = "M2B_64",
  "BE" = "M2B_65",
  "BF" = "M2B_66",
  "M2CPleaseindicateyouragreem" = "M2C_67_1",
  "BH" = "M2C_67_2",
  "BI" = "M2C_67_3",
  "BJ" = "M2C_67_4",
  "BK" = "M2C_68_1",
  "BL" = "M2C_68_2",
  "BM" = "M2C_68_3",
  "BN" = "M2C_68_4",
  "BO" = "M2C_69_1",
  "BP" = "M2C_69_2",
  "BQ" = "M2C_69_3",
  "BR" = "M2C_69_4",
  "M3APleaseindicateyouragreeme" = "M3A_70",
  "BT" = "M3A_71",
  "BU" = "M3A_72",
  "BV" = "M3A_73",
  "M5AAOurbodiesareinawar" = "M5A_75",
  "M5ABVaccinesarelikesecret" = "M5A_76",
  "M5ACVaccinesarelikeabull" = "M5A_77",
  "M5ADVaccinesarelikeherbic" = "M5A_78",
  "M6ATodevelopanewCOVID19v" = "M6A_79",
  "M6AToensurethatCOVID19vac" = "M6A_80",
  "M6ATostrengthenhealthsystem" = "M6A_81",
  "M6ATohavetightregulationsf" = "M6A_82",
  "M6BWhyisitsuggestedthatpe" = "M6B_83",
  "CN" = "M6B_84",
  "CO" = "M6B_85",
  "CP" = "M6B_86",
  "M7AInyouropinionhowbelieva" = "M7A_87",
  "CR" = "M7A_88",
  "CS" = "M7A_89",
  "CT" = "M7A_90",
  "CU" = "M7A_91",
  "M7Slider" = "M7_Slider",
  "M8AGeneralvaccineattitudes_I" = "M8A_94",
  "M8AGeneralvaccineattitudes_V" = "M8A_95",
  "M8AGeneralvaccineattitudes_E" = "M8A_96",
  "M8AGeneralvaccineattitudes_W" = "M8A_97",
  "DI" = "M8A_98",
  "M8BWhichofthefollowingdise" = "M8B_99",
  "DK" = "M8B_100",
  "DL" = "M8B_101",
  "DM" = "M8B_102",
  "DN" = "M8B_103",
  "DO" = "M8B_104",
  "DP" = "M8B_105",
  "M8CCOVID19threatperceptions" = "M8C_106",
  "DR" = "M8C_107",
  "DS" = "M8C_108",
  "M8DAreyoucurrentlyorhavey" = "M8D",
  "M8EWhichstatementsapplytoy" = "M8E_110",
  "DV" = "M8E_111",
  "DW" = "M8E_112",
  "DX" = "M8E_113",
  "M8EWhichstatementsapplyto" = "M8E_114",
  "M8fPerceivedsocialpressure_A" = "M8F",
  "M8gDoyouthinkmostofyourf" = "M8G",
  "MDHDoyouthinkmostofyourc" = "MDH",
  "MDIDoyouthinkyouremployer" = "MDI_118",
  "MDIHowwellinformeddoyoufe" = "MDI_120",
  "Module3_Idontlikesituations" = "Module3_121",
  "Module3_Idislikequestionswhi" = "Module3_122",
  "Module3_Ifindthatawellorde" = "Module3_123",
  "Module3_Ifeeluncomfortablewh" = "Module3_124",
  "Module3_Ifeelirritatedwheno" = "Module3_125",
  "Module3_Idontliketogointo" = "Module3_126",
  "Module3_WhenIhavemadeadeci" = "Module3_127",
  "Module3_WhenIamconfrontedwi" = "Module3_128",
  "Module3_Iwouldquicklybecome" = "Module3_129",
  "Module3_Idontliketobewith" = "Module3_130",
  "Module3_Idislikeitwhenaper" = "Module3_131",
  "Module3_Ifindthatestablishin" = "Module3_132",
  "Module3_Ienjoyhavingaclear" = "Module3_133",
  "Module3_Idonotusuallyconsul" = "Module3_134",
  "Module3_Idislikeunpredictable" = "Module3_135",
  "AbatandaballcostEuro1" = "Expanded_136",
  "Ifittakes5machines5minu" = "Expanded_137",
  "Inalakethereisapatcho" = "Expanded_138",
  "IfJohncandrinkonebarrel" = "Expanded_139",
  "Jerryreceivedboththe15th" = "Expanded_140",
  "AmanbuysapigforEURO60" = "Expanded_141",
  "SimondecidedtoinvestEURO" = "Expanded_142"
)
final_rename_map <- c(
  "female" = "S1",
  "age" = "S2",
  "education" = "S3",
  "employed" = "S5",
  "relationship" = "S8",
  "hhsize" = "S9",
  "religion" = "S12",
  "vaccinated" = "M1B",
  "vaccineintent" = "M2A_57",
  "Moderna" = "M2A_58",
  "AZ" = "M2A_59",
  "JJ" = "M2A_60",
  "Pfizer" = "M2A_61",
  "reason1" = "M2A_62",
  "reason2" = "M2A_63",
  "region" = "S4",
  "income" = "S11"
)

# Cleaning qualitative dataset -----------------------------------------------
qual <- readxl::read_excel(qualitative_file, sheet = "Codes", col_types = "text") %>%
  as_tibble() %>%
  mutate(across(everything(), as_excel_string))

qual$country <- "Sweden"

qual <- drop_any(qual, c("source", "number"))
qual <- drop_any(
  qual,
  c(
    "S1Whatisyourgender", "S2Whatisyourage", "S3Whatisyourhighesteducati",
    "S4Inwhichadministrativeuni", "H", "I", "J", "K", "L", "M",
    "S5Areyoucurrentlyemployed", "S8Areyoulivinginalongter",
    "S9Howmanypeoplelivepermane", "AE", "AF", "AG", "AH", "AI", "AJ",
    "S12Doyouidentifywithareli", "M1BHaveyoualreadybeenvacci",
    "M2AHowwouldyoudecideifyou", "Pleaseindicatewithwhichvacci", "BA", "BB", "BC",
    "Pleasenamereasonsforwhichyo", "Pleasenamereasonswhyyouwoul",
    "N", "O", "P", "Q", "R", "S"
  )
)
qual <- rename_by_map(qual, c("uniqueid" = "userid"))

qual <- replace_values(
  qual,
  c("S6Doyouworkinessentialser", "S7Doyoupracticeaprofessio"),
  from = c("1", "2", ""),
  to = c("Yes", "No", "N/A")
)

qual <- replace_values(
  qual,
  c("S10Pleaseindicatehowmanych", "Z", "AA", "AB", "AC"),
  from = c("-66"),
  to = c("N/A")
)

qual <- drop_any(qual, c("S11Ifyoutakealltheincomes", "Messageexperimenttestgroup"))

qual <- replace_values(
  qual,
  c("M1CHowmanypeoplediedfromC", "M1DHowmanypeoplediedfromC"),
  from = c(""),
  to = c("N/A")
)

qual <- replace_values(
  qual,
  c("M1EWhichaspectsofpublichea", "AQ", "AR", "AS", "AT", "M1FWhichactivitieswillonly", "AV", "AW", "AX"),
  from = c("1", "0", ""),
  to = c("Yes", "No", "N/A")
)

qual <- replace_values(
  qual,
  c(
    "M2BPleaseindicateyouragreem", "BG", "BH", "M2CPleaseindicateyouragreem", "BJ", "BK",
    "BL", "BM", "BN", "BO", "BP", "BQ", "BR", "BS", "BT", "M3APleaseindicateyouragreeme",
    "BV", "BW", "BX", "M8AGeneralvaccineattitudes_I", "M8AGeneralvaccineattitudes_V",
    "M8AGeneralvaccineattitudes_E", "M8AGeneralvaccineattitudes_W", "DK"
  ),
  from = c("1", "2", "3", "4", "5", "6", "7", "8", ""),
  to = c("1", "2", "3", "4", "5", "6", "7", "I have too little information/knowledge on this", "N/A")
)

qual <- drop_any(qual, c("M4DCEaboutconditionsoftran", "BZ", "CA", "CB", "CC", "CD", "CE", "CF"))

qual <- replace_values(
  qual,
  c("M6ATodevelopanewCOVID19v", "M6AToensurethatCOVID19vac", "M6ATostrengthenhealthsystem", "M6ATohavetightregulationsf"),
  from = c("1", "2", "3", "4", "5", ""),
  to = c("Not effective at all", "Slightly effective", "Moderately effective", "Very effective", "Not sure", "N/A")
)

qual <- replace_values(
  qual,
  c("M7AInyouropinionhowbelieva", "CT", "CU", "CV", "CW"),
  from = c("1", "2", "3", "4", "5", "6", "7", ""),
  to = c(
    "Extremely Unbelievable", "Unbelievable", "Somewhat Unbelievable", "Not sure",
    "Somewhat Believable", "Believable", "Extremely Believable", "N/A"
  )
)

qual <- drop_any(qual, c("M7Conjoint_1", "M7Conjoint_2", "M7Conjoint_3", "M7Conjoint_4", "M7Conjoint_5", "M7Conjoint_6", "M7Conjoint_7", "M7Conjoint_8"))

qual <- replace_values(
  qual,
  c("M8BWhichofthefollowingdise", "DM", "DN", "DO", "DP", "DQ", "DR", "M8gDoyouthinkmostofyourf", "MDHDoyouthinkmostofyourc", "MDIDoyouthinkyouremployer"),
  from = c("1", "2", "3", ""),
  to = c("Yes", "No", "I don't know", "N/A")
)

qual <- replace_values(
  qual,
  c(
    "M8CCOVID19threatperceptions", "DT", "DU", "Module3_Idontlikesituations", "Module3_Idislikequestionswhi",
    "Module3_Ifindthatawellorde", "Module3_Ifeeluncomfortablewh", "Module3_Ifeelirritatedwheno",
    "Module3_Idontliketogointo", "Module3_WhenIhavemadeadeci", "Module3_WhenIamconfrontedwi",
    "Module3_Iwouldquicklybecome", "Module3_Idontliketobewith", "Module3_Idislikeitwhenaper",
    "Module3_Ifindthatestablishin", "Module3_Ienjoyhavingaclear", "Module3_Idonotusuallyconsul",
    "Module3_Idislikeunpredictable"
  ),
  from = c(""),
  to = c("N/A")
)

qual <- apply_replacement_table(qual, qual_simple_replacements)
qual <- drop_any(qual, c("Additionalcomments", "MDJAccordingtotheinsertei"))
qual <- rename_by_map(qual, initial_rename_map)

if (all(c("S10Pleaseindicatehowmanych", "X", "Y", "Z", "AA") %in% names(qual))) {
  qual$children <- ""
  for (i in seq_len(nrow(children_conditions))) {
    mask <- qual$S10Pleaseindicatehowmanych == children_conditions$S10[[i]] &
      qual$X == children_conditions$X[[i]] &
      qual$Y == children_conditions$Y[[i]] &
      qual$Z == children_conditions$Z[[i]] &
      qual$AA == children_conditions$AA[[i]]
    qual$children[mask] <- children_conditions$children[[i]]
  }
  qual <- move_before(qual, "children", "S10Pleaseindicatehowmanych")
}

qual <- drop_any(qual, c("S10Pleaseindicatehowmanych", "X", "Y", "Z", "AA"))
qual <- apply_replacement_table(qual, qual_simple_replacements)

qual <- order_any(qual, c("country", "userid"))
qual <- rename_by_map(qual, qualitative_final_rename_map)

# Cleaning the main dataset ---------------------------------------------------
main <- read_dta_flexible(main_file) %>% haven::zap_labels() %>% as_tibble()

# Optional heterogeneity merge — only attempted if the file is present.
if (file.exists(heterogeneity_file) ||
    file.exists(paste0(heterogeneity_file, ".dta"))) {
  heterogeneity <- read_dta_flexible(heterogeneity_file) %>% haven::zap_labels() %>% as_tibble()
  main <- stata_merge_master_using(main, heterogeneity,
                                   by = c("userid", "sex", "age", "education"))
  main <- drop_any(main, "_merge")
}
main <- drop_any(main, "female")

main <- rename_by_map(main, c("sex" = "female"))
main <- recode_numeric_to_character_move_to_end(
  main, "female",
  values = c(1, 2, 3),
  labels = c("Women", "Men", "Other")
)

main <- recode_numeric_to_character_move_to_end(
  main, "age",
  values = c(1, 2, 3, 4, 5, 6, 7),
  labels = c("Under 18", "18-24", "25-34", "35-44", "45-54", "66-64", "65+")
)

main <- recode_numeric_to_character_move_to_end(
  main, "education",
  values = c(1, 2, 3, 4),
  labels = c(
    "Primary education (ages 5-11)",
    "Secondary education (ages 16-18 GCSE's)",
    "Further education (A-levels, GNVQ's, BTEC's)",
    "Higher education (Degree +)"
  )
)
if ("education" %in% names(main)) main$education[main$education == "."] <- "N/A"

for (var in c("regionfr", "regionita", "regionspa", "regionswe", "regionbul", "regionpol", "regionuk")) {
  if (var %in% names(main)) {
    main[[var]] <- to_stata_string(main[[var]])
    main <- move_to_end(main, var)
  }
}
main <- apply_replacement_table(main, region_replacements)

source_num <- if ("source" %in% names(main)) to_num(main$source) else rep(NA_real_, nrow(main))
src_eq <- function(value) !is.na(source_num) & source_num == value
main$region <- ""
if ("regionfr" %in% names(main)) main$region[src_eq(2)] <- main$regionfr[src_eq(2)]
if ("regionita" %in% names(main)) main$region[main$region == "" & main$regionita != "." & src_eq(3)] <- main$regionita[main$region == "" & main$regionita != "." & src_eq(3)]
if ("regionspa" %in% names(main)) main$region[main$region == "" & main$regionspa != "." & src_eq(4)] <- main$regionspa[main$region == "" & main$regionspa != "." & src_eq(4)]
if ("regionswe" %in% names(main)) main$region[main$region == "" & main$regionswe != "." & src_eq(5)] <- main$regionswe[main$region == "" & main$regionswe != "." & src_eq(5)]
if ("regionbul" %in% names(main)) main$region[main$region == "" & main$regionbul != "." & src_eq(7)] <- main$regionbul[main$region == "" & main$regionbul != "." & src_eq(7)]
if ("regionpol" %in% names(main)) main$region[main$region == "" & main$regionpol != "." & src_eq(6)] <- main$regionpol[main$region == "" & main$regionpol != "." & src_eq(6)]
if ("regionuk" %in% names(main)) main$region[main$region == "" & main$regionuk != "." & src_eq(1)] <- main$regionuk[main$region == "" & main$regionuk != "." & src_eq(1)]
main$region[src_eq(8)] <- "N/A"

main <- drop_any(main, c("regionfr", "regionita", "regionspa", "regionswe", "regionbul", "regionpol", "regionuk", "regionger"))

main <- recode_numeric_to_character_move_to_end(
  main, "relationship",
  values = c(0, 1, 2),
  labels = c("No", "Yes", "Yes")
)

main <- drop_any(main, "employed")
main <- rename_by_map(main, c("employment" = "employed"))
main <- recode_numeric_to_character_move_to_end(
  main, "employed",
  values = c(1, 2),
  labels = c("Yes", "No")
)

main <- recode_numeric_to_character_move_to_end(
  main, "hhsize",
  values = c(1, 2, 3, 4),
  labels = c("Just me", "2 persons", "3-4 persons", "More than 4 persons")
)

main <- recode_numeric_to_character_move_to_end(
  main, "religion",
  values = c(1, 2, 3, 4, 5),
  labels = c("Christian", "Islam", "Judaism", "Other", "No religious belies, Atheist - Agnostic")
)
if ("religion" %in% names(main)) main$religion[main$religion == "."] <- "N/A"

main <- recode_numeric_to_character_move_to_end(
  main, "vaccinated",
  values = c(0, 1, 2),
  labels = c("No", "Yes (one or two doses)", "Yes (one or two doses)")
)

main <- recode_numeric_to_character_move_to_end(
  main, "vaccineintent",
  values = c(0, 1, 2, 3, 4),
  labels = c(
    "N/A",
    "I would certainly get vaccinated, regardless of which of the three vaccine types I am offered",
    "Whether I would get vaccinated depends on the vaccine type I am offered",
    "I am unsure whether I would get vaccinated",
    "I would not get vaccinated under any circumstances, regardless of which of the three vaccine types was offered"
  )
)
if ("vaccineintent" %in% names(main)) main$vaccineintent[main$vaccineintent == "."] <- "N/A"

for (var in c("Moderna", "AZ", "JJ", "Pfizer")) {
  main <- recode_numeric_to_character_move_to_end(
    main, var,
    values = c(0, 1),
    labels = c("No", "Yes"),
    missing_label = "N/A"
  )
}

for (var in c("incomefr", "incomeita", "incomepol", "incomebul", "incomesp", "incomesw")) {
  if (var %in% names(main)) {
    main[[var]] <- to_stata_string(main[[var]])
    main <- move_to_end(main, var)
  }
}
main <- apply_replacement_table(main, income_replacements)

main$income <- ""
if ("incomefr" %in% names(main)) main$income[src_eq(2)] <- main$incomefr[src_eq(2)]
if ("incomeita" %in% names(main)) main$income[main$income == "" & main$incomeita != "." & src_eq(3)] <- main$incomeita[main$income == "" & main$incomeita != "." & src_eq(3)]
if ("incomepol" %in% names(main)) main$income[main$income == "" & main$incomepol != "." & src_eq(6)] <- main$incomepol[main$income == "" & main$incomepol != "." & src_eq(6)]
if ("incomebul" %in% names(main)) main$income[main$income == "" & main$incomebul != "." & src_eq(7)] <- main$incomebul[main$income == "" & main$incomebul != "." & src_eq(7)]
if ("incomesp" %in% names(main)) main$income[main$income == "" & main$incomesp != "." & src_eq(4)] <- main$incomesp[main$income == "" & main$incomesp != "." & src_eq(4)]
if ("incomesw" %in% names(main)) main$income[main$income == "" & main$incomesw != "." & src_eq(5)] <- main$incomesw[main$income == "" & main$incomesw != "." & src_eq(5)]
main$income[source_num %in% c(8, 1)] <- "N/A"

main <- drop_any(main, c("incomeita", "incomepol", "incomebul", "incomesp", "incomesw", "incomeger", "incomefr"))
main <- drop_any(main, c("christ", "muslim", "jew"))

main <- recode_numeric_to_character_move_to_end(
  main, "hesitancy",
  values = c(0, 1),
  labels = c("I am unsure/I will not get vaccinated", "I will get vaccinated"),
  missing_label = "N/A"
)

for (var in c("risk_trt", "passport_trt", "hedonism_trt", "altruism_trt", "vaccine_will", "curfew")) {
  main <- yes_no_na_move_to_end(main, var)
}

main$treatment <- ""
if ("risk_trt" %in% names(main)) main$treatment[main$risk_trt == "Yes"] <- "Risk reduction"
if ("passport_trt" %in% names(main)) main$treatment[main$passport_trt == "Yes"] <- "Vaccination certificate"
if ("hedonism_trt" %in% names(main)) main$treatment[main$hedonism_trt == "Yes"] <- "Hedonistic benefits"
if ("altruism_trt" %in% names(main)) main$treatment[main$altruism_trt == "Yes"] <- "Altruistic benefits"
main$treatment[main$treatment == ""] <- "Control"

main <- drop_any(main, c("trialarm", "risk_trt", "passport_trt", "hedonism_trt", "altruism_trt"))

country_source <- if ("country" %in% names(main)) main$country else rep(NA, nrow(main))
main$country <- to_stata_string(country_source)
main$country[num_eq(country_source, 1) | src_eq(1)] <- "UK"
main$country[num_eq(country_source, 2) | src_eq(2)] <- "France"
main$country[num_eq(country_source, 3) | src_eq(3)] <- "Italy"
main$country[num_eq(country_source, 8) | src_eq(8)] <- "Germany"
main$country[num_eq(country_source, 4) | src_eq(4)] <- "Spain"
main$country[num_eq(country_source, 7) | src_eq(7)] <- "Bulgaria"
main$country[num_eq(country_source, 5) | src_eq(5)] <- "Sweden"
main$country[num_eq(country_source, 6) | src_eq(6)] <- "Poland"
main <- drop_any(main, "source")
main <- move_to_end(main, "country")

main <- recode_numeric_to_character_move_to_end(
  main, "trustgov",
  values = c(1, 0),
  labels = c("High trust", "Low/Medium trust"),
  missing_label = "N/A"
)

main <- recode_numeric_to_character_move_to_end(
  main, "healthlit",
  values = c(1, 0),
  labels = c("High literacy", "Low/Medium literacy"),
  missing_label = "N/A"
)

main <- recode_numeric_to_character_move_to_end(
  main, "misinform",
  values = c(1, 0),
  labels = c("High misinformation exposure", "Low/Medium misinformation exposure"),
  missing_label = "N/A"
)

main <- drop_any(main, "cv19vac_type")
main <- replace_values(main, "reason1", from = c("-66"), to = c("N/A"))
main <- replace_values(main, "reason2", from = c("-66", "............................", "Xxf", "qdqfbgh,n;"), to = rep("N/A", 4))

if ("country" %in% names(main)) main <- main %>% filter(country == "Sweden")

# Adding questions and qualitative codes --------------------------------------
main <- add_blank_first_row(main)
if ("userid" %in% names(main)) main$userid <- as.character(main$userid)
if ("userid" %in% names(qual)) qual$userid <- as.character(qual$userid)

if ("userid" %in% names(qual) && "userid" %in% names(main)) {
  dat <- stata_merge_master_using(main, qual, by = "userid")
} else {
  # qualitative file has been used to populate sweden-only variables earlier;
  # if its userid column is absent, skip the merge and use main as-is.
  dat <- main
}
dat <- dat %>%
  mutate(across(everything(), as.character)) %>%
  mutate(across(everything(), ~ ifelse(is.na(.x), "", .x)))

dat <- apply_replacement_table(dat, final_question_replacements)

dat <- order_any(
  dat,
  c(
    "country", "userid", "female", "age", "education", "region", "employed", "curfew", "trustgov",
    "healthlit", "misinform", "S6", "S7", "relationship", "hhsize", "S10", "income", "religion",
    "vaccinated", "treatment", "M1C", "M1D", "M1E_48", "M1E_49", "M1E_50", "M1E_51",
    "M1E_52", "M1F_53", "M1F_54", "M1F_55", "M1F_56", "vaccineintent", "Moderna", "AZ",
    "JJ", "Pfizer", "reason1", "reason2"
  )
)

dat <- rename_by_map(dat, final_rename_map)
dat <- drop_any(dat, c("vaccine_will", "hesitancy"))

if ("userid" %in% names(dat)) {
  names(dat)[names(dat) == "userid"] <- "subject_id"
  dat$subject_id[1] <- "subject_id"
  dat <- dat[, c("subject_id", setdiff(names(dat), "subject_id")), drop = FALSE]
}
write_clean_csv(dat, output_file)
