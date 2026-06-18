# Data cleaning script translated from data_cleaning_Chamie.do
# Supplementary-materials version: retains the original Stata recodes,
# variable names, labels, and spelling/typographical choices.

# Required packages -----------------------------------------------------------
# install.packages(c("dplyr", "haven", "readxl", "readr", "stringr", "tidyr"))
library(dplyr)
library(haven)
library(readxl)
library(readr)
library(stringr)
library(tidyr)

# Paths -----------------------------------------------------------------------
if (!exists("read_mapping_csv")) source("preprocessing/utils.R")
data_path   <- file.path("data", "human", "rcts", "chamie_et_al_2021")
saving_path <- file.path("data", "processed", "rcts", "chamie_et_al_2021")
dir.create(saving_path, recursive = TRUE, showWarnings = FALSE)

trial_file <- file.path(data_path, "aim3trial_final.dta")
codebook_file <- file.path(data_path, "IBIS Health AIM3 Data Dictionary_12apr2021_PLOSupload.xlsx")
output_file <- file.path(saving_path, "chamie_et_al_2021_data.csv")

# Helper functions ------------------------------------------------------------
num_eq <- function(x, value) {
  y <- suppressWarnings(as.numeric(as.character(x)))
  !is.na(y) & y == value
}

to_stata_string <- function(x) {
  if (is.numeric(x)) {
    result <- format(x, scientific = FALSE, trim = TRUE)
    result[is.na(x)] <- NA_character_
    result
  } else {
    as.character(x)
  }
}

replace_num <- function(out, source, value, label) {
  out[num_eq(source, value)] <- label
  out
}

recode_numeric_to_character <- function(df, var, map) {
  stopifnot(var %in% names(df))
  source <- df[[var]]
  out <- to_stata_string(source)
  for (value in names(map)) {
    out <- replace_num(out, source, as.numeric(value), unname(map[[value]]))
  }
  df[[var]] <- out
  df
}

replace_string_values <- function(x, from, to) {
  out <- as.character(x)
  out[!is.na(out) & out %in% from] <- to
  out
}

substitute_codes <- function(x, replacements) {
  out <- as.character(x)
  non_na <- !is.na(out)
  for (code in names(replacements)) {
    out[non_na] <- str_replace_all(out[non_na], fixed(code), unname(replacements[[code]]))
  }
  out
}

drop_any <- function(df, vars) {
  dplyr::select(df, -any_of(vars))
}

order_any <- function(df, vars) {
  ordered <- intersect(vars, names(df))
  df[, c(ordered, setdiff(names(df), ordered)), drop = FALSE]
}

replace_like_stata_first_value <- function(df, var, value) {
  if (!var %in% names(df)) return(df)
  first_value <- df[[var]][1]
  if (is.na(first_value)) return(df)
  df[[var]][!is.na(df[[var]]) & df[[var]] == first_value] <- value
  df
}

as_output_character <- function(x) {
  as.character(x)
}

# Cleaning the dataset --------------------------------------------------------
dat <- read_dta(trial_file) %>% haven::zap_labels()

# sex
dat <- recode_numeric_to_character(dat, "sex", c("1" = "Male", "2" = "Female"))

# ageyrs
dat$ageyrs <- to_stata_string(dat$ageyrs)

# moneymade1week
money_source <- dat$moneymade1week
dat$moneymade1week <- to_stata_string(money_source)
money_numeric <- suppressWarnings(as.numeric(as.character(money_source)))
dat$moneymade1week[!is.na(money_numeric) & money_numeric < 0] <- NA_character_
rm(money_numeric)

# primarypartnerhiv
dat <- recode_numeric_to_character(dat, "primarypartnerhiv", c(
  "1" = "Yes", "2" = "No", "-8" = "Refused to answer"
))
# anyotherpartnershiv
dat <- recode_numeric_to_character(dat, "anyotherpartnershiv", c("1" = "Yes", "2" = "No"))

# pay4sex
dat$pay4sex_original <- dat$pay4sex
dat$pay4sex <- to_stata_string(dat$pay4sex_original)
dat$pay4sex[num_eq(dat$pay4sex_original, 1) | num_eq(dat$pay4sex_original, 2) | num_eq(dat$pay4sex_original, 3)] <- "Yes"
dat$pay4sex[num_eq(dat$pay4sex_original, 4)] <- "No"
dat$pay4sex[num_eq(dat$pay4sex_original, -8)] <- "Refused to answer"
dat <- drop_any(dat, "pay4sex_original")

# diagstd
dat <- recode_numeric_to_character(dat, "diagstd", c("1" = "Yes", "0" = "No"))

# studygroup
dat <- recode_numeric_to_character(dat, "studygroup", c(
  "1" = "Control", "3" = "Deposit", "2" = "Incentive"
))

# maritalcat
dat <- recode_numeric_to_character(dat, "maritalcat", c(
  "1" = "Married or cohabitating",
  "2" = "Divorced/separated/widowed",
  "3" = "Never married"
))
dat <- drop_any(dat, "marital")

# schoolcat
dat <- recode_numeric_to_character(dat, "schoolcat", c(
  "1" = "Less than primary or primary",
  "2" = "Secondary",
  "3" = "Tertiary"
))
dat <- drop_any(dat, "lvlsch")

# occupcat
dat <- recode_numeric_to_character(dat, "occupcat", c(
  "1" = "Bar owner/worker",
  "2" = "Boda/motorcycle transport",
  "3" = "Other"
))
dat <- drop_any(dat, c("occup", "occup_oth"))

# locationcat
dat <- recode_numeric_to_character(dat, "locationcat", c(
  "1" = "Bar", "2" = "Boda/Transport", "3" = "Other"
))

# retest indicators
dat <- recode_numeric_to_character(dat, "Mo3retest", c("0" = "No", "1" = "Yes"))
dat <- recode_numeric_to_character(dat, "Mo6retest", c("0" = "No", "1" = "Yes"))
dat <- recode_numeric_to_character(dat, "Mo3Mo6retest", c("0" = "No", "1" = "Yes"))

# scalerisks
dat$scalerisks <- to_stata_string(dat$scalerisks)

# Variables converted to character, with selected missing/refusal values recoded
vars_to_string <- c(
  "intid", "recruitmentcardnum", "fingerprint_status", "travelcost", "monthaway",
  "nuwcigarettes", "numsexualpartners", "totalvalue", "totalvalueweek",
  "numdrinks", "numhivtests", "numtested", "cows", "goats", "sheep",
  "chickens", "moredeposit"
)
for (var in vars_to_string) dat[[var]] <- to_stata_string(dat[[var]])

dat$numsexualpartners <- replace_string_values(dat$numsexualpartners, c("-7", "-8"), NA_character_)
dat$totalvalue        <- replace_string_values(dat$totalvalue,        "-8",           NA_character_)
dat$totalvalueweek    <- replace_string_values(dat$totalvalueweek,    c("-7", "-8"),  NA_character_)
dat$numdrinks         <- replace_string_values(dat$numdrinks,         "-7",           NA_character_)
dat$chickens          <- replace_string_values(dat$chickens,          "-7",           NA_character_)

# 0/1 No/Yes variables
yes_no_01_vars <- c(
  "bchildren", "feltsick", "seektreatment", "toosicktowork", "smoke",
  "currentlysmoke", "consumealcohol", "usecondom", "willingtodeposit",
  "sexpartner", "hivtest", "psexualpartner", "usecondom_pay4sex",
  "self_testing", "riskpref1", "riskpref2", "riskpref3", "riskpref4",
  "riskpref5", "riskpref6", "BLdeposit", "wilretest", "posexpartner",
  "diagnosedsti", "recievegiftsex", "paidgiftsex", "respondentyes",
  "negativehiv", "stayincommunity"
)
for (var in yes_no_01_vars) {
  dat <- recode_numeric_to_character(dat, var, c("0" = "No", "1" = "Yes"))
}


# 1/2 Yes/No variables
one_two_vars <- c(
  "knowanyprep", "heardprep", "takingprep", "heardart", "know_ppletakingart",
  "riskpref7", "riskpref8", "refundtoday"
)
for (var in one_two_vars) {
  dat <- recode_numeric_to_character(dat, var, c("2" = "No", "1" = "Yes"))
}
dat$refundtoday <- replace_string_values(dat$refundtoday, "-9", NA_character_)

# workplace
work_source <- dat$workplace
dat$workplace <- to_stata_string(work_source)
dat$workplace[num_eq(work_source, 2)] <- "Village of residence"
dat$workplace[num_eq(work_source, 4)] <- "Rubindi trading center"
dat$workplace[num_eq(work_source, 6)] <- "Ibanda town"
dat$workplace[num_eq(work_source, 7)] <- "Mbarara town"
dat$workplace[dat$workplace == "9"] <- "Rwensinga"
dat <- drop_any(dat, "workplaceother")

# hfacilityvisit
hf_source <- dat$hfacilityvisit
dat$hfacilityvisit <- to_stata_string(hf_source)
dat$hfacilityvisit[num_eq(hf_source, 1)] <- "HF level 4"
dat$hfacilityvisit[num_eq(hf_source, 2)] <- "HF level 3"
dat$hfacilityvisit[num_eq(hf_source, 3)] <- "HF level 2"
dat$hfacilityvisit[num_eq(hf_source, 5)] <- "Dispensary"
dat$hfacilityvisit[dat$hfacilityvisit_other %in% "DRUG SHOP"] <- "Drug Shop"
dat$hfacilityvisit[dat$hfacilityvisit_other %in% c("PRIVATE", "PRIVATE CLINIC", "PRIVATE CLINIC ", "CLINIC")] <- "Private clinic"
dat$hfacilityvisit[num_eq(hf_source, -9)] <- NA_character_
dat$hfacilityvisit[dat$hfacilityvisit_other %in% "WAS PIOSONED"] <- NA_character_
dat <- drop_any(dat, "hfacilityvisit_other")

# lowrisk
lowrisk_map <- c(
  "1" = "Not having sex",
  "2" = "Using condoms",
  "3" = "Having only one partner",
  "4" = "Limiting the number of partners",
  "5" = "Partner does not have any other partners",
  "6" = "Being cirmcumcised",
  "7" = "Other"
)
dat$lowrisk <- substitute_codes(dat$lowrisk, lowrisk_map)
dat$lowrisk[!is.na(dat$lowrisk) & dat$lowrisk == ""] <- NA_character_
dat <- drop_any(dat, c("nosex", "condom", "onepartner", "limitsno", "nopartiners", "circumcised", "lother", "lowriskother"))

# highrisk
highrisk_map <- c(
  "1" = "Not using condoms",
  "2" = "Having more than one sex partner",
  "3" = "Having an HIV positive partner",
  "4" = "Having blood transfusion/infection",
  "5" = "Not being circumcised",
  "6" = "Being a commercial sex worker",
  "7" = "Having partner who is a commercial sex worker",
  "8" = "Other"
)
dat$highrisk <- substitute_codes(dat$highrisk, highrisk_map)
dat$highrisk[!is.na(dat$highrisk) & dat$highrisk == ""] <- NA_character_
dat <- drop_any(dat, c("nocondom", "morepartiner", "hivpospartner", "bloodtrans", "uncircumcised", "sexworker", "partnersexworker", "hother", "highriskother"))

# resasonsnottest
dat <- recode_numeric_to_character(dat, "resasonsnottest", c(
  "3" = "Too time consuming",
  "4" = "Embarassment",
  "5" = "I know I am negative",
  "6" = "Lack of privacy/afraid to be seen",
  "7" = "Afraid to know the result",
  "-7" = NA_character_
))

# hivtestwhen
dat <- recode_numeric_to_character(dat, "hivtestwhen", c(
  "1" = "< 3 months",
  "2" = "3-6 months",
  "3" = "6-12 months",
  "4" = "12-24 months",
  "5" = "> 2 years"
))

# chancehiv
dat <- recode_numeric_to_character(dat, "chancehiv", c(
  "1" = "High",
  "2" = "Moderate",
  "3" = "Low",
  "4" = "No risk",
  "-7" = "Don't know"
))

# hivtestwhere
dat <- recode_numeric_to_character(dat, "hivtestwhere", c(
  "1" = "Public health facility",
  "2" = "Private clinic",
  "3" = "CHC",
  "4" = "Mobile clinic",
  "6" = "Self-test"
))
dat <- drop_any(dat, "hivtestwhere_other")

# hivresult disclosure and partner status
dat <- recode_numeric_to_character(dat, "hivresult_disclose2partner", c(
  "1" = "Yes", "2" = "No", "3" = "No primary partner"
))

dat <- recode_numeric_to_character(dat, "hivresult_disclose2other", c(
  "1" = "Yes", "2" = "No", "-8" = "Refused to answer"
))

dat <- recode_numeric_to_character(dat, "partnerhivstatus", c(
  "1" = "Yes", "2" = "No", "3" = "No primary partner"
))

# recentsexualintercourse
dat <- recode_numeric_to_character(dat, "recentsexualintercourse", c(
  "1" = "Past 7 days",
  "2" = "Past month",
  "3" = "Past 3 months",
  "4" = "Past 6 months",
  "5" = "Past 1 year"
))

# giftsexchanged
dat$giftsexchanged <- as.character(dat$giftsexchanged)
dat$giftsexchanged <- replace_string_values(dat$giftsexchanged, "-9", NA_character_)
dat$giftsexchanged <- replace_string_values(dat$giftsexchanged, "-8", "Don't know")
dat$giftsexchanged <- str_replace_all(dat$giftsexchanged, fixed("-8"), "Refuse to answer")
giftsexchanged_map <- c(
  "1" = "Cash",
  "2" = "Housing and/or utilities",
  "3" = "Food to eat",
  "4" = "Food to sell",
  "5" = "School fees",
  "6" = "To get a job, a work promotion, or to keep your job",
  "7" = "Other material goods (clothes, jewelry, makeup, electronics, etc.)",
  "8" = "Household items (soap, cleaning supplies, etc.)",
  "9" = "Other"
)
for (code in names(giftsexchanged_map)) {
  dat$giftsexchanged <- str_replace_all(dat$giftsexchanged, fixed(code), unname(giftsexchanged_map[[code]]))
}
dat <- drop_any(dat, c("gecash", "gehouse", "gefoodeat", "gefoodsell", "geschoolfees", "gejob", "geomg", "gehhitems", "geother", "geotherspecify"))

# hiv test likelihood variables
for (var in c("hivtest_3month", "hivtest_12month")) {
  dat <- recode_numeric_to_character(dat, var, c(
    "1" = "Very likely", "2" = "Somewhat likely", "3" = "Not sure"
  ))
}

# concernshivtest
concerns_map <- c(
  "1" = "Cost of travel to place where HIV testing is performed",
  "2" = "Cost of missing work",
  "3" = "Fearful of learning HIV status",
  "4" = "Fearful of someone else learning your HIV status",
  "5" = "Partner is opposed",
  "6" = "No concerns",
  "7" = "Other"
)
dat$concernshivtest <- substitute_codes(dat$concernshivtest, concerns_map)
dat$concernshivtest[!is.na(dat$concernshivtest) & dat$concernshivtest == ""] <- NA_character_
dat <- drop_any(dat, c("costoftravel", "missingwork", "fear", "fearother", "opposed", "noconcerns", "cother", "concerns_hivtestother"))

# mainmotivation
dat <- recode_numeric_to_character(dat, "mainmotivation", c(
  "1" = "Staying health/HIV",
  "2" = "Preventing my partners from infection",
  "3" = "Fearful of getting sick",
  "4" = "Change to receive an incentive",
  "5" = "Other"
))

# choicewheretest
dat <- recode_numeric_to_character(dat, "choicewheretest", c(
  "1" = "Self-test at home",
  "2" = "Test at health facility"
))

# receivedkit
dat <- recode_numeric_to_character(dat, "receivedkit", c(
  "1" = "Very likely",
  "2" = "Somewhat likely",
  "3" = "Not very likely"
))

# depositatenr and deposittoday
for (var in c("depositatenr", "deposittoday")) {
  dat <- recode_numeric_to_character(dat, var, c("1" = "Accept", "2" = "Decline"))
}

# fewertests
dat <- recode_numeric_to_character(dat, "fewertests", c("1" = "< 3 HIV test"))

# acceptdeposit
dat <- recode_numeric_to_character(dat, "acceptdeposit", c("1" = "Accepted", "2" = "Declined"))

# Drop unused raw variables
dat <- drop_any(dat, c(
  "choiceother", "stoptime", "date", "randomnum1", "randomnum2", "baselinetime",
  "baselinedate", "screeningtime", "interviewerid", "cardnumber", "agerange",
  "screeningdate", "comments", "reasondecline", "reasondeclinespecify",
  "recruitmentdate", "locationname", "locationtype", "randomnum24", "randomnum23",
  "randomnum22", "randomnum21", "sw_ver", "rateaminitials", "declinedeposit",
  "enrolled"
))

# Household items
item_df <- tibble(
  item1 = if_else(num_eq(dat$clock, 1), "Clock", NA_character_),
  item2 = if_else(num_eq(dat$electricity, 1), "Electricity", NA_character_),
  item3 = if_else(num_eq(dat$radio, 1), "Radio", NA_character_),
  item4 = if_else(num_eq(dat$tv, 1), "TV", NA_character_),
  item5 = if_else(num_eq(dat$phone, 1), "Phone", NA_character_),
  item6 = if_else(num_eq(dat$fridge, 1), "Fridge", NA_character_),
  item7 = if_else(num_eq(dat$solar, 1), "Solar", NA_character_),
  item8 = if_else(num_eq(dat$bicycle, 1), "Bicycle", NA_character_),
  item9 = if_else(num_eq(dat$motorcycle, 1), "Motorcycle", NA_character_),
  item10 = if_else(num_eq(dat$car, 1), "Car", NA_character_)
)
item_df[is.na(item_df)] <- ""
dat$item <- apply(item_df, 1, paste, collapse = ",")
dat$item[dat$item == ",,,,,,,,,"] <- NA_character_
for (i in seq_len(4)) dat$item <- str_replace_all(dat$item, fixed(",,"), ",")
dat$item <- str_replace(dat$item, ",$", "")
dat$item <- str_replace(dat$item, "^,+", "")
dat <- drop_any(dat, c("clock", "electricity", "radio", "tv", "phone", "fridge", "solar", "bicycle", "motorcycle", "car"))

# Follow-up 0/1 variables
followup_yes_no_vars <- c(
  "Mo3hivtestagain", "Mo3withinterest", "Mo3notdeposited", "Mo3deposit3mo",
  "Mo3noincentive", "Mo6hivtestagain", "Mo6withinterest", "Mo6notdeposited",
  "Mo6deposit6mo", "Mo6noincentive", "Mo3deposit_total"
)
for (var in followup_yes_no_vars) {
  dat <- recode_numeric_to_character(dat, var, c("1" = "Yes", "0" = "No"))
}
dat$Mo3withinterest <- replace_string_values(dat$Mo3withinterest, "7", "Did not make a deposit at last visit")
dat$Mo3notdeposited <- replace_string_values(dat$Mo3notdeposited, "7", "Did not make a deposit at last visit")

dat <- drop_any(dat, c(
  "Mo3interviewstart", "Mo3intid", "Mo3fingerprint_status", "Mo6interviewstart",
  "Mo6intid", "Mo6fingerprint_status", "Mo3highriskotherspecify", "Mo3hivtestother",
  "Mo3stoptime", "Mo3PPstudygroup", "Mo6orstayhealthy", "Mo6ornotinfectpartner",
  "Mo6orfearsick", "Mo6orincentive", "Mo6orother", "Mo6otherreasonsspecify",
  "Mo6stoptime", "Mo6hivtestother"
))

# Mo3/Mo6 chance of HIV
for (var in c("Mo3chancehiv", "Mo6chancehiv")) {
  dat <- recode_numeric_to_character(dat, var, c(
    "1" = "High", "2" = "Moderate", "3" = "Low", "4" = "No risk", "-7" = "Don't know"
  ))
}

# Mo3/Mo6 low-risk reasons
mo_lowrisk_map <- c(
  "1" = "Not having sex",
  "2" = "Using condoms",
  "3" = "Having only one partner",
  "4" = "Limiting the number of partners",
  "5" = "Partner having no other partners",
  "6" = "Having all partners who are HIV negative",
  "7" = "Being circumcised",
  "8" = "Other"
)
for (var in c("Mo3lowrisk", "Mo6lowrisk")) {
  dat[[var]] <- substitute_codes(dat[[var]], mo_lowrisk_map)
  dat[[var]][!is.na(dat[[var]]) & dat[[var]] == ""] <- NA_character_
}
dat <- drop_any(dat, c(
  "Mo3lowrisknosex", "Mo3lowriskusecondom", "Mo3lowriskonepartner",
  "Mo3lowrisklimitpartners", "Mo3lowrisknootherpartners", "Mo3lowriskhivneg",
  "Mo3lowriskcircum", "Mo3lowriskother", "Mo3lowriskotherspecify",
  "Mo6lowrisknosex", "Mo6lowriskusecondom", "Mo6lowriskonepartner",
  "Mo6lowrisklimitpartners", "Mo6lowrisknootherpartners", "Mo6lowriskhivneg",
  "Mo6lowriskcircum", "Mo6lowriskother", "Mo6lowriskotherspecify"
))

# Mo3/Mo6 high-risk reasons
mo_highrisk_map <- c(
  "1" = "Not using condoms",
  "2" = "Having more than one sex partner",
  "3" = "Having a HIV positive partner",
  "4" = "Having blood transfusion/injection",
  "5" = "Not being circumcised",
  "6" = "Being a commercial sex worker",
  "7" = "Having partner(s) who are commercial sex worker(s)",
  "8" = "Other"
)
for (var in c("Mo3highrisk", "Mo6highrisk")) {
  dat[[var]] <- substitute_codes(dat[[var]], mo_highrisk_map)
  dat[[var]][!is.na(dat[[var]]) & dat[[var]] == ""] <- NA_character_
}
dat <- drop_any(dat, c(
  "Mo3highrisknocondom", "Mo3highriskmultiplepartner", "Mo3highriskhivpartner",
  "Mo3highriskbloodtrans", "Mo3highriskcircumsised", "Mo3highrisksexworker",
  "Mo3highriskpartnersexworker", "Mo3highriskother", "Mo6highrisknocondom",
  "Mo6highriskmultiplepartner", "Mo6highriskhivpartner", "Mo6highriskbloodtrans",
  "Mo6highriskcircumsised", "Mo6highrisksexworker", "Mo6highriskpartnersexworker",
  "Mo6highriskother", "Mo6highriskotherspecify"
))

# Mo3/Mo6 main reason for HIV test today
for (var in c("Mo3hivtestoday", "Mo6hivtestoday")) {
  dat <- recode_numeric_to_character(dat, var, c(
    "1" = "Staying health/HIV negative",
    "5" = "Other",
    "3" = "Fearful of getting sick",
    "4" = "Chance to receive an incentive"
  ))
}

# Mo3/Mo6 thoughts about deposit amount
for (var in c("Mo3thoughtsdeposit1", "Mo6thoughtsdeposit1")) {
  dat <- recode_numeric_to_character(dat, var, c(
    "1" = "Too little", "2" = "Just right", "3" = "Too much"
  ))
}

# Mo3/Mo6 other reasons
other_reasons_map <- c(
  "1" = "Stayinh healthy",
  "2" = "Preventing my sexual partners from becoming HIV infected",
  "3" = "Fearful of getting sick",
  "4" = "Chance to receive an incentive",
  "5" = "Other",
  "-7" = "Don't know",
  "-8" = "Refused to answer"
)
for (var in c("Mo3otherreasons", "Mo6otherreasons")) {
  dat[[var]] <- substitute_codes(dat[[var]], other_reasons_map)
  dat[[var]][!is.na(dat[[var]]) & dat[[var]] == ""] <- NA_character_
}

dat <- drop_any(dat, c(
  "Mo3orstayhealthy", "Mo3ornotinfectpartner", "Mo3orfearsick", "Mo3orincentive",
  "Mo3orother", "Mo3otherreasonsspecify", "Mo3comments", "Mo6comments"
))

temp1 <- dat

# Cleaning the codebook -------------------------------------------------------
common_admin_drop <- c(
  "choiceother", "stoptime", "date", "randomnum1", "randomnum2", "baselinetime",
  "baselinedate", "screeningtime", "interviewerid", "cardnumber", "agerange",
  "screeningdate", "comments", "reasondecline", "reasondeclinespecify",
  "recruitmentdate", "locationname", "locationtype", "randomnum24", "randomnum23",
  "randomnum22", "randomnum21", "sw_ver", "rateaminitials", "declinedeposit",
  "enrolled"
)

codebook_base_clean <- function(sheet) {
  read_excel(codebook_file, sheet = sheet) %>%
    rename_with(~ gsub(" ", "", .x)) %>%
    select(-any_of(c("VariableType", "VariableCodes", "AcceptedValues", "Skip", "Comments"))) %>%
    mutate(VariableName = as.character(VariableName), Question = as.character(Question)) %>%
    filter(!is.na(VariableName), VariableName != "")
}

screening <- codebook_base_clean("screening") %>%
  filter(!is.na(VariableName), VariableName != "", !is.na(Question), Question != "") %>%
  filter(VariableName != "comments") %>%
  mutate(Question = if_else(
    Question == "[Interviewer] Does the respondent have a negative rapid HIV antibody result at time of enrollment?",
    "Does the respondent have a negative rapid HIV antibody result at time of enrollment?",
    Question
  )) %>%
  filter(!VariableName %in% c(common_admin_drop, "highrisk")) %>%
  mutate(source = "screening")

baseline <- codebook_base_clean("baseline") %>%
  filter(!VariableName %in% c("clock", "electricity", "radio", "tv", "phone", "fridge", "solar", "bicycle", "motorcycle", "car")) %>%
  filter(!VariableName %in% c(
    "Section A.  Demographic and socio-economic information",
    "Section B: Health and sexual behavior",
    "Section C.  Additional HIV testing and service delivery questions",
    "Section D. Loss aversion questions",
    "Section E:  RANDOMIZATION  ",
    "Section H: Interviewer's Observations and Interview Location ",
    "GENERATED VARIABLES", "", "comments", "Participant Identification"
  )) %>%
  filter(!is.na(Question), Question != "") %>%
  filter(!VariableName %in% c(
    "marital", "occup_oth", "workplaceother", "hfacilityvisit_other",
    "nosex", "condom", "onepartner", "limitsno", "nopartiners", "circumcised", "lother", "lowriskother",
    "nocondom", "morepartiner", "hivpospartner", "bloodtrans", "uncircumcised", "sexworker", "partnersexworker", "hother", "highriskother",
    "gecash", "gehouse", "gefoodeat", "gefoodsell", "geschoolfees", "gejob", "geomg", "gehhitems", "geother", "geotherspecify",
    "costoftravel", "missingwork", "fear", "fearother", "opposed", "noconcerns", "cother", "concerns_hivtestother",
    common_admin_drop, "screeningid"
  )) %>%
  mutate(source = "baseline")

deposit_change <- codebook_base_clean("deposit_change") %>%
  filter(!is.na(Question), Question != "") %>%
  filter(!VariableName %in% c(common_admin_drop, "intid", "subjid", "moredeposit")) %>%
  mutate(VariableName = if_else(VariableName == "BL deposit", "BL_deposit", VariableName)) %>%
  mutate(source = "deposit_change")

followup_prefix_vars <- c(
  "interviewstart", "intid", "fingerprint_status", "chancehiv", "lowrisk",
  "lowrisknosex", "lowriskusecondom", "lowriskonepartner", "lowrisklimitpartners",
  "lowrisknootherpartners", "lowriskhivneg", "lowriskcircum", "lowriskother",
  "lowriskotherspecify", "highrisk", "highrisknocondom", "highriskmultiplepartner",
  "highriskhivpartner", "highriskbloodtrans", "highriskcircumsised", "highrisksexworker",
  "highriskpartnersexworker", "highriskother", "highriskotherspecify", "hivtestoday",
  "hivtestother", "otherreasons", "orstayhealthy", "ornotinfectpartner", "orfearsick",
  "orincentive", "orother", "otherreasonsspecify", "hivtestagain", "withinterest",
  "notdeposited", "thoughtsdeposit1", "deposit3mo", "noincentive", "testelsewhere",
  "notestingreason", "notestingreasonspecify", "thoughtsdeposit2", "losingdeposit",
  "ldforget", "ldfaraway", "ldbusy", "ldsmallamnt", "ldstudymoney", "ldnodeposit",
  "ldnotest", "ldother", "ldotherspecify", "comments", "sw_ver", "stoptime"
)

followup_common_filter <- function(df, include_screeningid = FALSE) {
  drop_vars <- c(
    common_admin_drop,
    "marital", "occup_oth", "workplaceother", "hfacilityvisit_other",
    "nosex", "condom", "onepartner", "limitsno", "nopartiners", "circumcised", "lother", "lowriskother",
    "nocondom", "morepartiner", "hivpospartner", "bloodtrans", "uncircumcised", "sexworker", "partnersexworker", "hother", "highriskother",
    "gecash", "gehouse", "gefoodeat", "gefoodsell", "geschoolfees", "gejob", "geomg", "gehhitems", "geother", "geotherspecify",
    "costoftravel", "missingwork", "fear", "fearother", "opposed", "noconcerns", "cother", "concerns_hivtestother"
  )
  if (include_screeningid) drop_vars <- c(drop_vars, "screeningid")
  df %>% filter(!VariableName %in% drop_vars)
}

followup3 <- codebook_base_clean("followup") %>%
  filter(!is.na(Question), Question != "") %>%
  followup_common_filter(include_screeningid = TRUE) %>%
  mutate(VariableName = if_else(VariableName %in% followup_prefix_vars, paste0("Mo3", VariableName), VariableName)) %>%
  filter(!VariableName %in% c(
    "Mo3interviewstart", "Mo3intid", "Mo3fingerprint_status",
    "Mo3lowrisknosex", "Mo3lowriskusecondom", "Mo3lowriskonepartner",
    "Mo3lowrisklimitpartners", "Mo3lowrisknootherpartners", "Mo3lowriskhivneg",
    "Mo3lowriskcircum", "Mo3lowriskother", "Mo3lowriskotherspecify",
    "Mo3highrisknocondom", "Mo3highriskmultiplepartner", "Mo3highriskhivpartner",
    "Mo3highriskbloodtrans", "Mo3highriskcircumsised", "Mo3highrisksexworker",
    "Mo3highriskpartnersexworker", "Mo3highriskother", "Mo3highriskotherspecify",
    "Mo3orstayhealthy", "Mo3ornotinfectpartner", "Mo3orfearsick", "Mo3orincentive",
    "Mo3orother", "Mo3otherreasonsspecify", "Mo3comments", "Mo3stoptime",
    "Mo3hivtestother", "Mo3PPstudygroup", "subjid", "Mo6retest", "Mo3mo6retest"
  )) %>%
  mutate(source = "followup3")

followup6_prefix_vars <- c(followup_prefix_vars, "deposit6mo")
followup6 <- codebook_base_clean("followup") %>%
  filter(!is.na(Question), Question != "") %>%
  followup_common_filter(include_screeningid = FALSE) %>%
  mutate(VariableName = if_else(VariableName %in% followup6_prefix_vars, paste0("Mo6", VariableName), VariableName)) %>%
  filter(!VariableName %in% c(
    "Mo6interviewstart", "Mo6intid", "Mo6fingerprint_status",
    "Mo6lowrisknosex", "Mo6lowriskusecondom", "Mo6lowriskonepartner",
    "Mo6lowrisklimitpartners", "Mo6lowrisknootherpartners", "Mo6lowriskhivneg",
    "Mo6lowriskcircum", "Mo6lowriskother", "Mo6lowriskotherspecify",
    "Mo6highrisknocondom", "Mo6highriskmultiplepartner", "Mo6highriskhivpartner",
    "Mo6highriskbloodtrans", "Mo6highriskcircumsised", "Mo6highrisksexworker",
    "Mo6highriskpartnersexworker", "Mo6highriskother", "Mo6highriskotherspecify",
    "Mo6orstayhealthy", "Mo6ornotinfectpartner", "Mo6orfearsick", "Mo6orincentive",
    "Mo6orother", "Mo6otherreasonsspecify", "Mo6comments", "Mo6stoptime",
    "Mo6PPstudygroup", "Mo3retest", "Mo3deposit_total"
  )) %>%
  mutate(source = "followup6")

codebook_long <- bind_rows(screening, baseline, deposit_change, followup3, followup6) %>%
  distinct()

if (anyDuplicated(codebook_long$VariableName)) {
  duplicated_vars <- unique(codebook_long$VariableName[duplicated(codebook_long$VariableName)])
  stop(
    "The cleaned codebook contains duplicate VariableName entries: ",
    paste(duplicated_vars, collapse = ", "),
    ". Resolve these before transposing the codebook."
  )
}

codebook_row <- codebook_long %>%
  select(VariableName, Question) %>%
  mutate(Question = as.character(Question)) %>%
  pivot_wider(names_from = VariableName, values_from = Question)

if ("Mo3mo6retest" %in% names(codebook_row)) {
  names(codebook_row)[names(codebook_row) == "Mo3mo6retest"] <- "Mo3Mo6retest"
}
codebook_row$item <- ""

codebook_row <- drop_any(codebook_row, c(
  "transactionalsex", "numrisk", "agecat", "anyhivtest", "frequenttester",
  "highestrisk", "mobility", "lvlsch", "occup", "information1", "hivtestwhere_other",
  "information2", "BL_deposit", "deposit6mo", "Mo3testelsewhere", "Mo3notestingreason",
  "Mo3notestingreasonspecify", "Mo3thoughtsdeposit2", "Mo3losingdeposit",
  "Mo3ldforget", "Mo3ldfaraway", "Mo3ldbusy", "Mo3ldsmallamnt",
  "Mo3ldstudymoney", "Mo3ldnodeposit", "Mo3ldnotest", "Mo3ldother",
  "Mo3ldotherspecify", "subjid", "Mo6deposit3mo", "Mo6testelsewhere",
  "Mo6notestingreason", "Mo6notestingreasonspecify", "Mo6thoughtsdeposit2",
  "Mo6losingdeposit", "Mo6ldforget", "Mo6ldfaraway", "Mo6ldbusy",
  "Mo6ldsmallamnt", "Mo6ldstudymoney", "Mo6ldnodeposit", "Mo6ldnotest",
  "Mo6ldother", "Mo6ldotherspecify", "Mo6hivtestother"
))

final_order <- c(
  "intid", "studyid", "sex", "screeningid", "parishname", "villagename",
  "interviewstart", "recruitmentcardnum", "fingerprint_status", "ageyrs",
  "bchildren", "cows", "goats", "sheep", "chickens", "travelcost",
  "moneymade1week", "workplace", "monthaway", "feltsick", "seektreatment",
  "hfacilityvisit", "toosicktowork", "smoke", "currentlysmoke", "nuwcigarettes",
  "consumealcohol", "numdrinks", "chancehiv", "lowrisk", "highrisk", "hivtest",
  "resasonsnottest", "numtested", "hivtestwhen", "hivtestwhere",
  "hivresult_disclose2partner", "hivresult_disclose2other", "partnerhivstatus",
  "primarypartnerhiv", "anyotherpartnershiv", "heardart", "know_ppletakingart",
  "heardprep", "takingprep", "knowanyprep", "recentsexualintercourse", "usecondom",
  "psexualpartner", "numsexualpartners", "pay4sex", "usecondom_pay4sex",
  "giftsexchanged", "totalvalue", "totalvalueweek", "diagstd", "hivtest_3month",
  "hivtest_12month", "concernshivtest", "mainmotivation", "othermotivation",
  "self_testing", "receivedkit", "choicewheretest", "riskpref1", "riskpref2",
  "riskpref3", "riskpref4", "riskpref5", "riskpref6", "scalerisks",
  "riskpref7", "riskpref8", "studygroup", "acceptdeposit", "moredeposit",
  "willingtodeposit", "maritalcat", "schoolcat", "occupcat", "stayincommunity",
  "negativehiv", "numhivtests", "fewertests", "wilretest", "sexpartner",
  "posexpartner", "diagnosedsti", "recievegiftsex", "paidgiftsex", "respondentyes",
  "locationcat", "depositatenr", "deposittoday", "refundtoday", "BLdeposit",
  "Mo3chancehiv", "Mo3lowrisk", "Mo3highrisk", "Mo3hivtestoday",
  "Mo3otherreasons", "Mo3hivtestagain", "Mo3withinterest", "Mo3notdeposited",
  "Mo3thoughtsdeposit1", "Mo3deposit3mo", "Mo3noincentive", "Mo3retest",
  "Mo6chancehiv", "Mo6lowrisk", "Mo6highrisk", "Mo6hivtestoday",
  "Mo6otherreasons", "Mo6hivtestagain", "Mo6withinterest", "Mo6notdeposited",
  "Mo6thoughtsdeposit1", "Mo6deposit6mo", "Mo6noincentive", "Mo6retest",
  "Mo3deposit_total", "Mo3Mo6retest", "item"
)

codebook_row <- order_any(codebook_row, final_order)

# Appending cleaned codebook row and cleaned data -----------------------------
temp1_chr <- temp1 %>% mutate(across(everything(), as_output_character))
codebook_row_chr <- codebook_row %>% mutate(across(everything(), as_output_character))

final_df <- bind_rows(codebook_row_chr, temp1_chr)

# Cleaning questions ----------------------------------------------------------
question_overrides <- c(
  studyid = "What is your study ID number?",
  sex = "What is your gender?",
  screeningid = "What is your screening ID?",
  parishname = "What is the name of your parish?",
  villagename = "What is the name of your vallage?",
  ageyrs = "What is your age?",
  moneymade1week = "How much money do you normally make in one week of doing work and other income generating activities?  Please do not count money earned by your family, or money earned as interest or from renting goods and land to others.",
  resasonsnottest = "What is the main reason why you never tested for HIV before?",
  primarypartnerhiv = "Is your primary partner HIV positive?",
  anyotherpartnershiv = "Do you have any other sexual partners who you know are HIV positive?",
  numsexualpartners = "In total, with how many different persons have you had sexual intercourse in the past 12 months?",
  giftsexchanged = "The last time you paid or received gifts from someone in exchange for having sexual intercourse, what did you receive in exchange for having sex with this person?",
  hivtest_3month = "How likely would you say you are to go for an HIV test again sometime in the next 12 months?",
  hivtest_12month = "How likely would you say you are to go for an HIV test sometime in the future?",
  concernshivtest = "What concerns do you have about getting an HIV test?",
  scalerisks = "On a scale of 1-10, 10 being that you really like taking risks and 1 being that you do not like taking risks at all, how much do you like taking risks?",
  maritalcat = "What is your current marital status?",
  schoolcat = "What is your current educational level?",
  occupcat = "What is your primary occupation?",
  negativehiv = "Did you have a negative rapid HIV antibody result before this interview?",
  locationcat = "Where were you when recruited for this interview?",
  Mo3chancehiv = "After 3 months: Do you think your chances of having HIV today are high, moderate, low, or no risk at all?",
  Mo3lowrisk = "After 3 months: Why do you think that you have a low chance or no risk of having HIV today?",
  Mo3highrisk = "After 3 months: Why do you think that you have a moderate or high chance of having HIV today?",
  Mo3hivtestoday = "After 3 months: What is your main reason for coming for an HIV test today?",
  Mo3otherreasons = "After 3 months: What are other reasons you came for HIV testing today?",
  Mo3hivtestagain = "After 3 months: Do you think you would have tested for HIV again at this time if you were not in this study?",
  Mo3withinterest = "After 3 months: Did knowing that you would receive your deposit back with interest motivate you?",
  Mo3notdeposited = "After 3 months: Do you think you would have tested for HIV today if you had not deposited any money?",
  Mo3thoughtsdeposit1 = "After 3 months: What do you think about the amount of money you were asked to deposit when we first offered you HIV testing?",
  Mo3deposit3mo = "After 3 months: You can voluntarily make another deposit now, just as you did 3 months ago, that will be repaid to you with interest if you come to retest for HIV again in 3 months. Would you like to make another deposit now?",
  Mo3noincentive = "After 3 months: Do you think you would have tested for HIV again today if you were NOT offered an incentive for repeat testing?",
  Mo3retest = "After 3 months: Did you come back for Month 3 retesting?",
  Mo6chancehiv = "After 6 months: Do you think your chances of having HIV today are high, moderate, low, or no risk at all?",
  Mo6lowrisk = "After 6 months: Why do you think that you have a low chance or no risk of having HIV today?",
  Mo6highrisk = "After 6 months: Why do you think that you have a moderate or high chance of having HIV today?",
  Mo6hivtestoday = "After 6 months: What is your main reason for coming for an HIV test today?",
  Mo6otherreasons = "After 6 months: What are other reasons you came for HIV testing today?",
  Mo6hivtestagain = "After 6 months: Do you think you would have tested for HIV again at this time if you were not in this study?",
  Mo6withinterest = "After 6 months: Did knowing that you would receive your deposit back with interest motivate you?",
  Mo6notdeposited = "After 6 months: Do you think you would have tested for HIV today if you had not deposited any money?",
  Mo6thoughtsdeposit1 = "After 6 months: What do you think about the amount of money you were asked to deposit when we first offered you HIV testing?",
  Mo6deposit6mo = "After 6 months: Would you be willing to make a deposit again now if you had another chance to receive a payment with interest for retesting for HIV in 3 months?",
  Mo6noincentive = "After 6 months: Do you think you would have tested for HIV again today if you were NOT offered an incentive for repeat testing?",
  Mo6retest = "After 6 months: Did you come back for month 6 retesting?",
  Mo3deposit_total = "After 3 months: Did you make a deposit at 3 months",
  Mo3Mo6retest = "Did you come back for retesting after 3 months AND 6 months?",
  item = "Which items does your household have?",
  lowrisk = "Why do you think that you have a low chance or no risk of getting HIV/AIDS?",
  highrisk = "Why do you think that you have a moderate or high chance of having HIV/AIDS?",
  studygroup = "studygroup"
)
for (var in names(question_overrides)) {
  final_df <- replace_like_stata_first_value(final_df, var, unname(question_overrides[[var]]))
}

names(final_df)[names(final_df) == "studygroup"] <- "treatment"
final_df$treatment[1] <- "treatment"

final_df <- drop_any(final_df, c("interviewstart", "recruitmentcardnum", "fingerprint_status"))
final_df <- drop_any(final_df, "othermotivation")
final_df <- drop_any(final_df, c("fewertests", "respondentyes"))
final_df <- drop_any(final_df, c("intid", "Mo3Mo6retest"))

# Export ----------------------------------------------------------------------
if ("studyid" %in% names(final_df)) {
  names(final_df)[names(final_df) == "studyid"] <- "ID"
  final_df$ID[1] <- "ID"
  final_df <- final_df[, c("ID", setdiff(names(final_df), "ID")), drop = FALSE]
}
write_clean_csv(final_df, output_file)
