# Duch et al. (2025) — CANDOUR Wave 2, multi-country survey (Ghana + US).
#
# Reads:
#   data/human/surveys/duch_et_al_2025/candour_wave_2_ghana.csv
#   data/human/surveys/duch_et_al_2025/candour_wave_2_US.csv
# Writes (one per country):
#   data/processed/surveys/duch_et_al_2025/duch_et_al_2025_<country>_data.csv
#
# Per-country preserved differences: US has `past_vote` + `past_candidate`
# (vote in 2020), region/district terminology (state/county vs region/district),
# population reference (~130M households vs ~8M), country-name substitutions
# in question texts.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(tibble)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

source_id <- "duch_et_al_2025"
human_dir     <- file.path("data", "human", "surveys", source_id)
processed_dir <- file.path("data", "processed", "surveys", source_id)

# Per-country settings -------------------------------------------------------
country_settings <- list(
  ghana = list(
    raw_file        = file.path(human_dir, "candour_wave_2_ghana.csv"),
    country_filter  = "Ghana",
    country_name    = "Ghana",
    country_adj     = "Ghanaian",
    region_term     = "region",
    district_term   = "district",
    households_text = "approximately 8 million households in Ghana. Out of these 8 million",
    has_past_vote   = FALSE
  ),
  us = list(
    raw_file        = file.path(human_dir, "candour_wave_2_US.csv"),
    country_filter  = "US",
    country_name    = "the US",
    country_adj     = "US",
    region_term     = "state",
    district_term   = "county",
    households_text = "approximately 130 million households in the US. Out of these 130 million",
    has_past_vote   = TRUE
  )
)

# Variable selection ---------------------------------------------------------
common_select_vars <- c(
  "ID", "StartDate", "country",
  "age", "gender", "REGION_0", "REGION_1", "Q3.3",
  "ideology", "party", "party_commitment", "gov_rate", "gov_relect",
  "ethnicity", "marital_status", "religion", "dep_children", "dep_children_n", "HH_size",
  paste0("politics_", 1:10),
  paste0("eco_out_", c(1:4, 6:43)),
  "vac_hist_1", "vac_hist_2",
  paste0("vac_hist_3_", 1:12),
  paste0("vac_hist_4_", 1:12),
  paste0("vac_hist_6_", 1:8),
  paste0("vac_hist_7_", 1:7),
  paste0("vac_hist_8_", 1:15),
  "vac_hes_1",
  paste0("vac_hes_2_", 1:7),
  paste0("vac_hes_3_", 1:15),
  paste0("vac_hes_4_", 1:5),
  paste0("vac_hes_5_", 1:10),
  paste0("vac_hes_6_", 1:17),
  "vac_hes_7", paste0("vac_hes_8_", 1:3),
  "health_pol_1", "health_pol_2",
  "health_pol_10_1", "health_pol_10_2", "health_pol_10_3",
  "health_pol_11", "health_pol_12", "health_pol_13",
  "health_pol_17", "health_pol_18", "health_pol_19",
  "eco_att_1", "eco_att_2",
  paste0("eco_att_3_", 1:6),
  "eco_att_11_1", "eco_att_11_2", "eco_att_11_3",
  "eco_att_12_1", "eco_att_12_2", "eco_att_13",
  paste0("health_1_", 1:10),
  "health_2",
  paste0("health_3_", 1:5),
  paste0("health_4_", 1:5),
  "health_5", "health_6", "health_7", "health_8", "health_9",
  paste0("eq5d_", c("1_1","1_2","2_1","2_2","3_1","3_2","4_1","4_2","5_1","5_2","6","7"))
)

us_extra_vars <- c("past_vote", "past_candidate")

process_country <- function(setting) {
  raw <- read_csv(setting$raw_file, show_col_types = FALSE,
                  col_types = cols(.default = col_character()))

  vars_to_select <- if (setting$has_past_vote) {
    c(common_select_vars[1:13], us_extra_vars, common_select_vars[14:length(common_select_vars)])
  } else common_select_vars

  df <- raw %>%
    subset(country == setting$country_filter) %>%
    mutate(ID = as.character(seq_len(n()))) %>%
    select(all_of(vars_to_select)) %>%
    mutate(across(everything(), ~ str_replace_all(.x, c("’" = "'")))) %>%
    mutate(
      dep_children_n = ifelse(dep_children == "No", "0",
                       ifelse(dep_children == "Prefer not to say", dep_children, dep_children_n)),
      vac_hist_3_1 = paste(vac_hist_3_1, vac_hist_3_2, vac_hist_3_3, vac_hist_3_4, vac_hist_3_5,
                           vac_hist_3_6, vac_hist_3_7, vac_hist_3_8, vac_hist_3_9, vac_hist_3_10,
                           vac_hist_3_11, vac_hist_3_12, sep = ", "),
      vac_hist_4_1 = paste(vac_hist_4_1, vac_hist_4_2, vac_hist_4_3, vac_hist_4_4, vac_hist_4_5,
                           vac_hist_4_6, vac_hist_4_7, vac_hist_4_8, vac_hist_4_9, vac_hist_4_10,
                           vac_hist_4_11, vac_hist_4_12, sep = ", "),
      vac_hist_6_1 = paste(vac_hist_6_1, vac_hist_6_2, vac_hist_6_3, vac_hist_6_4, vac_hist_6_5,
                           vac_hist_6_6, vac_hist_6_7, vac_hist_6_8, sep = ", "),
      vac_hist_7_1 = paste(vac_hist_7_1, vac_hist_7_2, vac_hist_7_3, vac_hist_7_4, vac_hist_7_5,
                           vac_hist_7_6, vac_hist_7_7, sep = ", "),
      vac_hist_8_1 = paste(vac_hist_8_1, vac_hist_8_2, vac_hist_8_3, vac_hist_8_4, vac_hist_8_5,
                           vac_hist_8_6, vac_hist_8_7, vac_hist_8_8, vac_hist_8_9, vac_hist_8_10,
                           vac_hist_8_11, vac_hist_8_12, vac_hist_8_13, vac_hist_8_14, vac_hist_8_15,
                           sep = ", "),
      vac_hes_2_1 = paste(vac_hes_2_1, vac_hes_2_2, vac_hes_2_3, vac_hes_2_4, vac_hes_2_5,
                          vac_hes_2_6, vac_hes_2_7, sep = ", "),
      vac_hes_3_1 = paste(vac_hes_3_1, vac_hes_3_2, vac_hes_3_3, vac_hes_3_4, vac_hes_3_5,
                          vac_hes_3_6, vac_hes_3_7, vac_hes_3_8, vac_hes_3_9, vac_hes_3_10,
                          vac_hes_3_11, vac_hes_3_12, vac_hes_3_13, vac_hes_3_14, vac_hes_3_15,
                          sep = ", "),
      health_1_1 = paste(health_1_1, health_1_2, health_1_3, health_1_4, health_1_5,
                         health_1_6, health_1_7, health_1_8, health_1_9, health_1_10, sep = ", ")
    ) %>%
    mutate(across(c(vac_hist_3_1, vac_hist_4_1, vac_hist_6_1, vac_hist_7_1, vac_hist_8_1,
                    vac_hes_2_1, vac_hes_3_1, health_1_1),
                  ~ gsub(pattern = "NA, |, NA", replacement = "", x = .))) %>%
    mutate(across(c(vac_hist_3_1, vac_hist_4_1, vac_hist_6_1, vac_hist_7_1, vac_hist_8_1,
                    vac_hes_2_1, vac_hes_3_1, health_1_1),
                  ~ ifelse(. == "NA", NA, .))) %>%
    select(-dep_children,
           -matches("^vac_hist_3_[2-9]$|^vac_hist_3_1[0-2]$"),
           -matches("^vac_hist_4_[2-9]$|^vac_hist_4_1[0-2]$"),
           -matches("^vac_hist_6_[2-8]$"),
           -matches("^vac_hist_7_[2-7]$"),
           -matches("^vac_hist_8_[2-9]$|^vac_hist_8_1[0-5]$"),
           -matches("^vac_hes_2_[2-7]$"),
           -matches("^vac_hes_3_[2-9]$|^vac_hes_3_1[0-5]$"),
           -matches("^health_1_[2-9]$|^health_1_10$"),
           -vac_hist_3_1, -vac_hist_4_1, -vac_hist_7_1) %>%
    mutate(eco_out_4 = gsub("Other reason, please specify", "Other reason", eco_out_4))

  # Build per-country q_item with placeholders -------------------------------
  q_base <- list(
    "ID",
    "What date did this survey start?",
    "Which country do you live in?",
    "What is your current age?",
    "What is your gender?",
    paste0("Which ", setting$region_term, " do you live in?"),
    paste0("Which ", setting$district_term, " do you live in?"),
    "What is the highest educational qualification you have completed?",
    "The following is a scale from 0 to 10 that goes from left to right, where 0 means \"Left\" and 10 means \"Right\". Today when talking about political trends, many people talk about those who are more sympathetic to the left or the right. According to the sense that the terms \"Left\" and \"Right\" have for you when you think about your political point of view, where would you find yourself on this scale? Please, responde with a number.",
    paste0("Thinking generally about political parties in ", ifelse(setting$has_past_vote, "the United States", setting$country_name), " which would you describe yourself as?"),
    "Would you call yourself very committed, fairly committed, or not very committed to that party?",
    paste0("Overall, how would you rate the current ", setting$country_adj, " government on a scale of 0 (very low rating) to 100 (very high rating)?"),
    "Would you vote to re-elect this government in the next election?"
  )
  if (setting$has_past_vote) {
    q_base <- c(q_base, list(
      "Did you vote in the 2020 presidential election?",
      "Which presidential candidate did you vote for in 2020?"
    ))
  }
  q_base <- c(q_base, list(
    "Which, if any, best describes your ethnicity?",
    "Are you currently married, in a civil partnership, or living with a partner?",
    "What is your present religion, if any?",
    "How many dependent children do you have who live with you? (By 'dependent' children, we mean those who are not yet financially independent).",
    "Including yourself, how many adults live in your household? (This refers to all adults, including any children aged 18 or over, who live with you).",
    paste0("How bad or good would you rate the ", setting$country_adj, " government's handling of the COVID-19, pandemic? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD)."),
    "How bad or good would you rate other governments' handling of the COVID-19, pandemic? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD).",
    paste0("Would you say that the lockdown and quarantine policies of the ", setting$country_adj, " government were very bad or very good? Your answer that can range between 0 (VERY BAD) to 100 (VERY GOOD)."),
    "Would you say that the lockdown and quarantine policies of other governments, in general throughout the world, were very good or very bad? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD).",
    paste0("How bad or good would you rate the ", setting$country_adj, " government's handling of the COVID-19 vaccination campaign? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD)."),
    "How bad or good would you rate the other governments' handling of the COVID-19 vaccination campaign? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD).",
    paste0("How bad or good would you rate the ", setting$country_adj, " government's economic policies during the COVID-19 pandemic? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD)."),
    "How bad or good would you rate the other governments' economic policies during the COVID-19 pandemic? Your answer can range between 0 (VERY BAD) to 100 (VERY GOOD).",
    paste0("Do you think COVID-19-related deaths as a percentage of the population in ", setting$country_name, " have been very low or very high? Your answer can range between 0 (VERY LOW) to 100 (VERY HIGH)."),
    "Do you think COVID-19-related deaths as a percentage of the population in other countries in general throughout the world have been very low or very high? Your answer can range between 0 (VERY LOW) to 100 (VERY HIGH).",
    "Last week, did you do ANY work for either pay or profit?",
    "Are you employed by government, by a private company, a nonprofit organization or are you self-employed or working in a family business?",
    "What is your current occupation?",
    "What is your main reason for not working for pay or profit?",
    "How frequently did you work from home in the last four weeks?",
    "Over the past 6 months has your work from home increased or decreased?",
    "Gross PERSONAL income is an individual's total income received from wages and salaries BEFORE any taxes are paid and BEFORE any benefits are obtained. What is your gross annual personal income?",
    "Thinking back to 12 months ago, has your individual income increased or decreased since then?",
    "What is the percentage INCREASE in individual income since 12 months ago?",
    "What is the percentage DECREASE in individual income since 12 months ago?",
    "Gross HOUSEHOLD income combines your gross income with that of your partner or any other household member with whom you share financial responsibilities BEFORE any taxes are paid and BEFORE any benefits are obtained. What is your gross annual household income?",
    "Thinking back to 12 months ago, has your household income increased or decreased since then?",
    "What is the percentage INCREASE in household income since 12 months ago?",
    "What is the percentage DECREASE in household income since 12 months ago?",
    paste0("There are ", setting$households_text, ", how many do you think have income lower than yours?"),
    "During the last 12 months, was there a time when you or others in your household were worried you would not have enough food to eat because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "During the last 12 months, was there a time when you or others in your household were unable to eat healthy and nutritious food because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "During the last 12 months, was there a time when you or your household ate only a few kinds of foods because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "During the last 12 months, was there a time when you or others in your household had to skip a meal because there was not enough money or other resources to get food?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "During the last 12 months, was there a time when your household ran out of food because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "During the last 12 months, was there a time when you or others in your household ate less than you thought you should because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "How often did this happen?",
    "During the last 12 months, was there a time when you or others in your household were hungry but did not eat because there was not enough money or other resources for food?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "How often did this happen?",
    "During the last 12 months, was there a time when you or others in your household went without eating for a whole day because of a lack of money or other resources?",
    "Was this specifically due to the COVID-19 crisis?",
    "Did this happen in the past 4 weeks (30 days)?",
    "How often did this happen?",
    "Have you already been offered, or had an opportunity to receive, a COVID-19 vaccine?",
    "Have you received a COVID-19 vaccine?",
    "Why did you decide NOT to get vaccinated against COVID-19, what are / were reasons?",
    "Why did you decide to get vaccinated against COVID-19, what were the reasons?",
    "If a COVID-19 vaccine was available to you, would you definitely get it, probably get it, probably not get it or definitely not get it?",
    "What are your reasons for NOT getting vaccinated for COVID-19?",
    "What are your reasons for getting vaccinated for COVID-19?",
    "Since this time last year, do you believe you have been infected with the COVID-19 virus?",
    "Since this time last year, have you had a COVID-19 test that showed that you were infected with the virus?",
    "Since this time last year, has a relative of yours been infected with COVID-19?",
    "Since this time last year, have any friends or colleagues of yours been infected with COVID-19?",
    "Do you know anyone who has died from COVID-19?",
    "How much you agree or disagree with the statements: All school children should be required by law to get a COVID-19 vaccine. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: All health care workers who are in contact with patients should be required by law to get a COVID-19 vaccine. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Whether a person gets a COVID-19 vaccine or not should be a matter of personal choice. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Any individual over the age of 65 should be required by law to get a COVID-19 vaccine. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Employers should be allowed to require all employees to get a COVID-19 vaccine. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: The government should make COVID-19 vaccination mandatory for everybody. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Health clinics should be required by law to give a suitable version of a COVID-19 vaccine to all newborns, infants and pre-school children. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: People should not be allowed to travel to other countries unless they can demonstrate that they have been vaccinated against COVID-19. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Only people fully-vaccinated against COVID-19 should be allowed into large indoors events such as cinemas, night clubs and concerts. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Only people fully-vaccinated against COVID-19 should be allowed into cafes and restaurants. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "Since this time last year, have you canceled a doctor appointment as a measure in response to COVID-19?",
    "Since this time last year, have you worn a face mask appointment as a measure in response to COVID-19?",
    "Since this time last year, have you visited a doctor or hospital appointment as a measure in response to COVID-19?",
    "Since this time last year, have you canceled or postponed work activities appointment as a measure in response to COVID-19?",
    "Since this time last year, have you canceled or postponed school activities appointment as a measure in response to COVID-19?",
    "Since this time last year, have you canceled outside housekeepers or caregivers appointment as a measure in response to COVID-19?",
    "Since this time last year, have you avoided some or all restaurants appointment as a measure in response to COVID-19?",
    "Since this time last year, have you worked from home appointment as a measure in response to COVID-19?",
    "Since this time last year, have you studied from home appointment as a measure in response to COVID-19?",
    "Since this time last year, have you canceled or postponed pleasure, social, or recreational activites appointment as a measure in response to COVID-19?",
    "Since this time last year, have you stockpiled food or water appointment as a measure in response to COVID-19?",
    "Since this time last year, have you avoided public or crowded places appointment as a measure in response to COVID-19?",
    "Since this time last year, have you prayed appointment as a measure in response to COVID-19?",
    "Since this time last year, have you avoided contact with high-risk people appointment as a measure in response to COVID-19?",
    "Since this time last year, have you kept six feet distance from those outside my household appointment as a measure in response to COVID-19?",
    "Since this time last year, have you stayed home because I felt unwell appointment as a measure in response to COVID-19?",
    "Since this time last year, have you wiped packages entering my home appointment as a measure in response to COVID-19?",
    "How likely do you think it is that you will get COVID-19 in the next year? Your answer can range from 0 (will definitely not happen) to 100 (will definitely happen)",
    "How much you agree or disagree with the statements: The priority for vaccines should be first doses for those who want them before making booster shots available. You answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: If a booster shot was available to me today, I would get it. You answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: Vaccine booster shots will be needed at least every year to maintain protection against COVID-19. You answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "Governments will need to spend more money in the future on health care facilities and medical workers in order to prevent the spreading of another pandemic virus. Would you be willing to pay additional taxes in order to fund this pandemic prevention spending by government?",
    "Why would you not be willing to pay extra?",
    "How much you agree or disagree with the statements: COVID-19 treatments and vaccines should first be provided for those in the world who need them most. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: COVID-19 treatments and vaccines should first be provided for those around the world who cannot afford to buy them. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "How much you agree or disagree with the statements: COVID-19 treatments and vaccines should first be provided for those who live in the country in which they are first developed. Your answer can range from 0 which means very much disagree to 100 which means very much agree.",
    "Some people feel that the richer countries in the world should donate some of the COVID-19 vaccine doses they have purchased to the World Health Organization (WHO), to distribute in countries that can not afford their own vaccines. What is your view?",
    "Do you think the Government (richer countries) should donate any vaccine doses it has purchased in the future, after everyone in the country (those countries) has (have) had an opportunity to be vaccinated?",
    "Would you be willing to pay an additional tax on airline tickets in order to fund efforts to prevent transmission of the COVID-19 virus and future similar viruses?",
    "Governments are spending money on medical research to find a solution to the COVID-19 virus pandemic. Do you agree or disagree that the government spending will help to find a solution to the COVID-19 virus pandemic? Your answer can range from 0 which means strongly disagree to 100 which means strongly agree.",
    "And what about in the future? Do you think the government should spend more on medical research than they were spending before the pandemic? Spend the same amount on medical research as before the pandemic? Or spend less on medical research than before the pandemic?",
    "How many international flights did you take in the year 2019 - that is prior to the beginning of the COVID-19 pandemic?",
    "Please tell me, in general, how willing or unwilling you are to take risks, using a scale from 0 to 10, where 0 means you are \"completely unwilling to take risks\" and 10 means you are \"very willing to take risks.\" You can also use any number between 0 and 10 to indicate where you fall on the scale, using 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10.",
    "Some people say that people get ahead by their own hard work; others say that lucky breaks or help from other people are more important. Which do you think is most important?",
    "To what extent do you agree or disagree with the following statement: It is the responsibility of the government to reduce the differences in income between people with high incomes and those with low incomes",
    paste0("To what extent do you agree or disagree with the following statement: Differences in income in ", setting$country_name, " are too large"),
    paste0("To what extent do you agree or disagree with the following statement: The government in ", setting$country_name, " is very effective in limiting fraud, waste and abuse in the programs it administers"),
    "To what extent do you agree or disagree with the following statement: The government should provide a decent standard of living for the unemployed",
    "To what extent do you agree or disagree with the following statement: The government should provide decent housing to those who can't afford it",
    "To what extent do you agree or disagree with the following statement: People with high incomes should pay a larger share of their income in taxes than those with low incomes",
    "How well does this statement describes you as a person: When someone does me a favor, I am willing to return it. Please indicate your answer on a scale from 0 to 10. A 0 means \"does not describe me at all\", and a 10 means \"describes me perfectly\". You can use any position between 0 and 10 to indicate where you would put yourself on the scale.",
    "How well does this statement describes you as a person: I am very willing to give to good causes without expecting anything in return. Please indicate your answer on a scale from 0 to 10. A 0 means \"does not describe me at all\", and a 10 means \"describes me perfectly\". You can use any position between 0 and 10 to indicate where you would put yourself on the scale.",
    "How well does this statement describes you as a person: As long as I am not convinced otherwise, I assume that people have only the best intentions. Please indicate your answer on a scale from 0 to 10. A 0 means \"does not describe me at all\", and a 10 means \"describes me perfectly\". You can use any position between 0 and 10 to indicate where you would put yourself on the scale.",
    "How willing are you to punish someone who treats you unfairly, even if there may be costs for you? Please indicate your answer on a scale from 0 to 10. A 0 means \"completely unwilling to do so\", and a 10 means \"very willing to do so\". You can use any position between 0 and 10 to indicate where you would put yourself on the scale.",
    "How willing are you to take risks with your health? Please indicate your answer on a scale from 0 to 10. A 0 means \"completely unwilling to do so\", and a 10 means \"very willing to do so\". You can use any position between 0 and 10 to indicate where you would put yourself on the scale.",
    "Imagine the following situation: Today you unexpectedly received GH₵ 1,610. How much of this amount would you donate to a good cause?",
    "Which underlying health conditions do you have?",
    "How is your health in general?",
    "Please indicate how you have been feeling over the last two weeks: I have felt cheerful and in good spirits",
    "Please indicate how you have been feeling over the last two weeks: I have felt calm and relaxed",
    "Please indicate how you have been feeling over the last two weeks: I have felt active and vigorous",
    "Please indicate how you have been feeling over the last two weeks: I woke up feeling fresh and rested",
    "Please indicate how you have been feeling over the last two weeks: My daily life has been filled with things that interest me",
    "What is the total number of contacts you have had in hospital admissions (where you stayed in hospital one night or more) with health professionals since the beginning of the COVID-19 pandemic?",
    "What is the total number of contacts you have had in hospital clinic (when you did not stay overnight) with health professionals since the beginning of the COVID-19 pandemic?",
    "What is the total number of face-to-face contacts with a doctor you have had since the beginning of the COVID-19 pandemic?",
    "What is the total number of telephone contacts with a health professional (doctor or nurse) you have had since the beginning of the COVID-19 pandemic?",
    "What is the total number of internet contacts with a health professional (doctor or nurse) you have had since the beginning of the COVID-19 pandemic?",
    "Do you regularly take medications for any health condition?",
    "Since the beginning of the COVID-19 pandemic, have there been times when you have not been able to promptly obtain the medications for your condition when needed?",
    "Please indicate the extent to which you agree/disagree with the following statement about your personal experience with your health providers since the beginning of the COVID-19 pandemic. Getting a face-to-face appointment with my doctor has been difficult.",
    "Please indicate the extent to which you agree/disagree with the following statements about your personal experience with your health providers since the beginning of the COVID-19 pandemic. Getting a telephone/internet appointment with my doctor has been difficult.",
    "Please indicate the extent to which you agree/disagree with the following statements about your personal experience with your health providers since the beginning of the COVID-19 pandemic. Scheduling a surgery in a clinic/hospital has been difficult",
    "Which statement best describes your mobility A YEAR AGO?",
    "Which statement best describes your mobility TODAY?",
    "Which statement best describes your personal care A YEAR AGO?",
    "Which statement best describes your personal care TODAY?",
    "Which statement best describes your usual activities A YEAR AGO (e.g. work, study, housework, family or leisure activities)?",
    "Which statement best describes your usual activities TODAY (e.g. work, study, housework, family or leisure activities)?",
    "Which statement best describes your pain / discomfort A YEAR AGO?",
    "Which statement best describes your pain / discomfort TODAY?",
    "Which statement best describes your anxiety / depression A YEAR AGO?",
    "Which statement best describes your anxiety / depression TODAY?",
    "We would like to know how good or bad your health was A YEAR AGO. Please answer on a scale numbered 0 to 100. 100 means the best health you can imagine. 0 means the worst health you can imagine.",
    "We would like to know how good or bad your health is TODAY. Please answer on a scale numbered 0 to 100. 100 means the best health you can imagine. 0 means the worst health you can imagine."
  ))

  q_item <- unlist(q_base) %>%
    gsub(pattern = "Australia", replacement = setting$country_name, x = .) %>%
    t() %>% as.data.frame(stringsAsFactors = FALSE)

  if (length(q_item) != ncol(df)) {
    stop("q_item length (", ncol(q_item), ") does not match data frame columns (", ncol(df), ") for country ", setting$country_filter)
  }
  colnames(q_item) <- colnames(df)

  df_out <- rbind(q_item, as.data.frame(lapply(df, as.character), stringsAsFactors = FALSE))
  names(df_out)[names(df_out) == "ID"] <- "subject_id"
  df_out$subject_id[1] <- "subject_id"
  df_out <- df_out[, c("subject_id", setdiff(names(df_out), "subject_id")), drop = FALSE]

  output_path <- file.path(processed_dir,
                           paste0(source_id, "_", tolower(setting$country_filter), "_data.csv"))
  write_clean_csv(df_out, output_path)
}

for (key in names(country_settings)) {
  process_country(country_settings[[key]])
}
