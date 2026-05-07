# HPS 2021 Pulse — multi-week US household pulse survey.
#
# Reads:
#   data/human/surveys/hps_2021/pulse2021_puf_22.csv .. pulse2021_puf_27.csv
# Writes:
#   data/processed/surveys/hps_2021/hps_2021_data.csv

# load packages
suppressPackageStartupMessages({
  library(tidyverse)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

# Utils
merge_dummies <- function(data, prefix) {
  
  dummy_cols <- str_subset(names(data), paste0("^", prefix, "[0-9]+$"))
  label <- data[[prefix]][1]
  
  data %>%
    mutate(
      across(
        all_of(dummy_cols),
        ~ case_when(
          row_number() == 1 ~ .,  # Preserve first row
          . == "1" ~ first(.),    # Use first value where "1"
          is.na(.) ~ NA_character_,
          TRUE ~ "Not applicable"
        )
      )
    ) %>%
    mutate(
      !!prefix := if_else(
        row_number() == 1,
        label,
        apply(select(., all_of(dummy_cols)), 1, 
              function(x) paste(x[!is.na(x) & x != "Not applicable"], collapse = "; "))
      )
    )
}

# load data
hps_dir <- file.path("data", "human", "surveys", "hps_2021")
raw_data_22 <- read_csv(file.path(hps_dir, "pulse2021_puf_22.csv"))
raw_data_23 <- read_csv(file.path(hps_dir, "pulse2021_puf_23.csv"))
raw_data_24 <- read_csv(file.path(hps_dir, "pulse2021_puf_24.csv"))
raw_data_25 <- read_csv(file.path(hps_dir, "pulse2021_puf_25.csv"))
raw_data_26 <- read_csv(file.path(hps_dir, "pulse2021_puf_26.csv"))
raw_data_27 <- read_csv(file.path(hps_dir, "pulse2021_puf_27.csv"))

# combine data
# same colnames
raw_data <- rbind(raw_data_22,
                  raw_data_23,
                  raw_data_24,
                  raw_data_25,
                  raw_data_26,
                  raw_data_27)


vars_selected <- c(
  "SCRAM", "WEEK", "EST_ST", "EST_MSA",
  "TBIRTH_YEAR", "EGENDER", "RHISPANIC", "RRACE",
  "EEDUC", "MS", "THHLD_NUMPER", "THHLD_NUMKID", "RECVDVACC",
  "DOSES", "GETVACC",
  
  # Dummy set to merge
  "WHYNOT",
  
  # Dummy set to merge
  "WHYNOTB",
  
  "HADCOVID", "WRKLOSS", "EXPCTLOSS", "ANYWORK", "KINDWORK", "RSNNOWRK",
  "TW_START", "UI_APPLY", "UI_RECV", "SSA_RECV", "SSA_APPLY",
  
  "SSALIKELY", "SSADECISN", "EIP", "EXPNS_DIF",
  
  # Dummy set to merge
  "CHNGHOW",
  
  # Dummy set to merge
  "WHYCHNGD",
  
  # Dummy set to merge
  "SPNDSRC",
  
  "FEWRTRIPS", "FEWRTRANS", "CURFOODSUF", "CHILDFOOD",
  
  # Dummy set to merge
  "FOODSUFRSN",
  
  "FREEFOOD", "SNAP_YN", "TSPNDFOOD", "TSPNDPRPD",
  "ANXIOUS", "WORRY", "INTEREST", "DOWN",
  "PRIVHLTH", "PUBHLTH", "DELAY", "NOTGET", "PRESCRIPT", "MH_SVCS",
  "MH_NOTGET", "TENURE", "LIVQTR", "RENTCUR", "MORTCUR", "MORTCONF",
  
  # Dummy set to merge
  "ENROLL",
  
  # Dummy set to merge
  "TEACH",
  
  "INCOME"
)


dummy_vars_to_merge <- c(
  "WHYNOT",
  "WHYNOTB",
  "CHNGHOW",
  "WHYCHNGD",
  "SPNDSRC",
  "FOODSUFRSN",
  "ENROLL",
  "TEACH"
)

# Set seed for random sampling
set.seed(123)

# Create df
df <- raw_data %>%
  
  # Random sample of 1,500 observations
  sample_n(1500) %>%
  
  # Select variables
  select(all_of(setdiff(vars_selected, dummy_vars_to_merge)), starts_with(dummy_vars_to_merge)) %>%
  
  # Recode variables
  mutate(
    across(everything(), ~as.character(.)),
    EST_ST = case_when(
      EST_ST == "01" ~ "Albama",
      EST_ST == "02" ~ "Alaska",
      EST_ST == "04" ~ "Arizona",
      EST_ST == "05" ~ "Arkansas",
      EST_ST == "06" ~ "California",
      EST_ST == "08" ~ "Colorado",
      EST_ST == "09" ~ "Connecticut",
      EST_ST == "10" ~ "Delaware",
      EST_ST == "11" ~ "District of Columbia",
      EST_ST == "12" ~ "Florida",
      EST_ST == "13" ~ "Georgia",
      EST_ST == "15" ~ "Hawaii",
      EST_ST == "16" ~ "Idaho",
      EST_ST == "17" ~ "Illinois",
      EST_ST == "18" ~ "Indiana",
      EST_ST == "19" ~ "Iowa",
      EST_ST == "20" ~ "Kansas",
      EST_ST == "21" ~ "Kentucky",
      EST_ST == "22" ~ "Louisiana",
      EST_ST == "23" ~ "Maine",
      EST_ST == "24" ~ "Maryland",
      EST_ST == "25" ~ "Massachusetts",
      EST_ST == "26" ~ "Michigan",
      EST_ST == "27" ~ "Minnesota",
      EST_ST == "28" ~ "Mississippi",
      EST_ST == "29" ~ "Missouri",
      EST_ST == "30" ~ "Montana",
      EST_ST == "31" ~ "Nebraska",
      EST_ST == "32" ~ "Nevada",
      EST_ST == "33" ~ "New Hampshire",
      EST_ST == "34" ~ "New Jersey",
      EST_ST == "35" ~ "New Mexico",
      EST_ST == "36" ~ "New York",
      EST_ST == "37" ~ "North Carolina",
      EST_ST == "38" ~ "North Dakota",
      EST_ST == "39" ~ "Ohio",
      EST_ST == "40" ~ "Oklahoma",
      EST_ST == "41" ~ "Oregon",
      EST_ST == "42" ~ "Pennsylvania",
      EST_ST == "44" ~ "Rhode Island",
      EST_ST == "45" ~ "South Carolina",
      EST_ST == "46" ~ "South Dakota",
      EST_ST == "47" ~ "Tennessee",
      EST_ST == "48" ~ "Texas",
      EST_ST == "49" ~ "Utah",
      EST_ST == "50" ~ "Vermont",
      EST_ST == "51" ~ "Virginia",
      EST_ST == "53" ~ "Washington",
      EST_ST == "54" ~ "West Virginia",
      EST_ST == "55" ~ "Wisconsin",
      EST_ST == "56" ~ "Wyoming"),
    EST_MSA = case_when(
      EST_MSA == '35620' ~ "New York-Newark-Jersey City, NY-NJ-PA Metro Area",
      EST_MSA == '31080' ~ "Los Angeles-Long Beach-Anaheim, CA Metro Area",
      EST_MSA == '16980' ~ "Chicago-Naperville-Elgin, IL-IN-WI Metro Area",
      EST_MSA == '19100' ~ "Dallas-Fort Worth-Arlington, TX Metro Area",
      EST_MSA == '26420' ~ "Houston-The Woodlands-Sugar Land, TX Metro Area",
      EST_MSA == '47900' ~ "Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area",
      EST_MSA == '33100' ~ "Miami-Fort Lauderdale-Pompano Beach, FL Metro Area",
      EST_MSA == '37980' ~ "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metro Area",
      EST_MSA == '12060' ~ "Atlanta-Sandy Springs-Alpharetta, GA Metro Area",
      EST_MSA == '38060' ~ "Phoenix-Mesa-Chandler, AZ Metro Area",
      EST_MSA == '14460' ~ "Boston-Cambridge-Newton, MA-NH Metro Area",
      EST_MSA == '41860' ~ "San Francisco-Oakland-Berkeley, CA Metro Area",
      EST_MSA == '40140' ~ "Riverside-San Bernardino-Ontario, CA Metro Area",
      EST_MSA == '19820' ~ "Detroit-Warren-Dearborn, MI Metro Area",
      EST_MSA == '42660' ~ "Seattle-Tacoma-Bellevue, WA Metro Area",
      TRUE ~ NA),
    across(everything(), ~case_when(. == "-99" ~ NA,
                                    . == "-88" ~ "Not applicable",
                                    TRUE ~ .)),
    EGENDER = ifelse(EGENDER == "1", "MALE", "FEMALE"),
    RHISPANIC = ifelse(RHISPANIC == "1",
                       "No, not of Hispanic, Latino, or Spanish origin",
                       "Yes, of Hispanic, Latino, or Spanish origin"),
    RRACE = case_when(RRACE == "1" ~ "White, Alone",
                      RRACE == "2" ~ "Black, Alone",
                      RRACE == "3" ~ "Asian, Alone",
                      RRACE == "4" ~ "Any other race alone, or race in combination"),
    EEDUC = case_when(EEDUC == "1" ~ "Less than high school",
                      EEDUC == "2" ~ "Some high school",
                      EEDUC == "3" ~ "High school graduate or equivalent (e.g. GED)",
                      EEDUC == "4" ~ "Some college, but degree not received or is in progress",
                      EEDUC == "5" ~ "Associate's degree (e.g. AA, AS)",
                      EEDUC == "6" ~ "Bachelor's degree (e.g. BA, BS, AB)",
                      EEDUC == "7" ~ "Graduate degree (e.g. master's, professional, doctorate)"),
    MS = case_when(MS == "1" ~ "Now married",
                   MS == "2" ~ "Widowed",
                   MS == "3" ~ "Divorced",
                   MS == "4" ~ "Separated",
                   MS == "5" ~ "Never married",
                   TRUE ~ MS),
    across(matches("RECVDVACC|DOSES|WRKLOSS|EXPCTLOSS|ANYWORK|^UI_|^SSA_|FEWRTRIPS|FREEFOOD|SNAP_YN|DELAY|NOTGET|PRESCRIPT|^MH_|RENTCUR|MORTCUR"),
           ~case_when(as.character(.) == "1" ~ "Yes",
                      as.character(.) == "2" ~ "No",
                      TRUE ~ as.character(.))),
    GETVACC = case_when(GETVACC == "1" ~ "Definitely get a vaccine",
                        GETVACC == "2" ~ "Probably get a vaccine",
                        GETVACC == "3" ~ "Probably NOT get a vaccine",
                        GETVACC == "4" ~ "Definitely NOT get a vaccine",
                        TRUE ~ GETVACC),
    HADCOVID = case_when(HADCOVID == "1" ~ "Yes",
                         HADCOVID == "2" ~ "No",
                         HADCOVID == "3" ~ "Not sure",
                         TRUE ~ HADCOVID),
    KINDWORK = case_when(KINDWORK == "1" ~ "Government",
                         KINDWORK == "2" ~ "Private company",
                         KINDWORK == "3" ~ "Non-profit organization including tax exempt and charitable organizations",
                         KINDWORK == "4" ~ "Self-employed",
                         KINDWORK == "5" ~ "Working in a family business",
                         TRUE ~ KINDWORK),
    RSNNOWRK = case_when(RSNNOWRK == "1" ~ "I did not want to be employed at this time",
                         RSNNOWRK == "2" ~ "I am/was sick with coronavirus symptoms",
                         RSNNOWRK == "3" ~ "I am/was caring for someone with coronavirus symptoms",
                         RSNNOWRK == "4" ~ "I am/was caring for children not in school or daycare",
                         RSNNOWRK == "5" ~ "I am/was caring for an elderly person",
                         RSNNOWRK == "6" ~ "I am/was sick (not coronavirus related) or disabled",
                         RSNNOWRK == "7" ~ "I am retired",
                         RSNNOWRK == "8" ~ "My employer experienced a reduction in business (including furlough) due to coronavirus pandemic",
                         RSNNOWRK == "9" ~ "I am/was laid off due to coronavirus pandemic",
                         RSNNOWRK == "10" ~ "My employer closed temporarily due to the coronavirus pandemic",
                         RSNNOWRK == "11" ~ "My employer went out of business due to the coronavirus pandemic",
                         RSNNOWRK == "12" ~ "Other reason, please specify",
                         RSNNOWRK == "13" ~ "I was concerned about getting or spreading the coronavirus",
                         TRUE ~ RSNNOWRK),
    TW_START = case_when(TW_START == "1" ~ "Yes, at least one adult substituted some or all of their typical in-person work for telework",
                         TW_START == "2" ~ "No, no adults substituted their typical in-person work for telework",
                         TW_START == "3" ~ "No, there has been no change in telework",
                         TRUE ~ TW_START),
    SSALIKELY = case_when(SSALIKELY == "1" ~ "Extremely likely",
                          SSALIKELY == "2" ~ "Very likely",
                          SSALIKELY == "3" ~ "Somewhat likely",
                          SSALIKELY == "4" ~ "Not at all likely",
                          TRUE ~ SSALIKELY),
    SSADECISN = case_when(SSADECISN == "1" ~ "The coronavirus pandemic has not affected my decision about applying for benefits",
                          SSADECISN == "2" ~ "I have decided not to apply",
                          SSADECISN == "3" ~ "I applied or decided to apply earlier than expected",
                          SSADECISN == "4" ~ "I applied or decided to apply later than expected",
                          TRUE ~ SSADECISN
                          ),
    EIP = case_when(EIP == "1" ~ "Mostly spend it",
                    EIP == "2" ~ "Mostly save it",
                    EIP == "3" ~ "Mostly use it to pay off debt",
                    EIP == "4" ~ "Not applicable, I did not receive the stimulus payment",
                    TRUE ~ EIP),
    EXPNS_DIF = case_when(EXPNS_DIF == "1" ~ "Not at all difficult",
                          EXPNS_DIF == "2" ~ "A little difficult",
                          EXPNS_DIF == "3" ~ "Somewhat difficult",
                          EXPNS_DIF == "4" ~ "Very difficult",
                          TRUE ~ EXPNS_DIF),
    FEWRTRANS = case_when(FEWRTRANS == "1" ~ "Yes",
                          FEWRTRANS == "2" ~ "No",
                          FEWRTRANS == "3" ~ "Did not use before",
                          TRUE ~ FEWRTRANS),
    CURFOODSUF = case_when(CURFOODSUF == "1" ~ "Enough of the kinds of food (I/we) wanted to eat",
                           CURFOODSUF == "2" ~ "Enough, but not always the kinds of food (I/we) wanted to eat",
                           CURFOODSUF == "3" ~ "Sometimes not enough to eat",
                           CURFOODSUF == "4" ~ "Often not enough to eat",
                           TRUE ~ CURFOODSUF),
    CHILDFOOD = case_when(CHILDFOOD == "1" ~ "Often true",
                          CHILDFOOD == "2" ~ "Sometimes true",
                          CHILDFOOD == "3" ~ "Never true",
                          TRUE ~ CHILDFOOD),
    across(matches("ANXIOUS|WORRY|INTEREST|DOWN"),
           ~case_when(as.character(.) == "1" ~ "Not at all",
                      as.character(.) == "2" ~ "Several days",
                      as.character(.) == "3" ~ "More than half the days",
                      as.character(.) == "4" ~ "Nearly every day",
                      TRUE ~ as.character(.))),
    TENURE = case_when(TENURE == "1" ~ "Owned free and clear",
                       TENURE == "2" ~ "Owned with a mortgage or loan (including home equitly loans)",
                       TENURE == "3" ~ "Rented",
                       TENURE == "4" ~ "Occupied without payment of rent",
                       TRUE ~ TENURE),
    LIVQTR = case_when(LIVQTR == "1" ~ "A mobile home",
                       LIVQTR == "2" ~ "A one-family house detached from any other house",
                       LIVQTR == "3" ~ "A one-family house attached to one or more houses",
                       LIVQTR == "4" ~ "A building with 2 apartments",
                       LIVQTR == "5" ~ "A building with 3 or 4 apartment",
                       LIVQTR == "6" ~ "A building with 5 to 9 apartments",
                       LIVQTR == "7" ~ "A building with 10 to 19 apartments",
                       LIVQTR == "8" ~ "A building with 20 to 49 apartments",
                       LIVQTR == "9" ~ "A building with 50 or more apartments",
                       LIVQTR == "10" ~ "Boat, RV, van, etc.",
                       TRUE ~ LIVQTR),
    MORTCONF = case_when(MORTCONF == "1" ~ "No confidence",
                         MORTCONF == "2" ~ "Slight confidence",
                         MORTCONF == "3" ~ "Moderate confidence",
                         MORTCONF == "4" ~ "High confidence",
                         MORTCONF == "5" ~ "Payment is/will be deferred",
                         TRUE ~ MORTCONF),
    INCOME = case_when(INCOME == "1" ~ "Less than $25,000",
                       INCOME == "2" ~ "$25,000 - $34,999",
                       INCOME == "3" ~ "$35,000 - $49,999",
                       INCOME == "4" ~ "$50,000 - $74,999",
                       INCOME == "5" ~ "$75,000 - $99,999",
                       INCOME == "6" ~ "$100,000 - $149,999",
                       INCOME == "7" ~ "$150,000 - $199,999",
                       INCOME == "8" ~ "$200,000 and above",
                       TRUE ~ INCOME),
    PRIVHLTH = case_when(PRIVHLTH == "1" ~ "Yes, I have a Private Health Insurance",
                         PRIVHLTH == "2" ~ "No, I don't have a Private Health Insurance",
                         TRUE ~ NA),
    PUBHLTH = case_when(PUBHLTH == "1" ~ "Yes, I have a Public Health Insurance",
                        PUBHLTH == "2" ~ "No, I don't have a Public Health Insurance",
                        TRUE ~ NA)
    ) %>%
  
  # Cbind columns for dummy variables to merge
  add_column(
    !!!setNames(
      rep(list(NA_character_), length(dummy_vars_to_merge)),
      dummy_vars_to_merge
    )
  )

# Insert questionnaire item
q_item <- c(
  "Record identifier",
  "Week of interview",
  "State", "Metropolitan statistical area",
  "What year were you born? Please enter a number.",
  "Are you…?",
  "Are you of Hispanic, Latino, or Spanish origin?",
  "What is your race?",
  "What is the highest degree or level of school you have completed?",
  "What is your marital status?",
  "How many total people – adults and children – currently live in your household, including yourself? Please enter a number.",
  "How many people under 18 years-old currently live in your household? Please enter a number.",
  "Have you received a COVID-19 vaccine?",
  "Did you receive (or do you plan to receive) all required doses",
  "Once a vaccine to prevent COVID-19 is available to you, would you…",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Which of the following, if any, are reasons that you [only probably will /probably won’t/definitely won’t] [get a COVID-19 vaccine/won’t receive all required doses of a COVID-19 vaccine]?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "Has a doctor or other health care provider ever told you that you have COVID-19?",
  "Have you, or has anyone in your household experienced a loss of employment income since March 13, 2020?",
  "Do you expect that you or anyone in your household will experience a loss of employment income in the next 4 weeks because of the coronavirus pandemic?",
  "Now we are going to ask about your employment. In the last 7 days, did you do ANY work for either pay or profit?",
  "Are you employed by government, by a private company, a nonprofit organization or were you self-employed or working in a family business?",
  "What is your main reason for not working for pay or profit?",
  "Working from home is sometimes referred to as telework. Did any adults in this household substitute some or all of their typical in-person work for telework because of the coronavirus pandemic, including yourself?",
  "Since March 13, 2020, have you applied for Unemployment Insurance (UI) benefits?",
  "Since March 13, 2020, did you receive Unemployment Insurance (UI) benefits?",
  "Do you currently receive Social Security benefits (Retirement, Disability, or Survivors), Supplemental Security Income (SSI) benefits, or Medicare benefits?",
  "Did you apply or attempt to apply for Social Security benefits (Retirement, Disability, or Survivors), Supplemental Security Income (SSI) benefits, or Medicare benefits after March 13, 2020?",
  "How likely are you to apply for Social Security benefits (Retirement, Disability, or Survivors), Supplemental Security Income (SSI) benefits, or Medicare benefits in the next 12 months?",
  "How has the coronavirus pandemic affected your decision about applying or not applying for Social Security benefits (Retirement, Disability, or Survivors), Supplemental Security Income (SSI) benefits, or Medicare benefits?",
  "In the last 7 days, if you or anyone in your household received a “stimulus payment,” that is a coronavirus related Economic Impact Payment from the Federal Government, did you",
  "In the last 7 days, how difficult has it been for your household to pay for usual household expenses, including but not limited to food, rent or mortgage, car payments, medical expenses, student loans, and so on?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "In the last 7 days, have you taken fewer trips to stores than you normally would have because of the coronavirus pandemic? Curbside pick-up should be counted as trips to stores.",
  "In the last 7 days, have you taken fewer trips than you normally would have by bus, rail, or ride-sharing services, like Uber and Lyft, because of the coronavirus pandemic?",
  "In the last 7 days, which of these statements best describes the food eaten in your household?",
  "Please indicate whether the next statement was often true, sometimes true, or never true in the last 7 days for the children living in your household who are under 18 years old. “The children were not eating enough because we just couldn't afford enough food.”",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "During the last 7 days, did you or anyone in your household get free groceries or a free meal?",
  "Do you or does anyone in your household receive benefits from the Supplemental Nutrition Assistance Program (SNAP) or the Food Stamp Program?",
  "During the last 7 days, how much money did you and your household spend on food at supermarkets, grocery stores, online, and other places you buy food to prepare and eat at home? Please include purchases made with SNAP or food stamps. Enter amount.",
  "During the last 7 days, how much money did you or your household spend on prepared meals, including eating out, fast food, and carry out or delivered meals? Please include money spent in cafeterias at work or at school or on vending machines. Please do not include money you have already told us about in item Q28(above). Enter amount.",
  "Over the last 7 days, how often have you been bothered by the following problems ... Feeling nervous, anxious, or on edge?",
  "Over the last 7 days, how often have you been bothered by the following problems ... Not being able to stop or control worrying?",
  "Over the last 7 days, how often have you been bothered by ... having little interest or pleasure in doing things?",
  "Over the last 7 days, how often have you been bothered by ... feeling down, depressed, or hopeless?",
  "Recode of Q36 Health Insurance Variables (Private)",
  "Recode of Q36 Health Insurance Variables (Public)",
  "At any time in the last 4 weeks, did you DELAY getting medical care because of the coronavirus pandemic?",
  "At any time in the last 4 weeks, did you need medical care for something other than coronavirus, but DID NOT GET IT because of the coronavirus pandemic?",
  "At any time in the last 4 weeks, did you take prescription medication to help you with any emotions or with your concentration, behavior or mental health?",
  "At any time in the last 4 weeks, did you receive counseling or therapy from a mental health professional such as a psychiatrist, psychologist, psychiatric nurse, or clinical social worker?",
  "At any time in the last 4 weeks, did you need counseling or therapy from a mental health professional, but DID NOT GET IT for any reason?",
  "Is your house or apartment…?",
  "Which best describes this building? Include all apartments, flats, etc., even if vacant.",
  "Is this household currently caught up on rent payments?",
  "Is this household currently caught up on mortgage payments?",
  "How confident are you that your household will be able to pay your next rent or mortgage payment on time?",
  "At any time during the 2020-2021 school year, were, or will, any children in this household enrolled in a public school, enrolled in a private school, or educated in a homeschool setting in Kindergarten through 12th grade or grade equivalent?",
  "At any time during the 2020-2021 school year, were, or will, any children in this household enrolled in a public school, enrolled in a private school, or educated in a homeschool setting in Kindergarten through 12th grade or grade equivalent?",
  "At any time during the 2020-2021 school year, were, or will, any children in this household enrolled in a public school, enrolled in a private school, or educated in a homeschool setting in Kindergarten through 12th grade or grade equivalent?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?",
  "In 2019 what was your total household income before taxes?",
  "Which of the following, if any, are reasons for your previous response?",
  "Why do you believe that you don’t need a COVID-19 vaccine?",
  "In the last 7 days, which of the following changes have you or your household made to your spending or shopping?",
  "In the last 7 days, for which of the following reasons have you or your household changed spending?",
  "Thinking about your experience in the last 7 days, which of the following did you or your household members use to meet your spending needs?",
  "Why did you not have enough to eat (or not what you wanted to eat)?",
  "At any time during the 2020-2021 school year, were, or will, any children in this household enrolled in a public school, enrolled in a private school, or educated in a homeschool setting in Kindergarten through 12th grade or grade equivalent?",
  "How has the coronavirus pandemic affected how the children in this household received education for the 2020 – 2021 school year?"
  ) %>%
  
  # Transform into a data frame
  t() %>%
  as.data.frame(stringsAsFactors = FALSE)

# set column names
colnames(q_item) <- colnames(df)

# Add response levels
q_df <- q_item %>%
  pivot_longer(
    cols = c(
      all_of(setdiff(vars_selected, dummy_vars_to_merge)),
      starts_with(dummy_vars_to_merge)
    ),
    values_to = "question"
    ) %>%
  
  # Adapt questions
  mutate(question = case_when(
    name == "EGENDER" ~ "What is your gender?",
    name == "GETVACC" ~ "Once a vaccine to prevent COVID-19 is available to you, would you get vaccinated?",
    
    # Dummy set to decode and merge
    name == "WHYNOT1" ~ "Concerned about possible side effects of a COVID-19 vaccine",
    name == "WHYNOT2" ~ "Do not know if a COVID-19 vaccine will work",
    name == "WHYNOT3" ~ "Do not believe you need a COVID-19 vaccine",
    name == "WHYNOT4" ~ "Do not like vaccines",
    name == "WHYNOT5" ~ "My doctor has not recommended it",
    name == "WHYNOT6" ~ "Plan to wait and see if it is safe and may get it later",
    name == "WHYNOT7" ~ "Think other people need it more than I do right now",
    name == "WHYNOT8" ~ "Concerned about the cost of a COVID-19 vaccine",
    name == "WHYNOT9" ~ "Do not trust COVID-19 vaccines",
    name == "WHYNOT10" ~ "Do not trust the government",
    name == "WHYNOT11" ~ "Other reasons",
    
    # Dummy set to decode and merge
    name == "WHYNOTB1" ~ "Already had COVID-19", # paste(question, "Is it because ?"),
    name == "WHYNOTB2" ~ "Not a member of a high-risk group",
    name == "WHYNOTB3" ~ "Plan to use masks or other precautions instead",
    name == "WHYNOTB4" ~ "Do not believe COVID-19 is a serious illness",
    name == "WHYNOTB5" ~ "Do not think vaccines are beneficial",
    name == "WHYNOTB6" ~ "Other reasons",
    
    name == "EIP" ~ "In the last 7 days, if you or anyone in your household received a “stimulus payment,” that is a coronavirus related Economic Impact Payment from the Federal Government, how did you use it?",
    
    # Dummy set to decode and merge
    name == "CHNGHOW1" ~ "Made more purchases online (as opposed to in stores)",
    name == "CHNGHOW2" ~ "Made more purchases by curbside pick-up (as opposed to in store)",
    name == "CHNGHOW3" ~ "Made more purchases in-store (as opposed to purchases online or curbside pickup)",
    name == "CHNGHOW4" ~ "Increased use of credit cards or smartphone apps for purchases, instead of using cash",
    name == "CHNGHOW5" ~ "Increased use of cash instead of using credit cards or smartphone apps for purchases",
    name == "CHNGHOW6" ~ "Avoided eating at restaurants",
    name == "CHNGHOW7" ~ "Resumed eating at restaurants",
    name == "CHNGHOW8" ~ "Canceled or postponed in-person medical or dental appointments",
    name == "CHNGHOW9" ~ "Attended in-person medical or dental appointments",
    name == "CHNGHOW10" ~ "Canceled or postponed housekeeping or caregiving services",
    name == "CHNGHOW11" ~ "Resumed or started new housekeeping or caregiving services",
    name == "CHNGHOW12" ~ "Did not make any changes to spending or shopping behavior",
    
    # Dummy set to decode and merge
    name == "WHYCHNGD1" ~ "Usual shopping places were closed or had limited hours (e.g., restaurant, doctor/dentist office, health club, hair salon, child care center, etc.)",
    name == "WHYCHNGD2" ~ "Usual shopping places re-opened or increased hours",
    name == "WHYCHNGD3" ~ "Concerned about going to public or crowded places or having contact with high-risk people",
    name == "WHYCHNGD4" ~ "No longer concerned about going to public or crowded places or having contact with high-risk people",
    name == "WHYCHNGD5" ~ "Loss of income",
    name == "WHYCHNGD6" ~ "Increased income",
    name == "WHYCHNGD7" ~ "Concerned about bening laid off or having hours reduced",
    name == "WHYCHNGD8" ~ "No longer concerned about being laid off or having hours reduced",
    name == "WHYCHNGD9" ~ "Working from home/teleworking",
    name == "WHYCHNGD10" ~ "Resumed working onsite at workplace",
    name == "WHYCHNGD11" ~ "Concerns about the economy",
    name == "WHYCHNGD12" ~ "No longer concerned about the economy",
    name == "WHYCHNGD13" ~ "Other reasons",
    
    # Dummy set to decode and merge
    name == "SPNDSRC1" ~ "Regular income sources like those received before the pandemic",
    name == "SPNDSRC2" ~ "Credit cards or loans ",
    name == "SPNDSRC3" ~ "Money from savings or selling assets",
    name == "SPNDSRC4" ~ "Borrowing from friends or family",
    name == "SPNDSRC5" ~ "Unemployment insurance (UI) benefit payments",
    name == "SPNDSRC6" ~ "Stimulus (economic impact) payment",
    name == "SPNDSRC7" ~ "Money saved from deferred or forgiven payments (to meet your spending needs)",
    name == "SPNDSRC8" ~ "Supplemental Nutrition Assistance Program (SNAP)",
    
    # Dummy set to decode and merge
    name == "FOODSUFRSN1" ~ "Couldn't afford to buy more food",
    name == "FOODSUFRSN2" ~ "Couldn’t get out to buy food (for example, didn’t have transportation, or had mobility or health problems that prevented you from getting out)",
    name == "FOODSUFRSN3" ~ "Afraid to go or didn’t want to go out to buy food",
    name == "FOODSUFRSN4" ~ "Couldn’t get groceries or meals delivered to me",
    name == "FOODSUFRSN5" ~ "The stores didn’t have the food I wanted",
    
    name == "TSPNDPRPD" ~ "During the last 7 days, how much money did you or your household spend on prepared meals, including eating out, fast food, and carry out or delivered meals? Please include money spent in cafeterias at work or at school or on vending machines. Please do not include money you have already told us about in the previous question. Enter amount.",
    name == "PRIVHLTH" ~ "Do you have a private health insurance?",
    name == "PUBHLTH" ~ "Do you have a public health insurance?",
    name == "TENURE" ~ "Is your house or apartment owned or rented?",
    
    # Dummy set to decode and merge
    name == "ENROLL1" ~ "Enrolled in a public or private school",
    name == "ENROLL2" ~ "Homeschooled",
    name == "ENROLL3" ~ "No",
    
    # Dummy set to decode and merge
    name == "TEACH1" ~ "Classes normally taught in person at the school were canceled",
    name == "TEACH2" ~ "Classes normally taught in person moved to a distance-learning format using online resources, either self-paced or in real time",
    name == "TEACH3" ~ "Classes normally taught in person moved to a distance-learning format using paper materials sent home to children",
    name == "TEACH4" ~ "Classes normally taught in person changed in some other way",
    name == "TEACH5" ~ "The coronavirus pandemic did not affect how children in this household receive education",
    
    .default = question
  ))
  
q_df <- q_df %>%
  mutate(
    question = case_when(
      name == "WHYNOT" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "WHYNOT") & name != "WHYNOT" & !str_starts(name, "WHYNOTB")) %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "WHYNOTB" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "WHYNOTB") & name != "WHYNOTB") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "CHNGHOW" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "CHNGHOW") & name != "CHNGHOW") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "WHYCHNGD" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "WHYCHNGD") & name != "WHYCHNGD") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "SPNDSRC" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "SPNDSRC") & name != "SPNDSRC") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "FOODSUFRSN" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "FOODSUFRSN") & name != "FOODSUFRSN") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "ENROLL" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "ENROLL") & name != "ENROLL") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      name == "TEACH" ~ {
        why_not_options <- q_df %>%
          filter(str_starts(name, "TEACH") & name != "TEACH") %>%
          pull(question)
        
        paste(
          question,
          "Answer one of the following options:",
          paste(c(why_not_options, "Not applicable"), collapse = "; ")
        )
      },
      TRUE ~ question
    )
  ) %>%
  
  # Add response levels
  mutate(response_levels = case_when(
    name %in% c("SCRAM", "WEEK", "EST_ST", "EST_MSA", "TBIRTH_YEAR") ~ NA,
    name == "EGENDER" ~ paste(c(levels(as.factor(df$EGENDER))), collapse = "; "),
    name == "RHISPANIC" ~ paste(c(levels(as.factor(df$RHISPANIC))), collapse = "; "),
    name == "RRACE" ~ paste(c(levels(as.factor(df$RRACE))), collapse = "; "),
    name == "EEDUC" ~ "Less than high school; Some high school; High school graduate or equivalent (e.g. GED); Some college, but degree not received or is in progress; Associate's degree (e.g. AA, AS); Bachelor's degree (e.g. BA, BS, AB); Graduate degree (e.g. master's, professional, doctorate)",
    name == "MS" ~ "Now married; Widowed; Divorced; Separated; Never married",
    name %in% c("THHLD_NUMPER", "THHLD_NUMKID") ~ NA,
    name %in% c("RECVDVACC", "EXPCTLOSS", "ANYWORK", "UI_APPLY", "UI_RECV",
                "SSA_RECV", "SSA_APPLY", "FEWRTRIPS", "FEWRTRANS", "FREEFOOD",
                "SNAP_YN", "DELAY", "NOTGET", "PRESCRIPT", "MH_SVCS",
                "MH_NOTGET") ~ "Yes; No",
    name == "DOSES" ~ "Yes; No. Answer Not applicable if you have not received a COVID-19 vaccination",
    name == "GETVACC" ~ "Definitely get a vaccine; Probably get a vaccine; Probably NOT get a vaccine; Definitely NOT get a vaccine. Answer Not applicable if you have already received a COVID-19 vaccination",
    grepl("WHYNOTB", name) ~ NA,
    grepl("WHYNOT", name) ~ NA,
    name == "HADCOVID" ~ "Yes; No; Not sure",
    name == "WRKLOSS" ~ "Yes; No. Answer Not applicable if you were born in 2002",
    name == "KINDWORK" ~ "Government; Private company; Non-profit organization including tax exempt and charitable organizations; Self-employed; Working in a family business. Answer Not applicable if you did not do any work in the last 7 days",
    name == "RSNNOWRK" ~ "I did not want to be employed at this time; I am/was sick with coronavirus symptoms; I am/was caring for someone with coronavirus symptoms; I am/was caring for children not in school or daycare; I am/was caring for an elderly person; I am/was sick (not coronavirus related) or disabled; I am retired; My employer experienced a reduction in business (including furlough) due to coronavirus pandemic; I am/was laid off due to coronavirus pandemic; My employer closed temporarily due to the coronavirus pandemic; My employer went out of business due to the coronavirus pandemic; Other reason, please specify; I was concerned about getting or spreading the coronavirus. Answer Not applicable if you did any work in the last 7 days",
    name == "TW_START" ~ "Yes, at least one adult substituted some or all of their typical in-person work for telework; No, no adults substituted their typical in-person work for telework; No, there has been no change in telework",
    name == "SSALIKELY" ~ "Extremely likely; Very likely; Somewhat likely; Not at all likely. Answer Not applicable if you have already applied or attempted to apply for the benefits",
    name == "SSADECISN" ~ "The coronavirus pandemic has not affected my decision about applying for benefits; I have decided not to apply; I applied or decided to apply earlier than expected; I applied or decided to apply later than expected. Answer Not applicable if you have already applied or attempted to apply for the benefits",
    name == "EIP" ~ "Mostly spend it; Mostly save it; Mostly use it to pay off debt; Not applicable, I did not receive the stimulus payment",
    name == "EXPNS_DIF" ~ "Not at all difficult; A little difficult; Somewhat difficult; Very difficult",
    grepl("CHNGHOW", name) ~ NA,
    grepl("WHYCHNGD", name) ~ NA,
    grepl("SPNDSRC", name) ~ NA,
    name == "CURFOODSUF" ~ "Enough of the kinds of food (I/we) wanted to eat; Enough, but not always the kinds of food (I/we) wanted to eat; Sometimes not enough to eat; Often not enough to eat",
    name == "CHILDFOOD" ~ "Often true; Sometimes true; Never true. Answer Not applicable if your household has had sufficient food or you don't have any children under 18 in your household",
    grepl("FOODSUFRSN", name) ~ "Yes if applicable. Answer Not applicable if your household has had sufficient food",
    name %in% c("TSPNDFOOD", "TSPNDPRPD") ~ NA,
    name %in% c("ANXIOUS", "WORRY", "INTEREST", "DOWN") ~ "Not at all; Several days; More than half the days; Nearly every day",
    name == "TENURE" ~ "Owned free and clear; Owned with a mortgage or loan (including home equitly loans); Rented; Occupied without payment of rent",
    name == "LIVQTR" ~ "A mobile home; A one-family house detached from any other house; A one-family house attached to one or more houses; A building with 2 apartments; A building with 3 or 4 apartments; A building with 5 to 9 apartments; A building with 10 to 19 apartments; A building with 20 to 49 apartments; A building with 50 or more apartments; Boat, RV, van, etc.",
    name == "RENTCUR" ~ "Yes; No. Answer Not applicable if you are not paying rent",
    name == "MORTCUR" ~ "Yes; No. Answer Not applicable if you do not have a mortgage or loan",
    name == "MORTCONF" ~ "No confidence; Slight confidence; Moderate confidence; High confidence; Payment is/will be deferred. Answer Not applicable if you do not have a rent or mortgage",
    name %in% c("ENROLL1", "ENROLL2", "TEACH1", "TEACH2", "TEACH3", "TEACH4") ~ NA,
    name %in% c("ENROLL3", "TEACH5") ~ NA,
    name == "INCOME" ~ "Less than $25,000; $25,000 - $34,999; $35,000 - $49,999; $50,000 - $74,999; $75,000 - $99,999; $100,000 - $149,999; $150,000 - $199,999; $200,000 and above",
    name == "PRIVHLTH" ~ "Yes, I have a Private Health Insurance; No, I don't have a Private Health Insurance",
    name == "PUBHLTH" ~ "Yes, I have a Public Health Insurance; No, I don't have a Public Health Insurance"
    )) %>%
  
  mutate(question_prompt = ifelse(is.na(response_levels), question, paste(question, "Answer one of the following options:", response_levels))) %>%
  
  select(name, question_prompt) %>%
  
  pivot_wider(names_from = name,
              values_from = question_prompt)


# Combine survey items with data
df <- rbind(q_df, df)
  
# Merge dummy variables and preserve only selected variables in desired order
df_final <- df %>%
  
  # Merge dummy variables
  reduce(
    .x = dummy_vars_to_merge,
    .f = ~ merge_dummies(.x, .y),
    .init = df
  ) %>%
  
  # Preserve only selected variables in desired order
  select(!!vars_selected)

# Standardize: rename SCRAM -> subject_id (the record identifier).
names(df_final)[names(df_final) == "SCRAM"] <- "subject_id"
df_final$subject_id[1] <- "subject_id"
df_final <- df_final[, c("subject_id", setdiff(names(df_final), "subject_id")), drop = FALSE]

write_clean_csv(df_final,
                file.path("data", "processed", "surveys", "hps_2021", "hps_2021_data.csv"))
