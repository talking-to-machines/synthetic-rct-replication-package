# Afrobarometer Round 9 — currently Sierra Leone only.
#
# Reads:
#   data/human/surveys/afrobarometer/<R9 merged .sav> (file not yet in repo)
# Writes:
#   data/processed/surveys/afrobarometer/afrobarometer_sl_data.csv
#
# Multi-country planned (Ghana to be added later).

# load packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(foreign)
})

if (!exists("read_mapping_csv")) source("preprocessing/utils.R")

afro_dir <- file.path("data", "human", "surveys", "afrobarometer")
sav_files <- list.files(afro_dir, pattern = "\\.sav$", full.names = TRUE)
if (length(sav_files) == 0) {
  stop("No .sav file in ", afro_dir,
       " — drop the Afrobarometer R9 merged release into that folder.")
}
raw_data <- read.spss(sav_files[1], to.data.frame = TRUE)

# create df
df <- raw_data %>%
  # subset to Sierra Leone data
  subset(COUNTRY == "Sierra Leone") %>%
  # select variables
  dplyr::select(RESPNO, COUNTRY, URBRUR_COND, REGION, EA_SVC_A,
                EA_SVC_B, EA_SVC_C, EA_SVC_D, EA_FAC_B, EA_FAC_D,
                EA_FAC_F2, EA_FAC_G, DATEINTR, Q1, Q2,
                Q2OTHER, Q3, Q4A, Q4B, Q5A,
                Q5B, Q6A, Q6B, Q6C, Q6D,
                Q6E, Q8, Q9A, Q9B, Q9C,
                Q13, Q18, Q19A, Q33D, Q37A,
                Q37B, Q37D, Q37E, Q37F, Q37J,
                Q37K, Q41A, Q41B, Q41C, Q41D,
                Q41E, Q41F, Q41G, Q41H, Q45PT1,
                Q56C, Q57A, Q57B, Q58A, Q58B,
                Q58C, Q59, Q60, Q61A, Q61B,
                Q61C, Q62, Q63A, Q63B, Q63C,
                Q64A, Q64B, Q65A, Q65B, Q74A,
                Q74B, Q74C, Q74D, Q74E, Q75,
                Q84A, Q84AOTHER, Q85A, Q86A, Q86B,
                Q86C, Q86D, Q89A, Q89B, Q89BOTHER,
                Q90F, Q90G, Q90H, Q90I, Q93B,
                Q93BOTHER, Q94, Q95, Q95OTHER, Q100,
                Q101, Q101OTHER) %>%
  # fix date variable
  mutate(DATEINTR = as.POSIXct(DATEINTR, origin = "1582-10-14", tz = "UTC")) %>%
  # de-factor all variables
  lapply(as.character,
         stringAsFactors = FALSE) %>%
  as.data.frame() %>%
  # remove curly marks
  mutate(across(everything(), ~str_replace_all(., c("’" = "'")))) %>%
  # merge identical questions (red variables)
  mutate(Q2OTHER = trimws(Q2OTHER),
         Q84AOTHER = trimws(Q84AOTHER),
         Q89BOTHER = trimws(Q89BOTHER),
         Q93BOTHER = trimws(Q93BOTHER),
         Q95OTHER = trimws(Q95OTHER),
         Q101OTHER = trimws(Q101OTHER)) %>%
  mutate(Q2 = ifelse(Q2OTHER == "", Q2, paste0(substr(Q2OTHER, 1, 1), tolower(substr(Q2OTHER, 2, nchar(Q2OTHER))))),
         Q84A = ifelse(Q84AOTHER == "", Q84A, paste0(substr(Q84AOTHER, 1, 1), tolower(substr(Q84AOTHER, 2, nchar(Q84AOTHER))))),
         Q89B = ifelse(Q89BOTHER == "", Q89B, paste0(Q89BOTHER)),
         Q93B = ifelse(Q93BOTHER == "", Q93B, paste0(substr(Q93BOTHER, 1, 1), tolower(substr(Q93BOTHER, 2, nchar(Q93BOTHER))))),
         Q95 = ifelse(Q95OTHER == "", Q95, paste(Q95, Q95OTHER, sep = ", ")),
         Q101 = ifelse(Q101OTHER == "", Q101, paste(Q101, Q101OTHER, sep = ", "))
         ) %>%
  dplyr::select(-Q2OTHER, -Q84AOTHER, -Q89BOTHER, -Q93BOTHER, -Q95OTHER, -Q101OTHER) %>%
  # remove "duplicated_"
  mutate_all(~ gsub("_duplicated.*", "", .))

# insert questionnaire item
q_item <- c("Respondent Number", "Country", "PSU/EA", "Region/Province/State",
            "Are the following services present in the primary sampling unit/enumeration area: Electricity grid that most houses can access?",
            "Are the following services present in the primary sampling unit/enumeration area: Piped water system that most houses can access?",
            "Are the following services present in the primary sampling unit/enumeration area: Sewage system that most houses can access?",
            "Are the following services present in the primary sampling unit/enumeration area: Mobile phone service?",
            "Are the following facilities present in the primary sampling unit/enumeration area or in easy walking distance: School (private or public or both)?",
            "Are the following facilities present in the primary sampling unit/enumeration area or in easy walking distance: Health clinic (private or public or both)?",
            "Are the following facilities present in the primary sampling unit/enumeration area or in easy walking distance: A social center, government help center, or other government office where people can request help with problems?",
            "Are the following facilities present in the primary sampling unit/enumeration area or in easy walking distance: Is there any kind of paid transport, such as a bus, taxi, moped, or other form, available on a daily basis?",
            "Date of interview",
            "How old are you?",
            "What is the primary language you speak in your home?",
            "Let's start with your general view about the current direction of our country. Some people might think the country is going in the wrong direction. Others may feel it is going in the right direction. So let me ask YOU about the overall direction of the country: Would you say that the country is going in the wrong direction or going in the right direction?",
            "In general, how would you describe: the present economic condition of this country?",
            "In general, how would you describe: Your own present living conditions?",
            "Looking back, how do you rate economic conditions in this country compared to 12 months ago?",
            "Looking ahead, do you expect economic conditions in this country to be better or worse in 12 months' time?",
            "Over the past year, how often, if ever, have you or anyone in your family gone without: Enough food to eat?",
            "Over the past year, how often, if ever, have you or anyone in your family gone without: Enough clean water for home use?",
            "Over the past year, how often, if ever, have you or anyone in your family gone without: Medicines or medical treatment?",
            "Over the past year, how often, if ever, have you or anyone in your family gone without: Enough fuel to cook your food?",
            "Over the past year, how often, if ever, have you or anyone in your family gone without: A cash income?",
            "When you get together with your friends or family, how often would you say you discuss political matters?",
            "In this country, how free are you: to say what you think?",
            "In this country, how free are you: to join any political organization you want?",
            "In this country, how free are you: to choose who to vote for without feeling pressured?",
            "Let's talk about the last national election held in [YEAR]. People are not always able to vote in elections, for example, because they weren't registered, they were unable to go, or someone prevented them from voting. How about you? In the last national election held in [YEAR], did you vote, or not, or were you too young to vote? Or can’t you remember whether you voted?",
            "Which of the following statements is closest to your view? Choose Statement 1 or Statement 2. Statement 1: It is more important to have a government that can get things done, even if we have no influence over what it does. Statement 2: It is more important for citizens to be able to hold government accountable, even if that means it makes decisions more slowly.",
            "Which of the following statements is closest to your view? Choose Statement 1 or Statement 2. Statement 1: The government is like the people's boss. People should respect the government and do what it directs. Statement 2: The government is like the people's employee. It should respect the citizens and do what they request.",
            "In your opinion, how often, in this country: do people have to be careful of what they say about politics?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: the [president]?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: [Parliament]?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: your [local government council]?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: the ruling party?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: opposition political parties?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: traditional leaders?",
            "How much do you trust each of the following, or haven't you heard enough about them to say: religious leaders?",
            "In the past 12 months have you had contact with a public clinic or hospital?",
            "How easy or difficult was it to obtain the medical care or services you needed?",
            "How often, if ever, did you have to pay a bribe, give a gift, or do a favour for a health worker or clinic or hospital staff in order to get the medical care or services you needed?",
            "In general, when dealing with health workers and clinic or hospital staff, how much do you feel that they treat you with respect?",
            "And have you encountered any of these problems with a public clinic or hospital during the past 12 months: lack of medicines or other supplies?",
            "And have you encountered any of these problems with a public clinic or hospital during the past 12 months: absence of doctors or other medical personnel?",
            "And have you encountered any of these problems with a public clinic or hospital during the past 12 months: long waiting time?",
            "And have you encountered any of these problems with a public clinic or hospital during the past 12 months: poor condition of facilities?",
            "In your opinion, what are the most important problems facing this country that government should address?",
            "For each of the following statements, please tell me whether you agree or disagree: in my community, children and adults who have mental or emotional problems are generally able to get the help they need to have a good life.",
            "Please tell me whether you personally or any other or any other member of your household have been affected in any of the following ways by the COVID-19 pandemic: became ill with, or tested positive for, COVID-19?",
            "Please tell me whether you personally or any other or any other member of your household have been affected in any of the following ways by the COVID-19 pandemic: temporarily or permanently lost a job, business, or primary source of income?",
            "Have you received a vaccination against COVID-19, either one or two doses?",
            "If a vaccine for COVID-19 is available, how likely are you to try to get vaccinated?",
            "What is the main reason that you would be unlikely to get a COVID-19 vaccine?",
            "How much do you trust the government to ensure that any vaccine for COVID-19 that is developed or offered to [Kenyan] citizens is safe before it is used in this country?",
            "How well or badly would you say the current government has managed the response to the COVID-19 pandemic?",
            "How satisfied or dissatisfied are you with the government's response to COVID-19 in the following areas: providing relief to vulnerable households?",
            "How satisfied or dissatisfied are you with the government's response to COVID-19 in the following areas: ensuring that disruptions to children's education are kept to a minimum?",
            "How satisfied or dissatisfied are you with the government's response to COVID-19 in the following areas: making sure that health facilities have adequate resources to respond to the COVID-19 pandemic?",
            "Considering all of the funds and resources that were available to the government for combating and responding to the COVID-19 pandemic, how much do you think was lost or stolen due to corruption?",
            "When the country is facing a public health emergency like the COVID-19 pandemic, do you agree or disagree that it is justified for the government to temporarily limit democracy or democratic freedoms by taking the following measures: censoring media reporting?",
            "When the country is facing a public health emergency like the COVID-19 pandemic, do you agree or disagree that it is justified for the government to temporarily limit democracy or democratic freedoms by taking the following measures: using the police and security forces to enforce public health mandates like restrictions on public gatherings or wearing face masks?",
            "When the country is facing a public health emergency like the COVID-19 pandemic, do you agree or disagree that it is justified for the government to temporarily limit democracy or democratic freedoms by taking the following measures: postponing elections?",
            "After experiencing the COVID-19 pandemic in [Kenya], how prepared or unprepared do you think the government will be to deal with future public health emergencies?",
            "Do you agree or disagree wit the following statement: our government needs to invest more of our health resources in special preparations to respond to health emergencies like COVID-19, even if it means fewer resources are available for other health services?",
            "Since the start of the COVID-19 pandemic, have you or your household received any assistance from government, like food, cash payments, relief from bill payments, or other assistance that you were not normally receiving before the pandemic?",
            "Do you think that the distribution of government support to people during the COVID-19 pandemic, for example through food packages or cash payments, has been fair or unfair?",
            "Now let us talk about the media and how you get information about politics and other issues. How often do you get news from the following sources: radio?",
            "How often do you get news from the following sources: television?",
            "How often do you get news from the following sources: print newspapers?",
            "How often do you get news from the following sources: internet?",
            "How often do you get news from the following sources: social media such as Facebook, Twitter, WhatsApp, or others?",
            "Do you agree or disagree with the following statement: information held by public authorities is only for use by government officials; it should not have to be shared with the public.",
            "Let's go back to talking about you. What is your ethnic community or cultural group?",
            "Please tell me whether you agree or disagree with the following statement: I feel strong ties with other [Kenyans].",
            "How much do you trust each of the following types of people: other [Kenyans]?",
            "How much do you trust each of the following types of people: your relatives?",
            "How much do you trust each of the following types of people: your neighbours?",
            "How much do you trust each of the following types of people: other people you know?",
            "Do you feel close to any particular political party?",
            "Which party is that?",
            "Which of these things do you personally own? [If no, ask:] Does anyone else in your household own one: mobile phone?",
            "Does your phone have access to the internet?",
            "How often do you use: a mobile phone?",
            "How often do you use: the Internet?",
            "What is your main occupation? [If unemployed, retired, or disabled, ask:] What was your last main occupation?",
            "What is your highest level of education?",
            "What is your religion, if any?",
            "Respondent's gender",
            "Respondent's race") %>%
  # replace square-bracketed words
  gsub(pattern = "\\[YEAR\\]", replacement = "2018", x = .) %>%
  gsub(pattern = "\\[Kenya\\]", replacement = "Sierra Leone", x = .) %>%
  gsub(pattern = "\\[Kenyan\\]", replacement = "Sierra Leonean", x = .) %>%
  gsub(pattern = "\\[Kenyans\\]", replacement = "Sierra Leoneans", x = .) %>%
  # transform into a dataframe
  t() %>%
  as.data.frame(stringsAsFactors = FALSE)
# set column names
colnames(q_item) <- colnames(df)

# add response levels
q_item <- q_item %>%
  pivot_longer(cols = c(RESPNO, COUNTRY, URBRUR_COND, REGION, EA_SVC_A,
                        EA_SVC_B, EA_SVC_C, EA_SVC_D, EA_FAC_B, EA_FAC_D,
                        EA_FAC_F2, EA_FAC_G, DATEINTR, Q1, Q2,
                        Q3, Q4A, Q4B, Q5A, Q5B,
                        Q6A, Q6B, Q6C, Q6D, Q6E,
                        Q8, Q9A, Q9B, Q9C, Q13,
                        Q18, Q19A, Q33D, Q37A, Q37B,
                        Q37D, Q37E, Q37F, Q37J, Q37K,
                        Q41A, Q41B, Q41C, Q41D, Q41E,
                        Q41F, Q41G, Q41H, Q45PT1, Q56C,
                        Q57A, Q57B, Q58A, Q58B, Q58C,
                        Q59, Q60, Q61A, Q61B, Q61C,
                        Q62, Q63A, Q63B, Q63C, Q64A,
                        Q64B, Q65A, Q65B, Q74A, Q74B,
                        Q74C, Q74D, Q74E, Q75, Q84A, 
                        Q85A, Q86A, Q86B, Q86C, Q86D,
                        Q89A, Q89B, Q90F, Q90G, Q90H,
                        Q90I, Q93B, Q94, Q95, Q100,
                        Q101),
               values_to = "question") %>%
  # adapt Q90F
  mutate(question = ifelse(name == "Q90F", "Do you personally own a mobile phone or does anyone in your household own one?", question)) %>%
  # add response levels
  mutate(response_levels = case_when(
    name %in% c("RESPNO", "COUNTRY", "URBRUR_COND", "REGION", "DATEINTR") ~ NA,
    name %in% c("EA_SVC_A", "EA_SVC_B", "EA_SVC_C", "EA_SVC_D", "EA_FAC_B", "EA_FAC_D", "EA_FAC_F2", "EA_FAC_G") ~ "No; Yes; Can't determine",
    name == "Q1" ~ "an integer above 17; Refused; Don't know",
    name == "Q2" ~ paste(c(levels(as.factor(df$Q2)), "Refused", "Don't know"), collapse = "; "),
    name == "Q3" ~ "Going in the wrong direction; Going in the right direction; Refused; Don't know",
    name %in% c("Q4A", "Q4B") ~ "Very bad; Fairly bad; Neither good nor bad; Fairly good; Very good; Refused; Don't know",
    name %in% c("Q5A", "Q5B") ~ "Much worse; Worse; Same; Better; Much better; Refused; Don't know",
    name %in% c("Q6A", "Q6B", "Q6C", "Q6D", "Q6E") ~ "Never; Just once or twice; Several times; Many times; Always; Refused; Don't
know",
    name == "Q8" ~ "Never; Occasionally; Frequently; Refused; Don't know",
    name %in% c("Q9A", "Q9B", "Q9C") ~ "Not at all free; Not very free; Somewhat free; Completely free; Refused; Don't
know",
    name == "Q13" ~ "I did not vote; I was too young to vote; I can't remember whether I voted; I voted in the election; Refused; Don't know",
    name == "Q18" ~ "Strongly agree with Statement 1 (It is more important to have a government that can get things done, even if we have no influence over what it does);
    Agree with Statement 1 (It is more important to have a government that can get things done, even if we have no influence over what it does);
    Agree with Statement 2 (It is more important for citizens to be able to hold government accountable, even if that means it makes decisions more slowly);
    Strongly agree with Statement 2 (It is more important for citizens to be able to hold government accountable, even if that means it makes decisions more slowly);
    Agree with neither; Refused; Don't know",
    name == "Q19A" ~ "Strongly agree with Statement 1 (The government is like the people's boss. People should respect the government and do what it directs);
    Agree with Statement 1 (The government is like the people's boss. People should respect the government and do what it directs);
    Agree with Statement 2 (The government is like the people's employee. It should respect the citizens and do what they request);
    Strongly agree with Statement 2 (The government is like the people's employee. It should respect the citizens and do what they request);
    Agree with neither; Refused; Don't know",
    name == "Q33D" ~ "Never; Rarely; Often; Always; Refused; Don't know",
    name %in% c("Q37A", "Q37B", "Q37D", "Q37E", "Q37F", "Q37J", "Q37K") ~ "Not at all; Just a little; Somewhat; A lot; Refused; Don't know/Haven't heard enough",
    name %in% c("Q41A", "Q58A", "Q65A") ~ "No; Yes; Refused; Don't know",
    name == "Q41B" ~ "Very easy; Easy; Difficult; Very difficult; Refused; Don't know. Answer No contact if you haven't had any contact with a public clinic or hospital in the past 12 months",
    name %in% c("Q41C", "Q41E", "Q41F", "Q41G", "Q41H") ~ "Never; Once or twice; A few times; Often; Refused; Don't know. Answer No contact if you haven't had any contact with a public clinic or hospital in the past 12 months",
    name == "Q41D" ~ "Not at all; A little bit; Somewhat; A lot; Refused; Don't know. Answer No contact if you haven't had any contact with a public clinic or hospital in the past 12 months",
    name == "Q41PT1" ~ "Nothing/no problems; Management of the economy; Wages, incomes, and salaries; Unemployment; Poverty/Destitution; Rates and taxes; Loans/Credit; Farming/Agriculture;
    Food shortage/Famine; Drought; Land; Transportation; Communications; Infrastructure/Roads; Education;
    Housing; Electricity; Water supply; Orphans/Street children/Homeless children; Services (other); Health; AIDS;
    Sickness/Disease; Crime and security; Corruption; Political violence; Political instability/Political divisions/Ethnic tensions; Discrimination/Inequality; Gender issues/Women's rights;
    Democracy/Political rights; War (international); Civil war; Agricultural marketing; Climate change; COVID-19; Internally displaced;
    Pollution; Drug abuse; Other; Refused; Don't know",
    name %in% c("Q56C", "Q63A", "Q63B", "Q63C", "Q64B", "Q75", "Q85A") ~ "Strongly disagree; Disagree; Neither agree nor disagree; Agree; Strongly agree; Refused; Don't know",
    name %in% c("Q57A", "Q57B") ~ "Yes; No; Refused; Don't know",
    name == "Q58B" ~ "Very unlikely; Somewhat unlikely; Somewhat likely; Very likely; Refused; Don't know. Answer Not applicable if you have never received a COVID-19 vaccination",
    name == "Q58C" ~ "COVID doesn't exist/COVID is not real; Not worried about COVID/COVID is not serious or life-threatening/not deadly; I am at no risk or low risk for getting COVID/Small chance of contracting COVID;
    I already had COVID and believe I am immune; God will protect me; Don't trust the vaccine/worried about getting fake or counterfeit vaccine;
    Don't trust the government to ensure the vaccine is safe; Vaccine is not safe; Vaccine was developed too quickly;
    Vaccine is not effective/Vaccinated people can still get COVID; Vaccine may cause COVID; Vaccine may cause infertility;
    Vaccine may cause other bad side effects; Vaccines are being used to control or track people; People are being experimented on with vaccines;
    Afraid of vaccines in general; Allergic to vaccines; Don't like needles;
    Don't trust the vaccine source/will wait for other vaccines; Effective treatments for COVID are or will be available; It is too difficult to get the vaccine, e.g. have to travel far;
    Vaccine will be too expensive; I don't know how to get the vaccine; I will wait until others have been vaccinated;
    I will get the vaccine later; Religious objections to vaccines in general or to the COVID vaccine; Some other reason;
    Don't know. Answer Not applicable if you've already been vaccinated or have answered you're likely to get vaccinated",
    name %in% c("Q59", "Q86A", "Q86B", "Q86C", "Q86D") ~ "Not at all; Just a little; Somewhat; A lot; Refused; Don't know",
    name == "Q60" ~ "Very badly; Fairly badly; Fairly well; Very well; Refused; Don't know",
    name %in% c("Q61A", "Q61B", "Q61C") ~ "Not at all satisfied; Not very satisfied; Fairly satisfied; Very satisfied; Refused; Don't know",
    name == "Q62" ~ "A lot; Some; A little; None; Refused; Don't know",
    name == "Q64A" ~ "Very unprepared; Somewhat unprepared; Somewhat prepared; Very prepared; Refused; Don't know",
    name == "Q65B" ~ "Very unfair; Somewhat unfair; Somewhat fair; Very fair; Refused; Don't know",
    name %in% c("Q74A", "Q74B", "Q74C", "Q74D", "Q74E", "Q90H", "Q90I") ~ "Never; Less than once a month; A few times a month; A few times a week; Every day; Refused; Don't know",
    name == "Q84A" ~ paste(c(levels(as.factor(df$Q84A))[levels(as.factor(df$Q84A)) != "Refused to answer"], "Refused to answer", "Don't know"), collapse = "; "),
    name == "Q89A" ~ "No (does NOT feel close to ANY party); Yes (feels close to a party); Refused to answer; Don't know",
    name == "Q89B" ~ paste0(paste(c(levels(as.factor(df$Q89B))[!levels(as.factor(df$Q89B)) %in% c("Refused", "Don't know", "Not Applicable")], "Refused", "Don't know"), collapse = "; "), ". Answer Not applicable if you don't feel close to any party"),
    name == "Q90F" ~ "No one in household owns; Yes, someone else in household owns; Yes (personally owns); Refused; Don't know",
    name == "Q90G" ~ "No (Does not have Internet access); Yes (Has Internet access); Refused; Don't know. Answer Not applicable (does not personally have mobile phone) if you don't personally own a mobile phone",
    
    name == "Q93B" ~ "Never had a job; Student; Housewife/Homemaker;
    Agriculture/Farming/Fishing/Forestry; Trader/Hawker/Vendor; Retail/Shop;
    Unskilled manual worker (e.g. cleaner, laborer, domestic help, unskilled manufacturing worker); Artisan or skilled manual worker (e.g. trades like electrician, mechanic, mechanic, machinist, or skilled manufacturing worker); Clerical or secretarial;
    Supervisor/Foreman/Senior manager; Security services; Mid-level professional (e.g. teacher, nurse, mid-level government officer);
    Upper-level professional (e.g. banker/finance, doctor, lawyer, engineer, accountant, professor, senior-level government officer); Refused; Don't know;
    Retired",
    name == "Q94" ~ "No formal schooling; Informal schooling only (including Koranic schooling); Some primary schooling;
    Primary school completed; Intermediate school or some secondary school/high school; Secondary school/high school completed;
    Post-secondary qualifications other than university, e.g. a diploma or degree from a polytechnic or college; Some university; University completed;
    Post-graduate; Refused; Don't know",
    name == "Q95" ~ "None; Christian only (i.e., without specific sub-group identification); Roman Catholic; Orthodox; Anglican;
    Lutheran; Methodist; Baptist; Evangelical; Pentecostal (e.g. “born again” and/or “saved”); 
    Jehovah's Witness; Muslim only (i.e., without specific sub-group identification); Sunni only (i.e., without specific sub-group identification); Qadiriya Brotherhood; Traditional/Ethnic religion;
    Agnostic (Do not know if there is a God); Atheist (Do not believe in a God); Church of Christ; Refused; Don't know",
    name == "Q100" ~ "Man; Woman",
    name == "Q101" ~"Black/African; White/European; Coloured/Mixed race; Arab/Lebanese/North African;
    South Asian (Indian, Pakistani, etc.); East Asian (Chinese, Korean, Indonesian, etc.); Don't know"
    )) %>%
  mutate(question_prompt = ifelse(is.na(response_levels), question, paste(question, "Answer one of the following options:", response_levels))) %>%
  dplyr::select(name, question_prompt) %>%
  pivot_wider(names_from = name,
              values_from = question_prompt)

# combine prompt with data
df <-
  rbind(q_item,
        as.data.frame(lapply(df, as.character),
                      stringsAsFactors = FALSE))

names(df)[names(df) == "RESPNO"] <- "ID"
df$ID[1] <- "ID"
df <- df[, c("ID", setdiff(names(df), "ID")), drop = FALSE]

write_clean_csv(df,
                file.path("data", "processed", "surveys", "afrobarometer",
                          "afrobarometer_sl_data.csv"))
