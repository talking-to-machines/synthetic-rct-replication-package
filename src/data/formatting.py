import json
import pandas as pd
from typing import Any

duch_2023_logit_treatments = {
    "CDC Health": "Health authorities are working hard to distribute the COVID-19 vaccines free for everyone with no strings attached. COVID 19 vaccines are safe and effective. After you have been fully vaccinated you can resume activities that you did prior to the pandemic. Getting the COVID-19 vaccine will help prevent you from getting COVID-19 and reduce your risk of being hospitalized with COVID-19. COVID 19 vaccine help you to protect yourself your environment and your loved ones from COVID-19 exposure.\nWe indicated that we will follow up with you in six weeks. We will contact you in order to verify your vaccination status. If you can provide us with your COVID-19 vaccination carnet at the time, we will upload a copy of the vaccination carnet to our secure server for verification.",
    "Placebo": "The Sun lights up our lives for business for education even for socializing but when the Sun sets many people use candles who are quality battery-operated torches and kerosene lamps as inefficient and expensive ways to create light. What if you can take some Sun with you at night?  You can with portable solar products there are different types, but each portable solar product is made up of three basic parts: a small solar panel, a modern rechargeable battery and an LED bulb. The solar panel catches the light from the Sun and stores this energy in the battery. This can now be used for much needed light when it's dark. Many can even charge phones portable solar products should be reliable affordable and warranted be sure to demand top quality solar products look for these products lighting Africa shining the way.\nWe indicated that we will follow up with you in six weeks. We will contact you in order to verify your vaccination status. If you can provide us with your COVID-19 vaccination carnet at the time, we will upload a copy of the vaccination carnet to our secure server for verification.",
    "Low Cash": "Health authorities are working hard to distribute the COVID-19 vaccines free for everyone with no strings attached. COVID-19 vaccines are safe and effective. After you have been fully vaccinated you can resume activities that you did prior to the pandemic. If you have at least one COVID-19 vaccine shot you will receive 20 Cedi. If you get vaccinated, you will get rewarded.\nWe indicated that we will follow up with you in six weeks. We will contact you in order to verify your vaccination status. If you can provide us with your COVID-19 vaccination carnet at the time, we will upload a copy of the vaccination carnet to our secure server for verification and you will be paid your 20 Cedi via cell phone money payment or by cash if you prefer.",
    "High Cash": "Health authorities are working hard to distribute the COVID-19 vaccines free for everyone with no strings attached. COVID-19 vaccines are safe and effective. After you have been fully vaccinated you can resume activities that you did prior to the pandemic. If you have at least one COVID-19 vaccine shot you will receive 60 Cedi. If you get vaccinated, you will get rewarded.\nWe indicated that we will follow up with you in six weeks. We will contact you in order to verify your vaccination status. If you can provide us with your COVID-19 vaccination carnet at the time, we will upload a copy of the vaccination carnet to our secure server for verification and you will be paid your 60 Cedi via cell phone money payment or by cash if you prefer.",
}


def generate_qna_format(demographic_info: pd.Series) -> str:
    """
    Formats the demographic information of a subject in a Q&A format.

    Parameters:
        demographic_info (pd.Series): A pandas Series containing the demographic information of the subject.

    Returns:
        str: The formatted survey response.
    """
    survey_response = ""
    counter = 1
    for question, response in demographic_info.items():
        if pd.isnull(response) or response == "NA":
            continue

        if type(response) == str and "\n" in response:
            response = response.split("\n")[0].replace("\r", "")

        survey_response += f"{counter}) Interviewer: {question} Me: {response} "
        counter += 1

    return survey_response


def generate_demographic_prompt(row: pd.Series, excluded_columns: list) -> str:
    """Build a numbered interview-style prompt string from a row's demographic columns.

    Iterates over the row's columns (excluding those in `excluded_columns`), and
    formats each non-null value as a numbered interviewer-respondent exchange.
    """
    demographic_questions = [q for q in list(row.index) if q not in excluded_columns]
    demographic_prompt = ""
    counter = 1
    for question in demographic_questions:
        if pd.isnull(row[question]) or row[question] == "NA" or row[question] == "N/A":
            continue
        demographic_prompt += f"{counter}) Interviewer: {question} Me: {row[question]} "
        counter += 1
    return demographic_prompt


def construct_system_message_with_treatment(
    survey_context: str, demographic_prompt: str, treatment: str, study: str
) -> str:
    """
    Constructs system message by combining the survey context, demographic prompt and treatment prompt.

    Parameters:
        survey_context (str): The context of the survey.
        demographic_prompt (str): The prompt for demographic information.
        treatment (str): The treatment applied to the agent
        study (str): The study type of interest

    Returns:
        str: The constructed prompt.
    """
    if study in ["duch_2023_logit"]:
        return f"{survey_context}\n{demographic_prompt}\n\nDuring the survey, you received the following information:\n{duch_2023_logit_treatments[treatment]}\n\nYou were told you would be contacted in six weeks to verify your vaccination status. Six weeks have now passed."

    else:
        raise ValueError(f"Study {study} is not supported.")


def generate_synthetic_experiment_prompts(
    data: pd.DataFrame,
    survey_context: str,
    demographic_questions: list,
    question: Any,
    study: str = "duch_2023",
) -> pd.DataFrame:
    """
    Generates synthetic experiment prompts for each participant in a survey dataset.

    Args:
        data (pd.DataFrame): DataFrame containing survey participant data.
        survey_context (str): Context or description of the survey.
        demographic_questions (list): List of column names representing demographic questions.
        question (Any): The main survey question(s). Can be a string or list of strings.
        study (str, optional): The study design. Defaults to "duch_2023".

    Returns:
        pd.DataFrame: DataFrame of generated prompts per participant.

    Raises:
        ValueError: If an unsupported study type is provided.
    """
    if isinstance(question, list):
        question = " ".join(question)

    prompts = []
    custom_id_counter = 0
    for i in range(len(data)):
        if study in ["duch_2023_logit"]:
            question_prompt = f"{question} Please give your response to the question in the structured format below:\nYes/No"
        else:
            raise ValueError(f"Study {study} not supported.")

        prompts.append(
            {
                "custom_id": f"{custom_id_counter}",
                "ID": data.loc[i, "ID"],
                "survey_context": survey_context,
                "demographic_info": generate_qna_format(
                    data.loc[i, demographic_questions]
                ),
                "treatment": data.loc[i, "treatment"],
                "question": question,
                "question_prompt": question_prompt,
            }
        )
        custom_id_counter += 1
    prompts = pd.DataFrame(prompts)

    prompts["system_message"] = prompts.apply(
        lambda row: construct_system_message_with_treatment(
            row["survey_context"],
            row["demographic_info"],
            row["treatment"],
            study=study,
        ),
        axis=1,
    )

    return prompts


def format_duch_2023_system_prompt(row: pd.Series) -> str:
    """Format the system prompt for a Duch et al. 2023 respondent.

    Populates the system prompt template with the respondent's demographic
    profile and the treatment-specific transcript they received.
    """
    system_prompt = (
        "You participate in a healthcare survey in Ghana and have the following profile: \n"
        "{demographic_prompt}\n\n"
        "During the survey, you received the following information:\n"
        "{treatment_prompt}\n\n"
        "You were told you would be contacted in six weeks to verify your vaccination status. Six weeks have now passed."
    )
    return system_prompt.format(
        demographic_prompt=row["demographic_prompt"],
        treatment_prompt=duch_2023_logit_treatments[row["treatment"]],
    )


def format_duch_2023_user_prompt(row: pd.Series, target_outcome: str) -> str:
    """Build the fine-tuning message sequence for a single respondent.

    Returns a JSON string encoding a list of system/user/assistant messages.
    """
    question_prompt = f"{target_outcome} Please give your response to the question in the structured format below:\n[Yes/No]"
    response = row[target_outcome]

    prompt = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": question_prompt},
        {"role": "assistant", "content": response},
    ]
    return json.dumps(prompt)
