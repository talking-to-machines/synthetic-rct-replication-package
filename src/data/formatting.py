import json
import pandas as pd


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
    system_template: str,
    demographic_prompt: str,
    treatment: str,
    treatment_transcripts: dict,
) -> str:
    """Fill `{profile}` and `{treatment}` placeholders in `system_template`.

    Parameters:
        system_template: System-message template with `{profile}` and
            `{treatment}` named placeholders (loaded from the RCT prompt JSON).
        demographic_prompt: Formatted demographic Q&A block.
        treatment: Treatment-arm label (key into `treatment_transcripts`).
        treatment_transcripts: Mapping from treatment label to transcript text.

    Returns:
        The fully-rendered system message.
    """
    return system_template.format(
        profile=demographic_prompt,
        treatment=treatment_transcripts[treatment],
    )


def generate_synthetic_experiment_prompts(
    data: pd.DataFrame,
    demographic_questions: list,
    system_template: str,
    user_template: str,
    treatment_transcripts: dict,
    id_column: str = "ID",
    treatment_column: str = "treatment",
) -> pd.DataFrame:
    """Generate per-participant system + user prompts for an RCT.

    Templates and transcripts are supplied by the caller (loaded from the RCT
    prompt JSON). The system template fills `{profile}` with the demographic
    Q&A block and `{treatment}` with the assigned treatment transcript. The
    user template is used verbatim.

    Args:
        data: DataFrame with one row per subject.
        demographic_questions: Column names to format into the profile block.
        system_template: Template string with `{profile}`/`{treatment}` placeholders.
        user_template: Literal user message (already includes formatting instruction).
        treatment_transcripts: Mapping from treatment label to transcript text.
        id_column: Column name for the subject identifier.
        treatment_column: Column name for the treatment-arm label.

    Returns:
        DataFrame with columns: custom_id, <id_column>, demographic_info,
        treatment, question_prompt, system_message.
    """
    prompts = []
    for custom_id_counter in range(len(data)):
        prompts.append(
            {
                "custom_id": f"{custom_id_counter}",
                id_column: data.loc[custom_id_counter, id_column],
                "demographic_info": generate_qna_format(
                    data.loc[custom_id_counter, demographic_questions]
                ),
                "treatment": data.loc[custom_id_counter, treatment_column],
                "question_prompt": user_template,
            }
        )
    prompts = pd.DataFrame(prompts)

    prompts["system_message"] = prompts.apply(
        lambda row: construct_system_message_with_treatment(
            system_template,
            row["demographic_info"],
            row["treatment"],
            treatment_transcripts,
        ),
        axis=1,
    )

    return prompts


def format_duch_2023_system_prompt(
    row: pd.Series,
    system_template: str,
    treatment_transcripts: dict,
) -> str:
    """Render the system prompt for a Duch et al. 2023 respondent.

    Fills the `{profile}` and `{treatment}` placeholders in `system_template`
    using the respondent's pre-built demographic prompt and their assigned
    treatment transcript.
    """
    return system_template.format(
        profile=row["demographic_prompt"],
        treatment=treatment_transcripts[row["treatment"]],
    )


def format_duch_2023_user_prompt(
    row: pd.Series,
    user_template: str,
    target_outcome: str,
) -> str:
    """Build the fine-tuning message sequence for a single respondent.

    Returns a JSON string encoding a list of system/user/assistant messages.
    """
    prompt = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": user_template},
        {"role": "assistant", "content": row[target_outcome]},
    ]
    return json.dumps(prompt)
