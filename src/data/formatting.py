import json
import pandas as pd

from src.data.cleaning import load_data


def generate_qna_format(
    profile_info: pd.Series,
    var_labels: dict | None = None,
) -> str:
    """Format the profile information of a subject in a Q&A format.

    Parameters:
        profile_info (pd.Series): A pandas Series containing the profile
            information of the subject. Series index = variable codes (short
            form when read via `load_data`).
        var_labels (dict | None): Optional mapping from variable code to the
            long-form survey question. When provided, the long-form text is
            used in the rendered Q&A so the prompt is human-readable. Falls
            back to the variable code when a key is missing.

    Returns:
        str: The formatted survey response.
    """
    survey_response = ""
    counter = 1
    for var, response in profile_info.items():
        if pd.isnull(response) or response == "NA":
            continue

        if type(response) == str and "\n" in response:
            response = response.split("\n")[0].replace("\r", "")

        question = (var_labels or {}).get(var, var)
        survey_response += f"{counter}) Interviewer: {question} Me: {response} \n"
        counter += 1

    return survey_response


def generate_profile_prompt(row: pd.Series, excluded_columns: list) -> str:
    """Build a numbered interview-style prompt string from a row's profile columns.

    Iterates over the row's columns (excluding those in `excluded_columns`), and
    formats each non-null value as a numbered interviewer-respondent exchange.
    """
    profile_vars = [q for q in list(row.index) if q not in excluded_columns]
    profile_prompt = ""
    counter = 1
    for question in profile_vars:
        if pd.isnull(row[question]) or row[question] == "NA" or row[question] == "N/A":
            continue
        profile_prompt += f"{counter}) Interviewer: {question} Me: {row[question]} "
        counter += 1
    return profile_prompt


def construct_system_message_with_treatment(
    system_template: str,
    profile_prompt: str,
    treatment: str,
    treatment_transcripts: dict,
) -> str:
    """Fill `{profile}` and `{treatment}` placeholders in `system_template`.

    Parameters:
        system_template: System-message template with `{profile}` and
            `{treatment}` named placeholders (loaded from the RCT prompt JSON).
        profile_prompt: Formatted profile Q&A block.
        treatment: Treatment-arm label (key into `treatment_transcripts`).
        treatment_transcripts: Mapping from treatment label to transcript text.

    Returns:
        The fully-rendered system message.
    """
    return system_template.format(
        profile=profile_prompt,
        treatment=treatment_transcripts[treatment],
    )


def generate_synthetic_experiment_prompts(
    data: pd.DataFrame,
    profile_vars: list,
    system_template: str,
    user_template: str,
    treatment_transcripts: dict,
    id_column: str = "SubjectID",
    treatment_column: str = "individual_treatment",
    var_labels: dict | None = None,
) -> pd.DataFrame:
    """Generate per-participant system + user prompts for an RCT.

    Templates and transcripts are supplied by the caller (loaded from the RCT
    prompt JSON). The system template fills `{profile}` with the profile Q&A
    block and `{treatment}` with the assigned treatment transcript. The user
    template is used verbatim.

    Args:
        data: DataFrame with one row per subject (short-code columns).
        profile_vars: Column names (short codes) to format into the profile block.
        system_template: Template string with `{profile}`/`{treatment}` placeholders.
        user_template: Literal user message (already includes formatting instruction).
        treatment_transcripts: Mapping from treatment label to transcript text.
        id_column: Column name for the subject identifier.
        treatment_column: Column name for the treatment-arm label.
        var_labels: Optional mapping from variable code -> long-form question.
            When supplied, the rendered profile block uses long-form text for
            readability (recommended; produced by `load_data`).

    Returns:
        DataFrame with columns: custom_id, <id_column>, profile_info,
        treatment, question_prompt, system_message.
    """
    prompts = []
    for custom_id_counter in range(len(data)):
        prompts.append(
            {
                "custom_id": f"{custom_id_counter}",
                id_column: data.loc[custom_id_counter, id_column],
                "profile_info": generate_qna_format(
                    data.loc[custom_id_counter, profile_vars],
                    var_labels=var_labels,
                ),
                "treatment": data.loc[custom_id_counter, treatment_column],
                "question_prompt": user_template,
            }
        )
    prompts = pd.DataFrame(prompts)

    prompts["system_message"] = prompts.apply(
        lambda row: construct_system_message_with_treatment(
            system_template,
            row["profile_info"],
            row["treatment"],
            treatment_transcripts,
        ),
        axis=1,
    )

    return prompts


def format_system_prompt(
    row: pd.Series,
    system_template: str,
    treatment_transcripts: dict,
) -> str:
    """Render the system prompt for one RCT respondent.

    Fills the `{profile}` and `{treatment}` placeholders in `system_template`
    using the respondent's pre-built profile prompt and their assigned
    treatment transcript.
    """
    return system_template.format(
        profile=row["profile_prompt"],
        treatment=treatment_transcripts[row["treatment"]],
    )


def format_user_prompt(
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


def build_finetune_source_records(
    source_id: str, source_cfg: dict, kind: str
) -> list[dict]:
    """Build per-subject {"messages": [...]} records for one fine-tuning source.

    Expects `source_cfg` to provide `data_file`, `prompt_file`, and `outcome`,
    and the prompt JSON to provide `profile_vars`, `system_template`,
    `user_template`. RCT prompt JSONs additionally provide `treatment`
    (transcripts dict); when present, `{treatment}` in `system_template` is
    filled from the row's `treatment_column`. Subjects with a missing outcome
    are dropped.
    """
    data_file = source_cfg.get("data_file")
    prompt_file = source_cfg.get("prompt_file")
    outcome = source_cfg.get("outcome")
    if not data_file or not prompt_file or not outcome:
        raise ValueError(
            f"{kind}/{source_id} is missing data_file, prompt_file, or outcome "
            f"in config.yaml (got data_file={data_file!r}, "
            f"prompt_file={prompt_file!r}, outcome={outcome!r})."
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_cfg = json.load(f)
    profile_vars = prompt_cfg["profile_vars"]
    system_template = prompt_cfg["system_template"]
    user_template = prompt_cfg["user_template"]
    if kind == "rcts":
        treatment_transcripts = prompt_cfg.get("treatment")
        treatment_col = prompt_cfg.get("treatment_column", "treatment")
    else:
        treatment_transcripts = None

    data, var_labels = load_data(data_file)

    records: list[dict] = []
    for _, row in data.iterrows():
        outcome_val = row.get(outcome)
        if pd.isnull(outcome_val) or str(outcome_val).strip() in ("", "NA", "N/A"):
            continue
        profile_prompt = generate_qna_format(row[profile_vars], var_labels=var_labels)
        if treatment_transcripts is not None:
            system_msg = system_template.format(
                profile=profile_prompt,
                treatment=treatment_transcripts[row[treatment_col]],
            )
        else:
            system_msg = system_template.format(profile=profile_prompt)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_template},
                    {"role": "assistant", "content": str(outcome_val).strip()},
                ]
            }
        )
    return records


def format_instruction_messages(record: dict, system_prompt: str) -> list[dict]:
    """Format an Alpaca-style instruction record as system/user/assistant messages."""
    instruction = record["instruction"].strip()
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"].strip()},
    ]
