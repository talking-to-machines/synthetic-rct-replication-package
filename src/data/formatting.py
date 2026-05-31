import json
import pandas as pd
from src.data.cleaning import load_data


def generate_qna_format(
    profile_info: pd.Series,
    var_labels: dict | None = None,
) -> str:
    """Format a subject's profile fields as an interviewer Q&A block.

    Args:
        profile_info: Series whose index is variable codes (short form when
            read via `load_data`) and values are the subject's responses.
        var_labels: Optional mapping from variable code to long-form survey
            question. When supplied, the rendered text uses the long-form
            label; falls back to the code when a label is missing.

    Returns:
        A multi-line string with one numbered Q&A pair per non-null field.
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


def construct_system_message_with_treatment(
    system_template: str,
    profile_prompt: str,
    treatment: str,
    treatment_transcripts: dict,
) -> str:
    """Fill `{profile}` and `{treatment}` placeholders in `system_template`.

    Args:
        system_template: System-message template with `{profile}` and
            `{treatment}` named placeholders.
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
    id_column: str = "ID",
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


def build_finetune_source_records(
    source_id: str, source_cfg: dict, kind: str
) -> tuple[list[dict], list[str] | None]:
    """Build per-subject `{"messages": [...]}` records for one source.

    Subjects whose `outcome` is null or blank are dropped. For RCT sources,
    `{treatment}` in the system template is filled from the row's treatment
    column, and the per-record treatment labels are returned alongside so
    callers can drive stratified splitting.

    Args:
        source_id: Identifier of the source (used in error messages).
        source_cfg: Per-source config block with `data_file`, `prompt_file`,
            and `outcome` keys.
        kind: Either `"rcts"` or `"surveys"`. RCT sources fill the treatment
            placeholder; survey sources do not.

    Returns:
        Tuple `(records, treatments)`. `records` is the list of
        chat-completion-shaped dicts. `treatments` is parallel to `records`
        for RCT sources, or `None` for survey sources.

    Raises:
        ValueError: If `source_cfg` is missing `data_file`, `prompt_file`,
            or `outcome`.
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
        treatment_col = None

    data, var_labels = load_data(data_file)

    records: list[dict] = []
    treatments: list[str] = []
    for _, row in data.iterrows():
        outcome_val = row.get(outcome)
        if pd.isnull(outcome_val) or str(outcome_val).strip() in ("", "NA", "N/A"):
            continue
        profile_prompt = generate_qna_format(row[profile_vars], var_labels=var_labels)
        if treatment_transcripts is not None:
            treatment_label = row[treatment_col]
            system_msg = system_template.format(
                profile=profile_prompt,
                treatment=treatment_transcripts[treatment_label],
            )
            treatments.append(str(treatment_label))
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

    return records, (treatments if treatment_col is not None else None)
