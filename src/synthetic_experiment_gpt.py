import os
import json
import math
import pandas as pd
from src.data_processing import load_data, create_batch_file, include_variable_names
from src.prompt_generation import generate_synthetic_experiment_prompts
from src.api_interaction import batch_query
from openai import OpenAI
from config.settings import OPENAI_API_KEY


def main(request):
    # Load OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )

    # Load and preprocess data
    data = load_data(data_file_path, drop_first_row=drop_first_row)

    # Generate demographic prompts
    if request["study"] in ["duch_2023_logit"]:
        prompts = generate_synthetic_experiment_prompts(
            data,
            request["survey_context"],
            request["demographic_questions"],
            request["question"],
            study=request["study"],
        )

    else:
        raise ValueError(f"Study {study} is not supported.")

    # Perform batch query for survey questions
    is_logit = request["study"] == "duch_2023_logit"
    batch_file_dir = create_batch_file(
        prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        batch_file_name="batch_input_llm_replication_experiment.jsonl",
        logit=is_logit,
    )

    llm_responses = batch_query(
        client,
        batch_input_file_dir=batch_file_dir,
        batch_output_file_dir="batch_output_llm_replication_experiment.jsonl",
        logit=is_logit,
    )

    llm_responses.rename(columns={"query_response": "llm_response"}, inplace=True)

    prompts_with_responses = pd.merge(left=prompts, right=llm_responses, on="custom_id")
    data_with_responses = pd.merge(
        left=data, right=prompts_with_responses, on="ID", suffixes=("", "_y")
    )

    if request["study"] == "duch_2023_logit":
        data_with_responses["user_response"] = data_with_responses[
            request["question"][0]
        ]

    else:
        raise ValueError(f"Study {study} is not supported.")

    # Post-processing for logit studies: parse JSON response into separate columns
    if is_logit:

        def parse_logit_response(json_str):
            try:
                parsed = json.loads(json_str)
                actual_response = parsed.get("response", "")
                top_logprobs = parsed.get("top_logprobs", [])

                yes_probs = []
                no_probs = []
                for entry in top_logprobs:
                    token = entry["token"].strip().lower()
                    if token == "yes":
                        yes_probs.append(math.exp(entry["logprob"]))
                    elif token == "no":
                        no_probs.append(math.exp(entry["logprob"]))

                prob_yes = sum(yes_probs) if yes_probs else None
                prob_no = sum(no_probs) if no_probs else None

                return pd.Series(
                    {
                        "llm_response_parsed": actual_response,
                        "logprob_yes": (
                            math.log(prob_yes) if prob_yes is not None else None
                        ),
                        "logprob_no": (
                            math.log(prob_no) if prob_no is not None else None
                        ),
                        "prob_yes": prob_yes,
                        "prob_no": prob_no,
                    }
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                return pd.Series(
                    {
                        "llm_response_parsed": "",
                        "logprob_yes": None,
                        "logprob_no": None,
                        "prob_yes": None,
                        "prob_no": None,
                    }
                )

        logit_columns = data_with_responses["llm_response"].apply(parse_logit_response)
        data_with_responses = pd.concat([data_with_responses, logit_columns], axis=1)

    # Include model and experiment information
    data_with_responses["model"] = model
    data_with_responses["scenario"] = scenario

    # Include variable names as new column headers
    if drop_first_row:
        data_with_response_headers = include_variable_names(
            data_with_responses, data_file_path
        )

    # Save prompts with responses into Excel file
    prompts_response_file_path = os.path.join(
        current_dir, f"../results/{request['experiment_round']}/{version}.xlsx"
    )
    data_with_response_headers.to_excel(prompts_response_file_path, index=False)


if __name__ == "__main__":
    study = "duch_2023_logit"  # duch_2023_logit
    model_name = "gpt4o_instruct"
    scenario_name = "s1"
    current_dir = os.path.dirname(__file__)
    experiment_round = "duch_2023_logit"
    scenario = "S1 (Instruct Model)"
    model = "gpt-4o-2024-08-06"

    if study == "duch_2023_logit":
        ### Configuration for COVID-19 Vaccination RCT (START) ###
        version = f"{study}_{model_name}_{scenario_name}"  # Vaccination Outcome
        data_file_path = os.path.join(
            current_dir,
            "../data/duch_et_al_2023_holdout_1537.csv",
        )
        treatment_assignment_column = "treatment"
        drop_first_row = True

        input_data = {
            "data_file_path": data_file_path,
            "study": study,
            "treatment_assignment_column": treatment_assignment_column,
            "experiment_round": experiment_round,
            "demographic_questions": [
                "Start Date",
                "What is your current age?",
                "What is your gender?",
                "What is the highest educational qualification you have completed?",
                "Which region do you live in?",
                "Which distric do you live in?",
                "What is the name of the community you live in?",
                "How many people live in your village?",
                "What is the distance in km of the nearest health clinic from where you live?",
                "How many people live in the house together with you (NOT including you) at this moment?",
                "How many children below 18 years old are currently living in your home?",
                "What is your current working situation?",
                "How much (in Ghanaian Cedis) on average does your household spend in a typical week on food?",
                "How much (in Ghanaian Cedis) on average does your household spend in a typical week on non-food items (electricity, water, rent, school fees)?",
                "How would you rate the overall economic or financial condition of your household today?",
                "Do you have a registered mobile number?",
                "How many family members do you have in another village?",
                "How many friends and acquaintances who are not part of your family do you have in another village?",
                "How many individuals can you identify in your social network? Think of friends and relatives that live close to you",
                "How often do you use social media?",
            ],
            "question": [
                "Have you received a COVID-19 vaccine, as verified in the records of the Ghanaian District Health Offices?"
            ],
            "survey_context": "You participate in a healthcare survey in Ghana and have the following profile:",
        }
        ### Configuration for COVID-19 Vaccination RCT (END) ###

    else:
        raise ValueError(f"Study {study} is not supported.")

    main(input_data)
