import os
import json
import math
import pandas as pd
from src.data_processing import load_data, include_variable_names
from src.prompt_generation import generate_synthetic_experiment_prompts
from src.api_interaction import inference_endpoint_query


def main(request):
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
        raise ValueError(f"Study {request['study']} is not supported.")

    # Perform query for survey questions
    prompts_with_responses = inference_endpoint_query(
        endpoint_url=request["api_url"],
        prompts=prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        experiment_round=request["experiment_round"],
        experiment_version=version,
        model_name=request["model_name"],
    )

    data_with_responses = pd.merge(
        left=data,
        right=prompts_with_responses,
        on="ID",
        suffixes=("", "_y"),
    )

    if request["study"] in ["duch_2023_logit"]:
        data_with_responses["user_response"] = data_with_responses[request["question"]]

    else:
        raise ValueError(f"Study {request['study']} is not supported.")

    # Post-processing for logit studies: parse JSON response into separate columns
    if request["study"] in ["duch_2023_logit"]:

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
    data_with_responses["finetuning_approach"] = finetuning_approach
    data_with_responses["epochs"] = epochs
    data_with_responses["checkpoints"] = checkpoints
    data_with_responses["evaluations"] = evaluations
    data_with_responses["batch_size"] = batch_size
    data_with_responses["lora_rank"] = lora_rank
    data_with_responses["lora_alpha"] = lora_alpha
    data_with_responses["lora_dropout"] = lora_dropout
    data_with_responses["lora_trainable_models"] = lora_trainable_models
    data_with_responses["train_on_inputs"] = train_on_inputs
    data_with_responses["learning_rate"] = learning_rate
    data_with_responses["learning_rate_scheduler"] = learning_rate_scheduler
    data_with_responses["warmup_ratio"] = warmup_ratio
    data_with_responses["max_gradient_norm"] = max_gradient_norm
    data_with_responses["weight_decay"] = weight_decay

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
    study = "duch_2023_logit"
    model_name = "llama31_70b_instruct"
    scenario_name = "s11_finetuning_ep6_r32"
    current_dir = os.path.dirname(__file__)
    experiment_round = "round11"
    scenario = "S11 (Fine-tuned Model Epoch=6 R=32)"
    model = "Llama 3.1 70B Instruct"
    api_url = ""  # HF dedicated inference endpoint
    model_name = "together_logit"  # together_logit
    drop_first_row = True
    treatment_assignment_column = "treatment"
    finetuning_approach = "LoRa"
    epochs = 6
    checkpoints = 1
    evaluations = 0
    batch_size = 8
    lora_rank = 32
    lora_alpha = 64
    lora_dropout = 0.1
    lora_trainable_models = "all-linear"
    train_on_inputs = "auto"
    learning_rate = 0.00002
    learning_rate_scheduler = "linear"
    warmup_ratio = 0.1
    max_gradient_norm = 1
    weight_decay = 0

    if study == "duch_2023_logit":
        version = f"{study}_{model_name}_{scenario_name}"  # Vaccination Outcome
        data_file_path = os.path.join(
            current_dir,
            "../data/duch_et_al_2023_holdout_1537.csv",
        )

        input_data = {
            "data_file_path": data_file_path,
            "study": study,
            "treatment_assignment_column": treatment_assignment_column,
            "api_url": api_url,
            "model_name": model_name,
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

    else:
        raise ValueError(f"Study {study} is not supported.")

    main(input_data)
