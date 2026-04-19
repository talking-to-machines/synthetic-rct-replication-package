"""Inference for all model x condition x RCT combinations.

Reads:  data/processed/rcts/*/test.jsonl, data/prompts/rcts/
Writes: data/synthetic/, outputs/logs/inference/

Extracted from archived `synthetic_experiment_gpt.py` and
`synthetic_experiment_togetherai.py`. Entry points preserved as
`run_gpt_inference` and `run_togetherai_inference` for backwards
compatibility with those scripts.
"""

import json
import math
import os

import pandas as pd
from openai import OpenAI

from src.data.cleaning import load_data, include_variable_names
from src.data.formatting import generate_synthetic_experiment_prompts
from src.models.api_client import (
    batch_query,
    create_batch_file,
    inference_endpoint_query,
)
from src.utils.config import OPENAI_API_KEY


def _parse_logit_response(json_str: str) -> pd.Series:
    """Parse Yes/No logit distribution from a JSON string of logprobs."""
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
                "logprob_yes": math.log(prob_yes) if prob_yes is not None else None,
                "logprob_no": math.log(prob_no) if prob_no is not None else None,
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


def run_gpt_inference(request: dict) -> pd.DataFrame:
    """Run inference for GPT models via the OpenAI Batch API.

    Expects `request` with keys: study, survey_context, demographic_questions,
    question, data_file_path, drop_first_row, experiment_round, version,
    model, scenario.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    data = load_data(
        request["data_file_path"], drop_first_row=request["drop_first_row"]
    )

    if request["study"] not in ["duch_2023_logit"]:
        raise ValueError(f"Study {request['study']} is not supported.")

    prompts = generate_synthetic_experiment_prompts(
        data,
        request["survey_context"],
        request["demographic_questions"],
        request["question"],
        study=request["study"],
    )

    is_logit = request["study"] == "duch_2023_logit"
    batch_file_dir = create_batch_file(
        prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        batch_file_name="batch_input_llm_replication_experiment.jsonl",
        logit=is_logit,
        model=request["model"],
    )

    llm_responses = batch_query(
        client,
        batch_input_file_dir=batch_file_dir,
        batch_output_file_dir="batch_output_llm_replication_experiment.jsonl",
        logit=is_logit,
    )
    llm_responses.rename(columns={"query_response": "llm_response"}, inplace=True)

    prompts_with_responses = pd.merge(prompts, llm_responses, on="custom_id")
    data_with_responses = pd.merge(
        data, prompts_with_responses, on="ID", suffixes=("", "_y")
    )

    data_with_responses["user_response"] = data_with_responses[request["question"][0]]

    if is_logit:
        logit_columns = data_with_responses["llm_response"].apply(_parse_logit_response)
        data_with_responses = pd.concat([data_with_responses, logit_columns], axis=1)

    data_with_responses["model"] = request["model"]
    data_with_responses["scenario"] = request["scenario"]

    if request["drop_first_row"]:
        data_with_responses = include_variable_names(
            data_with_responses, request["data_file_path"]
        )

    output_path = os.path.join(
        "data/synthetic", f"{request['experiment_round']}_{request['version']}.xlsx"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_with_responses.to_excel(output_path, index=False)
    return data_with_responses


def run_togetherai_inference(request: dict) -> pd.DataFrame:
    """Run inference for open models (Llama/Qwen) via Together AI dedicated endpoints.

    Expects `request` with keys as in `run_gpt_inference` plus: api_url,
    model_name (backend, e.g. "together_logit"), finetuning_approach, epochs,
    checkpoints, evaluations, batch_size, lora_rank, lora_alpha, lora_dropout,
    lora_trainable_models, train_on_inputs, learning_rate,
    learning_rate_scheduler, warmup_ratio, max_gradient_norm, weight_decay.
    """
    data = load_data(
        request["data_file_path"], drop_first_row=request["drop_first_row"]
    )

    if request["study"] not in ["duch_2023_logit"]:
        raise ValueError(f"Study {request['study']} is not supported.")

    prompts = generate_synthetic_experiment_prompts(
        data,
        request["survey_context"],
        request["demographic_questions"],
        request["question"],
        study=request["study"],
    )

    prompts_with_responses = inference_endpoint_query(
        endpoint_url=request["api_url"],
        prompts=prompts,
        system_message_field="system_message",
        user_message_field="question_prompt",
        experiment_round=request["experiment_round"],
        experiment_version=request["version"],
        model_name=request["model_name"],
    )

    data_with_responses = pd.merge(
        data, prompts_with_responses, on="ID", suffixes=("", "_y")
    )
    data_with_responses["user_response"] = data_with_responses[request["question"]]

    logit_columns = data_with_responses["llm_response"].apply(_parse_logit_response)
    data_with_responses = pd.concat([data_with_responses, logit_columns], axis=1)

    for key in (
        "model",
        "scenario",
        "finetuning_approach",
        "epochs",
        "checkpoints",
        "evaluations",
        "batch_size",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_trainable_models",
        "train_on_inputs",
        "learning_rate",
        "learning_rate_scheduler",
        "warmup_ratio",
        "max_gradient_norm",
        "weight_decay",
    ):
        data_with_responses[key] = request.get(key)

    if request["drop_first_row"]:
        data_with_responses = include_variable_names(
            data_with_responses, request["data_file_path"]
        )

    output_path = os.path.join(
        "data/synthetic", f"{request['experiment_round']}_{request['version']}.xlsx"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_with_responses.to_excel(output_path, index=False)
    return data_with_responses


if __name__ == "__main__":
    raise NotImplementedError(
        "Wire this entry point to config.yaml to fan out inference across models x conditions x RCTs."
    )
