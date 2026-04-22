import json
import os
import time

import pandas as pd
from openai import OpenAI
from together import Together
from tqdm import tqdm

from src.utils.config import TOGETHER_API_KEY

tqdm.pandas()


def create_batch_file(
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str = "question_prompt",
    batch_file_name: str = "batch_tasks.jsonl",
    logit: bool = False,
    model: str = "gpt-4o-2024-08-06",
) -> str:
    """
    Create a JSONL batch file from the prompts DataFrame for the OpenAI Batch API.

    Parameters:
        prompts (pd.DataFrame): The DataFrame containing prompts.
        system_message_field (str): The column name indicating the system message.
        user_message_field (str): The column name indicating the user message.
        batch_file_name (str): The name of the batch file.
        logit (bool): Whether to include logprob parameters (max_tokens=1, logprobs=True, top_logprobs=5).
        model (str): Model id for the batch request body.

    Returns:
        str: The path to the created JSONL batch file.
    """
    tasks = []
    for i in range(len(prompts)):
        body = {
            "model": model,
            "temperature": 1.0,
            "messages": [
                {"role": "system", "content": prompts.loc[i, system_message_field]},
                {"role": "user", "content": prompts.loc[i, user_message_field]},
            ],
        }
        if logit:
            body["max_tokens"] = 1
            body["logprobs"] = True
            body["top_logprobs"] = 5
        task = {
            "custom_id": f'{prompts.loc[i, "custom_id"]}',
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        tasks.append(task)

    current_dir = os.path.dirname(__file__)
    batch_file_name = os.path.join(current_dir, f"../../batch_files/{batch_file_name}")
    os.makedirs(os.path.dirname(batch_file_name), exist_ok=True)
    with open(batch_file_name, "w") as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    return batch_file_name


def batch_query(
    client: OpenAI,
    batch_input_file_dir: str,
    batch_output_file_dir: str,
    logit: bool = False,
) -> pd.DataFrame:
    """
    Query the LLM using OpenAI batch processing and return the responses after completion.

    Parameters:
        batch_input_file_dir (str): The directory containing the batch input file.
        batch_output_file_dir (str): The directory containing the batch output file.
        logit (bool): Whether to extract logprob data from the response.

    Returns:
        pd.DataFrame: The prompts with the corresponding LLM responses.
    """
    batch_file = client.files.create(
        file=open(batch_input_file_dir, "rb"), purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch job status: {batch_job.status}")
        if batch_job.status == "completed":
            break
        elif batch_job.status == "failed":
            raise Exception("Batch job failed.")
        else:
            time.sleep(300)

    result_file_id = batch_job.output_file_id
    results = client.files.content(result_file_id).content

    current_dir = os.path.dirname(__file__)
    batch_output_dir = os.path.join(
        current_dir, f"../../batch_files/{batch_output_file_dir}"
    )
    os.makedirs(os.path.dirname(batch_output_dir), exist_ok=True)
    with open(batch_output_dir, "wb") as file:
        file.write(results)

    response_list = []
    with open(batch_output_dir, "r") as file:
        for line in file:
            result = json.loads(line.strip())
            choice = result["response"]["body"]["choices"][0]
            actual_response = choice["message"]["content"]

            if logit:
                top_logprobs_data = []
                logprobs_obj = choice.get("logprobs", {})
                if logprobs_obj and logprobs_obj.get("content"):
                    for item in logprobs_obj["content"][0]["top_logprobs"]:
                        top_logprobs_data.append(
                            {
                                "token": item["token"],
                                "logprob": item["logprob"],
                            }
                        )
                query_response = json.dumps(
                    {
                        "response": actual_response,
                        "top_logprobs": top_logprobs_data,
                    }
                )
            else:
                query_response = actual_response

            response_list.append(
                {
                    "custom_id": f'{result["custom_id"]}',
                    "query_response": query_response,
                }
            )

    return pd.DataFrame(response_list)


def inference_endpoint_query(
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str,
    experiment_round: str,
    experiment_version: str,
    model_name: str,
    together_model_id: str,
) -> pd.DataFrame:
    """
    Query a dedicated inference endpoint (Together AI) and return the responses.

    Saves per-row progress to resume interrupted runs.

    Parameters:
        prompts (pd.DataFrame): The DataFrame containing prompts.
        system_message_field (str): The column name indicating the system message.
        user_message_field (str): The column name indicating the user message.
        experiment_round (str): The round of the experiment.
        experiment_version (str): The experiment/model version.
        model_name (str): The name of the LLM backend ("together_logit").
        together_model_id (str): Together AI model id to query.

    Returns:
        pd.DataFrame: The prompts with the corresponding LLM responses.
    """
    current_dir = os.path.dirname(__file__)
    progress_dir = os.path.join(
        current_dir, f"../../outputs/logs/inference/{experiment_round}/progress"
    )
    progress_file = os.path.join(progress_dir, f"{experiment_version}.csv")

    os.makedirs(progress_dir, exist_ok=True)

    if os.path.exists(progress_file):
        processed_prompts = pd.read_csv(progress_file)
        processed_prompts["ID"] = processed_prompts["ID"].astype("int64")
        prompts = prompts.merge(
            processed_prompts[["ID", "llm_response"]], on="ID", how="left"
        )
    else:
        prompts["llm_response"] = None

    def together_logit_query(row: pd.Series):
        if not pd.isnull(row["llm_response"]):
            return row["llm_response"]

        messages = [
            {"role": "system", "content": row[system_message_field]},
            {"role": "user", "content": row[user_message_field]},
        ]

        response = client.chat.completions.create(
            model=together_model_id,
            messages=messages,
            stream=False,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            temperature=1.0,
        )

        actual_response = response.choices[0].message.content
        top_logprobs_data = []

        logprobs_obj = response.choices[0].logprobs
        if logprobs_obj:
            top_lp = getattr(logprobs_obj, "top_logprobs", None)
            if top_lp:
                first_pos = top_lp[0]
                if isinstance(first_pos, dict):
                    for token, logprob in first_pos.items():
                        top_logprobs_data.append({"token": token, "logprob": logprob})
                elif isinstance(first_pos, list):
                    for item in first_pos:
                        if isinstance(item, dict):
                            top_logprobs_data.append(
                                {
                                    "token": item.get("token"),
                                    "logprob": item.get("logprob"),
                                }
                            )
            else:
                tokens = getattr(logprobs_obj, "tokens", None) or []
                token_logprobs = getattr(logprobs_obj, "token_logprobs", None) or []
                if tokens and token_logprobs:
                    top_logprobs_data.append(
                        {"token": tokens[0], "logprob": token_logprobs[0]}
                    )

        result = json.dumps(
            {
                "response": actual_response,
                "top_logprobs": top_logprobs_data,
            }
        )

        row["llm_response"] = result

        row.to_frame().T.to_csv(
            progress_file,
            mode="a",
            header=not os.path.exists(progress_file),
            index=False,
        )

        return result

    if model_name == "together_logit":
        client = Together(api_key=TOGETHER_API_KEY)
        prompts["llm_response"] = prompts.progress_apply(together_logit_query, axis=1)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return prompts
