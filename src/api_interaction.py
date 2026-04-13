import pandas as pd
import os
from config.settings import TOGETHER_API_KEY
import time
import json
from openai import OpenAI
from tqdm import tqdm
from together import Together

tqdm.pandas()


def batch_query(
    client: OpenAI,
    batch_input_file_dir: str,
    batch_output_file_dir: str,
    logit: bool = False,
) -> pd.DataFrame:
    """
    Query the LLM using batch processing and return the responses after completion.

    Parameters:
        batch_input_file_dir (str): The directory containing the batch input file.
        batch_output_file_dir (str): The directory containing the batch output file.
        logit (bool): Whether to extract logprob data from the response.

    Returns:
        pd.DataFrame: The prompts with the corresponding LLM responses.
    """
    # Upload batch input file
    batch_file = client.files.create(
        file=open(batch_input_file_dir, "rb"), purpose="batch"
    )

    # Create batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    # Check batch status
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch job status: {batch_job.status}")
        if batch_job.status == "completed":
            break
        elif batch_job.status == "failed":
            raise Exception("Batch job failed.")
        else:
            # Wait for 5 minutes before checking again
            time.sleep(300)

    # Retrieve batch results
    result_file_id = batch_job.output_file_id
    results = client.files.content(result_file_id).content

    # Save the batch output
    current_dir = os.path.dirname(__file__)
    batch_output_dir = os.path.join(
        current_dir, f"../batch_files/{batch_output_file_dir}"
    )
    with open(batch_output_dir, "wb") as file:
        file.write(results)

    # Loading data from saved output file
    response_list = []
    with open(batch_output_dir, "r") as file:
        for line in file:
            # Parsing the JSON result string into a dict
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
    endpoint_url: str,
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str,
    experiment_round: str,
    experiment_version: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Query dedicated inference endpoint API and return the responses after completion for HuggingFace and Deepseek.

    Parameters:
        endpoint_url (str): The endpoint URL
        prompts (pd.DataFrame): The DataFrame containing prompts.
        system_message_field (str): The column name indicating the system message.
        user_message_field (str): The column name indicating the user message.
        experiment_round (str): The round of the experiment
        experiment_version (str): The experiment/model version
        model_name (str): The name of the LLM

    Returns:
        pd.DataFrame: The prompts with the corresponding LLM responses.
    """
    current_dir = os.path.dirname(__file__)
    progress_dir = os.path.join(current_dir, f"../results/{experiment_round}/progress")
    progress_file = os.path.join(
        current_dir, f"../results/{experiment_round}/progress/{experiment_version}.csv"
    )

    # Check and create the progress folder if it doesn't exist
    os.makedirs(progress_dir, exist_ok=True)

    # Load progress if exists
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
            model="iamraymondlow/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-c6090e90",  # meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
            messages=messages,
            stream=False,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            temperature=1.0,
        )

        actual_response = response.choices[0].message.content
        top_logprobs_data = []

        # logprobs_obj = response.choices[0].logprobs
        # print(logprobs_obj)
        # if logprobs_obj and logprobs_obj.top_logprobs:
        #     for token, logprob in logprobs_obj.top_logprobs[0].items():
        #         top_logprobs_data.append({
        #             "token": token,
        #             "logprob": logprob,
        #         })

        # print(response)
        # logprobs_obj = response.choices[0].logprobs.content[0]["top_logprobs"] # 70B fine-tune
        # if logprobs_obj:
        #     for item in logprobs_obj:
        #         top_logprobs_data.append({
        #             "token": item["token"],
        #             "logprob": item["logprob"],
        #         })

        logprobs_obj = response.choices[0].logprobs  # 70B instruct
        if logprobs_obj:
            top_logprobs_data.append(
                {
                    "token": logprobs_obj.tokens[0],
                    "logprob": logprobs_obj.token_logprobs[0],
                }
            )

        result = json.dumps(
            {
                "response": actual_response,
                "top_logprobs": top_logprobs_data,
            }
        )

        row["llm_response"] = result

        # Save progress
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
