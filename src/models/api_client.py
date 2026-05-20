import json
import os
import time
import pandas as pd
from openai import OpenAI
from together import Together
from tqdm import tqdm
from src.utils.config import HF_TOKEN, TOGETHER_API_KEY

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


def _parse_openai_logprobs(logprobs_obj) -> list:
    """Parse OpenAI chat-completion logprobs (used by HF TGI / OpenAI).

    Returns the same per-position structure as the Together parser:
    a list of {"sampled_token", "sampled_logprob", "top_logprobs"} dicts.
    """
    per_position = []
    if not logprobs_obj or not getattr(logprobs_obj, "content", None):
        return per_position
    for tok in logprobs_obj.content:
        entries = []
        seen = set()

        def _add(t, lp):
            if t is None or lp is None:
                return
            key = t.strip().lower()
            if key in seen:
                return
            seen.add(key)
            entries.append({"token": t, "logprob": lp})

        _add(tok.token, tok.logprob)
        for alt in tok.top_logprobs or []:
            _add(alt.token, alt.logprob)

        per_position.append(
            {
                "sampled_token": tok.token,
                "sampled_logprob": tok.logprob,
                "top_logprobs": entries,
            }
        )
    return per_position


def inference_endpoint_query(
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str,
    experiment_round: str,
    experiment_version: str,
    model_name: str,
    model_id: str,
    temperature: float = 1.0,
    max_tokens: int = 1,
    logprobs_top_k: int = 5,
) -> pd.DataFrame:
    """
    Query a dedicated inference endpoint (Together AI or HuggingFace).

    Saves per-row progress to resume interrupted runs.

    Parameters:
        prompts (pd.DataFrame): The DataFrame containing prompts.
        system_message_field (str): The column name indicating the system message.
        user_message_field (str): The column name indicating the user message.
        experiment_round (str): The round of the experiment.
        experiment_version (str): The experiment/model version.
        model_name (str): Backend identifier — "together_logit" or "hf_logit".
        model_id (str): For "together_logit", the Together model name. For
            "hf_logit", the base URL of the HuggingFace dedicated endpoint
            (TGI ignores the `model` field).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens generated per query.
        logprobs_top_k: Number of top-k logprobs returned at each position.

    Returns:
        pd.DataFrame: The prompts with the corresponding LLM responses.
    """
    current_dir = os.path.dirname(__file__)
    progress_dir = os.path.join(
        current_dir, f"../../outputs/logs/inference/{experiment_round}/progress"
    )
    progress_file = os.path.join(progress_dir, f"{experiment_version}.csv")

    os.makedirs(progress_dir, exist_ok=True)

    prompts["ID"] = prompts["ID"].astype(str)
    if os.path.exists(progress_file):
        processed_prompts = pd.read_csv(progress_file)
        processed_prompts["ID"] = processed_prompts["ID"].astype(str)
        prompts = prompts.merge(
            processed_prompts[["ID", "llm_response"]], on="ID", how="left"
        )
    else:
        prompts["llm_response"] = None

    def _save_progress(row: pd.Series, result: str) -> str:
        row["llm_response"] = result
        row.to_frame().T.to_csv(
            progress_file,
            mode="a",
            header=not os.path.exists(progress_file),
            index=False,
        )
        return result

    def together_logit_query(row: pd.Series):
        """Query Together AI for one prompt row and capture per-position logprobs."""
        if not pd.isnull(row["llm_response"]):
            return row["llm_response"]

        messages = [
            {"role": "system", "content": row[system_message_field]},
            {"role": "user", "content": row[user_message_field]},
        ]

        create_kwargs = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "logprobs": logprobs_top_k,
            "temperature": temperature,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**create_kwargs)

        actual_response = response.choices[0].message.content

        # Together's chat API with logprobs=k returns, for each generated
        # position, the sampled token (tokens[i]/token_logprobs[i]) plus the
        # top-k most-likely alternatives (top_logprobs[i]). Store per-position
        # so the parser can scan the sequence for Yes/No.
        per_position_logprobs = []
        logprobs_obj = response.choices[0].logprobs
        if logprobs_obj:
            tokens = getattr(logprobs_obj, "tokens", None) or []
            token_logprobs = getattr(logprobs_obj, "token_logprobs", None) or []
            top_lp = getattr(logprobs_obj, "top_logprobs", None) or []

            for i, sampled_token in enumerate(tokens):
                sampled_logprob = token_logprobs[i] if i < len(token_logprobs) else None
                entries = []
                seen_tokens = set()

                def _add(tok, lp):
                    if tok is None or lp is None:
                        return
                    key = tok.strip().lower()
                    if key in seen_tokens:
                        return
                    seen_tokens.add(key)
                    entries.append({"token": tok, "logprob": lp})

                _add(sampled_token, sampled_logprob)
                if i < len(top_lp):
                    first_pos = top_lp[i]
                    if isinstance(first_pos, dict):
                        for tok, lp in first_pos.items():
                            _add(tok, lp)
                    elif isinstance(first_pos, list):
                        for item in first_pos:
                            if isinstance(item, dict):
                                _add(item.get("token"), item.get("logprob"))

                per_position_logprobs.append(
                    {
                        "sampled_token": sampled_token,
                        "sampled_logprob": sampled_logprob,
                        "top_logprobs": entries,
                    }
                )

        result = json.dumps(
            {
                "response": actual_response,
                "per_position_logprobs": per_position_logprobs,
            }
        )
        return _save_progress(row, result)

    def hf_logit_query(row: pd.Series):
        """Query a HuggingFace dedicated endpoint (OpenAI-compatible TGI)."""
        if not pd.isnull(row["llm_response"]):
            return row["llm_response"]

        messages = [
            {"role": "system", "content": row[system_message_field]},
            {"role": "user", "content": row[user_message_field]},
        ]

        create_kwargs = {
            "model": hf_served_model,
            "messages": messages,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": logprobs_top_k,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(**create_kwargs)

        choice = response.choices[0]
        per_position_logprobs = _parse_openai_logprobs(choice.logprobs)
        result = json.dumps(
            {
                "response": choice.message.content,
                "per_position_logprobs": per_position_logprobs,
            }
        )
        return _save_progress(row, result)

    if model_name == "together_logit":
        client = Together(api_key=TOGETHER_API_KEY)
        prompts["llm_response"] = prompts.progress_apply(together_logit_query, axis=1)
    elif model_name == "hf_logit":
        client = OpenAI(
            base_url=model_id.rstrip("/") + "/v1",
            api_key=HF_TOKEN,
        )
        models_listed = client.models.list().data
        if not models_listed:
            raise RuntimeError(
                f"Endpoint {model_id} returned no models at /v1/models. "
                "Check the endpoint is running and HF_TOKEN is valid."
            )
        hf_served_model = models_listed[0].id
        prompts["llm_response"] = prompts.progress_apply(hf_logit_query, axis=1)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return prompts
