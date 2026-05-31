import json
import os
import time
import boto3
import pandas as pd
import sagemaker
from botocore.exceptions import ClientError
from openai import OpenAI
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from tqdm import tqdm
from src.utils.config import AWS_DEFAULT_REGION, HF_TOKEN, SM_ROLE_ARN

tqdm.pandas()


def create_batch_file(
    prompts: pd.DataFrame,
    system_message_field: str,
    user_message_field: str = "question_prompt",
    batch_file_name: str = "batch_tasks.jsonl",
    logit: bool = False,
    model: str = "gpt-4o-2024-08-06",
) -> str:
    """Create a JSONL batch file for the OpenAI Batch API.

    Args:
        prompts: DataFrame with one row per request; must contain `custom_id`
            and the columns named by `system_message_field` and
            `user_message_field`.
        system_message_field: Column holding the system message.
        user_message_field: Column holding the user message.
        batch_file_name: Filename written under `batch_files/`.
        logit: When True, sets `max_tokens=1`, `logprobs=True`, `top_logprobs=5`
            for a single-token logit pass.
        model: Model id placed in each request body.

    Returns:
        Absolute path to the written JSONL batch file.
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
        tasks.append(
            {
                "custom_id": f'{prompts.loc[i, "custom_id"]}',
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )

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
    """Submit an OpenAI batch job and return responses once it completes.

    Polls the batch endpoint until the job is `completed`, then writes the
    raw output JSONL alongside the input and returns a parsed DataFrame.

    Args:
        client: Authenticated `openai.OpenAI` client.
        batch_input_file_dir: Path to the JSONL file produced by
            `create_batch_file`.
        batch_output_file_dir: Filename for the downloaded output JSONL,
            written under `batch_files/`.
        logit: When True, includes top-k logprobs from the first generated
            token in the returned `query_response` JSON string.

    Returns:
        DataFrame with columns `custom_id` and `query_response`.

    Raises:
        Exception: If the batch job reaches the `failed` status.
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

    results = client.files.content(batch_job.output_file_id).content

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
                            {"token": item["token"], "logprob": item["logprob"]}
                        )
                query_response = json.dumps(
                    {"response": actual_response, "top_logprobs": top_logprobs_data}
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


def _delete_leftover_endpoint(sm_client, name: str) -> None:
    """Remove an endpoint and its config + model objects if they exist.

    Used at deploy time to recover from a previously failed CreateEndpoint,
    which can leave the three objects stranded under the same name.

    Args:
        sm_client: A boto3 `sagemaker` client.
        name: The shared name of the endpoint, endpoint-config, and model.
    """
    for delete, kwargs in [
        (sm_client.delete_endpoint, {"EndpointName": name}),
        (sm_client.delete_endpoint_config, {"EndpointConfigName": name}),
        (sm_client.delete_model, {"ModelName": name}),
    ]:
        try:
            delete(**kwargs)
        except ClientError as e:
            msg = str(e)
            if "Could not find" not in msg and "does not exist" not in msg:
                raise


def deploy_sagemaker_endpoint(
    huggingface_model_id: str,
    endpoint_name: str,
    instance_type: str,
    max_input_tokens: int = 2048,
    max_total_tokens: int = 4096,
    dtype: str = "bfloat16",
    startup_timeout: int = 900,
    num_shard: int | None = None,
) -> tuple:
    """Deploy `huggingface_model_id` on a real-time SageMaker TGI endpoint.

    TGI pulls `HF_MODEL_ID` from the Hub at container start and exposes an
    OpenAI-compatible chat-completions API. Blocks until the endpoint is
    InService.

    Args:
        huggingface_model_id: Hub repo id served via `HF_MODEL_ID`.
        endpoint_name: Stable SageMaker endpoint name (also used for the
            EndpointConfig and Model objects).
        instance_type: SageMaker inference instance type (e.g. `ml.g5.2xlarge`).
        max_input_tokens: Per-request input cap passed as `MAX_INPUT_TOKENS`.
        max_total_tokens: Per-request input+output cap passed as
            `MAX_TOTAL_TOKENS`; also used for `MAX_BATCH_PREFILL_TOKENS`.
        dtype: TGI `DTYPE` env var. Typically `bfloat16` for our models.
        startup_timeout: Seconds allowed for the container to reach a healthy
            state (large models need >10 min for weight download + warmup).
        num_shard: Number of GPUs for tensor parallelism (TGI `NUM_SHARD`).
            When None, TGI auto-detects from `CUDA_VISIBLE_DEVICES`.

    Returns:
        `(predictor, endpoint_name)` where `predictor.delete_endpoint()`
        tears down the endpoint, config, and model objects in one call.

    Raises:
        RuntimeError: If `SM_ROLE_ARN` or `HF_TOKEN` is unset.
    """
    if not SM_ROLE_ARN:
        raise RuntimeError("SM_ROLE_ARN is unset. Set it in your .env / environment.")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is unset. Set it in your .env / environment.")

    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
    sess = sagemaker.Session(boto_session=boto_session)

    sm_client = boto_session.client("sagemaker")
    _delete_leftover_endpoint(sm_client, endpoint_name)

    tgi_image = get_huggingface_llm_image_uri("huggingface", session=sess)

    tgi_env = {
        "HF_MODEL_ID": huggingface_model_id,
        "HF_TOKEN": HF_TOKEN,
        "DTYPE": dtype,
        "MAX_INPUT_TOKENS": str(max_input_tokens),
        "MAX_TOTAL_TOKENS": str(max_total_tokens),
        "MAX_BATCH_PREFILL_TOKENS": str(max_total_tokens),
    }
    if num_shard is not None:
        tgi_env["NUM_SHARD"] = str(num_shard)

    hf_model = HuggingFaceModel(
        image_uri=tgi_image,
        role=SM_ROLE_ARN,
        env=tgi_env,
        sagemaker_session=sess,
    )

    print(f"Deploying {huggingface_model_id} -> {endpoint_name} ({instance_type})")
    predictor = hf_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=startup_timeout,
    )
    print(f"Endpoint InService: {endpoint_name}")
    return predictor, endpoint_name


def _parse_invoke_logprobs(logprobs_obj) -> list:
    """Convert TGI's chat-completion logprobs into the per-position schema.

    Args:
        logprobs_obj: The `logprobs` dict from a TGI chat-completion choice,
            shaped `{"content": [{"token", "logprob", "top_logprobs": [...]}, ...]}`.

    Returns:
        A list of `{sampled_token, sampled_logprob, top_logprobs}` dicts,
        one per generated position. Empty if `logprobs_obj` is falsy.
    """
    per_position = []
    if not logprobs_obj:
        return per_position
    for tok in logprobs_obj.get("content", []) or []:
        entries = []
        seen = set()

        def _add(t, lp):
            """Append a `(token, logprob)` pair, deduped by token (case-insensitive)."""
            if t is None or lp is None:
                return
            key = t.strip().lower()
            if key in seen:
                return
            seen.add(key)
            entries.append({"token": t, "logprob": lp})

        _add(tok.get("token"), tok.get("logprob"))
        for alt in tok.get("top_logprobs") or []:
            _add(alt.get("token"), alt.get("logprob"))

        per_position.append(
            {
                "sampled_token": tok.get("token"),
                "sampled_logprob": tok.get("logprob"),
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
    endpoint_name: str,
    temperature: float = 1.0,
    max_tokens: int = 1,
    logprobs_top_k: int = 5,
) -> pd.DataFrame:
    """Query an InService SageMaker TGI endpoint, one prompt per row.

    Persists per-row results to
    `outputs/logs/inference/{experiment_round}/progress/{experiment_version}.csv`
    so a re-invocation with the same arguments skips already-completed rows.

    Args:
        prompts: DataFrame containing `ID`, `system_message_field`, and
            `user_message_field`. Mutated in place to add an `llm_response`
            column.
        system_message_field: Column holding the system message.
        user_message_field: Column holding the user message.
        experiment_round: RCT identifier used in the progress-file path.
        experiment_version: Experiment slug used in the progress-file name.
        endpoint_name: Name of an InService endpoint (deploy via
            `deploy_sagemaker_endpoint`).
        temperature: Sampling temperature passed to the chat-completion call.
        max_tokens: Maximum tokens generated per query.
        logprobs_top_k: Number of top-k logprobs returned per position.

    Returns:
        The input DataFrame with an `llm_response` column whose values are
        JSON strings `{"response": str, "per_position_logprobs": [...]}`.
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

    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
    sm_runtime = boto_session.client("sagemaker-runtime")

    def _save_progress(row: pd.Series, result: str) -> str:
        """Append `result` to the progress CSV and return it for assignment."""
        row["llm_response"] = result
        row.to_frame().T.to_csv(
            progress_file,
            mode="a",
            header=not os.path.exists(progress_file),
            index=False,
        )
        return result

    def sagemaker_logit_query(row: pd.Series):
        """Invoke the endpoint for one prompt row; skip if already answered."""
        if not pd.isnull(row["llm_response"]):
            return row["llm_response"]

        payload = {
            "messages": [
                {"role": "system", "content": row[system_message_field]},
                {"role": "user", "content": row[user_message_field]},
            ],
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": logprobs_top_k,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        body = json.loads(resp["Body"].read())
        choice = body["choices"][0]
        per_position = _parse_invoke_logprobs(choice.get("logprobs"))
        result = json.dumps(
            {
                "response": choice["message"]["content"],
                "per_position_logprobs": per_position,
            }
        )
        return _save_progress(row, result)

    prompts["llm_response"] = prompts.progress_apply(sagemaker_logit_query, axis=1)
    return prompts


def delete_sagemaker_endpoint(predictor=None, endpoint_name: str | None = None) -> None:
    """Tear down a SageMaker endpoint and its config + model objects.

    Args:
        predictor: Predictor returned by `deploy_sagemaker_endpoint`. Calls
            `predictor.delete_endpoint()` which cleans up all three objects.
        endpoint_name: Endpoint name to delete via the boto SageMaker client.
            Use when the predictor handle is unavailable. No-op on a 404.

    Raises:
        ValueError: If neither `predictor` nor `endpoint_name` is provided.
    """
    if predictor is not None:
        predictor.delete_endpoint()
        return
    if endpoint_name is None:
        raise ValueError("Provide either `predictor` or `endpoint_name`.")
    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
    sm_client = boto_session.client("sagemaker")
    _delete_leftover_endpoint(sm_client, endpoint_name)
