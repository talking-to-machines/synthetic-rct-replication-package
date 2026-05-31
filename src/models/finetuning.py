import time
from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

from src.utils.config import AWS_DEFAULT_REGION, HF_TOKEN, HF_USERNAME, SM_ROLE_ARN

# `src/` is uploaded to /opt/ml/code as source_dir; train.py is the entry_point
# and self-dispatches between launcher mode and in-container TRL mode.
SAGEMAKER_SOURCE_DIR = Path(__file__).resolve().parents[1]


def resolve_params(cfg: dict, model_key: str) -> dict:
    """Merge top-level and per-model fine-tuning overrides.

    Args:
        cfg: Parsed `config.yaml` dict. Must contain `models`, `training`,
            `lora`, and `sagemaker` blocks.
        model_key: Key under `cfg["models"]` (e.g. `"llama_8b"`).

    Returns:
        Dict with `base_model`, `family`, `training`, `lora`,
        `training_instance_type`, and `sagemaker` keys.

    Raises:
        KeyError: If `model_key` is not registered, or the model is missing
            `training_instance_type`.
        ValueError: If the model is not declared as `type: open`.
    """
    if model_key not in cfg["models"]:
        raise KeyError(
            f"Model {model_key!r} not in config.yaml. "
            f"Known: {sorted(cfg['models'])}"
        )
    model_cfg = cfg["models"][model_key]
    if model_cfg.get("type") != "open":
        raise ValueError(
            f"Model {model_key!r} is not an open-source model and cannot be fine-tuned."
        )

    training = {**cfg["training"], **model_cfg.get("training", {})}
    lora = {**cfg["lora"], **model_cfg.get("lora", {})}

    instance_type = model_cfg.get("training_instance_type")
    if not instance_type:
        raise KeyError(
            f"Model {model_key!r} is missing `training_instance_type` in config.yaml."
        )

    return {
        "base_model": model_cfg["base_model"],
        "family": model_cfg.get("family", ""),
        "training": training,
        "lora": lora,
        "training_instance_type": instance_type,
        "sagemaker": cfg.get("sagemaker", {}),
    }


def _resolve_training_image(sm_cfg: dict, region: str) -> str:
    """Substitute `{region}` into the configured training-DLC URI.

    Args:
        sm_cfg: The `sagemaker` block from config.yaml.
        region: AWS region for the DLC's ECR registry.

    Returns:
        Fully-qualified training-DLC image URI.

    Raises:
        KeyError: If `training_image_uri` is missing from `sm_cfg`.
    """
    template = sm_cfg.get("training_image_uri", "")
    if not template:
        raise KeyError("`sagemaker.training_image_uri` missing from config.yaml.")
    return template.format(region=region)


def launch_finetune(
    params: dict,
    training_jsonl: Path,
    model_key: str,
    rct_id: str | None = None,
) -> tuple[str, "HuggingFace"]:
    """Submit a SageMaker fine-tuning job for `params["base_model"]`.

    Uploads `training_jsonl` to S3, then submits a HuggingFace estimator job
    whose entry_point is `src/train.py` running in its in-container branch.
    That branch merges the LoRA adapter into the base model and pushes the
    merged model to `{HF_USERNAME}/{model_key}-{rct_id}` on the Hub.

    Args:
        params: Output of `resolve_params(...)`.
        training_jsonl: Local path to the combined fine-tuning corpus.
        model_key: Key under `cfg["models"]`. Used in the Hub repo name and
            the SageMaker job name.
        rct_id: Optional RCT identifier appended to the Hub repo name.

    Returns:
        `(job_name, estimator)`. `job_name` is the SageMaker training job
        name, which the caller persists for resumption / polling.

    Raises:
        RuntimeError: If `HF_TOKEN`, `HF_USERNAME`, or `SM_ROLE_ARN` is unset.
    """
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is unset. Set it in your .env / environment.")
    if not HF_USERNAME:
        raise RuntimeError("HF_USERNAME is unset. Set it in your .env / environment.")
    if not SM_ROLE_ARN:
        raise RuntimeError("SM_ROLE_ARN is unset. Set it in your .env / environment.")

    training = params["training"]
    lora = params["lora"]
    sm_cfg = params["sagemaker"]
    instance_type = params["training_instance_type"]

    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
    sess = sagemaker.Session(boto_session=boto_session)
    bucket = sess.default_bucket()

    base_model = params["base_model"]
    model_slug = base_model.split("/")[-1]
    s3_prefix = sm_cfg.get("s3_prefix", "synthetic-rct")
    s3_key_prefix = f"{s3_prefix}/{rct_id or 'combined'}/{model_slug}/train"

    s3_train_uri = sess.upload_data(
        path=str(training_jsonl),
        bucket=bucket,
        key_prefix=s3_key_prefix,
    )
    print(f"Training data uploaded to {s3_train_uri}")

    hub_repo_suffix = f"-{rct_id}" if rct_id else ""
    hub_model_id = f"{HF_USERNAME}/{model_key}{hub_repo_suffix}"

    hyperparameters = {
        "base_model": base_model,
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "gradient_accumulation": training.get("gradient_accumulation_steps", 1),
        "lr": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_ratio": training["warmup_ratio"],
        "max_grad_norm": training["max_grad_norm"],
        "max_seq_length": training["max_seq_length"],
        "seed": (training.get("seeds") or [42])[0],
        "lora_r": lora["r"],
        "lora_alpha": lora["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": ",".join(lora["target_modules"]),
        "hub_model_id": hub_model_id,
    }
    if training.get("gradient_checkpointing"):
        hyperparameters["gradient_checkpointing"] = "true"
    if training.get("fsdp"):
        hyperparameters["fsdp"] = "true"

    # SageMaker training-job names must match [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62};
    # config keys (e.g. "olmo2_7b") contain underscores so we sanitise here.
    base_job_name = f"{model_key}{hub_repo_suffix}".replace("_", "-").replace(".", "")[
        :50
    ]

    estimator_kwargs = dict(
        entry_point="train.py",
        source_dir=str(SAGEMAKER_SOURCE_DIR),
        image_uri=_resolve_training_image(sm_cfg, AWS_DEFAULT_REGION),
        py_version=sm_cfg.get("training_py_version", "py312"),
        instance_type=instance_type,
        instance_count=1,
        role=SM_ROLE_ARN,
        hyperparameters=hyperparameters,
        environment={"HF_TOKEN": HF_TOKEN},
        max_run=sm_cfg.get("max_run_seconds", 6 * 60 * 60),
        sagemaker_session=sess,
        base_job_name=base_job_name,
    )
    # FSDP requires torchrun (one process per GPU); otherwise the in-container
    # Trainer raises "Using fsdp only works in distributed training."
    if training.get("fsdp"):
        estimator_kwargs["distribution"] = {"torch_distributed": {"enabled": True}}

    estimator = HuggingFace(**estimator_kwargs)
    estimator.fit({"training": s3_train_uri}, wait=False)
    job_name = estimator.latest_training_job.name
    print(f"Launched SageMaker training job: {job_name}")
    print(f"  Hub repo (will be pushed by job): https://huggingface.co/{hub_model_id}")
    return job_name, estimator


def poll_finetune_until_done(
    job_name: str,
    poll_interval: int = 60,
) -> dict | None:
    """Block until a SageMaker training job reaches a terminal state.

    Polls `describe_training_job` every `poll_interval` seconds and prints
    status transitions.

    Args:
        job_name: Training job name returned by `launch_finetune`.
        poll_interval: Seconds to sleep between status checks.

    Returns:
        Dict with `s3_model_artifact` on success, or None if the job failed
        or was stopped.
    """
    boto_session = boto3.Session(region_name=AWS_DEFAULT_REGION)
    sm_client = boto_session.client("sagemaker")

    terminal = {"Completed", "Failed", "Stopped"}
    last = None
    while True:
        desc = sm_client.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus")
        if (status, secondary) != last:
            print(f"  job {job_name}: {status} / {secondary}")
            last = (status, secondary)
        if status in terminal:
            break
        time.sleep(poll_interval)

    if status != "Completed":
        print(f"  FailureReason: {desc.get('FailureReason')}")
        return None

    return {"s3_model_artifact": desc["ModelArtifacts"]["S3ModelArtifacts"]}
