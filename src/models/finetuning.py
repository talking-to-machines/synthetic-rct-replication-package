"""Together AI fine-tuning helpers.

Config resolution merges top-level training/lora defaults with per-model
overrides. The launch and poll helpers wrap Together AI's fine-tuning API
so train.py can stay thin.
"""

import time
import together


def resolve_params(cfg: dict, model_key: str) -> dict:
    """Merge top-level and per-model training/lora overrides for fine-tuning.

    Looks up the model entry under `cfg["models"][model_key]`, validates it
    is open-source (only open models can be fine-tuned via Together AI), and
    returns the merged hyperparameter dict used by `launch_finetune`.

    Args:
        cfg: Parsed `config.yaml` dict (must contain `models`, `training`,
            and `lora` blocks).
        model_key: Key under `cfg["models"]` (e.g. `"llama_8b"`).

    Returns:
        Dict with `base_model` (str), `training` (dict), and `lora` (dict)
        keys, ready to pass to `launch_finetune`.

    Raises:
        KeyError: If `model_key` is not registered under `cfg["models"]`.
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

    training = {**cfg["training"]}
    lora = {**cfg["lora"], **model_cfg.get("lora", {})}
    return {
        "base_model": model_cfg["base_model"],
        "training": training,
        "lora": lora,
    }


def launch_finetune(
    client: together.Together,
    file_id: str,
    params: dict,
    suffix: str,
) -> str:
    """Launch a Together AI LoRA fine-tuning job.

    Args:
        client: Authenticated Together AI client.
        file_id: ID of the previously uploaded training file (returned by
            `client.files.upload`).
        params: Hyperparameter dict from `resolve_params` containing
            `base_model`, `training`, and `lora` sub-dicts.
        suffix: Suffix appended to the resulting fine-tuned model name.

    Returns:
        The Together AI fine-tuning job id, used for polling and retrieval.
    """
    training = params["training"]
    lora = params["lora"]
    modules_str = ",".join(lora["target_modules"])

    ft_resp = client.fine_tuning.create(
        training_file=file_id,
        model=params["base_model"],
        suffix=suffix,
        n_epochs=training["epochs"],
        n_checkpoints=training["n_checkpoints"],
        n_evals=training["n_evals"],
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        lora=True,
        lora_r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        lora_trainable_modules=modules_str,
        train_on_inputs=training["train_on_inputs"],
    )
    return ft_resp.id


def poll_finetune_until_done(
    client: together.Together,
    job_id: str,
    poll_interval: int = 60,
) -> str | None:
    """Block until a Together AI fine-tuning job reaches a terminal state.

    Polls `client.fine_tuning.retrieve` every `poll_interval` seconds and
    prints status transitions. Returns when the job has completed, failed,
    or been cancelled.

    Args:
        client: Authenticated Together AI client.
        job_id: Fine-tuning job id returned by `launch_finetune`.
        poll_interval: Seconds to sleep between status checks.

    Returns:
        The fine-tuned model name (`status.model_output_name`) on success,
        or None if the job failed or was cancelled.
    """
    while True:
        status = client.fine_tuning.retrieve(id=job_id)
        status_str = str(status.status).upper()
        print(f"  job {job_id}: {status_str}")
        if "COMPLETED" in status_str:
            return status.model_output_name
        if "FAILED" in status_str or "CANCELLED" in status_str:
            return None
        time.sleep(poll_interval)
