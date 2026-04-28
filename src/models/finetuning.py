"""Together AI fine-tuning helpers.

Config resolution merges top-level training/lora defaults with per-model
(and, for instruction tuning, per-dataset) overrides. The launch and poll
helpers wrap Together AI's fine-tuning API so train.py and
instruction_tuning.py can stay thin.
"""

import time
from pathlib import Path

import together


def resolve_train_params(cfg: dict, model_key: str) -> dict:
    """Merge top-level and per-model training/lora overrides for fine-tuning."""
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


def resolve_instruction_tune_params(
    cfg: dict, model_key: str, dataset_key: str
) -> dict:
    """Merge top-level, per-model, and per-dataset training/lora overrides."""
    if model_key not in cfg["models"]:
        raise KeyError(
            f"Model {model_key!r} not in config.yaml. "
            f"Known: {sorted(cfg['models'])}"
        )
    it_cfg = cfg.get("instruction_tuning", {})
    datasets = it_cfg.get("datasets", {})
    if dataset_key not in datasets:
        raise KeyError(
            f"Dataset {dataset_key!r} not in config.yaml instruction_tuning.datasets. "
            f"Known: {sorted(datasets)}"
        )

    model_cfg = cfg["models"][model_key]
    dataset_cfg = datasets[dataset_key]

    training = {
        **cfg["training"],
        **model_cfg.get("training", {}),
        **dataset_cfg.get("training", {}),
    }
    lora = {
        **cfg["lora"],
        **model_cfg.get("lora", {}),
        **dataset_cfg.get("lora", {}),
    }
    return {
        "base_model": model_cfg["base_model"],
        "training": training,
        "lora": lora,
        "data_file": Path(dataset_cfg["data_file"]),
        "output_jsonl": Path(dataset_cfg["output_jsonl"]),
        "system_prompt": it_cfg.get(
            "system_prompt", "You are a helpful assistant that follows instructions."
        ),
    }


def launch_finetune(
    client: together.Together,
    file_id: str,
    params: dict,
    suffix: str,
) -> str:
    """Launch a Together AI LoRA fine-tuning job and return the job id."""
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
    """Poll a Together AI fine-tuning job until terminal. Returns the model name on success."""
    while True:
        status = client.fine_tuning.retrieve(id=job_id)
        status_str = str(status.status).upper()
        print(f"  job {job_id}: {status_str}")
        if "COMPLETED" in status_str:
            return status.model_output_name
        if "FAILED" in status_str or "CANCELLED" in status_str:
            return None
        time.sleep(poll_interval)
