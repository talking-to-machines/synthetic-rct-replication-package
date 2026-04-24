"""Instruction-tune a base pretrain model via Together AI.

Generalises over any base-model entry in config.yaml (--model-key) and any
dataset registered under `instruction_tuning.datasets` in config.yaml
(--dataset). Defaults target `llama_8b_base` on the Alpaca dataset.

Reads:  config.yaml (training, lora, instruction_tuning blocks)
        {dataset.data_file}
Writes: {dataset.output_jsonl}
        outputs/logs/training/{model_key}_{dataset}_ft_job.pkl

Training data is emitted as {"messages": [system, user, assistant]} JSONL
(Together AI's chat-messages format) so the base pretrain learns the
instruct-style conversation template during LoRA fine-tuning.

Config resolution order (most specific wins):
  top-level training/lora defaults
  <- models.{model_key}.{training,lora}
  <- instruction_tuning.datasets.{dataset}.{training,lora}
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import together
import yaml

from src.utils.config import TOGETHER_API_KEY


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_params(cfg: dict, model_key: str, dataset_key: str) -> dict:
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


def format_instruction_messages(record: dict, system_prompt: str) -> list[dict]:
    instruction = record["instruction"].strip()
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"].strip()},
    ]


def build_jsonl(src_json: Path, dst_jsonl: Path, system_prompt: str) -> int:
    with open(src_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            messages = format_instruction_messages(r, system_prompt)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    return len(records)


def launch_finetune(
    client: together.Together,
    file_id: str,
    params: dict,
    suffix: str,
) -> str:
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


def poll_until_done(
    client: together.Together,
    job_id: str,
    poll_interval: int = 60,
) -> str | None:
    while True:
        status = client.fine_tuning.retrieve(id=job_id)
        status_str = str(status.status).upper()
        print(f"  job {job_id}: {status_str}")
        if "COMPLETED" in status_str:
            return status.model_output_name
        if "FAILED" in status_str or "CANCELLED" in status_str:
            return None
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-key", type=str, default="llama_8b_base")
    parser.add_argument(
        "--dataset",
        type=str,
        default="alpaca",
        help="Dataset key under instruction_tuning.datasets in config.yaml.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Override dataset.data_file.",
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=None,
        help="Override dataset.output_jsonl.",
    )
    parser.add_argument("--job-pkl", type=Path, default=None)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = resolve_params(cfg, args.model_key, args.dataset)

    input_json = args.input_json or params["data_file"]
    train_jsonl = args.train_jsonl or params["output_jsonl"]
    job_pkl = args.job_pkl or Path(
        f"outputs/logs/training/{args.model_key}_{args.dataset}_ft_job.pkl"
    )

    print(f"Model: {args.model_key} -> {params['base_model']}")
    print(f"Dataset: {args.dataset} ({input_json})")
    print(f"Training: {params['training']}")
    print(f"LoRA: {params['lora']}")

    n = build_jsonl(input_json, train_jsonl, params["system_prompt"])
    print(f"Wrote {n} examples to {train_jsonl}")

    client = together.Together(api_key=TOGETHER_API_KEY)
    resp = client.files.upload(file=str(train_jsonl), purpose="fine-tune")
    print(f"Uploaded training file: {resp.id}")

    full_suffix = f"{args.model_key}_{args.dataset}"
    job_id = launch_finetune(client, resp.id, params, full_suffix)
    print(f"Launched fine-tuning job: {job_id}")

    job_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(job_pkl, "wb") as f:
        pickle.dump(
            {
                "job_id": job_id,
                "model_key": args.model_key,
                "dataset": args.dataset,
                "suffix": full_suffix,
                "base_model": params["base_model"],
                "params": {
                    k: v
                    for k, v in params.items()
                    if k not in ("data_file", "output_jsonl")
                },
            },
            f,
        )

    model_name = poll_until_done(client, job_id, args.poll_interval)
    if model_name is None:
        raise RuntimeError(f"Fine-tuning job {job_id} failed or was cancelled.")
    print(f"Fine-tuned model: {model_name}")


if __name__ == "__main__":
    main()
