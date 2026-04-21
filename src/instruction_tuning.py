"""Instruction-tune a base pretrain model via Together AI.

Generalises over any base-model entry in config.yaml (resolved by --model-key).
Defaults target `llama_8b_base` on the Alpaca dataset.

Reads:  data/human/alpaca/alpaca_data.json
        config.yaml
Writes: data/finetuning/alpaca/alpaca_train.jsonl
        outputs/logs/training/{model_key}_{suffix}_ft_job.pkl

Training data is emitted as {"messages": [system, user, assistant]} JSONL
(Together AI's chat-messages format) so the base pretrain learns the
instruct-style conversation template during LoRA fine-tuning.
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import together
import yaml

from src.utils.config import TOGETHER_API_KEY


SYSTEM_PROMPT = "You are a helpful assistant that follows instructions."


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_model_params(cfg: dict, model_key: str) -> dict:
    if model_key not in cfg["models"]:
        raise KeyError(
            f"Model {model_key!r} not in config.yaml. "
            f"Known: {sorted(cfg['models'])}"
        )
    model_cfg = cfg["models"][model_key]
    training = {**cfg["training"], **model_cfg.get("training", {})}
    lora = {**cfg["lora"], **model_cfg.get("lora", {})}
    return {"base_model": model_cfg["base_model"], "training": training, "lora": lora}


def format_alpaca_messages(record: dict) -> list[dict]:
    instruction = record["instruction"].strip()
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": record["output"].strip()},
    ]


def build_jsonl(src_json: Path, dst_jsonl: Path) -> int:
    with open(src_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            messages = format_alpaca_messages(r)
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
        n_checkpoints=1,
        n_evals=0,
        batch_size=training["batch_size"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        max_grad_norm=1.0,
        lora_r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        lora_trainable_modules=modules_str,
        train_on_inputs="auto",
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
        "--input-json", type=Path, default=Path("data/human/alpaca/alpaca_data.json")
    )
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("data/finetuning/alpaca/alpaca_train.jsonl"),
    )
    parser.add_argument("--suffix", type=str, default="alpaca")
    parser.add_argument("--job-pkl", type=Path, default=None)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    job_pkl = args.job_pkl or Path(
        f"outputs/logs/training/{args.model_key}_{args.suffix}_ft_job.pkl"
    )

    cfg = load_config(args.config)
    params = resolve_model_params(cfg, args.model_key)
    print(f"Model: {args.model_key} -> {params['base_model']}")
    print(f"Training: {params['training']}")
    print(f"LoRA: {params['lora']}")

    n = build_jsonl(args.input_json, args.train_jsonl)
    print(f"Wrote {n} examples to {args.train_jsonl}")

    client = together.Together(api_key=TOGETHER_API_KEY)
    resp = client.files.upload(file=str(args.train_jsonl), purpose="fine-tune")
    print(f"Uploaded training file: {resp.id}")

    full_suffix = f"{args.model_key}_{args.suffix}"
    job_id = launch_finetune(client, resp.id, params, full_suffix)
    print(f"Launched fine-tuning job: {job_id}")

    job_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(job_pkl, "wb") as f:
        pickle.dump(
            {
                "job_id": job_id,
                "model_key": args.model_key,
                "suffix": full_suffix,
                "base_model": params["base_model"],
                "params": params,
            },
            f,
        )

    model_name = poll_until_done(client, job_id, args.poll_interval)
    if model_name is None:
        raise RuntimeError(f"Fine-tuning job {job_id} failed or was cancelled.")
    print(f"Fine-tuned model: {model_name}")


if __name__ == "__main__":
    main()
