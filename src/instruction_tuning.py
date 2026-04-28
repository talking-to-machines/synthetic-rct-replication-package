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
import pickle
from pathlib import Path

import together

from src.build_corpus import build_instruction_corpus
from src.models.finetuning import (
    launch_finetune,
    poll_finetune_until_done,
    resolve_instruction_tune_params,
)
from src.utils.config import TOGETHER_API_KEY
from src.utils.io import load_yaml


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

    cfg = load_yaml(args.config)
    params = resolve_instruction_tune_params(cfg, args.model_key, args.dataset)

    input_json = args.input_json or params["data_file"]
    train_jsonl = args.train_jsonl or params["output_jsonl"]
    job_pkl = args.job_pkl or Path(
        f"outputs/logs/training/{args.model_key}_{args.dataset}_ft_job.pkl"
    )

    print(f"Model: {args.model_key} -> {params['base_model']}")
    print(f"Dataset: {args.dataset} ({input_json})")
    print(f"Training: {params['training']}")
    print(f"LoRA: {params['lora']}")

    n = build_instruction_corpus(input_json, train_jsonl, params["system_prompt"])
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

    model_name = poll_finetune_until_done(client, job_id, args.poll_interval)
    if model_name is None:
        raise RuntimeError(f"Fine-tuning job {job_id} failed or was cancelled.")
    print(f"Fine-tuned model: {model_name}")


if __name__ == "__main__":
    main()
