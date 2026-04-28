"""LoRA fine-tune a model on the combined survey + RCT corpus via Together AI.

Generalises over any open-model entry in config.yaml (--model-key). Sources
contributing to the corpus are listed in `finetuning.{surveys,rcts}`. For each
source, this script reads its `data_file` and `prompt_file` from config.yaml,
formats per-subject {"messages": [...]} records, randomly splits them into
train/test using `finetuning.test_fraction` (and `finetuning.seed`), writes
per-source JSONLs to `data/processed/{kind}/{id}/`, then concatenates the
per-source training JSONLs into a single corpus before launching the job.

Reads:  config.yaml (training, lora, finetuning, {rcts,surveys} blocks)
        {source.data_file}, {source.prompt_file}
Writes: data/processed/{kind}/{id}/{id}_train.jsonl
        data/processed/{kind}/{id}/{id}_test.jsonl
        data/finetuning/train.jsonl
        outputs/logs/training/{model_key}_ft_job.pkl

Config resolution order (most specific wins):
  top-level training/lora defaults
  <- models.{model_key}.{training,lora}
"""

import argparse
import pickle
from pathlib import Path

import together

from src.build_corpus import build_finetune_corpus
from src.models.finetuning import (
    launch_finetune,
    poll_finetune_until_done,
    resolve_train_params,
)
from src.utils.config import TOGETHER_API_KEY
from src.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-key", type=str, default="llama_8b")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/finetuning/train.jsonl"),
        help="Combined corpus output path.",
    )
    parser.add_argument("--job-pkl", type=Path, default=None)
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    params = resolve_train_params(cfg, args.model_key)
    job_pkl = args.job_pkl or Path(f"outputs/logs/training/{args.model_key}_ft_job.pkl")

    print(f"Model: {args.model_key} -> {params['base_model']}")
    print(f"Training: {params['training']}")
    print(f"LoRA: {params['lora']}")

    n = build_finetune_corpus(cfg, args.output_jsonl)
    if n == 0:
        raise RuntimeError(
            f"No training examples produced. Check finetuning sources in {args.config}."
        )
    print(f"Wrote {n} examples to {args.output_jsonl}")

    client = together.Together(api_key=TOGETHER_API_KEY)
    resp = client.files.upload(file=str(args.output_jsonl), purpose="fine-tune")
    print(f"Uploaded training file: {resp.id}")

    job_id = launch_finetune(client, resp.id, params, args.model_key)
    print(f"Launched fine-tuning job: {job_id}")

    job_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(job_pkl, "wb") as f:
        pickle.dump(
            {
                "job_id": job_id,
                "model_key": args.model_key,
                "suffix": args.model_key,
                "base_model": params["base_model"],
                "params": params,
            },
            f,
        )

    model_name = poll_finetune_until_done(client, job_id, args.poll_interval)
    if model_name is None:
        raise RuntimeError(f"Fine-tuning job {job_id} failed or was cancelled.")
    print(f"Fine-tuned model: {model_name}")


if __name__ == "__main__":
    main()
