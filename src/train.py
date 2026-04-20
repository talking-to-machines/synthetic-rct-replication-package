"""LoRA fine-tuning entry point (Together AI).

Reads:  data/finetuning/train.jsonl
Writes: outputs/adapters/, outputs/logs/training/

Extracted from archived notebooks `finetune_llama3.1_8b.ipynb` and
`finetune_llama3.1_70b.ipynb`. Those scripts launched a grid of jobs over
(epochs, lora_r) combinations on Together AI; the entry point here keeps
that flow but parameterises the base model.
"""

import pickle
import time
from itertools import product

import together

from src.models.lora import LoRAConfig, TrainingConfig
from src.utils.config import TOGETHER_API_KEY


def run_together_finetune_grid(
    train_file_path: str,
    base_model: str,
    epochs_list: list,
    r_list: list,
    output_pkl: str,
    suffix_prefix: str = "duch2023",
    poll_interval: int = 60,
) -> dict:
    """Launch a grid of Together AI fine-tuning jobs and poll until all finish."""
    client = together.Together(api_key=TOGETHER_API_KEY)

    resp = client.files.upload(file=train_file_path, purpose="fine-tune")
    print(f"File ID: {resp.id}")

    ft_jobs = {}
    for n_epochs, r in product(epochs_list, r_list):
        lora = LoRAConfig(lora_r=r, lora_alpha=2 * r, lora_dropout=0.05)
        schedule = TrainingConfig(n_epochs=n_epochs)
        suffix = f"{suffix_prefix}_ep{n_epochs}_r{r}"

        ft_resp = client.fine_tuning.create(
            training_file=resp.id,
            model=base_model,
            suffix=suffix,
            **schedule.to_dict(),
            **lora.to_dict(),
        )
        ft_jobs[suffix] = ft_resp.id
        print(f"Created job: {suffix} -> {ft_resp.id}")

    with open(output_pkl, "wb") as f:
        pickle.dump(ft_jobs, f)
    print(f"\nTotal jobs created: {len(ft_jobs)}")

    ft_models = {}
    while True:
        all_done = True
        for key, job_id in ft_jobs.items():
            if key in ft_models:
                continue
            status = client.fine_tuning.retrieve(id=job_id)
            status_str = str(status.status).upper()
            print(f"  {key} (job {job_id}): {status_str}")
            if "COMPLETED" in status_str:
                ft_models[key] = status.model_output_name
            elif "FAILED" in status_str or "CANCELLED" in status_str:
                ft_models[key] = None
            else:
                all_done = False
        if all_done:
            break
        time.sleep(poll_interval)

    return ft_models


if __name__ == "__main__":
    raise NotImplementedError(
        "Wire this entry point to config.yaml to launch fine-tuning for each model."
    )
