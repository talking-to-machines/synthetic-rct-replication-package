"""Build the fine-tuning corpus from survey training files and RCT training splits.

Reads:  data/processed/*/train.jsonl
Writes: data/finetuning/train.jsonl

Extracted from the archived notebook `prepare_fine_tuning_data.ipynb`.
"""

import json
import os

import pandas as pd

from src.data.formatting import (
    format_system_prompt,
    format_user_prompt,
    generate_profile_prompt,
)
from src.utils.io import save_jsonl
from src.utils.seed import RANDOM_STATE, set_seed

set_seed(RANDOM_STATE)


def build_rct_corpus(
    source_csv: str,
    output_dir: str,
    prompt_file: str,
    target_outcome: str,
) -> str:
    """Construct the JSONL fine-tuning corpus for an RCT.

    Reads profile_vars, system_template, user_template, and
    treatment (transcripts dict) from `prompt_file` (RCT prompt JSON).
    """
    with open(prompt_file) as f:
        prompt_cfg = json.load(f)
    profile_vars = prompt_cfg["profile_vars"]
    system_template = prompt_cfg["system_template"]
    user_template = prompt_cfg["user_template"]
    treatment_transcripts = prompt_cfg["treatment"]
    treatment_col = prompt_cfg.get("treatment_column", "treatment")

    data = pd.read_csv(source_csv, header=1)
    data[target_outcome] = data[target_outcome].replace("NA", None)

    training = data[profile_vars + [target_outcome, treatment_col]].copy()
    training["profile_prompt"] = training.apply(
        generate_profile_prompt, axis=1, args=([target_outcome, treatment_col],)
    )
    training["system_prompt"] = training.apply(
        format_system_prompt,
        axis=1,
        args=(system_template, treatment_transcripts),
    )
    training["text"] = training.apply(
        format_user_prompt, axis=1, args=(user_template, target_outcome)
    )
    training = training.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "train.csv")
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    training[["text"]].to_csv(csv_path, index=False)
    save_jsonl(training[["text"]], jsonl_path, text_column="text")
    return jsonl_path


if __name__ == "__main__":
    raise NotImplementedError(
        "Wire this entry point to config.yaml to assemble data/finetuning/train.jsonl."
    )
