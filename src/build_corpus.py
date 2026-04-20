"""Build the fine-tuning corpus from survey training files and RCT training splits.

Reads:  data/processed/*/train.jsonl
Writes: data/finetuning/train.jsonl

Extracted from the archived notebook `prepare_fine_tuning_data.ipynb`.
"""

import os

import pandas as pd

from src.data.formatting import (
    format_duch_2023_system_prompt,
    format_duch_2023_user_prompt,
    generate_demographic_prompt,
)
from src.utils.io import save_jsonl
from src.utils.seed import RANDOM_STATE, set_seed

set_seed(RANDOM_STATE)


def build_duch_2023_corpus(
    source_csv: str,
    output_dir: str,
    demographic_questions: list,
    target_outcome: str,
) -> str:
    """Construct the JSONL fine-tuning corpus for the Duch et al. 2023 RCT."""
    data = pd.read_csv(source_csv, header=1)
    data[target_outcome] = data[target_outcome].replace("NA", None)

    training = data[demographic_questions + [target_outcome, "treatment"]].copy()
    training["demographic_prompt"] = training.apply(
        generate_demographic_prompt, axis=1, args=([target_outcome, "treatment"],)
    )
    training["system_prompt"] = training.apply(format_duch_2023_system_prompt, axis=1)
    training["text"] = training.apply(
        format_duch_2023_user_prompt, axis=1, args=(target_outcome,)
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
