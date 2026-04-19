"""Clean and split raw human data (surveys and RCTs).

Reads:  data/human/
Writes: data/processed/

Duch et al. 2023 ships already-split files in the archive
(`duch_et_al_2023_training_1538.csv` and `duch_et_al_2023_holdout_1537.csv`),
so the function below reproduces that convention: a training split feeds
fine-tuning, a holdout split feeds inference.
"""

import os

import pandas as pd

from src.data.cleaning import load_data
from src.utils.seed import RANDOM_STATE


def split_train_holdout(
    data: pd.DataFrame,
    holdout_size: int | float,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/holdout split by count or fraction."""
    shuffled = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if isinstance(holdout_size, float):
        n_holdout = int(round(len(shuffled) * holdout_size))
    else:
        n_holdout = int(holdout_size)
    holdout = shuffled.iloc[:n_holdout].reset_index(drop=True)
    train = shuffled.iloc[n_holdout:].reset_index(drop=True)
    return train, holdout


def preprocess_rct(
    raw_csv: str,
    output_dir: str,
    holdout_size: int | float,
    drop_first_row: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[str, str, str]:
    """Clean one RCT CSV and write clean/train/test CSVs under ``output_dir``."""
    data = load_data(raw_csv, drop_first_row=drop_first_row)
    os.makedirs(output_dir, exist_ok=True)

    rct_name = os.path.basename(output_dir)
    clean_path = os.path.join(output_dir, f"{rct_name}_clean.csv")
    train_path = os.path.join(output_dir, f"{rct_name}_train.csv")
    test_path = os.path.join(output_dir, f"{rct_name}_test.csv")

    data.to_csv(clean_path, index=False)
    train, holdout = split_train_holdout(data, holdout_size, random_state=random_state)
    train.to_csv(train_path, index=False)
    holdout.to_csv(test_path, index=False)
    return clean_path, train_path, test_path


if __name__ == "__main__":
    raise NotImplementedError(
        "Wire this entry point to config.yaml to preprocess each survey and RCT."
    )
