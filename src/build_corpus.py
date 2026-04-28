"""Build fine-tuning corpora from survey/RCT splits and instruction-tune datasets.

Reads:  data/processed/*/train.jsonl, instruction-tune dataset JSON files
Writes: data/finetuning/train.jsonl, instruction-tune dataset JSONL files

Originally extracted from the archived notebook `prepare_fine_tuning_data.ipynb`.
"""

import json
import os
from pathlib import Path

import pandas as pd

from src.data.cleaning import split_records
from src.data.formatting import (
    build_finetune_source_records,
    format_instruction_messages,
    format_system_prompt,
    format_user_prompt,
    generate_profile_prompt,
)
from src.utils.io import concatenate_jsonls, save_jsonl, write_jsonl
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


def build_finetune_corpus(cfg: dict, output_jsonl: Path) -> int:
    """Process every source in `finetuning`, split, write per-source JSONLs,
    and concatenate the train splits into `output_jsonl`."""
    finetuning = cfg.get("finetuning", {})
    test_fraction = finetuning.get("test_fraction", 0.2)
    seed = finetuning.get("seed", 42)

    sources: list[tuple[str, str]] = []
    for source_id in finetuning.get("surveys") or []:
        sources.append(("surveys", source_id))
    for source_id in finetuning.get("rcts") or []:
        sources.append(("rcts", source_id))

    train_paths: list[Path] = []
    for kind, source_id in sources:
        if kind not in cfg or source_id not in cfg[kind]:
            raise KeyError(
                f"{kind}/{source_id} listed in finetuning but missing from "
                f"cfg[{kind!r}]. Add a {kind}.{source_id} block to config.yaml."
            )
        records = build_finetune_source_records(source_id, cfg[kind][source_id], kind)
        train, test = split_records(records, test_fraction, seed)

        out_dir = Path("data/processed") / kind / source_id
        train_path = out_dir / f"{source_id}_train.jsonl"
        test_path = out_dir / f"{source_id}_test.jsonl"
        write_jsonl(train, train_path)
        write_jsonl(test, test_path)
        print(f"  {kind}/{source_id}: {len(train)} train, {len(test)} test")
        train_paths.append(train_path)

    return concatenate_jsonls(train_paths, output_jsonl)


def build_instruction_corpus(
    src_json: Path, dst_jsonl: Path, system_prompt: str
) -> int:
    """Convert an Alpaca-style JSON dataset into a chat-messages JSONL."""
    with open(src_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    dst_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            messages = format_instruction_messages(r, system_prompt)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    raise NotImplementedError(
        "Wire this entry point to config.yaml to assemble data/finetuning/train.jsonl."
    )
