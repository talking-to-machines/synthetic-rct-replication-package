"""Build the fine-tuning corpus from survey + RCT sources.

Reads:  source data files + prompt JSONs declared in config.yaml
Writes: data/processed/{kind}/{id}/{id}_train.json (per-source training set)
        data/processed/{kind}/{id}/{id}_test.csv   (per-source holdout, when split)
        {output_jsonl}                              (combined JSONL for Together AI)

Originally extracted from the archived notebook `prepare_fine_tuning_data.ipynb`.
"""

from pathlib import Path
from src.data.cleaning import split_indices
from src.data.formatting import build_finetune_source_records
from src.utils.io import write_jsonl, write_records_csv, write_records_json
from src.utils.seed import RANDOM_STATE, set_seed

set_seed(RANDOM_STATE)


def build_corpus(
    cfg: dict,
    output_jsonl: Path,
    train_test_split: bool = True,
) -> int:
    """Build the combined fine-tuning corpus from configured sources.

    Iterates over every survey/RCT listed under `cfg["finetuning"]`, builds
    per-subject `{"messages": [...]}` records, optionally splits each source
    into train/test using `finetuning.test_fraction` and `finetuning.seed`,
    writes per-source files, and concatenates the training portions into a
    single JSONL ready to upload to Together AI.

    Per-source files written under `data/processed/{kind}/{id}/`:
        - `{id}_train.json` (always; JSON array)
        - `{id}_test.csv`   (only when `train_test_split=True`; CSV with
          `treatment`, `system`, `user`, `assistant` columns when the source
          has a treatment column, else just the message-role columns)

    When a source has a treatment column, the train/test split is stratified
    on treatment (equal arm balance across train and test). Otherwise the
    split is a uniform random shuffle.

    Args:
        cfg: Parsed `config.yaml` dict. Must contain a `finetuning` block
            (with optional `surveys`/`rcts` lists, `test_fraction`, `seed`)
            and matching `surveys`/`rcts` entries with `data_file`,
            `prompt_file`, and `outcome` fields.
        output_jsonl: Path for the combined training corpus written as JSONL.
        train_test_split: When True (default) hold out a per-source test
            split; when False, use every record for training and skip test
            files entirely.

    Returns:
        Number of training examples written to `output_jsonl`.

    Raises:
        KeyError: If a source listed under `finetuning` is missing from the
            corresponding `surveys`/`rcts` block in config.
    """
    finetuning = cfg.get("finetuning", {})
    test_fraction = finetuning.get("test_fraction", 0.2)
    seed = finetuning.get("seed", 42)

    sources: list[tuple[str, str]] = []
    for source_id in finetuning.get("surveys") or []:
        sources.append(("surveys", source_id))
    for source_id in finetuning.get("rcts") or []:
        sources.append(("rcts", source_id))

    combined_train: list[dict] = []
    for kind, source_id in sources:
        if kind not in cfg or source_id not in cfg[kind]:
            raise KeyError(
                f"{kind}/{source_id} listed in finetuning but missing from "
                f"cfg[{kind!r}]. Add a {kind}.{source_id} block to config.yaml."
            )
        records, treatments = build_finetune_source_records(
            source_id, cfg[kind][source_id], kind
        )

        out_dir = Path("data/processed") / kind / source_id
        train_path = out_dir / f"{source_id}_train.json"

        if train_test_split:
            train_idx, test_idx = split_indices(
                len(records), test_fraction, seed, strata=treatments
            )
            train = [records[i] for i in train_idx]
            test = [records[i] for i in test_idx]
            test_treatments = (
                [treatments[i] for i in test_idx] if treatments is not None else None
            )

            test_path = out_dir / f"{source_id}_test.csv"
            write_records_json(train, train_path)
            write_records_csv(test, test_path, treatments=test_treatments)
            print(f"  {kind}/{source_id}: {len(train)} train, {len(test)} test")
        else:
            train = records
            write_records_json(train, train_path)
            print(f"  {kind}/{source_id}: {len(train)} train (no holdout)")

        combined_train.extend(train)

    output_jsonl = Path(output_jsonl)
    write_jsonl(combined_train, output_jsonl)
    return len(combined_train)
