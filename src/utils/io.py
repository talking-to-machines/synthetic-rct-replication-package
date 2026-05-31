import json
import os
from pathlib import Path
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict.

    Args:
        path: Path to a `.yaml`/`.yml` file (e.g. `config.yaml`).

    Returns:
        The parsed YAML document as a Python dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write an iterable of dict records to a JSONL file.

    Each record is serialised on its own line via `json.dumps`. Parent
    directories are created if they do not exist.

    Args:
        records: Records to serialise (one JSON object per line).
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_records_json(records: list[dict], path: str | Path) -> None:
    """Write records as a single indented JSON array.

    Used for per-source training files that humans inspect directly. Parent
    directories are created if they do not exist.

    Args:
        records: Records to serialise as a JSON array.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_records_csv(
    records: list[dict],
    path: str | Path,
    treatments: list[str] | None = None,
) -> None:
    """Write {"messages": [...]} records to CSV.

    Emits one row per record with one column per message role (`system`,
    `user`, `assistant`) plus an optional `treatment` column when treatment
    labels are supplied. Used for per-source test (holdout) files so they can
    be inspected and consumed by downstream evaluation code.

    Args:
        records: Records each containing a `messages` list of role/content dicts.
        path: Destination CSV path.
        treatments: Optional treatment labels parallel to `records`. When
            provided, a leading `treatment` column is included.

    Raises:
        ValueError: If `treatments` is provided but its length does not match
            `records`.
    """
    if treatments is not None and len(treatments) != len(records):
        raise ValueError(
            f"treatments length {len(treatments)} does not match records length "
            f"{len(records)}."
        )

    rows: list[dict] = []
    for i, rec in enumerate(records):
        row: dict = {}
        if treatments is not None:
            row["treatment"] = treatments[i]
        for msg in rec.get("messages", []):
            row[msg["role"]] = msg["content"]
        rows.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
