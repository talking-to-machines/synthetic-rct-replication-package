import json
import os
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_jsonl(df: pd.DataFrame, path: str, text_column: str = "text") -> None:
    """Save rows as JSONL lines: {"messages": [...]}.

    Expects df[text_column] to be a JSON string or a Python list of message dicts.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            val = row.get(text_column)
            if isinstance(val, str):
                try:
                    messages = json.loads(val)
                except Exception:
                    messages = [{"role": "assistant", "content": val}]
            elif isinstance(val, list):
                messages = val
            else:
                messages = [{"role": "assistant", "content": str(val)}]

            out = {"messages": messages}
            json.dump(out, f, ensure_ascii=False)
            f.write("\n")


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write an iterable of dict records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def concatenate_jsonls(paths: list[Path], output_jsonl: str | Path) -> int:
    """Concatenate JSONL files into a single file. Returns the line count written."""
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for src_path in paths:
            with open(src_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    if not line.endswith("\n"):
                        line += "\n"
                    out_f.write(line)
                    total += 1
    return total
