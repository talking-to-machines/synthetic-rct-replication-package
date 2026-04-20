import json
import os

import pandas as pd


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
