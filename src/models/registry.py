"""Model identifiers used across fine-tuning and inference.

Values are lifted from the archived notebooks/scripts to keep the live
pipeline consistent with what produced the original results.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    key: str
    base_id: str
    provider: str
    finetuned_id: str | None = None


REGISTRY: dict[str, ModelEntry] = {
    "llama_8b": ModelEntry(
        key="llama_8b",
        base_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        provider="together",
    ),
    "llama_70b": ModelEntry(
        key="llama_70b",
        base_id="meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        provider="together",
        finetuned_id="iamraymondlow/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo-c6090e90",
    ),
    "llama_8b_turbo": ModelEntry(
        key="llama_8b_turbo",
        base_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        provider="together",
    ),
    "gpt_4o": ModelEntry(
        key="gpt_4o",
        base_id="gpt-4o-2024-08-06",
        provider="openai",
    ),
}


def get_model(key: str) -> ModelEntry:
    """Return the registry entry for a model key."""
    if key not in REGISTRY:
        raise KeyError(f"Model {key!r} not registered. Known: {sorted(REGISTRY)}")
    return REGISTRY[key]
