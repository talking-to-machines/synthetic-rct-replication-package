"""Factory helpers for the API clients used in the archived scripts.

The archive uses two providers:
- OpenAI (batch inference for GPT models)
- Together AI (fine-tuning and dedicated inference for Llama/Qwen)
"""

from openai import OpenAI
from together import Together

from src.models.registry import ModelEntry, get_model
from src.utils.config import OPENAI_API_KEY, TOGETHER_API_KEY


def get_openai_client() -> OpenAI:
    """Return an authenticated OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)


def get_together_client() -> Together:
    """Return an authenticated Together AI client."""
    return Together(api_key=TOGETHER_API_KEY)


def get_client_for(model_key: str):
    """Return the appropriate client for a registered model key."""
    entry: ModelEntry = get_model(model_key)
    if entry.provider == "openai":
        return get_openai_client()
    if entry.provider == "together":
        return get_together_client()
    raise ValueError(f"Unknown provider {entry.provider!r} for model {model_key!r}.")
