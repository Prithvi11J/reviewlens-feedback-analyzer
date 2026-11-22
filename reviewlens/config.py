"""Central place for settings and for creating the LLM.

Every other module asks this file for the model, so switching models
(or providers) later means changing one function, not the whole codebase.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Read key=value pairs from .env into environment variables.
# Real environment variables win, which is how cloud hosts inject secrets.
load_dotenv()

DEFAULT_MODEL = "gpt-5.4-mini"


class ConfigError(RuntimeError):
    """Raised when required settings are missing."""


def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key == "sk-your-key-here":
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env "
            "and paste your OpenAI key into it."
        )
    return key


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_llm() -> ChatOpenAI:
    """Return a ready-to-use chat model.

    max_retries handles temporary API problems (rate limits, timeouts)
    with automatic backoff, which matters when we send hundreds of reviews.
    """
    return ChatOpenAI(
        model=get_model_name(),
        api_key=get_api_key(),
        max_retries=3,
        timeout=60,
    )
