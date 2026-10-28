"""Shared Groq client configuration for AI Resume + Job Matcher."""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# The previous model was shut down for developer/free usage in August 2026.
DEFAULT_MODEL = "openai/gpt-oss-20b"
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_client() -> OpenAI:
    """Create a client on demand, with a clear error if the key is missing."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to a local .env file in the "
            "project root (GROQ_API_KEY=your_key) and restart Streamlit."
        )
    return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)
