"""Configuration for LLM evaluation, read once from environment variables at import time.

A module already behaves like a cached singleton in Python, so there is no
need to wrap these values in a class with private attributes and read-only
properties — `config.llm_provider` works the same either way.
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv not installed, continue without .env loading
    pass

# LLM provider settings
llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

# OpenAI settings
openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama settings
ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:1b")

# Groq settings
groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Evaluation settings
use_llm_judge: bool = os.getenv("USE_LLM_JUDGE", "false").lower() == "true"
judge_model: Optional[str] = os.getenv("JUDGE_MODEL")  # unset -> judge with the same model being tested
