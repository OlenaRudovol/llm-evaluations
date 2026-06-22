import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    # python-dotenv not installed, continue without .env loading
    pass


class Config:
    """Configuration management for LLM evaluation. Values are cached at initialization."""

    def __init__(self):
        """Initialize config by reading environment variables once."""
        # LLM Provider settings
        self._llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        
        # Anthropic settings
        self._anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        # OpenAI settings
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Google settings
        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        self._google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

        # Ollama settings
        self._ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
        self._ollama_host = os.getenv("OLLAMA_HOST")
        self._ollama_port = os.getenv("OLLAMA_PORT")

        if self._llm_provider == "anthropic" and not self._anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider")
        if self._llm_provider == "openai" and not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        if self._llm_provider == "google" and not self._google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google provider")
    
    @property
    def llm_provider(self) -> str:
        return self._llm_provider

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self._anthropic_api_key

    @property
    def anthropic_model(self) -> str:
        return self._anthropic_model

    @property
    def openai_api_key(self) -> Optional[str]:
        return self._openai_api_key

    @property
    def openai_model(self) -> str:
        return self._openai_model

    @property
    def google_api_key(self) -> Optional[str]:
        return self._google_api_key

    @property
    def google_model(self) -> str:
        return self._google_model

    @property
    def ollama_model(self) -> str:
        """Get cached Ollama model (default: gemma3:1b)."""
        return self._ollama_model

    @property
    def ollama_host(self) -> Optional[str]:
        """Get cached Ollama host (optional)."""
        return self._ollama_host

    @property
    def ollama_port(self) -> Optional[str]:
        """Get cached Ollama port (optional)."""
        return self._ollama_port


# Global config instance
config = Config()