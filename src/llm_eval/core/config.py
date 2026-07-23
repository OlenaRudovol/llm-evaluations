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
        
        # OpenAI settings
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Ollama settings
        self._ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:1b")

        # Validate OpenAI config if provider is OpenAI
        if self._llm_provider == "openai" and not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
    
    @property
    def llm_provider(self) -> str:
        """Get cached LLM provider ('ollama' or 'openai')."""
        return self._llm_provider

    @property
    def openai_api_key(self) -> Optional[str]:
        """Get cached OpenAI API key."""
        return self._openai_api_key

    @property
    def openai_model(self) -> str:
        """Get cached OpenAI model (default: gpt-4o-mini)."""
        return self._openai_model

    @property
    def ollama_model(self) -> str:
        """Get cached Ollama model (default: gemma3:1b)."""
        return self._ollama_model


# Global config instance
config = Config()