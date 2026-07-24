"""Adapter for Groq's OpenAI-compatible API.

Groq hosts open models (Llama, Gemma, Mixtral, ...) behind an API that speaks
the same protocol as OpenAI's, so this adapter is a near-twin of
`OpenAIAdapter` — only the base URL, API key, and error labels differ.
"""

from openai import OpenAI

from .base import LLMAdapter

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqAdapter(LLMAdapter):
    """Adapter for Groq (OpenAI-compatible endpoint)."""

    def __init__(self, model: str, api_key: str):
        """
        Initialize Groq adapter.

        Args:
            model: Model name (e.g., 'llama-3.1-8b-instant')
            api_key: Groq API key

        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq provider")
        self._model = model
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL, timeout=30.0)

    def ask(self, question: str) -> str:
        """Query the Groq model."""
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"Groq API error: Invalid response format - {e}") from e
        except Exception as e:
            raise RuntimeError(f"Groq API error: {type(e).__name__} - {e}") from e

    @property
    def model_name(self) -> str:
        return self._model
