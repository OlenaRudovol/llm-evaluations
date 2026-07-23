"""Adapter for OpenAI API."""

from openai import OpenAI

from .base import LLMAdapter


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""

    def __init__(self, model: str, api_key: str):
        """
        Initialize OpenAI adapter.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            api_key: OpenAI API key

        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        self._model = model
        self.client = OpenAI(api_key=api_key, timeout=30.0)

    def ask(self, question: str) -> str:
        """Query the OpenAI model."""
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"OpenAI API error: Invalid response format - {e}") from e
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {type(e).__name__} - {e}") from e

    @property
    def model_name(self) -> str:
        return self._model