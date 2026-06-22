"""Adapter for Anthropic Claude API."""

import anthropic

from .base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, model: str, api_key: str):
        """
        Initialize Anthropic adapter.

        Args:
            model: Model name (e.g., 'claude-3-5-haiku-20241022')
            api_key: Anthropic API key
        """
        self._model = model
        self.client = anthropic.Anthropic(api_key=api_key)

    def ask(self, question: str) -> str:
        """Query the Anthropic Claude model."""
        try:
            response = self.client.messages.create(
                model=self._model,
                max_tokens=256,
                messages=[{"role": "user", "content": question}],
            )
            return response.content[0].text.strip()
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"Anthropic API error: Invalid response format - {e}")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {type(e).__name__} - {e}")

    @property
    def model_name(self) -> str:
        return self._model
