"""Adapter for Ollama local LLM server."""

import ollama

from .base import LLMAdapter


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama local LLM server."""

    def __init__(self, model: str):
        """
        Initialize Ollama adapter.

        Args:
            model: Model name (e.g., 'gemma3:1b')
        """
        self._model = model

    def ask(self, question: str) -> str:
        """Query the Ollama model."""
        try:
            response = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": question}],
                options={"temperature": 0.0},  # for stable responses
            )
            return response["message"]["content"].strip()
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Ollama API error: Invalid response format - {e}")
        except ConnectionError as e:
            raise RuntimeError(f"Ollama API error: Connection failed - {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {type(e).__name__} - {e}")

    @property
    def model_name(self) -> str:
        return self._model