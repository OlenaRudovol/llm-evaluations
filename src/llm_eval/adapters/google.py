"""Adapter for Google Gemini API (AI Studio)."""

import google.generativeai as genai

from .base import LLMAdapter


class GoogleAdapter(LLMAdapter):
    """Adapter for Google Gemini API via AI Studio."""

    def __init__(self, model: str, api_key: str):
        self._model = model
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def ask(self, question: str) -> str:
        """Query the Google Gemini model."""
        try:
            response = self.client.generate_content(question)
            return response.text.strip()
        except AttributeError as e:
            raise RuntimeError(f"Google API error: Invalid response format - {e}")
        except Exception as e:
            raise RuntimeError(f"Google API error: {type(e).__name__} - {e}")

    @property
    def model_name(self) -> str:
        return self._model
