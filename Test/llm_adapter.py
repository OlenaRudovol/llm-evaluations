"""
Unified adapter interface for different LLM providers.

This module provides a consistent interface for interacting with different LLM services
(OpenAI, Ollama, etc.), eliminating code duplication across evaluators.
"""

from abc import ABC, abstractmethod
from typing import Optional
import ollama
from openai import OpenAI


class LLMAdapter(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def ask(self, question: str) -> str:
        """Submit a question to the LLM and return the response."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model being used."""
        pass


class OllamaAdapter(LLMAdapter):
    """Adapter for Ollama local LLM server."""

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        """
        Initialize Ollama adapter.

        Args:
            model: Model name (e.g., 'gemma3:1b')
            host: Ollama server host (default: localhost)
            port: Ollama server port (default: 11434)
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

    @property
    def model_name(self) -> str:
        return self._model


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API."""

    def __init__(self, model: str, api_key: str):
        """
        Initialize OpenAI adapter.

        Args:
            model: Model name (e.g., 'gpt-4o-mini')
            api_key: OpenAI API key
        """
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
            raise RuntimeError(f"OpenAI API error: Invalid response format - {e}")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {type(e).__name__} - {e}")

    @property
    def model_name(self) -> str:
        return self._model
