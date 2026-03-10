"""
Unified adapter interface for different LLM providers.

This module provides a consistent interface for interacting with different LLM services
(OpenAI, Ollama, etc.), eliminating code duplication across evaluators.
"""

from abc import ABC, abstractmethod


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