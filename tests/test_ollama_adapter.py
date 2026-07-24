from unittest.mock import patch

import ollama
import pytest

from llm_eval.adapters.ollama import OllamaAdapter

MODEL = "gemma3:1b"


class TestOllamaAdapter:
    @pytest.fixture
    def adapter(self):
        return OllamaAdapter(model=MODEL)

    def test_model_name(self, adapter):
        assert adapter.model_name == MODEL

    @patch("llm_eval.adapters.ollama.ollama.chat")
    def test_ask_returns_stripped_content(self, mock_chat, adapter):
        mock_chat.return_value = {"message": {"content": "  Price, Quality  \n"}}

        answer = adapter.ask("What aspects?")

        assert answer == "Price, Quality"
        mock_chat.assert_called_once_with(
            model=MODEL,
            messages=[{"role": "user", "content": "What aspects?"}],
            options={"temperature": 0.0},
        )

    @patch("llm_eval.adapters.ollama.ollama.chat")
    def test_ask_wraps_malformed_response(self, mock_chat, adapter):
        mock_chat.return_value = {"message": {}}  # missing "content" -> KeyError

        with pytest.raises(RuntimeError, match="Invalid response format"):
            adapter.ask("What aspects?")

    @patch("llm_eval.adapters.ollama.ollama.chat")
    def test_ask_wraps_connection_error(self, mock_chat, adapter):
        mock_chat.side_effect = ConnectionError("server unreachable")

        with pytest.raises(RuntimeError, match="Connection failed"):
            adapter.ask("What aspects?")

    @patch("llm_eval.adapters.ollama.ollama.chat")
    def test_ask_wraps_ollama_response_error(self, mock_chat, adapter):
        mock_chat.side_effect = ollama.ResponseError("model not found", status_code=404)

        with pytest.raises(RuntimeError, match="Ollama API error"):
            adapter.ask("What aspects?")

    @patch("llm_eval.adapters.ollama.ollama.chat")
    def test_ask_preserves_exception_chain(self, mock_chat, adapter):
        original = ConnectionError("server unreachable")
        mock_chat.side_effect = original

        with pytest.raises(RuntimeError) as exc_info:
            adapter.ask("What aspects?")

        assert exc_info.value.__cause__ is original
