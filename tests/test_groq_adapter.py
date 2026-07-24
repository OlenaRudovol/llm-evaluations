from unittest.mock import MagicMock

import pytest

from llm_eval.adapters.groq import GROQ_BASE_URL, GroqAdapter

MODEL = "llama-3.1-8b-instant"


class TestGroqAdapter:
    @pytest.fixture
    def adapter(self):
        return GroqAdapter(model=MODEL, api_key="gsk-test")

    @pytest.mark.parametrize("api_key", ["", None])
    def test_missing_api_key_raises(self, api_key):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqAdapter(model=MODEL, api_key=api_key)

    def test_model_name(self, adapter):
        assert adapter.model_name == MODEL

    def test_uses_groq_base_url(self, adapter):
        assert str(adapter.client.base_url) == GROQ_BASE_URL + "/"

    def test_ask_returns_stripped_content(self, adapter):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Price, Quality  \n"
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        answer = adapter.ask("What aspects?")

        assert answer == "Price, Quality"
        adapter.client.chat.completions.create.assert_called_once_with(
            model=MODEL,
            messages=[{"role": "user", "content": "What aspects?"}],
            temperature=0.0,
        )

    def test_ask_wraps_malformed_response(self, adapter):
        mock_response = MagicMock()
        mock_response.choices = []  # choices[0] -> IndexError
        adapter.client.chat.completions.create = MagicMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match="Invalid response format"):
            adapter.ask("What aspects?")

    def test_ask_wraps_api_errors(self, adapter):
        adapter.client.chat.completions.create = MagicMock(side_effect=TimeoutError("timed out"))

        with pytest.raises(RuntimeError, match="Groq API error"):
            adapter.ask("What aspects?")

    def test_ask_preserves_exception_chain(self, adapter):
        original = TimeoutError("timed out")
        adapter.client.chat.completions.create = MagicMock(side_effect=original)

        with pytest.raises(RuntimeError) as exc_info:
            adapter.ask("What aspects?")

        assert exc_info.value.__cause__ is original
